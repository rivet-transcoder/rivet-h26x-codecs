//! Deblocking the encoder's reconstruction — with the decoder's filter.
//!
//! The slice headers this encoder writes used to disable the loop filter,
//! because the encoder did not run one and SELF demands that the encoder's
//! reconstruction equal what a decoder produces byte for byte. This module
//! buys the filter back the only way that property survives: by running
//! the *decoder's* `deblock_mb_rows` (src/h264/deblock.rs; conformance-
//! proven, clause 8.7) over the encoder's own reconstruction, fed the
//! same per-macroblock state a decoder derives while parsing.
//!
//! That state is the whole game. The filter's boundary strengths read the
//! macroblock kind (intra edges are bS 3/4 unconditionally), the
//! nonzero-coefficient mask (bS 2), and the motion field (bS 1/0), plus
//! each macroblock's QPs for the thresholds — and it reads all of it out
//! of the decoder's own `PicInfo` and per-4x4 motion, which the picture
//! walks now fill as they code ([`PicMotion`]). Nothing is synthesised
//! here any more: there is no encoder-side summary of a macroblock left to
//! be subtly different from what a decoder derived.
//!
//! Ordering mirrors the decoder's: it filters a row only after the row
//! below is reconstructed (intra prediction reads unfiltered neighbours)
//! and extends borders after filtering. The encoder has the simpler shape
//! of the same discipline — reconstruct the whole picture, then filter it
//! whole, then crop and store — one `deblock_mb_rows` call over every
//! row, after the last macroblock and before the reconstruction becomes a
//! reference.
//!
//! The picture is wrapped in a decoder `Frame` by *moving* the planes in
//! and out (`Vec` swaps, no copies); the per-picture `PicInfo` and motion
//! allocations are small beside the planes and can join a pool the day a
//! profile says so.

use crate::dsp::h264::H264Dsp;
use crate::encode::h264_pic::PicMotion;
use crate::encode::h264_syntax::{Geometry, Recon};
use crate::h264::deblock::{DeblockParams, deblock_mb_rows};
use crate::h264::frame::Frame;

/// Run the decoder's deblocking filter over a coded picture's
/// reconstruction, in place.
///
/// Everything the filter reads is already in `pm`: the per-macroblock
/// `MbInfo` the walks filled as they coded, and the per-4x4 motion in the
/// decoder's own layout. The planes and the motion are *moved* into a
/// decoder frame and back out (`Vec` swaps, no copies) — the geometry is
/// identical by construction, since both sides build planes of the coded
/// size with the decoder's borders.
///
/// The slice parameters are the ones the header writes when its `deblock`
/// flag is true — filter on, both offsets zero — and the two must stay in
/// step: the writers call this unconditionally for exactly that reason.
pub fn deblock_recon(dsp: &H264Dsp<u8>, g: &Geometry, pm: &mut PicMotion, rec: &mut [Recon]) {
    let (mbw, mbh) = (g.mbs_wide as usize, g.mbs_high as usize);
    debug_assert_eq!(pm.info.mbs.len(), mbw * mbh, "one MbInfo per macroblock");

    let PicMotion { info, frame: src } = pm;
    let mut frame = Frame::<u8>::empty();
    frame.mb_width = mbw;
    frame.mb_height = mbh;
    frame.chroma = g.chroma;
    frame.bit_depth = 8;
    std::mem::swap(&mut frame.motion, &mut src.motion);
    std::mem::swap(&mut frame.mb_intra, &mut src.mb_intra);
    std::mem::swap(&mut frame.y, &mut rec[0]);
    if rec.len() > 1 {
        std::mem::swap(&mut frame.cb, &mut rec[1]);
        std::mem::swap(&mut frame.cr, &mut rec[2]);
    }

    deblock_mb_rows(dsp, &mut frame, info, &[DeblockParams::default()], 0, mbh);

    std::mem::swap(&mut frame.motion, &mut src.motion);
    std::mem::swap(&mut frame.mb_intra, &mut src.mb_intra);
    std::mem::swap(&mut frame.y, &mut rec[0]);
    if rec.len() > 1 {
        std::mem::swap(&mut frame.cb, &mut rec[1]);
        std::mem::swap(&mut frame.cr, &mut rec[2]);
    }
}

/// The nonzero mask of a macroblock from a decision's per-block counts:
/// `derive()`'s own formula (src/h264/recon.rs, where `MbInfo::nz_mask`
/// is built), including its 8x8 case.
///
/// A 4x4-transform macroblock sets one bit per block with coefficients.
/// An 8x8-transform one spreads each 8x8's answer over all four of its
/// 4x4s — quoting the decoder's constants rather than deriving them
/// again, since the point is to be the same mask and not merely an
/// equivalent one. That matters even though the caller's counts are
/// per-sub-scan: an 8x8 whose coefficients all land in one sub-scan is
/// coded for the whole 8x8, and the filter must see it that way.
pub fn nz_mask_of(nz: &[u8; 16], transform_8x8: bool) -> u16 {
    let mut mask = 0u16;
    for (b, &n) in nz.iter().enumerate() {
        mask |= ((n != 0) as u16) << b;
    }
    if !transform_8x8 {
        return mask;
    }
    let q = |bits: u16| -> u16 { if bits != 0 { 0x33 } else { 0 } };
    q(mask & 0x0033) | (q(mask & 0x00cc) << 2) | (q(mask & 0x3300) << 8) | (q(mask & 0xcc00) << 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::frame::{BlockMotion, Mv, PARITY_FRAME};
    use crate::h264::mb::{MbInfo, MbKind};

    /// A picture of `n` identical macroblocks, in the state the walks
    /// would have committed for them.
    fn uniform(n: usize, mbw: usize, kind: MbKind, nz_mask: u16, l0: Option<Mv>) -> PicMotion {
        let mut pm = PicMotion::new(mbw, n / mbw);
        let mut mot = [[BlockMotion::default(); 16]; 2];
        if let Some(mv) = l0 {
            mot[0] = [BlockMotion { mv, ref_idx: 0, ref_parity: PARITY_FRAME, ref_id: 1 }; 16];
        }
        let qpc = crate::h264::mb::chroma_qp(26, 0, 0) as i8;
        for addr in 0..n {
            pm.commit(
                addr,
                MbInfo {
                    kind,
                    decoded: true,
                    slice: 0,
                    qp: 26,
                    qpc: [qpc; 2],
                    nz_mask,
                    ..MbInfo::default()
                },
                &mot,
            );
        }
        pm
    }

    /// The mask formula agrees with the decoder's own in `derive()`: bit
    /// `b` set exactly when block `b` has coefficients.
    #[test]
    fn the_nz_mask_matches_the_derivations_formula() {
        let mut nz = [0u8; 16];
        assert_eq!(nz_mask_of(&nz, false), 0);
        nz[0] = 3;
        nz[7] = 1;
        nz[15] = 16;
        assert_eq!(nz_mask_of(&nz, false), 1 | (1 << 7) | (1 << 15));
        let full = [1u8; 16];
        assert_eq!(nz_mask_of(&full, false), 0xffff);
    }

    /// Under the 8x8 transform each 8x8's answer covers all four of its
    /// 4x4s — the decoder's `derive()` spread, checked against blocks
    /// picked so that a mask which merely *looked* right per-block would
    /// be wrong: one coefficient in one corner of an 8x8 lights the whole
    /// 8x8, and an empty 8x8 stays dark beside it.
    #[test]
    fn the_nz_mask_spreads_over_an_8x8() {
        let mut nz = [0u8; 16];
        // 8x8 block 0 (rasters 0,1,4,5): one coefficient, in its
        // bottom-right 4x4.
        nz[5] = 1;
        // 8x8 block 3 (rasters 10,11,14,15): one, in its top-left.
        nz[10] = 4;
        assert_eq!(nz_mask_of(&nz, true), 0x0033 | 0xcc00);
        // The same counts without the flag light only the two blocks.
        assert_eq!(nz_mask_of(&nz, false), (1 << 5) | (1 << 10));
        assert_eq!(nz_mask_of(&[0; 16], true), 0);
        assert_eq!(nz_mask_of(&[1; 16], true), 0xffff);
    }

    /// A picture of skipped macroblocks with one shared vector filters to
    /// itself: every edge has one partition, the same reference, matching
    /// motion and no coefficients — bS 0 across the board — so the filter
    /// must not touch a sample. This pins the plumbing (the frame
    /// wrapping, the motion fill, the info fill) rather than the filter,
    /// which is the decoder's and proven elsewhere.
    #[test]
    fn an_all_skip_picture_filters_to_itself() {
        use crate::encode::h264_syntax::recon_plane;
        use crate::h264::frame::{CHROMA_PAD, LUMA_PAD};
        let g = Geometry::new(&crate::encode::Config {
            width: 48,
            height: 48,
            ..crate::encode::Config::default()
        });
        let dsp = H264Dsp::<u8>::new(crate::dsp::Cpu::SCALAR);
        let mut rec = vec![
            recon_plane(48, 48, LUMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
        ];
        let mut seed = 11u64;
        for p in rec.iter_mut() {
            for v in p.data.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = (seed >> 33) as u8;
            }
        }
        let before: Vec<Vec<u8>> = rec.iter().map(|p| p.data.clone()).collect();
        let mut pm = uniform(9, 3, MbKind::PSkip, 0, Some(Mv::new(6, -2)));
        deblock_recon(&dsp, &g, &mut pm, &mut rec);
        for (p, b) in rec.iter().zip(&before) {
            assert_eq!(&p.data, b, "bS 0 everywhere must leave every sample alone");
        }
    }

    /// The same picture declared intra must change: intra edges are bS 4/3
    /// whatever the coefficients, and random content at QP 26 is far above
    /// the thresholds. This is the cheap tripwire for the filter being
    /// silently disconnected — an encoder that stops filtering would pass
    /// every "nothing changed" test and fail only in the gate.
    #[test]
    fn an_intra_picture_with_edges_actually_filters() {
        use crate::encode::h264_syntax::recon_plane;
        use crate::h264::frame::{CHROMA_PAD, LUMA_PAD};
        let g = Geometry::new(&crate::encode::Config {
            width: 48,
            height: 48,
            ..crate::encode::Config::default()
        });
        let dsp = H264Dsp::<u8>::new(crate::dsp::Cpu::SCALAR);
        let mut rec = vec![
            recon_plane(48, 48, LUMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
        ];
        // Blocky content with *small* steps: a checkerboard of flat 4x4
        // blocks six levels apart. Small matters — the filter smooths only
        // steps below its alpha threshold (about fifteen at QP 26) and
        // deliberately leaves larger ones alone as real edges, so big
        // steps here would assert that nothing happened and prove nothing.
        let o = rec[0].origin();
        let stride = rec[0].stride;
        for y in 0..48 {
            for x in 0..48 {
                let v = 100 + ((x / 4 + y / 4) % 2) as u8 * 6;
                rec[0].data[o + y * stride + x] = v;
            }
        }
        let before = rec[0].data.clone();
        let mut pm = uniform(9, 3, MbKind::I4x4, 0xffff, None);
        deblock_recon(&dsp, &g, &mut pm, &mut rec);
        assert_ne!(rec[0].data, before, "intra edges at QP 26 must filter");
    }
}
