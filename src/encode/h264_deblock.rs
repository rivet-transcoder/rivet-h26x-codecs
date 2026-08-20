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
//! each macroblock's QPs for the thresholds. [`FilterMb`] carries exactly
//! that, filled by the picture writers from their decisions; get any of it
//! wrong — the classic being a stale or empty `nz_mask` — and the filter
//! runs with different strengths than the decoder's, which SELF reports on
//! the first picture that codes a residual next to an edge.
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
use crate::encode::h264_syntax::{Geometry, Recon};
use crate::h264::deblock::{DeblockParams, deblock_mb_rows};
use crate::h264::frame::{BlockMotion, Frame, Mv, PARITY_FRAME};
use crate::h264::mb::{MbKind, PicInfo, chroma_qp};

/// What the deblocking filter needs to know about one coded macroblock,
/// recorded by the picture writers in raster order as they code.
#[derive(Clone, Copy)]
pub struct FilterMb {
    /// The macroblock kind, in the decoder's own vocabulary — what bS
    /// derivation switches on. This encoder produces `I4x4`, `I16x16`,
    /// `Inter16x16` and `PSkip`.
    pub kind: MbKind,
    /// The "has coefficients" mask of the 4x4 luma blocks (raster), the
    /// same derivation `derive()` stores in `MbInfo::nz_mask`
    /// (src/h264/recon.rs): one bit per block with a nonzero count. Zero
    /// for a skipped macroblock. See [`nz_mask_of`].
    pub nz_mask: u16,
    /// The list-0 vector of an inter macroblock — for `PSkip` the
    /// *derived* skip vector, because that is the motion a decoder stores
    /// and compares across edges. Ignored for intra kinds.
    pub mv: Mv,
}

/// The nonzero mask of [`FilterMb`] from a decision's per-block counts:
/// the `derive()` formula for a 4x4-transform macroblock
/// (src/h264/recon.rs — this encoder has no 8x8 transform to spread).
pub fn nz_mask_of(nz: &[u8; 16]) -> u16 {
    let mut mask = 0u16;
    for (b, &n) in nz.iter().enumerate() {
        mask |= ((n != 0) as u16) << b;
    }
    mask
}

/// Run the decoder's deblocking filter over a coded picture's
/// reconstruction, in place.
///
/// `qp` is the slice QP, which is every macroblock's QP while nothing
/// writes a nonzero `mb_qp_delta`; the chroma QPs derive from it with the
/// zero offsets the PPS declares. The slice parameters are the ones the
/// header writes when its `deblock` flag is true — filter on, both
/// offsets zero — and the two must stay in step: the writers call this
/// unconditionally for exactly that reason.
pub fn deblock_recon(dsp: &H264Dsp<u8>, g: &Geometry, qp: u8, mbs: &[FilterMb], rec: &mut [Recon]) {
    let (mbw, mbh) = (g.mbs_wide as usize, g.mbs_high as usize);
    debug_assert_eq!(mbs.len(), mbw * mbh, "one FilterMb per macroblock");

    // A decoder-shaped frame around the encoder's planes. The geometry is
    // identical by construction — both sides build planes of the coded
    // size with the decoder's borders — so the planes move, not copy.
    let mut frame = Frame::<u8>::empty();
    frame.mb_width = mbw;
    frame.mb_height = mbh;
    frame.chroma = g.chroma;
    frame.bit_depth = 8;
    let n = mbw * mbh;
    frame.motion = [
        vec![BlockMotion::default(); n * 16],
        vec![BlockMotion::default(); n * 16],
    ];
    std::mem::swap(&mut frame.y, &mut rec[0]);
    if rec.len() > 1 {
        std::mem::swap(&mut frame.cb, &mut rec[1]);
        std::mem::swap(&mut frame.cr, &mut rec[2]);
    }

    let mut info = PicInfo::new(mbw, mbh);
    let qpc = chroma_qp(qp as i32, 0, 0) as i8;
    for (addr, m) in mbs.iter().enumerate() {
        let mi = &mut info.mbs[addr];
        mi.kind = m.kind;
        mi.decoded = true;
        mi.slice = 0;
        mi.qp = qp as i8;
        mi.qpc = [qpc, qpc];
        mi.nz_mask = m.nz_mask;
        // `part_edges` stays [0, 0] — the derivation's statement that one
        // partition covers the macroblock, which is true of everything
        // this encoder codes — and `transform_8x8` stays false.
        if !m.kind.is_intra() {
            // One motion for all sixteen blocks, reference index 0 of the
            // one reference every inter macroblock uses: `ref_id` is an
            // identity the filter only ever compares for equality, so a
            // constant says "the same picture" exactly as the decoder's
            // real ids do when there is one reference.
            let bm = BlockMotion { mv: m.mv, ref_idx: 0, ref_parity: PARITY_FRAME, ref_id: 0 };
            for blk in 0..16 {
                frame.motion[0][addr * 16 + blk] = bm;
            }
        }
    }

    deblock_mb_rows(dsp, &mut frame, &info, &[DeblockParams::default()], 0, mbh);

    std::mem::swap(&mut frame.y, &mut rec[0]);
    if rec.len() > 1 {
        std::mem::swap(&mut frame.cb, &mut rec[1]);
        std::mem::swap(&mut frame.cr, &mut rec[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask formula agrees with the decoder's own in `derive()`: bit
    /// `b` set exactly when block `b` has coefficients.
    #[test]
    fn the_nz_mask_matches_the_derivations_formula() {
        let mut nz = [0u8; 16];
        assert_eq!(nz_mask_of(&nz), 0);
        nz[0] = 3;
        nz[7] = 1;
        nz[15] = 16;
        assert_eq!(nz_mask_of(&nz), 1 | (1 << 7) | (1 << 15));
        let full = [1u8; 16];
        assert_eq!(nz_mask_of(&full), 0xffff);
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
        let mbs = vec![
            FilterMb { kind: MbKind::PSkip, nz_mask: 0, mv: Mv::new(6, -2) };
            9
        ];
        deblock_recon(&dsp, &g, 26, &mbs, &mut rec);
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
        let mbs = vec![
            FilterMb { kind: MbKind::I4x4, nz_mask: 0xffff, mv: Mv::ZERO };
            9
        ];
        deblock_recon(&dsp, &g, 26, &mbs, &mut rec);
        assert_ne!(rec[0].data, before, "intra edges at QP 26 must filter");
    }
}
