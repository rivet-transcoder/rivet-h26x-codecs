//! Encoder-side deblocking for H.265 — the decoder's own filter, driven
//! over the encoder's reconstruction.
//!
//! Today `write_pps` sets `pps_deblocking_filter_disabled_flag` purely
//! because the encoder did not filter what it reconstructs: a decoder
//! filters *its* reconstruction, so an unfiltered encoder-held picture
//! would fail SELF on every coded edge while CROSS stayed green — the
//! decoders outvoting the encoder two to nothing, as the first H.265
//! stream demonstrated. Declaring the filter off made the streams honest
//! at a real quality cost. This module buys that back the way everything
//! here is built: the filter is `hevc::deblock::deblock_rows`,
//! the decoder's own, conformance-proven; what the encoder contributes is
//! the *state* the filter reads, derived from its [`CuDecision`]s by
//! mirroring exactly the bookkeeping the decoder's `transform_unit` and
//! `coding_unit` perform — TB edge flags, per-4x4 cbf, QP, intra flags,
//! transquant-bypass exemption.
//!
//! **Order matters and is the caller's contract**: intra prediction reads
//! the *unfiltered* reconstruction, so [`deblock_picture`] must run after
//! every CTU of the picture has been coded, and the filtered planes are
//! then what a decoder outputs — the reconstruction SELF compares
//! against. Filtering earlier would desync every prediction that reads a
//! filtered neighbour.
//!
//! What the state derivation is worth testing is spelled by the boundary
//! strengths (8.7.2.4, `boundary_strengths` in the decoder): in an
//! all-intra picture every flagged edge has bS 2 — the intra case
//! short-circuits *before* `cbf_luma` or motion is consulted. So the
//! live inputs here are the edge flags, the intra pred_mode, the QP and
//! the bypass exemption, and the tests break when any of them is
//! mis-derived; the cbf mirror is faithfully kept but is inert until the
//! encoder produces inter CUs, and no all-intra test can catch dropping
//! it. Said here so nobody writes that vacuous test believing it bites.
//!
//! **SAO stays off**, and stays named: it is the other in-loop filter, a
//! separate lever with its own SPS flag (`write_sps` keeps it 0), its own
//! parameter search and its own wiring day. Nothing in this module
//! prepares or precludes it.

use std::sync::Arc;

use crate::encode::h265_intra::{luma_tbs, z_within_ctb, CuDecision, IntraCtx, IntraPicture};
use crate::hevc::deblock::{deblock_rows, DeblockScratch};
use crate::hevc::pic::{Geometry, PicInfo};
use crate::hevc::pps::Pps;
use crate::sample::Sample;

/// Deblock the encoder's reconstruction in place, exactly as a decoder
/// will deblock its own.
///
/// Call once per picture, **after** the last CTU has been coded (the
/// module docs say why the order is load-bearing), with the decisions in
/// the raster order `code_ctu` produced them. Transquant-bypass CUs are
/// exempt sample-for-sample, as the standard requires — a lossless
/// picture passes through unchanged — and a decision's transform tree
/// gates exactly which interior edges exist, so an unsplit CU's inside
/// never moves. Chroma follows the picture's format through the
/// decoder's own per-format grids. The slice filter parameters are the
/// ones our headers declare: zero beta/tc offsets, zero chroma QP
/// offsets, one slice, one tile.
pub fn deblock_picture<S: Sample>(ctx: &IntraCtx<'_, S>, pic: &mut IntraPicture<S>, decisions: &[CuDecision]) {
    let h4 = pic.recon.height / 4;
    let info = build_info(ctx, pic, decisions);
    // The one thing `deblock_rows` reads from the PPS is
    // `loop_filter_across_tiles`, but it wants the real struct: parse the
    // PPS our own writer emits, which is also the header a decoder of
    // this stream will hold.
    let pps = Pps::parse(&crate::nal::unescape_rbsp(&crate::encode::h265_syntax::write_pps(ctx.qp.clamp(0, 51) as u8, ctx.bypass, true)))
        .expect("the encoder's own PPS parses");
    let mut scratch = DeblockScratch::default();
    deblock_rows(ctx.dsp, &mut scratch, &mut pic.recon, &info, &pps, ctx.bit_depth, ctx.bit_depth, 0, h4);
}

/// Build the per-picture side data the decoder's filter reads, from the
/// decisions — the encoder's copy of what `coding_unit` and
/// `transform_unit` record as they parse:
///
/// - **TB edges**: `edges |= 1` down a transform block's left column and
///   `|= 4` along its top row, per luma TB of every decision's tree
///   ([`luma_tbs`] is the one enumeration of that tree). Prediction-unit
///   edge bits are inter machinery; an intra CU's prediction blocks are
///   never finer than its transform blocks, so TB flags are the complete
///   set here.
/// - **cbf_luma** filled over each TB that carries levels, **pred_mode**
///   1 (intra) and **qp_y** over each CU, **filter_exempt** 3 over
///   bypass CUs — the value `coding_unit` records, and what keeps the
///   filter off lossless samples.
/// - **One slice, one tile**: every CTB assigned slice 0 with default
///   filter parameters (`SliceFilterParams::default()` is exactly the
///   all-zero-offset, filtering-enabled set our slice header implies).
///   `PicInfo::new` marks CTBs as belonging to *no* slice, which the
///   filter treats as "do not touch" — a decoded picture earns filtering
///   CTB by CTB, and an encoder that forgot this fill would silently
///   ship an unfiltered picture; the step test exists to catch exactly
///   that silence.
///
/// The geometry is assembled directly for the one-slice raster walk this
/// encoder does — identity CTB scan, single tile, and `min_tb_addr_zs`
/// from [`z_within_ctb`], the interleave the availability tests hold
/// against `Geometry`'s own construction.
fn build_info<S: Sample>(ctx: &IntraCtx<'_, S>, pic: &IntraPicture<S>, decisions: &[CuDecision]) -> PicInfo {
    let (w, h) = (pic.recon.width, pic.recon.height);
    let log2 = pic.log2_cu;
    let n = 1usize << log2;
    let (w4, h4) = (w / 4, h / 4);
    let (wc, hc) = (w >> log2, h >> log2);
    let shift = log2 - 2;
    let mut min_tb_addr_zs = vec![0u32; w4 * h4];
    for y4 in 0..h4 {
        for x4 in 0..w4 {
            let ctb = (y4 >> shift) * wc + (x4 >> shift);
            min_tb_addr_zs[y4 * w4 + x4] = ((ctb as u32) << (2 * shift)) + z_within_ctb(log2, x4 * 4, y4 * 4);
        }
    }
    let nc = wc * hc;
    let geo = Geometry {
        w4,
        h4,
        wc,
        hc,
        log2_ctb: log2,
        ctb_tile: vec![0; nc],
        min_tb_addr_zs,
        ctb_rs_to_ts: (0..nc as u32).collect(),
        ctb_ts_to_rs: (0..nc as u32).collect(),
        tile_id_ts: vec![0; nc],
    };
    let mut info = PicInfo::new(Arc::new(geo));
    info.ctb_slice.fill(0);
    info.ctb_slice_addr.fill(0);
    // `slices[0]` stays `SliceFilterParams::default()`: filtering enabled,
    // zero offsets — the parameters our headers declare.

    let mut di = 0;
    for cy in 0..hc {
        for cx in 0..wc {
            let d = &decisions[di];
            di += 1;
            let (x0, y0) = (cx * n, cy * n);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 1u8);
            PicInfo::fill4(&mut info.qp_y, w4, x0, y0, n, n, ctx.qp as i8);
            if d.bypass {
                PicInfo::fill4(&mut info.filter_exempt, w4, x0, y0, n, n, 3u8);
            }
            for (tx, ty, tlog2, cbf) in luma_tbs(d, x0, y0) {
                let tn = 1usize << tlog2;
                // The mirror of transform_unit's edge bookkeeping: bit 1
                // down the TB's left column, bit 4 along its top row.
                for yy in (ty..ty + tn).step_by(4) {
                    let i = (yy >> 2) * w4 + (tx >> 2);
                    info.edges[i] |= 1;
                }
                for xx in (tx..tx + tn).step_by(4) {
                    let i = (ty >> 2) * w4 + (xx >> 2);
                    info.edges[i] |= 4;
                }
                if cbf {
                    PicInfo::fill4(&mut info.cbf_luma, w4, tx, ty, tn, tn, 1u8);
                }
            }
        }
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hevc::HevcDsp;
    use crate::dsp::hevc_enc::HevcEncDsp;
    use crate::dsp::distortion::DistortionDsp;
    use crate::dsp::Cpu;
    use crate::picture::ChromaFormat;

    struct Kit {
        dsp: HevcDsp<u8>,
        enc: HevcEncDsp,
        dist: DistortionDsp<u8>,
    }

    impl Kit {
        fn new() -> Self {
            Kit { dsp: HevcDsp::new(Cpu::SCALAR), enc: HevcEncDsp::scalar(), dist: DistortionDsp::scalar() }
        }
        fn ctx(&self, qp: i32, bypass: bool) -> IntraCtx<'_, u8> {
            IntraCtx {
                dsp: &self.dsp,
                enc: &self.enc,
                dist: &self.dist,
                qp,
                bit_depth: 8,
                strong_smoothing: false,
                bypass,
            }
        }
    }

    /// A synthetic "coded picture": the reconstruction planes written by
    /// hand, and one trivial decision per CTU describing the transform
    /// structure the state builder should see. What the filter does is
    /// the decoder's business (conformance-proven); what these tests pin
    /// is that the *derived state* makes it act exactly where a decoder
    /// of this stream would.
    fn synthetic(w: usize, h: usize, log2_cu: u32, chroma: ChromaFormat, luma: &dyn Fn(usize, usize) -> u8) -> (IntraPicture<u8>, Vec<CuDecision>) {
        let mut pic = IntraPicture::<u8>::new_with_chroma(w, h, log2_cu, 8, chroma);
        let off = pic.recon.y.origin();
        let stride = pic.recon.y.stride;
        for y in 0..h {
            for x in 0..w {
                pic.recon.y.data[off + y * stride + x] = luma(x, y);
            }
        }
        for plane in [&mut pic.recon.cb, &mut pic.recon.cr] {
            let o = plane.origin();
            let s = plane.stride;
            for y in 0..plane.height {
                for x in 0..plane.width {
                    plane.data[o + y * s + x] = 128;
                }
            }
        }
        let n = 1usize << log2_cu;
        let count = (w >> log2_cu) * (h >> log2_cu);
        let _ = n;
        let decisions = vec![CuDecision { log2_cu, ..CuDecision::default() }; count];
        (pic, decisions)
    }

    fn luma_snapshot(pic: &IntraPicture<u8>) -> Vec<u8> {
        let off = pic.recon.y.origin();
        let stride = pic.recon.y.stride;
        let (w, h) = (pic.recon.width, pic.recon.height);
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend_from_slice(&pic.recon.y.data[off + y * stride..off + y * stride + w]);
        }
        out
    }

    /// A flat picture must pass through unchanged: every CU edge carries
    /// bS 2 (all-intra), so the filter *runs* everywhere the state says
    /// an edge is — and changes nothing, because there is no step to
    /// smooth. Catches spurious filtering (wrong offsets, wrong planes)
    /// rather than missing filtering, which the step test covers.
    #[test]
    fn a_flat_reconstruction_is_left_alone() {
        let kit = Kit::new();
        let ctx = kit.ctx(30, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let (mut pic, decisions) = synthetic(w, h, log2_cu, ChromaFormat::Yuv420, &|_, _| 128);
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            assert_eq!(before, luma_snapshot(&pic), "log2_cu={log2_cu}");
            for plane in [&pic.recon.cb, &pic.recon.cr] {
                let o = plane.origin();
                for y in 0..plane.height {
                    for x in 0..plane.width {
                        assert_eq!(plane.data[o + y * plane.stride + x], 128);
                    }
                }
            }
        }
    }

    /// The h264-me lesson, H.265 edition: a small step at a CU boundary
    /// must actually be smoothed — a test that only shows large steps
    /// surviving proves nothing, because leaving real edges alone is what
    /// the filter is *for*. Two CTUs, luma 120 left and 126 right: the
    /// boundary columns must move toward each other and the interior must
    /// not move at all. Then the same picture with a 120/250 cliff must
    /// pass through untouched.
    #[test]
    fn a_small_step_is_smoothed_and_a_cliff_is_kept() {
        let kit = Kit::new();
        let ctx = kit.ctx(32, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, n);
            let (mut pic, decisions) = synthetic(w, h, log2_cu, ChromaFormat::Yuv420, &|x, _| if x < n { 120 } else { 126 });
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            let after = luma_snapshot(&pic);
            assert_ne!(before, after, "log2_cu={log2_cu}: the small step was not filtered");
            for y in 0..h {
                // The boundary pair moved toward each other...
                assert!(after[y * w + n - 1] > 120, "left edge column did not rise at row {y}");
                assert!(after[y * w + n] < 126, "right edge column did not fall at row {y}");
                // ...and nothing farther than the filter's three-sample
                // reach moved at all.
                for x in 0..w {
                    let dist = (x as i32 - n as i32).min(n as i32 - 1 - x as i32).abs();
                    if !(x as i32 >= n as i32 - 3 && (x as i32) < n as i32 + 3) {
                        assert_eq!(after[y * w + x], before[y * w + x], "({x},{y}) is {dist} from the edge and moved");
                    }
                }
            }

            let (mut pic, decisions) = synthetic(w, h, log2_cu, ChromaFormat::Yuv420, &|x, _| if x < n { 120 } else { 250 });
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            assert_eq!(before, luma_snapshot(&pic), "log2_cu={log2_cu}: a real edge was smoothed away");
        }
    }

    /// The transform tree gates interior edges: the same mid-CU step is
    /// untouchable while the CU is one TU (no edge exists there) and
    /// filtered the moment the decision splits — nothing changed but the
    /// *state*, which is precisely what this module derives. Dropping the
    /// edge bookkeeping, the pred_mode fill or the QP fill all fail here
    /// (bS gate, bS value, and beta/tc respectively).
    #[test]
    fn the_transform_tree_gates_interior_edges() {
        let kit = Kit::new();
        let ctx = kit.ctx(32, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let step = &|_: usize, y: usize| if y < n / 2 { 120u8 } else { 126 };

            let (mut pic, decisions) = synthetic(w, h, log2_cu, ChromaFormat::Yuv420, step);
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            assert_eq!(before, luma_snapshot(&pic), "log2_cu={log2_cu}: an unsplit CU has no interior edge to filter");

            let (mut pic, mut decisions) = synthetic(w, h, log2_cu, ChromaFormat::Yuv420, step);
            decisions[0].split_tu = true;
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            let after = luma_snapshot(&pic);
            assert_ne!(before, after, "log2_cu={log2_cu}: the split's interior edge was not filtered");
            let mid = n / 2;
            for x in 0..w {
                assert!(after[(mid - 1) * w + x] > 120, "top side did not rise at column {x}");
                assert!(after[mid * w + x] < 126, "bottom side did not fall at column {x}");
            }
        }
    }

    /// Transquant-bypass CUs are exempt sample for sample: the same step
    /// that the filter smooths in a lossy CU passes through a lossless
    /// one untouched — anything else would un-lose lossless output.
    #[test]
    fn bypass_reconstruction_is_exempt() {
        let kit = Kit::new();
        let ctx = kit.ctx(32, true);
        let (w, h) = (32, 16);
        let (mut pic, mut decisions) = synthetic(w, h, 4, ChromaFormat::Yuv420, &|x, _| if x < 16 { 120 } else { 126 });
        for d in &mut decisions {
            d.bypass = true;
        }
        let before = luma_snapshot(&pic);
        deblock_picture(&ctx, &mut pic, &decisions);
        assert_eq!(before, luma_snapshot(&pic), "a lossless picture was filtered");
    }

    /// A really coded picture, end to end: code a gentle ramp at a high
    /// QP — the classic blocky reconstruction, each TB quantising to its
    /// own DC so the picture comes out as a staircase — then filter, and
    /// require both that something changed (a run that alters nothing
    /// means the state never reached the filter; the first synthetic
    /// content here was uniform noise, which the filter's local-flatness
    /// test correctly refuses to touch, a lesson worth this comment) and
    /// that every change stays within the filter's three-sample reach of
    /// an 8-aligned grid line, which is as far as any legal deblocking
    /// can act.
    #[test]
    fn a_coded_picture_filters_near_the_grid_only() {
        let kit = Kit::new();
        let ctx = kit.ctx(40, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let mut src = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    src[y * w + x] = (60 + (x + y) / 2).min(230) as u8;
                }
            }
            let cbs = vec![128u8; w * h / 4];
            let crs = vec![128u8; w * h / 4];
            let mut pic = IntraPicture::<u8>::new(w, h, log2_cu, 8);
            pic.split_depth = 2;
            let mut decisions = Vec::new();
            for cy in 0..h / n {
                for cx in 0..w / n {
                    decisions.push(pic.code_ctu(&ctx, cx, cy, &src, w, &cbs, &crs, w / 2));
                }
            }
            let before = luma_snapshot(&pic);
            deblock_picture(&ctx, &mut pic, &decisions);
            let after = luma_snapshot(&pic);
            assert_ne!(before, after, "log2_cu={log2_cu}: filtering a coded noisy picture changed nothing");
            // Within three samples of an interior 8-grid line, on either
            // axis: the farthest any legal luma deblocking reaches.
            let near = |c: usize, limit: usize| -> bool {
                let m = c % 8;
                (m >= 5 && c - m + 8 < limit) || (m <= 2 && c >= 8)
            };
            for y in 0..h {
                for x in 0..w {
                    if before[y * w + x] != after[y * w + x] {
                        assert!(near(x, w) || near(y, h), "({x},{y}) changed but is not within 3 of an interior 8-grid line");
                    }
                }
            }
        }
    }
}
