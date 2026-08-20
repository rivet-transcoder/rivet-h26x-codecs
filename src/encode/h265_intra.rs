//! Intra mode decision and reconstruction for H.265.
//!
//! This is the *deciding* half of coding a CTU: which prediction modes,
//! what the quantised coefficients are, and what the reconstruction looks
//! like. Turning that into bits belongs to the CABAC coding-tree writer,
//! and the seam between the two is [`CuDecision`] — data, not a shared
//! file, so the decision cannot drift from the writer that serialises it.
//!
//! The two rules of [`super::h264_intra`] hold here unchanged:
//!
//! **The prediction is the decoder's own.** `hevc::intra::predict` is
//! conformance-proven against the JCT-VC suites and runs here as-is,
//! reading its reference samples out of the reconstruction plane with the
//! same availability, substitution, smoothing and boundary-filter rules it
//! applies when decoding. An encoder that predicted even slightly
//! differently would desync, and the desync would surface as drift in a
//! SELF check far from the cause.
//!
//! **So is the reconstruction.** After quantising, the levels go back
//! through the decoder's own `scale_coefficients` and inverse transforms
//! (`HevcDsp::{idct, idst4}`, `add_residual`), so the reconstruction this
//! block's neighbours predict from is what a decoder will hold — by
//! construction, not by hope.
//!
//! Candidates are scored by SATD on the prediction, for the reason the
//! H.264 side gives: it prices a residual the way the transform that will
//! code it does. All 35 modes are searched exhaustively — no MPM-style
//! shortlist yet; at these block counts the full search is cheap and it
//! removes a heuristic from the path while the module is young.
//!
//! # Simplifications — read before extending
//!
//! This is a first cut with a deliberately fixed geometry, and every one of
//! these is a simplification to be lifted, not a design position:
//!
//! - **One CU per CTU, no quadtree.** The CU size *is* the CTB size,
//!   `log2_cu` in 3..=5. Pictures must be whole multiples of it.
//! - **Fixed partitioning per size.** `log2_cu` 4 or 5 codes `PART_2Nx2N`
//!   with a single luma TU the size of the CU; `log2_cu` 3 codes
//!   `PART_NxN` with four 4x4 luma TUs — the standard's only route to the
//!   4x4 DST, which keeps that path honest. No `split_transform_flag`
//!   choices are searched.
//! - **4:2:0 only** (`chroma_array_type` 1), one chroma TU per component
//!   at half the CU size.
//! - **One slice, one tile, raster CTU order.** Availability reduces to
//!   picture geometry plus z-scan order, mirrored from the decoder.
//! - **Flat scaling lists, no transform skip, no RDPCM, no rotation** —
//!   matching what `write_sps` / `write_pps` currently emit (no scaling
//!   lists, `transform_skip_enabled_flag` 0, no range extensions).
//! - **Fixed QP** — no `cu_qp_delta`, so a decision carries no QP field.
//! - **Lossless is a whole-picture switch** (`IntraCtx::bypass`): every CU
//!   gets `cu_transquant_bypass_flag`, the residual is carried raw, and the
//!   PPS must set `transquant_bypass_enabled_flag` to match.
//!
//! Deblocking and SAO are downstream concerns: H.265 intra prediction reads
//! the *pre-filter* reconstruction, which is exactly what this module
//! holds, so the in-loop filters can be applied behind it later without
//! touching anything here.

use crate::dsp::distortion::DistortionDsp;
use crate::dsp::hevc::HevcDsp;
use crate::dsp::hevc_enc::{HevcEncDsp, qbits, quant_offset, quant_scale};
use crate::hevc::ctu::chroma_qp;
use crate::hevc::frame::{Frame, Plane16};
use crate::hevc::intra::{IntraScratch, RefAvail, predict};
use crate::hevc::pic::PicInfo;
use crate::hevc::residual::{ScalingSource, scale_coefficients};
use crate::picture::ChromaFormat;
use crate::sample::Sample;

/// How a CTU (here: one CU) was coded, in the form the coding-tree writer
/// needs. One is produced per CTU and meant to be consumed immediately —
/// the coefficient arrays are a few kilobytes, so buffering a slice's
/// worth would be megabytes of nothing.
///
/// What is *not* here, and why:
///
/// - **No scan order.** Coefficients are raster within each TU. H.265's
///   scan is mode-dependent (7.4.9.11: 4x4 TUs, and 8x8 luma TUs, of an
///   intra CU scan vertically for modes 6..=14 and horizontally for
///   22..=30, diagonally otherwise), so the writer derives `scanIdx` from
///   the modes stored here rather than this side pre-scanning.
/// - **No QP delta.** QP is fixed per picture (no `cu_qp_delta_enabled`).
/// - **No `transform_skip_flag` / RDPCM / rotation fields.** Never emitted
///   under the current SPS/PPS; add them beside `bypass` when they are.
/// - **No split flags.** The coding-tree geometry is fixed (CU == CTB;
///   `PART_NxN` implies the forced transform split, `PART_2Nx2N` a single
///   TU), so the writer derives `split_cu_flag` / `split_transform_flag`
///   from `log2_cu` and `nxn` against the SPS it wrote.
#[derive(Clone)]
pub struct CuDecision {
    /// log2 of the CU (== CTB) size this decision describes: 3, 4 or 5.
    pub log2_cu: u32,
    /// `part_mode`: true is `PART_NxN` (four 4x4 luma PBs/TUs, only ever
    /// produced at `log2_cu == 3`), false is `PART_2Nx2N` (one luma PB and
    /// one luma TU the size of the CU).
    pub nxn: bool,
    /// `cu_transquant_bypass_flag`. When set, every `luma` / `chroma`
    /// entry is a raw spatial residual sample (source minus prediction,
    /// raster), not a transform level.
    pub bypass: bool,
    /// Chosen luma prediction modes (0 planar, 1 DC, 2..=34 angular), one
    /// per prediction block in z-order. `PART_2Nx2N` has one prediction
    /// block; its mode is replicated across all four entries so a reader
    /// need not branch.
    pub luma_modes: [u8; 4],
    /// The same choices as the syntax carries them, one per prediction
    /// block in z-order (only `[0]` is meaningful for `PART_2Nx2N`). The
    /// MPM list is derived here, where the neighbour state lives, so two
    /// writers cannot derive it differently.
    pub luma_syntax: [LumaModeSyntax; 4],
    /// `intra_chroma_pred_mode` as coded: 0..=3 pick planar/26/10/1 (with
    /// 34 substituted where the pick equals the luma mode), 4 derives from
    /// luma. One per CU — 4:4:4 `PART_NxN` would need four.
    pub chroma_syntax: u8,
    /// The derived `IntraPredModeC` — what `chroma_syntax` decodes to
    /// (8.4.3), stored so the writer's mode-dependent scan for 4x4 chroma
    /// TUs does not re-derive it.
    pub chroma_mode: u8,
    /// `cbf_luma` per luma TU in z-order: whether the TU carries any
    /// nonzero level. `PART_2Nx2N` uses only `[0]`; the rest stay false.
    pub cbf_luma: [bool; 4],
    /// `cbf_cb`, `cbf_cr` for the single chroma TU of each component.
    pub cbf_chroma: [bool; 2],
    /// Quantised luma levels (or raw residual when `bypass`). One TU of
    /// `n*n` entries at `[0..n*n]` for `PART_2Nx2N` (`n = 1 << log2_cu`);
    /// four 4x4 TUs at `[16*i..16*i + 16]` in z-order for `PART_NxN`.
    /// Raster order within each TU. Entries beyond the described TUs are
    /// zero and meaningless.
    pub luma: [i16; 1024],
    /// Quantised chroma levels per component (`[0]` Cb, `[1]` Cr), one TU
    /// of `nc*nc` entries at `[0..nc*nc]`, `nc = 1 << (log2_cu - 1)`,
    /// raster within the TU. Raw residual when `bypass`.
    pub chroma: [[i16; 256]; 2],
}

impl Default for CuDecision {
    fn default() -> Self {
        CuDecision {
            log2_cu: 0,
            nxn: false,
            bypass: false,
            luma_modes: [1; 4],
            luma_syntax: [LumaModeSyntax::default(); 4],
            chroma_syntax: 4,
            chroma_mode: 1,
            cbf_luma: [false; 4],
            cbf_chroma: [false; 2],
            luma: [0; 1024],
            chroma: [[0; 256]; 2],
        }
    }
}

/// A luma mode as the syntax carries it: either an index into the MPM
/// candidate list, or the remainder after the (sorted) candidates are
/// removed from the mode numbering. The decoder's inverse is in
/// `hevc::ctu::coding_unit`: sort the three candidates ascending, then
/// bump `rem` past each candidate it reaches.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct LumaModeSyntax {
    /// `prev_intra_luma_pred_flag`.
    pub prev_flag: bool,
    /// `mpm_idx`, 0..=2, when `prev_flag`.
    pub mpm_idx: u8,
    /// `rem_intra_luma_pred_mode`, 0..=31, when `!prev_flag`.
    pub rem: u8,
}

/// Everything the mode decision needs that does not change per CTU.
pub struct IntraCtx<'a, S: Sample> {
    /// Decode-side kernels: the inverse transforms and residual add the
    /// reconstruction goes through.
    pub dsp: &'a HevcDsp<S>,
    /// Forward transforms and quantisation.
    pub enc: &'a HevcEncDsp,
    /// Distortion metrics, for scoring candidates.
    pub dist: &'a DistortionDsp<S>,
    /// The signalled luma QP, 0..=51. The bit-depth offset
    /// (`6 * (BitDepth - 8)`) and the chroma mapping (Table 8-10) are
    /// applied internally, the way the decoder applies them.
    pub qp: i32,
    /// Sample bit depth, luma and chroma alike (8..=14).
    pub bit_depth: u32,
    /// `strong_intra_smoothing_enabled_flag` of the SPS this stream will
    /// carry — `write_sps` currently writes false.
    pub strong_smoothing: bool,
    /// Code every CU with `cu_transquant_bypass_flag` (lossless). The PPS
    /// must set `transquant_bypass_enabled_flag` to match.
    pub bypass: bool,
}

/// The fixed picture geometry the availability rules need, copied out of
/// [`IntraPicture`] so the free functions below can borrow the planes and
/// the mode grid independently.
#[derive(Clone, Copy)]
struct Geo {
    /// log2 of the CU == CTB size.
    log2_cu: u32,
    /// CTUs per row.
    wc: usize,
    /// 4x4 blocks per row.
    w4: usize,
    /// Luma picture width and height in samples.
    width: usize,
    height: usize,
}

/// Per-picture state of the all-intra walk: the reconstruction the
/// predictions read, the chosen-mode grid the MPM derivation reads, and
/// the reference-sample scratch. The caller walks CTUs in raster order and
/// calls [`IntraPicture::code_ctu`] for each; anything else is a geometry
/// this module does not model yet.
pub struct IntraPicture<S: Sample> {
    /// The reconstruction, decoder-identical by construction. Prediction
    /// reads it; the in-loop filters (not this module's concern) would run
    /// over a copy of it afterwards.
    pub recon: Frame<S>,
    /// log2 of the fixed CU size, 3..=5.
    pub log2_cu: u32,
    /// Chosen luma mode per 4x4 block — the encoder's copy of the
    /// decoder's `PicInfo::intra_mode`, and like it filled per prediction
    /// block as modes are chosen so the MPM derivation for later blocks
    /// reads it through the same rules.
    modes: Vec<u8>,
    /// Reference-sample scratch, reused across blocks exactly as the
    /// decoder reuses its own.
    scratch: IntraScratch,
    geo: Geo,
}

impl<S: Sample> IntraPicture<S> {
    /// State for a picture of `width` by `height` luma samples, both
    /// multiples of the CU size — the fixed-geometry simplification above.
    pub fn new(width: usize, height: usize, log2_cu: u32, bit_depth: u32) -> Self {
        assert!((3..=5).contains(&log2_cu), "log2_cu {log2_cu} outside 3..=5");
        let n = 1usize << log2_cu;
        assert!(width.is_multiple_of(n) && height.is_multiple_of(n), "{width}x{height} is not a whole number of {n}x{n} CTUs");
        let w4 = width / 4;
        let h4 = height / 4;
        IntraPicture {
            recon: Frame::new(width, height, ChromaFormat::Yuv420, bit_depth),
            log2_cu,
            // 1 (DC) everywhere, as the decoder initialises intra_mode; the
            // availability test keeps uncoded entries from ever being read.
            modes: vec![1; w4 * h4],
            scratch: IntraScratch::default(),
            geo: Geo { log2_cu, wc: width >> log2_cu, w4, width, height },
        }
    }

    /// Decide and code the CTU at `(cu_x, cu_y)` (in CTU units), leaving
    /// its reconstruction in `recon`. The sources are whole planes with
    /// their strides; CTUs must arrive in raster order, because the
    /// availability rules assume everything before the current one in that
    /// order is reconstructed.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu(
        &mut self,
        ctx: &IntraCtx<'_, S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        let geo = self.geo;
        let n = 1usize << geo.log2_cu;
        let (x0, y0) = (cu_x * n, cu_y * n);
        let mut out = CuDecision { log2_cu: geo.log2_cu, bypass: ctx.bypass, ..CuDecision::default() };

        // Luma: PART_NxN with four 4x4 DST TUs at the smallest size,
        // PART_2Nx2N with one CU-sized TU otherwise (see the module docs).
        let IntraPicture { recon, modes, scratch, .. } = self;
        if geo.log2_cu == 3 {
            out.nxn = true;
            for pb in 0..4 {
                // z-order within the CU, which is decode order: each block
                // predicts from the reconstruction of those before it.
                let (px, py) = (x0 + (pb & 1) * 4, y0 + (pb >> 1) * 4);
                fill_ref_avail(geo, &mut scratch.avail, px, py, 4, 1, 1);
                let cands = mpm_candidates(geo, modes, px, py);
                let soff = py * y_stride + px;
                let (mode, cbf) = decide_luma_tu(
                    ctx,
                    &mut recon.y,
                    scratch,
                    px,
                    py,
                    2,
                    &src_y[soff..],
                    y_stride,
                    cands,
                    &mut out.luma[pb * 16..pb * 16 + 16],
                );
                out.luma_modes[pb] = mode;
                out.luma_syntax[pb] = as_syntax(mode, cands);
                out.cbf_luma[pb] = cbf;
                // The decoder records each PU's mode as it derives it, so
                // the next PU's MPM list sees this one; mirror that.
                PicInfo::fill4(modes, geo.w4, px, py, 4, 4, mode);
            }
        } else {
            fill_ref_avail(geo, &mut scratch.avail, x0, y0, n, 1, 1);
            let cands = mpm_candidates(geo, modes, x0, y0);
            let soff = y0 * y_stride + x0;
            let (mode, cbf) = decide_luma_tu(
                ctx,
                &mut recon.y,
                scratch,
                x0,
                y0,
                geo.log2_cu,
                &src_y[soff..],
                y_stride,
                cands,
                &mut out.luma[..n * n],
            );
            out.luma_modes = [mode; 4];
            out.luma_syntax[0] = as_syntax(mode, cands);
            out.cbf_luma[0] = cbf;
            PicInfo::fill4(modes, geo.w4, x0, y0, n, n, mode);
        }

        // Chroma: one TU per component at half the CU size, mode from the
        // standard candidate set against the first luma mode. Its reference
        // samples all lie outside the CU, so unlike the decoder's TU walk
        // (which predicts chroma after the last luma TU) the order against
        // luma does not matter.
        let nc = n / 2;
        let (cx, cy) = (x0 / 2, y0 / 2);
        // Availability is derived in luma coordinates with the 4:2:0
        // subsampling, as intra_predict_block does; one derivation serves
        // both components because their geometry is identical.
        fill_ref_avail(geo, &mut scratch.avail, x0, y0, nc, 2, 2);
        let coff = cy * c_stride + cx;
        decide_chroma(
            ctx,
            &mut recon.cb,
            &mut recon.cr,
            scratch,
            cx,
            cy,
            geo.log2_cu - 1,
            out.luma_modes[0],
            &src_cb[coff..],
            &src_cr[coff..],
            c_stride,
            &mut out,
        );
        out
    }

    /// The MPM candidate list for the prediction block at luma position
    /// `(xp, yp)` — exposed for a replaying test or writer that needs to
    /// re-derive what [`LumaModeSyntax`] indexes into.
    pub fn mpm_list(&self, xp: usize, yp: usize) -> [u32; 3] {
        mpm_candidates(self.geo, &self.modes, xp, yp)
    }
}

/// z-scan address of the 4x4 block holding luma sample `(x, y)` within its
/// CTB — the within-CTB part of `Geometry::min_tb_addr_zs` (6.5.2),
/// computed by the same bit interleave that table is built from: bit `i`
/// of the 4x4 x-coordinate contributes `4^i`, of the y-coordinate
/// `2 * 4^i`.
fn z_within_ctb(log2_ctb: u32, x: usize, y: usize) -> u32 {
    let mask = (1usize << log2_ctb) - 1;
    let x4 = (x & mask) >> 2;
    let y4 = (y & mask) >> 2;
    let mut v = 0u32;
    for i in 0..log2_ctb - 2 {
        let m = 1usize << i;
        if x4 & m != 0 {
            v += (m * m) as u32;
        }
        if y4 & m != 0 {
            v += (2 * m * m) as u32;
        }
    }
    v
}

/// Whether the block holding luma sample `(xn, yn)` is reconstructed by
/// the time the block at `(xc, yc)` is coded: the z-scan availability test
/// of 6.4.1, as `PicInfo::available_at` performs it, reduced to the one
/// slice, one tile, raster-CTU-walk geometry this encoder has — CTB raster
/// order between CTBs, `min_tb_addr_zs` order within one (the decoder
/// tests `zs[neighbour] <= zs[current]`, and so does this).
fn decoded_before(geo: Geo, xc: usize, yc: usize, xn: i32, yn: i32) -> bool {
    if xn < 0 || yn < 0 || xn as usize >= geo.width || yn as usize >= geo.height {
        return false;
    }
    let (xn, yn) = (xn as usize, yn as usize);
    let ctb_c = (yc >> geo.log2_cu) * geo.wc + (xc >> geo.log2_cu);
    let ctb_n = (yn >> geo.log2_cu) * geo.wc + (xn >> geo.log2_cu);
    if ctb_n != ctb_c {
        return ctb_n < ctb_c;
    }
    z_within_ctb(geo.log2_cu, xn, yn) <= z_within_ctb(geo.log2_cu, xc, yc)
}

/// Fill the per-sample reference availability for a transform block at
/// luma position `(xl, yl)` covering `n` *component* samples with `(sw,
/// sh)` subsampling — the mirror of `intra_predict_block`'s `side` closure
/// in `hevc::ctu`, walking each edge one 4x4 luma block at a time (its
/// uniform-availability fast path is an optimisation of the same rule;
/// this takes the plain path unconditionally). Constrained intra
/// prediction is off, so availability is pure decode-order geometry.
fn fill_ref_avail(geo: Geo, avail: &mut RefAvail, xl: usize, yl: usize, n: usize, sw: usize, sh: usize) {
    avail.corner = decoded_before(geo, xl, yl, xl as i32 - 1, yl as i32 - 1);
    // Left samples y = 0..2n and top samples x = 0..2n, in component
    // coordinates; one availability answer covers each 4x4 luma block's
    // worth of them.
    let unit_v = 4 / sh;
    let mut y = 0;
    while y < 2 * n {
        let a = decoded_before(geo, xl, yl, xl as i32 - 1, (yl + y * sh) as i32);
        for k in 0..unit_v {
            avail.left[y + k] = a;
        }
        y += unit_v;
    }
    let unit_h = 4 / sw;
    let mut x = 0;
    while x < 2 * n {
        let a = decoded_before(geo, xl, yl, (xl + x * sw) as i32, yl as i32 - 1);
        for k in 0..unit_h {
            avail.top[x + k] = a;
        }
        x += unit_h;
    }
}

/// The three MPM candidates (8.4.2) for the prediction block at `(xp,
/// yp)`: the left and above neighbours' modes (DC where a neighbour is
/// unavailable, and for an above neighbour outside the current CTB row),
/// expanded to three by the standard's formula. The mirror of
/// `mpm_candidates` in `hevc::ctu`; its not-intra check has no counterpart
/// here because everything this encoder codes is intra.
fn mpm_candidates(geo: Geo, modes: &[u8], xp: usize, yp: usize) -> [u32; 3] {
    let cand = |xn: i32, yn: i32, is_above: bool| -> u32 {
        if !decoded_before(geo, xp, yp, xn, yn) {
            return 1;
        }
        if is_above && (yn as usize) < (yp >> geo.log2_cu) << geo.log2_cu {
            return 1;
        }
        modes[(yn as usize >> 2) * geo.w4 + (xn as usize >> 2)] as u32
    };
    let a = cand(xp as i32 - 1, yp as i32, false);
    let b = cand(xp as i32, yp as i32 - 1, true);
    mpm_from_pair(a, b)
}

/// The candidate-list formula of 8.4.2 given the two neighbour modes —
/// split out pure so the syntax round-trip can be tested over every pair.
fn mpm_from_pair(a: u32, b: u32) -> [u32; 3] {
    if a == b {
        if a < 2 {
            [0, 1, 26]
        } else {
            [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
        }
    } else {
        let c = if a != 0 && b != 0 {
            0
        } else if a != 1 && b != 1 {
            1
        } else {
            26
        };
        [a, b, c]
    }
}

/// Turn a chosen mode into the flag/index/remainder the syntax carries.
/// The remainder is the mode's rank once the candidates are removed from
/// the numbering — the inverse of the decoder's sorted-bump loop, which a
/// test round-trips against.
fn as_syntax(mode: u8, cands: [u32; 3]) -> LumaModeSyntax {
    if let Some(i) = cands.iter().position(|&c| c == mode as u32) {
        return LumaModeSyntax { prev_flag: true, mpm_idx: i as u8, rem: 0 };
    }
    let below = cands.iter().filter(|&&c| c < mode as u32).count() as u8;
    LumaModeSyntax { prev_flag: false, mpm_idx: 0, rem: mode - below }
}

/// How a candidate mode will be signalled, for the rate side of a cost.
enum ModeSignal {
    /// Luma mode found in the MPM list at this index.
    LumaMpm(u8),
    /// Luma mode signalled as a 5-bit remainder.
    LumaEscape,
    /// `intra_chroma_pred_mode` 4: derived from luma.
    ChromaDerived,
    /// `intra_chroma_pred_mode` 0..=3.
    ChromaExplicit,
}

/// The rate half of a candidate's cost — **a placeholder for real RD**.
/// It counts the mode-signalling bins (treating the one context-coded bin
/// as a bit) and ignores the residual entirely, weighted by the
/// conventional Lagrangian `0.85 * 2^((QP - 12) / 3)` so it lives in the
/// same units as a SATD. It is deliberately not pretending to be more:
/// every rate heuristic in this module is this one function, so replacing
/// it with a real bit count is one edit.
fn mode_signalling_cost(qp: i32, signal: ModeSignal) -> f32 {
    let bins = match signal {
        // prev_intra_luma_pred_flag plus the truncated-Rice mpm_idx.
        ModeSignal::LumaMpm(0) => 2,
        ModeSignal::LumaMpm(_) => 3,
        // The flag plus five fixed bits of rem_intra_luma_pred_mode.
        ModeSignal::LumaEscape => 6,
        // intra_chroma_pred_mode's first bin, alone or with two more.
        ModeSignal::ChromaDerived => 1,
        ModeSignal::ChromaExplicit => 3,
    };
    let lambda = 0.85f32 * ((qp - 12) as f32 / 3.0).exp2();
    lambda * bins as f32
}

/// Forward-code and reconstruct one transform block whose *prediction is
/// already in the plane*: residual against `src`, forward transform (DST
/// for 4x4 intra luma, DCT otherwise) and quantisation from `hevc_enc`,
/// then reconstruction through the decoder's own `scale_coefficients`,
/// inverse transform and `add_residual` — the path `residual_block` in
/// `hevc::ctu` takes, so the two cannot disagree. Under `bypass` the
/// residual is carried and added raw, which is what the decoder does with
/// a `cu_transquant_bypass` block. Returns the TU's cbf.
#[allow(clippy::too_many_arguments)]
fn code_residual<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    plane: &mut Plane16<S>,
    x: usize,
    y: usize,
    log2: u32,
    c_idx: usize,
    qp: i32,
    src: &[S],
    src_stride: usize,
    levels: &mut [i16],
) -> bool {
    let n = 1usize << log2;
    let off = plane.offset(x as isize, y as isize);
    let stride = plane.stride;
    let max = (1i32 << ctx.bit_depth) - 1;

    let mut work = [0i16; 1024];
    for yy in 0..n {
        for xx in 0..n {
            work[yy * n + xx] =
                (src[yy * src_stride + xx].to_i32() - plane.data[off + yy * stride + xx].to_i32()) as i16;
        }
    }

    if ctx.bypass {
        // Lossless: the residual is the coefficients. The decoder skips
        // scaling and transform for a bypass block and adds them as they
        // are; prediction plus residual is the source, so the clip in
        // add_residual never bites and the round trip is exact.
        levels[..n * n].copy_from_slice(&work[..n * n]);
        (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
        return levels[..n * n].iter().any(|&v| v != 0);
    }

    if c_idx == 0 && log2 == 2 {
        (ctx.enc.fdst4)(&mut work, ctx.bit_depth);
    } else {
        (ctx.enc.fdct[(log2 - 2) as usize])(&mut work, log2, ctx.bit_depth);
    }
    let qb = qbits(qp, log2, ctx.bit_depth);
    let nz = (ctx.enc.quant)(
        &work,
        levels,
        n,
        quant_scale((qp % 6) as usize),
        qb,
        quant_offset(qb, true),
    );

    // Reconstruct through the decoder's own dequantisation and inverse
    // transform, so the plane holds what a decoder will hold.
    work[..n * n].copy_from_slice(&levels[..n * n]);
    scale_coefficients(&mut work, log2, qp, ctx.bit_depth, ScalingSource::Flat, false, n - 1, n - 1);
    let bd_shift = 20 - ctx.bit_depth as i32;
    if c_idx == 0 && log2 == 2 {
        (ctx.dsp.idst4)(&mut work, bd_shift, n - 1, n - 1);
    } else {
        (ctx.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
    }
    (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
    nz != 0
}

/// Choose the luma mode for one transform block by SATD over all 35
/// candidate predictions, then code and reconstruct the winner. The
/// availability in `sc.avail` must already describe this block. Trial
/// predictions are written into the reconstruction plane and scored
/// before the next overwrites them; the block's own reference samples lie
/// outside it, so the trials never disturb what they read.
#[allow(clippy::too_many_arguments)]
fn decide_luma_tu<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    x: usize,
    y: usize,
    log2: u32,
    src: &[S],
    src_stride: usize,
    cands: [u32; 3],
    levels: &mut [i16],
) -> (u8, bool) {
    let n = 1usize << log2;
    let off = plane.offset(x as isize, y as isize);
    let mut best = (f32::MAX, 1u8);
    for mode in 0..35u8 {
        // The decoder's flags for a luma block under this SPS: reference
        // smoothing on (predict itself skips DC and 4x4), boundary filter
        // on (no implicit RDPCM to suspend it).
        predict(plane, sc, x, y, n, mode as u32, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
        let satd = (ctx.dist.satd)(src, src_stride, &plane.data[off..], plane.stride, n, n);
        let signal = match cands.iter().position(|&c| c == mode as u32) {
            Some(i) => ModeSignal::LumaMpm(i as u8),
            None => ModeSignal::LumaEscape,
        };
        let cost = satd as f32 + mode_signalling_cost(ctx.qp, signal);
        if cost < best.0 {
            best = (cost, mode);
        }
    }
    let mode = best.1;
    // Re-predict the winner (the plane holds the last trial), then code.
    predict(plane, sc, x, y, n, mode as u32, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
    let qp = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
    let cbf = code_residual(ctx, plane, x, y, log2, 0, qp, src, src_stride, levels);
    (mode, cbf)
}

/// The derived chroma mode (`IntraPredModeC`, 8.4.3) for a syntax value
/// against the luma mode — the mapping `hevc::ctu::coding_unit` applies,
/// 4:2:0 so without the Table 8-3 remap: 0..=3 pick planar/26/10/1 with 34
/// substituted where the pick equals luma, 4 is luma itself.
fn chroma_mode_for(syntax: u8, luma: u8) -> u8 {
    let m = match syntax {
        0 => 0,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => luma,
    };
    if syntax < 4 && m == luma { 34 } else { m }
}

/// Choose the chroma mode over the five codable candidates by SATD summed
/// across both components, then code and reconstruct each. `sc.avail`
/// must already describe the chroma block (one derivation serves both
/// planes — their geometry is identical).
#[allow(clippy::too_many_arguments)]
fn decide_chroma<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    cb: &mut Plane16<S>,
    cr: &mut Plane16<S>,
    sc: &mut IntraScratch,
    cx: usize,
    cy: usize,
    log2c: u32,
    luma0: u8,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
    out: &mut CuDecision,
) {
    let nc = 1usize << log2c;
    let mut best = (f32::MAX, 4u8);
    for syntax in 0..5u8 {
        let mode = chroma_mode_for(syntax, luma0) as u32;
        let mut satd = 0u32;
        for (plane, src) in [(&mut *cb, src_cb), (&mut *cr, src_cr)] {
            // The decoder's flags for a 4:2:0 chroma block: no reference
            // smoothing, no boundary filter.
            predict(plane, sc, cx, cy, nc, mode, 1, false, false, ctx.bit_depth, ctx.strong_smoothing);
            let off = plane.offset(cx as isize, cy as isize);
            satd += (ctx.dist.satd)(src, c_stride, &plane.data[off..], plane.stride, nc, nc);
        }
        let signal = if syntax == 4 { ModeSignal::ChromaDerived } else { ModeSignal::ChromaExplicit };
        let cost = satd as f32 + mode_signalling_cost(ctx.qp, signal);
        if cost < best.0 {
            best = (cost, syntax);
        }
    }
    let syntax = best.1;
    let mode = chroma_mode_for(syntax, luma0);
    // QP for chroma as the decoder derives it: the bit-depth offset comes
    // off, Table 8-10 maps, and it goes back on. No PPS or slice offsets.
    let bd_off = 6 * (ctx.bit_depth as i32 - 8);
    let qp_c = chroma_qp(1, ctx.qp.clamp(-bd_off, 57)) + bd_off;
    for (comp, (plane, src)) in [(&mut *cb, src_cb), (&mut *cr, src_cr)].into_iter().enumerate() {
        predict(plane, sc, cx, cy, nc, mode as u32, 1 + comp, false, false, ctx.bit_depth, ctx.strong_smoothing);
        out.cbf_chroma[comp] = code_residual(
            ctx,
            plane,
            cx,
            cy,
            log2c,
            1 + comp,
            qp_c,
            src,
            c_stride,
            &mut out.chroma[comp][..nc * nc],
        );
    }
    out.chroma_syntax = syntax;
    out.chroma_mode = mode;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;

    fn lcg(s: &mut u64) -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    fn noise(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..w * h).map(|_| lcg(&mut s) as u8).collect()
    }

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

    /// Code a whole picture in raster CTU order, returning the state and
    /// the decisions.
    fn code_picture(
        ctx: &IntraCtx<'_, u8>,
        w: usize,
        h: usize,
        log2_cu: u32,
        src_y: &[u8],
        src_cb: &[u8],
        src_cr: &[u8],
    ) -> (IntraPicture<u8>, Vec<CuDecision>) {
        let mut pic = IntraPicture::new(w, h, log2_cu, 8);
        let n = 1usize << log2_cu;
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu(ctx, cx, cy, src_y, w, src_cb, src_cr, w / 2));
            }
        }
        (pic, decisions)
    }

    /// Every claim of an available neighbour must point at a block that
    /// really is earlier in decode order — checked against the corner
    /// cases the z-scan rules exist to encode, on a 2x2-CTU picture of
    /// 8x8 CTUs (so the within-CTB z-order has two levels to get wrong).
    #[test]
    fn availability_follows_the_z_scan_order() {
        let geo = Geo { log2_cu: 3, wc: 2, w4: 4, width: 16, height: 16 };
        // TU1 of CTU (0,0), at (4,0): its left column is TU0, decoded.
        assert!(decoded_before(geo, 4, 0, 3, 0));
        // Its below-left samples are TU2's, which come later in z-order.
        assert!(!decoded_before(geo, 4, 0, 3, 4));
        // TU2 at (0,4): the whole row above it is decoded, including the
        // top-right samples that fall in TU1.
        assert!(decoded_before(geo, 0, 4, 4, 3));
        // TU3 at (4,4): top-right would be in the next CTU, not decoded.
        assert!(!decoded_before(geo, 4, 4, 8, 3));
        // In CTU (1,1), below-left of its TU0 falls in CTU (0,1) — earlier
        // in the raster walk, so genuinely available.
        assert!(decoded_before(geo, 8, 8, 7, 12));
        // But below-left of the picture's first CTU is nothing.
        assert!(!decoded_before(geo, 0, 0, -1, 8));
        // And the z-order claim is the interleave the decoder builds:
        // block (4,4) is z 3, after (0,4) at z 2.
        assert_eq!(z_within_ctb(3, 4, 4), 3);
        assert_eq!(z_within_ctb(3, 0, 4), 2);
        assert_eq!(z_within_ctb(3, 4, 0), 1);
        // At 32x32 CTBs the interleave has three levels: block (28, 24) is
        // 4x4 coordinates (7, 6), whose interleave y2 x2 y1 x1 y0 x0 is
        // 110111 with y0 = 0.
        assert_eq!(z_within_ctb(5, 28, 24), 32 + 16 + 8 + 4 + 1);
    }

    /// The availability mirror held against the decoder itself: build the
    /// decoder's z-scan tables (`Geometry`, 6.5.2) from an SPS/PPS pair
    /// our own writers emitted, assign every CTB to one slice as a decoded
    /// picture would have it, and ask `PicInfo::available_at` — the
    /// function `decoded_before` mirrors — about every (current block,
    /// neighbour) pair of 4x4 blocks, including the out-of-picture ring.
    /// The two must agree everywhere, partial CTBs at the picture edge
    /// included. (The writer picks the CTB size from the picture, so the
    /// two shapes here cover 16x16 and 32x32 CTBs with several CTB rows
    /// and columns; 8x8 CTBs are below what it emits and rest on
    /// `availability_follows_the_z_scan_order`.)
    #[test]
    fn availability_mirror_agrees_with_the_decoders_tables() {
        use crate::encode::Config;
        use crate::encode::h265_syntax::{Geometry as SynGeometry, write_pps, write_sps};
        use crate::hevc::pic::Geometry as PicGeometry;
        use crate::hevc::pps::Pps;
        use crate::hevc::sps::Sps;

        for (w, h) in [(48u32, 24u32), (40, 80)] {
            let cfg = Config { width: w, height: h, ..Config::default() };
            let syn = SynGeometry::new(&cfg);
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &syn, 8))).unwrap();
            let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false))).unwrap();
            pps.resolve_tiles(&sps).unwrap();
            let geo_dec = std::sync::Arc::new(PicGeometry::new(&sps, &pps));
            let mut info = PicInfo::new(geo_dec);
            // One decoded slice covering the picture, every block written,
            // as the raster walk guarantees for everything before the
            // current block: the mirror encodes that guarantee, so the
            // decoder's is-it-written check (`pred_mode != 2`) must see
            // written blocks to be comparing the same question.
            info.ctb_slice_addr.fill(0);
            info.pred_mode.fill(1);

            let (pw, ph) = (sps.width as usize, sps.height as usize);
            let geo = Geo {
                log2_cu: sps.log2_ctb_size,
                wc: sps.pic_width_in_ctbs() as usize,
                w4: pw.div_ceil(4),
                width: pw,
                height: ph,
            };
            for yc in (0..ph).step_by(4) {
                for xc in (0..pw).step_by(4) {
                    let ac = info.avail_ctx(xc as i32, yc as i32, pw as i32, ph as i32);
                    for yn in (-4..ph as i32 + 4).step_by(4) {
                        for xn in (-4..pw as i32 + 4).step_by(4) {
                            assert_eq!(
                                decoded_before(geo, xc, yc, xn, yn),
                                info.available_at(&ac, xn, yn),
                                "{w}x{h} cur=({xc},{yc}) neighbour=({xn},{yn})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The mode-to-syntax mapping must round-trip through the decoder's
    /// own reconstruction rule (sort the candidates, bump the remainder
    /// past each) for every candidate list a neighbour pair can produce.
    #[test]
    fn mode_syntax_round_trips() {
        for a in 0..35u32 {
            for b in 0..35u32 {
                let cands = mpm_from_pair(a, b);
                // The syntax relies on the three candidates being distinct.
                assert!(cands[0] != cands[1] && cands[1] != cands[2] && cands[0] != cands[2], "a={a} b={b} {cands:?}");
                for mode in 0..35u8 {
                    let s = as_syntax(mode, cands);
                    let back = if s.prev_flag {
                        cands[s.mpm_idx as usize]
                    } else {
                        // The decoder's loop, verbatim.
                        let mut sorted = cands;
                        sorted.sort_unstable();
                        let mut m = s.rem as u32;
                        for c in sorted {
                            if m >= c {
                                m += 1;
                            }
                        }
                        m
                    };
                    assert_eq!(back, mode as u32, "a={a} b={b} mode={mode}");
                    assert!(s.prev_flag || s.rem < 32);
                }
            }
        }
    }

    /// A flat picture costs nothing: every candidate predicts the flat
    /// value exactly (the first CTU from substituted references, later
    /// ones from flat reconstructions), so every level is zero, every cbf
    /// is clear, DC ties the winner at zero distortion, and the
    /// reconstruction is the source.
    #[test]
    fn a_flat_picture_codes_to_nothing_and_dc_ties() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, false);
        for log2_cu in 3..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);

            // Every mode reconstructs flat, DC among them: predict all 35
            // with the decoder's own predictor on the first block, whose
            // absent neighbours substitute to 1 << (bit_depth - 1) = 128.
            // The plane is left at its zeroed default, so a predictor that
            // read a sample availability said it could not would show up
            // here as a non-128 output. With every distortion zero, DC
            // ties the winner by definition.
            let mut probe = IntraPicture::<u8>::new(w, h, log2_cu, 8);
            let geo = probe.geo;
            let IntraPicture { recon, scratch, .. } = &mut probe;
            fill_ref_avail(geo, &mut scratch.avail, 0, 0, n, 1, 1);
            for mode in 0..35u32 {
                predict(&mut recon.y, scratch, 0, 0, n, mode, 0, true, true, 8, false);
                let off = recon.y.origin();
                for yy in 0..n {
                    for xx in 0..n {
                        assert_eq!(recon.y.data[off + yy * recon.y.stride + xx], 128, "mode {mode} log2_cu={log2_cu}");
                    }
                }
            }

            let y = vec![128u8; w * h];
            let c = vec![128u8; w * h / 4];
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, &y, &c, &c);
            for d in &decisions {
                assert!(d.cbf_luma.iter().all(|&f| !f), "log2_cu={log2_cu}");
                assert!(d.cbf_chroma.iter().all(|&f| !f));
                assert!(d.luma.iter().all(|&v| v == 0));
                // With every candidate at zero distortion the rate term
                // decides, and no rate estimate prices an escape below an
                // MPM hit — likewise the derived chroma mode below three
                // explicit bins.
                let pbs = if d.nxn { 4 } else { 1 };
                for pb in 0..pbs {
                    assert!(d.luma_syntax[pb].prev_flag, "log2_cu={log2_cu} pb={pb}");
                }
                assert_eq!(d.chroma_syntax, 4);
            }
            let off = pic.recon.y.origin();
            for yy in 0..h {
                for xx in 0..w {
                    assert_eq!(pic.recon.y.data[off + yy * pic.recon.y.stride + xx], 128, "log2_cu={log2_cu}");
                }
            }
        }
    }

    /// The reconstruction property, made a real check: an independent
    /// walk predicts every block afresh with the decoder's predictor
    /// (from its own reconstruction, not the encoder's), derives each
    /// mode from the stored *syntax*, runs the stored levels through the
    /// decoder's dequantisation and inverse transforms, and must land on
    /// byte-identical planes. A desync anywhere — availability, MPM,
    /// prediction flags, quantisation — shows up here.
    ///
    /// The replay alone has a blind spot, so a distortion bound rides
    /// along: a wrong *forward* transform (say, the DCT where the 4x4 DST
    /// belongs) is invisible to any self-consistency check, because the
    /// encoder and the replay would both push the same wrong levels
    /// through the same inverse and agree perfectly. What such a fault
    /// cannot survive is closeness to the source — quantisation is the
    /// only loss in the loop, so per-sample error is bounded by the step
    /// (the bound `dsp::hevc_enc`'s round-trip tests established), and a
    /// mismatched transform pair turns coefficients into noise that blows
    /// straight through it.
    #[test]
    fn reconstruction_matches_a_fresh_decoder_side_replay() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 12i32), (3, 37), (4, 26), (4, 45), (5, 30), (5, 8)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = noise(w, h, 0x5eed ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w / 2, h / 2, 0xcb);
            let crs = noise(w / 2, h / 2, 0xc7);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, &y, &cbs, &crs);
            let replayed = replay(&ctx, w, h, log2_cu, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);

            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(1, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            for (name, plane, src, pw, ph, pqp) in [
                ("y", &pic.recon.y, &y, w, h, qp),
                ("cb", &pic.recon.cb, &cbs, w / 2, h / 2, qp_c),
                ("cr", &pic.recon.cr, &crs, w / 2, h / 2, qp_c),
            ] {
                let step = 1i32 << (pqp / 6);
                let off = plane.origin();
                let mut worst = 0i32;
                for yy in 0..ph {
                    for xx in 0..pw {
                        let d = plane.data[off + yy * plane.stride + xx] as i32 - src[yy * pw + xx] as i32;
                        worst = worst.max(d.abs());
                    }
                }
                assert!(worst <= 8 * step + 16, "{name} log2_cu={log2_cu} qp={pqp} worst={worst} step={step}");
            }
        }
    }

    /// Transquant bypass is exactly lossless: prediction plus the raw
    /// residual is the source, sample for sample, whatever the content.
    /// (RDPCM is not wired — the SPS carries no range extension.)
    #[test]
    fn lossless_bypass_reconstructs_the_source_exactly() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for log2_cu in 3..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let y = noise(w, h, 99 + log2_cu as u64);
            let cbs = noise(w / 2, h / 2, 7);
            let crs = noise(w / 2, h / 2, 8);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, &y, &cbs, &crs);
            for (name, plane, src, pw, ph) in [
                ("y", &pic.recon.y, &y, w, h),
                ("cb", &pic.recon.cb, &cbs, w / 2, h / 2),
                ("cr", &pic.recon.cr, &crs, w / 2, h / 2),
            ] {
                let off = plane.origin();
                for yy in 0..ph {
                    for xx in 0..pw {
                        assert_eq!(
                            plane.data[off + yy * plane.stride + xx],
                            src[yy * pw + xx],
                            "{name} ({xx},{yy}) log2_cu={log2_cu}"
                        );
                    }
                }
            }
            // And the replay agrees, which exercises the bypass path of
            // the reconstruction contract too.
            let replayed = replay(&ctx, w, h, log2_cu, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 26);
        }
    }

    /// A cbf flag is a statement about the levels beside it, nothing more:
    /// set exactly when its TU holds a nonzero level.
    #[test]
    fn cbf_flags_state_exactly_which_tus_hold_levels() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 30i32), (4, 40), (5, 51)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let y = noise(w, h, 0xcbf ^ log2_cu as u64);
            let cbs = noise(w / 2, h / 2, 1);
            let crs = noise(w / 2, h / 2, 2);
            let (_, decisions) = code_picture(&ctx, w, h, log2_cu, &y, &cbs, &crs);
            let mut some_set = false;
            let mut some_clear = false;
            for d in &decisions {
                let tus: &[(usize, usize)] = if d.nxn { &[(0, 16), (16, 32), (32, 48), (48, 64)] } else { &[(0, n * n)] };
                for (i, &(s, e)) in tus.iter().enumerate() {
                    let any = d.luma[s..e].iter().any(|&v| v != 0);
                    assert_eq!(d.cbf_luma[i], any, "luma tu {i} log2_cu={log2_cu} qp={qp}");
                    some_set |= any;
                    some_clear |= !any;
                }
                let nc = n / 2;
                for comp in 0..2 {
                    let any = d.chroma[comp][..nc * nc].iter().any(|&v| v != 0);
                    assert_eq!(d.cbf_chroma[comp], any, "chroma {comp} log2_cu={log2_cu} qp={qp}");
                }
            }
            // The test only means something if both flag values occurred
            // somewhere across the sweep; noise at these QPs produces both.
            assert!(some_set, "qp={qp}: nothing coded, the check is vacuous");
            let _ = some_clear;
        }
    }

    /// Replay a coded picture the way a decoder would see it: modes from
    /// the stored syntax (asserted against the stored modes), predictions
    /// from the decoder's predictor over the replay's own reconstruction,
    /// residuals from the stored levels through the decoder's inverse
    /// path. Deliberately does not touch the encoder's planes or call
    /// `code_residual`.
    fn replay(ctx: &IntraCtx<'_, u8>, w: usize, h: usize, log2_cu: u32, decisions: &[CuDecision]) -> Frame<u8> {
        let mut pic = IntraPicture::<u8>::new(w, h, log2_cu, 8);
        let geo = pic.geo;
        let n = 1usize << log2_cu;
        let qp_y = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let bd_off = 6 * (ctx.bit_depth as i32 - 8);
        let qp_c = chroma_qp(1, ctx.qp.clamp(-bd_off, 57)) + bd_off;
        let mut di = 0;
        for cy in 0..h / n {
            for cx in 0..w / n {
                let d = &decisions[di];
                di += 1;
                let (x0, y0) = (cx * n, cy * n);
                let IntraPicture { recon, modes, scratch, .. } = &mut pic;
                let luma_tus: &[(usize, usize, usize, u32)] = if d.nxn {
                    &[(x0, y0, 0, 2), (x0 + 4, y0, 1, 2), (x0, y0 + 4, 2, 2), (x0 + 4, y0 + 4, 3, 2)]
                } else {
                    &[(x0, y0, 0, log2_cu)]
                };
                for &(px, py, pb, log2) in luma_tus {
                    let tn = 1usize << log2;
                    // Derive the mode from the syntax, the decoder's way,
                    // and hold it against what the encoder said it chose.
                    let cands = mpm_candidates(geo, modes, px, py);
                    let s = d.luma_syntax[pb];
                    let mode = if s.prev_flag {
                        cands[s.mpm_idx as usize]
                    } else {
                        let mut sorted = cands;
                        sorted.sort_unstable();
                        let mut m = s.rem as u32;
                        for c in sorted {
                            if m >= c {
                                m += 1;
                            }
                        }
                        m
                    };
                    assert_eq!(mode, d.luma_modes[pb] as u32, "syntax and mode disagree at ({px},{py})");
                    fill_ref_avail(geo, &mut scratch.avail, px, py, tn, 1, 1);
                    predict(&mut recon.y, scratch, px, py, tn, mode, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
                    let base = if d.nxn { pb * 16 } else { 0 };
                    add_tu(ctx, &mut recon.y, px, py, log2, 0, qp_y, d.bypass, &d.luma[base..base + tn * tn]);
                    PicInfo::fill4(modes, geo.w4, px, py, tn, tn, mode as u8);
                }
                let mode = chroma_mode_for(d.chroma_syntax, d.luma_modes[0]);
                assert_eq!(mode, d.chroma_mode, "chroma syntax and mode disagree");
                let nc = n / 2;
                let (ccx, ccy) = (x0 / 2, y0 / 2);
                fill_ref_avail(geo, &mut scratch.avail, x0, y0, nc, 2, 2);
                for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                    predict(plane, scratch, ccx, ccy, nc, mode as u32, 1 + comp, false, false, ctx.bit_depth, ctx.strong_smoothing);
                    add_tu(ctx, plane, ccx, ccy, log2_cu - 1, 1 + comp, qp_c, d.bypass, &d.chroma[comp][..nc * nc]);
                }
            }
        }
        std::mem::replace(&mut pic.recon, Frame::empty())
    }

    /// The decoder-side inverse for one TU of stored levels: scale,
    /// inverse-transform, add — or add raw under bypass.
    #[allow(clippy::too_many_arguments)]
    fn add_tu(ctx: &IntraCtx<'_, u8>, plane: &mut Plane16<u8>, x: usize, y: usize, log2: u32, c_idx: usize, qp: i32, bypass: bool, levels: &[i16]) {
        let n = 1usize << log2;
        let off = plane.offset(x as isize, y as isize);
        let max = (1i32 << ctx.bit_depth) - 1;
        let mut work = [0i16; 1024];
        work[..n * n].copy_from_slice(levels);
        if !bypass {
            scale_coefficients(&mut work, log2, qp, ctx.bit_depth, ScalingSource::Flat, false, n - 1, n - 1);
            let bd_shift = 20 - ctx.bit_depth as i32;
            if c_idx == 0 && log2 == 2 {
                (ctx.dsp.idst4)(&mut work, bd_shift, n - 1, n - 1);
            } else {
                (ctx.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
            }
        }
        (ctx.dsp.add_residual)(&mut plane.data[off..], plane.stride, &work, n, max);
    }

    fn assert_planes_equal(a: &Frame<u8>, b: &Frame<u8>, log2_cu: u32, qp: i32) {
        for (name, pa, pb) in [("y", &a.y, &b.y), ("cb", &a.cb, &b.cb), ("cr", &a.cr, &b.cr)] {
            let (oa, ob) = (pa.origin(), pb.origin());
            for y in 0..pa.height {
                for x in 0..pa.width {
                    assert_eq!(
                        pa.data[oa + y * pa.stride + x],
                        pb.data[ob + y * pb.stride + x],
                        "{name} ({x},{y}) log2_cu={log2_cu} qp={qp}"
                    );
                }
            }
        }
    }
}
