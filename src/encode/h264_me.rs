//! Motion estimation and inter mode decision for H.264 P macroblocks.
//!
//! This is the inter counterpart of [`super::h264_intra`]: the *deciding*
//! half of coding a P macroblock — which vector, whether the macroblock
//! skips, what the quantised coefficients are, and what the reconstruction
//! looks like. Turning that into bits belongs to the entropy coders, and
//! the seam is [`InterDecision`] — data, not a shared file, shaped after
//! [`super::h264_intra::MbDecision`] so the serialisers meet a familiar
//! layout.
//!
//! The same two rules shape everything here:
//!
//! **The prediction is the decoder's own.** The chosen vector's prediction
//! runs through the decoder's [`crate::dsp::h264::H264Dsp`] quarter-sample
//! and chroma kernels, addressed exactly as `predict_partition` in
//! `src/h264/inter.rs` addresses them, reading the *reconstructed*
//! reference plane. An encoder
//! that predicted from source samples, or interpolated even one rounding
//! differently, would desync — and the desync would surface as SELF
//! failures hundreds of blocks from the cause, because inter prediction
//! compounds picture over picture where intra drift stays inside one.
//!
//! **So is the reconstruction.** The quantised residual goes back through
//! the decoder's dequantisation and inverse transform
//! ([`crate::dsp::h264::H264Dsp::residual4`]) to build the reconstruction
//! the next macroblock — and the next picture — predicts from.
//!
//! # The search, named
//!
//! Full-sample: a greedy small-diamond descent on SAD, seeded at the median
//! predictor (and at the zero vector when it lies in range), confined to
//! ±16 full samples around the seed and to the window the decoder can
//! legally read (picture plus replicated border). Then two SATD refinement
//! rings: the eight half-sample neighbours, then the eight quarter-sample
//! neighbours of the half-sample winner. 16x16 partitions only, one
//! reference (list 0, `ref_idx` 0), progressive frames only.
//!
//! Its limits are the classic ones: a greedy diamond finds the basin it is
//! seeded in, so untextured or periodic content can stall it at a local
//! minimum; there is no rate term on the vector (SAD/SATD only — the mvd
//! bit cost belongs to a rate-distortion pass that does not exist yet); and
//! one partition size cannot follow motion boundaries inside a macroblock.
//!
//! # Search cost at this scope (arithmetic, not measurement)
//!
//! Per macroblock, worst case: 2 seed SADs, then each diamond round scores
//! at most 4 new positions and must strictly improve to continue — the walk
//! is confined to a ±16 box, so at most 32 improving rounds plus the final
//! non-improving one: `2 + 4 * 33 = 134` 16x16 SADs. A macroblock whose
//! seed is already best costs `2 + 4 = 6`. Refinement always costs
//! `1 + 8 + 8 = 17` 16x16 SATDs, each behind one quarter-sample
//! interpolation. Chroma is not scored during the search (luma-only ME);
//! it is interpolated once, for the chosen vector.
//!
//! # Window legality
//!
//! A full-sample position `f` is searchable when the six-tap window of
//! every refinement reachable from it — `(21 + 21)`-sized at
//! `floor(mv / 4) - 2`, where refinement can lower the floor by one — stays
//! inside the padded plane the decoder reads without clamping. With the
//! decoder's own borders (`LUMA_PAD` = 32, `CHROMA_PAD` = 16, both in
//! `src/h264/frame.rs`) and this module's ±16 range, the chroma window is
//! then inside its border by construction. The
//! prediction helpers still carry the decoder's clamp fallback (mirroring
//! `interp` in `src/h264/inter.rs`), so a vector outside the border is
//! wrong nowhere — merely slower.

use crate::dsp::distortion::DistortionDsp;
use crate::dsp::h264::{NO_DC, PRED_STRIDE};
use crate::dsp::h264_enc::{qbits4, quant_offset};
use crate::encode::h264_intra::IntraCtx;
use crate::encode::h264_syntax::Recon;
use crate::h264::frame::Mv;
use crate::h264::mb::{NbMotion, median_mvp};
use crate::h264::transform::{chroma_dc_transform_420, chroma_dc_transform_422};

/// Everything inter coding needs that does not change per macroblock —
/// the *same* struct the intra module takes, aliased rather than repeated:
/// the quantisation tables already carry the inter lists (3..6), and one
/// context serving both modules means the picture loop cannot hand the two
/// halves of a macroblock decision different QPs.
pub type MeCtx<'a> = IntraCtx<'a>;

/// How a P macroblock was coded, in the form an entropy coder needs.
///
/// The inter counterpart of [`super::h264_intra::MbDecision`], and every
/// field the two share means the same thing, in the same layout, so a
/// serialiser reads both shapes with one set of conventions. Produced once
/// per macroblock and consumed immediately.
#[derive(Clone)]
pub struct InterDecision {
    /// The macroblock-level choice. For [`InterMbKind::UseIntra`] every
    /// other field except `mv` and `mvd` is meaningless: the caller runs
    /// the intra decision instead, and *its* output — including the
    /// reconstruction, which this module has not touched in that case — is
    /// what gets coded.
    pub kind: InterMbKind,
    /// The chosen motion vector, quarter luma samples, list 0. Filled for
    /// every kind, including `PSkip` (the derived skip vector), because
    /// the next macroblock's prediction and the deblocking filter need the
    /// motion whether or not the syntax carries it. The caller feeds it
    /// back as the neighbouring motion of later macroblocks.
    pub mv: Mv,
    /// `mv` minus the median predictor: what `mvd_l0` carries for
    /// `P16x16`. Meaningless for `PSkip` (the syntax carries nothing) —
    /// and not necessarily zero there, because the skip vector can be the
    /// zero vector while the median predictor is not (8.4.1.1).
    pub mvd: Mv,
    /// Reference index into list 0. Always 0 for now: this module searches
    /// exactly one reference picture. The field exists so the serialiser's
    /// contract does not change when more arrive.
    pub ref_idx: i8,
    /// `CodedBlockPatternLuma`: one bit per 8x8, set when any of its 4x4
    /// blocks has a nonzero level.
    pub cbp_luma: u8,
    /// `CodedBlockPatternChroma`: 0 none, 1 DC only, 2 DC and AC.
    pub cbp_chroma: u8,
    /// `mb_qp_delta`. Zero for now — constant-QP coding.
    pub qp_delta: i8,
    /// Luma levels per 4x4 block (raster within the macroblock), each
    /// block in raster order within itself. The zig-zag scan belongs to
    /// the entropy writer, exactly as with `MbDecision`. Inter blocks keep
    /// their own DC at position 0 — there is no Intra_16x16-style DC
    /// split, so there is no `luma_dc` field.
    pub luma: [[i16; 16]; 16],
    /// Chroma DC levels per component: four entries used in 4:2:0, eight
    /// in 4:2:2. Same layout as `MbDecision::chroma_dc`.
    pub chroma_dc: [[i16; 8]; 2],
    /// Chroma AC levels per component, per 4x4 block, position 0 zeroed
    /// (the DC lives in `chroma_dc`). Same layout as
    /// `MbDecision::chroma_ac`.
    pub chroma_ac: [[[i16; 16]; 8]; 2],
    /// Nonzero count per luma 4x4 block (raster), which CAVLC's `nC` needs
    /// from the neighbours and which is free to count while quantising.
    pub nz_luma: [u8; 16],
    /// The same per chroma 4x4 block.
    pub nz_chroma: [[u8; 8]; 2],
}

impl Default for InterDecision {
    fn default() -> Self {
        InterDecision {
            kind: InterMbKind::P16x16,
            mv: Mv::ZERO,
            mvd: Mv::ZERO,
            ref_idx: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
            qp_delta: 0,
            luma: [[0; 16]; 16],
            chroma_dc: [[0; 8]; 2],
            chroma_ac: [[[0; 16]; 8]; 2],
            nz_luma: [0; 16],
            nz_chroma: [[0; 8]; 2],
        }
    }
}

/// The macroblock types this module decides between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterMbKind {
    /// `P_Skip`: no syntax at all — the writer extends `mb_skip_run`. Legal
    /// only because the chosen vector equals the derived skip vector *and*
    /// no residual survived quantisation; this module asserts both before
    /// choosing it.
    PSkip,
    /// `P_L0_16x16`: one vector, one reference, coefficients as `cbp` says.
    P16x16,
    /// Inter lost: code this macroblock with the intra decision instead.
    /// See [`placeholder_inter_or_intra`] for how little that choice
    /// currently knows.
    UseIntra,
}

/// The neighbouring motion a 16x16 partition predicts from: the
/// macroblocks holding luma samples (-1, 0), (0, -1), (16, -1) and
/// (-1, -1) — A, B, C, D of 6.4.11.7. This module's coding is 16x16-only,
/// so a neighbour's motion is its whole macroblock's motion, and the
/// caller (which walks the picture) supplies the four values via
/// [`nb_inter`], [`nb_intra`] and [`nb_absent`].
#[derive(Clone, Copy)]
pub struct MotionNeighbours {
    /// Left (the macroblock holding sample (-1, 0)).
    pub a: NbMotion,
    /// Above (sample (0, -1)).
    pub b: NbMotion,
    /// Above-right (sample (16, -1)).
    pub c: NbMotion,
    /// Above-left (sample (-1, -1)) — the fallback when C is unavailable.
    pub d: NbMotion,
}

/// A neighbour coded inter with `ref_idx` 0 (the only reference this
/// module uses) and the given vector.
pub fn nb_inter(mv: Mv) -> NbMotion {
    NbMotion { avail: true, ref_idx: 0, mv }
}

/// A neighbour coded intra: available, but "not used for inter prediction"
/// — reference index -1, zero vector — exactly the value the decoder's
/// `MotionCache::gather` (src/h264/mb.rs) stores for one.
pub fn nb_intra() -> NbMotion {
    NbMotion { avail: true, ref_idx: -1, mv: Mv::ZERO }
}

/// A neighbour that does not exist: outside the picture, or not yet coded
/// in this slice.
pub fn nb_absent() -> NbMotion {
    NbMotion::NONE
}

/// The median motion vector prediction for a 16x16 partition (8.4.1.3).
///
/// Mirrors `prediction_neighbours` in `src/h264/mb.rs` — C falling back to
/// D when unavailable — and then runs the decoder's own `median_mvp`
/// (src/h264/mb.rs, 8.4.1.3.1) rather than restating the median rules.
/// `prediction_neighbours` itself is not called because it reads a
/// `MotionCache` only the decoder's slice loop can fill; the test
/// `the_predictor_and_skip_mv_agree_with_the_decoders_derivation` holds
/// this mirror against the real thing.
pub fn mv_predictor_16x16(nb: &MotionNeighbours) -> Mv {
    let c = if nb.c.avail { nb.c } else { nb.d };
    median_mvp(nb.a, nb.b, c, 0)
}

/// The P_Skip motion vector (8.4.1.1): the zero vector when either edge
/// neighbour is missing or is a zero-motion reference-0 block, else the
/// 16x16 median prediction.
///
/// Mirrors the decoder's `p_skip_mv` in `src/h264/mb.rs`, which the same
/// test holds this against. Getting this wrong is the classic desync: a
/// decoder *derives* the skip vector, so an encoder that skips with any
/// other vector has silently told the decoder a lie it cannot detect.
pub fn skip_mv_16x16(nb: &MotionNeighbours) -> Mv {
    if !nb.a.avail
        || !nb.b.avail
        || (nb.a.ref_idx == 0 && nb.a.mv == Mv::ZERO)
        || (nb.b.ref_idx == 0 && nb.b.mv == Mv::ZERO)
    {
        return Mv::ZERO;
    }
    mv_predictor_16x16(nb)
}

/// Replicate every reference plane's edges into its border, which is the
/// state the decoder's motion compensation assumes. Call once per
/// reference picture, after its last macroblock is reconstructed and
/// before anything searches against it — the decoder does the same as its
/// rows complete (`Frame::extend_rows`, src/h264/frame.rs).
pub fn prepare_reference(planes: &mut [Recon]) {
    for p in planes.iter_mut() {
        p.extend_edges(false);
    }
}

// ---------------------------------------------------------------------------
// Prediction — the decoder's kernels, the decoder's addressing
// ---------------------------------------------------------------------------

/// Whether a `ww x hh` read at `(x0, y0)` stays inside the padded plane.
/// The direct-read condition of `interp` in `src/h264/inter.rs`, for a
/// progressive frame (full border trusted on all four sides).
fn window_ok(r: &Recon, x0: i32, y0: i32, ww: i32, hh: i32) -> bool {
    let pad = r.pad as i32;
    x0 >= -pad && y0 >= -pad && x0 + ww <= r.width as i32 + pad && y0 + hh <= r.height as i32 + pad
}

/// Run one interpolation whose `ww x hh` source window starts at
/// `(x0, y0)`: a direct read when the window is inside the padded plane,
/// else a sample-by-sample clamp to the picture. Mirrors `interp` in
/// `src/h264/inter.rs` exactly, so a far-out vector reads the same samples
/// a decoder would.
fn interp(r: &Recon, x0: i32, y0: i32, ww: usize, hh: usize, out: &mut [u8], kernel: impl FnOnce(&mut [u8], &[u8], usize)) {
    if window_ok(r, x0, y0, ww as i32, hh as i32) {
        kernel(out, &r.data[r.offset(x0 as isize, y0 as isize)..], r.stride);
    } else {
        let (pw, ph) = (r.width as i32, r.height as i32);
        let mut window = [0u8; 32 * 32];
        for y in 0..hh {
            let yy = (y0 + y as i32).clamp(0, ph - 1) as isize;
            for x in 0..ww {
                let xx = (x0 + x as i32).clamp(0, pw - 1) as isize;
                window[y * 32 + x] = r.data[r.offset(xx, yy)];
            }
        }
        kernel(out, &window, 32);
    }
}

/// Interpolate the 16x16 luma prediction for `mv` at macroblock position
/// `(x, y)` into a [`PRED_STRIDE`]-strided scratch block, through the
/// decoder's own quarter-sample kernel — the addressing is
/// `predict_partition`'s (src/h264/inter.rs): six-tap window two left of
/// and two above the integer position, kernel picked by the two fraction
/// bits of each component.
fn luma_pred_into(ctx: &MeCtx, r: &Recon, x: i32, y: i32, mv: Mv, dst: &mut [u8; 16 * PRED_STRIDE]) {
    let xi = x + (mv.x as i32 >> 2);
    let yi = y + (mv.y as i32 >> 2);
    let pos = ((mv.y & 3) as usize) * 4 + (mv.x & 3) as usize;
    let k = ctx.dsp.qpel[pos];
    interp(r, xi - 2, yi - 2, 16 + 5, 16 + 5, dst, |o, s, st| k(o, s, st, 16, 16, 255));
}

/// Interpolate one chroma component's prediction (8 wide, `ch` high) for
/// luma vector `mv` at chroma position `(cx, cy)` into a scratch block,
/// through the decoder's bilinear kernel. The vector conversion is
/// `predict_partition`'s (src/h264/inter.rs, 8.4.1.4), progressive frames
/// only: eighth-sample fractions in 4:2:0; in 4:2:2 the vertical component
/// is in quarter *chroma* samples, so the fraction doubles.
fn chroma_pred_into(ctx: &MeCtx, r: &Recon, cx: i32, cy: i32, mv: Mv, ch: usize, dst: &mut [u8; 16 * PRED_STRIDE]) {
    let xci = cx + (mv.x as i32 >> 3);
    let (yci, yf) = if ch == 8 {
        (cy + (mv.y as i32 >> 3), (mv.y & 7) as i32)
    } else {
        (cy + (mv.y as i32 >> 2), ((mv.y & 3) << 1) as i32)
    };
    let xf = (mv.x & 7) as i32;
    let kc = ctx.dsp.chroma;
    interp(r, xci, yci, 8 + 1, ch + 1, dst, |o, s, st| kc(o, s, st, 8, ch, xf, yf));
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

/// Search range: full samples either side of the (clamped) predictor.
const RANGE: i32 = 16;

/// Find the 16x16 vector for the macroblock at luma position `(x, y)`,
/// returning it with its SATD. See the module docs for the algorithm, its
/// limits and its cost.
fn search_16x16(ctx: &MeCtx, r: &Recon, x: i32, y: i32, src: &[u8], src_stride: usize, pred: Mv) -> (Mv, u32) {
    // The searchable full-sample window: every position whose own six-tap
    // window is a direct read, shrunk by one on the low side so quarter
    // refinement (which can lower the floor by one) stays a direct read
    // too. See "Window legality" in the module docs.
    let pad = r.pad as i32;
    let (lo_x, hi_x) = (3 - pad - x, r.width as i32 + pad - 19 - x);
    let (lo_y, hi_y) = (3 - pad - y, r.height as i32 + pad - 19 - y);
    debug_assert!(lo_x <= hi_x && lo_y <= hi_y, "plane too small to search");

    // Seed at the predictor rounded to full samples, clamped legal; the
    // box is ±RANGE around that seed, intersected with the legal window.
    let cx = ((pred.x as i32 + 2) >> 2).clamp(lo_x, hi_x);
    let cy = ((pred.y as i32 + 2) >> 2).clamp(lo_y, hi_y);
    let (bx0, bx1) = ((cx - RANGE).max(lo_x), (cx + RANGE).min(hi_x));
    let (by0, by1) = ((cy - RANGE).max(lo_y), (cy + RANGE).min(hi_y));

    let sad = |fx: i32, fy: i32| -> u32 {
        // Full-sample position (0, 0) of the qpel table is a plain copy,
        // so scoring against the plane directly is the same arithmetic
        // without the copy.
        let off = r.offset((x + fx) as isize, (y + fy) as isize);
        (ctx.dist.sad)(src, src_stride, &r.data[off..], r.stride, 16, 16)
    };

    // Greedy small diamond on SAD. `visited` is indexed relative to the
    // box corner; the box is at most (2 * RANGE + 1)^2.
    const SIDE: usize = (2 * RANGE as usize) + 1;
    let mut visited = [false; SIDE * SIDE];
    let mark = |vis: &mut [bool; SIDE * SIDE], fx: i32, fy: i32| {
        vis[(fy - by0) as usize * SIDE + (fx - bx0) as usize] = true;
    };
    mark(&mut visited, cx, cy);
    let mut best = (sad(cx, cy), (cx, cy));
    if (bx0..=bx1).contains(&0) && (by0..=by1).contains(&0) && (cx, cy) != (0, 0) {
        mark(&mut visited, 0, 0);
        let s = sad(0, 0);
        if s < best.0 {
            best = (s, (0, 0));
        }
    }
    loop {
        let centre = best.1;
        let mut improved = false;
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (fx, fy) = (centre.0 + dx, centre.1 + dy);
            if !(bx0..=bx1).contains(&fx) || !(by0..=by1).contains(&fy) {
                continue;
            }
            let idx = (fy - by0) as usize * SIDE + (fx - bx0) as usize;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            let s = sad(fx, fy);
            if s < best.0 {
                best = (s, (fx, fy));
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // Refinement: half- then quarter-sample rings, scored by SATD through
    // the decoder's own interpolation. The full-sample winner is rescored
    // with SATD first, because SAD and SATD are not on the same scale.
    let mut scratch = [0u8; 16 * PRED_STRIDE];
    let mut satd_of = |mv: Mv| -> u32 {
        luma_pred_into(ctx, r, x, y, mv, &mut scratch);
        (ctx.dist.satd)(src, src_stride, &scratch, PRED_STRIDE, 16, 16)
    };
    let mut mv = Mv::new((best.1.0 * 4) as i16, (best.1.1 * 4) as i16);
    let mut cost = satd_of(mv);
    for step in [2i16, 1] {
        let base = mv;
        for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
            let cand = Mv::new(base.x + dx * step, base.y + dy * step);
            if !window_ok(r, x + (cand.x as i32 >> 2) - 2, y + (cand.y as i32 >> 2) - 2, 21, 21) {
                continue;
            }
            let s = satd_of(cand);
            if s < cost {
                cost = s;
                mv = cand;
            }
        }
    }
    (mv, cost)
}

// ---------------------------------------------------------------------------
// Residual coding — forward here, the decoder's inverse back
// ---------------------------------------------------------------------------

/// Forward-transform and quantise one 4x4 block whose prediction is
/// already in `rec` at `off`, with the inter tables (`list` 3..6, inter
/// dead zone). Returns the levels and their nonzero count. The shape of
/// `code_block_4x4` in `src/encode/h264_intra.rs` (private there), minus
/// the Intra_16x16 DC split: an inter block keeps its own DC.
fn code_inter_4x4(ctx: &MeCtx, rec: &Recon, off: usize, src: &[u8], src_stride: usize, list: usize, qp: i32, keep_dc: bool) -> ([i16; 16], u32, i32) {
    let mut residual = [0i16; 16];
    for y in 0..4 {
        for x in 0..4 {
            residual[y * 4 + x] = src[y * src_stride + x] as i16 - rec.data[off + y * rec.stride + x] as i16;
        }
    }
    let mut coeffs = [0i32; 16];
    (ctx.enc.fdct4)(&residual, &mut coeffs);
    let dc = coeffs[0];
    let m = (qp % 6) as usize;
    let qbits = qbits4(qp);
    let offset = quant_offset(qbits, false);
    let mut levels = [0i16; 16];
    let mut nz = (ctx.enc.quant4)(&coeffs, &mut levels, &ctx.quant.mf4[list][m], qbits, offset);
    if !keep_dc {
        if levels[0] != 0 {
            nz -= 1;
        }
        levels[0] = 0;
    }
    (levels, nz, dc)
}

/// Dequantise levels and add the inverse transform to the prediction
/// already in `rec` — the decoder's own path
/// ([`crate::dsp::h264::H264Dsp::residual4`], 8.5.12), so the two cannot
/// disagree. The shape of `reconstruct_4x4` in
/// `src/encode/h264_intra.rs` (private there).
fn add_residual_4x4(ctx: &MeCtx, rec: &mut Recon, off: usize, levels: &[i16; 16], list: usize, qp: i32, dc: Option<i32>) {
    let m = (qp % 6) as usize;
    let scale = &ctx.dequant.scale4[list][m];
    let q6 = qp / 6;
    let mut coefs = [0i32; 16];
    for i in 0..16 {
        let c = levels[i] as i32 * scale[i];
        coefs[i] = if qp >= 24 {
            c << (q6 - 4)
        } else {
            (c + (1 << (3 - q6))) >> (4 - q6)
        };
    }
    (ctx.dsp.residual4)(&mut rec.data[off..], rec.stride, &coefs, dc.unwrap_or(NO_DC), 255);
}

/// Code the chroma of the macroblock for the chosen vector: prediction
/// through the decoder's bilinear kernel, per-4x4 forward transform with
/// the DC pulled into the 2x2 (4:2:0) or 2x4 (4:2:2) Hadamard, and the
/// reconstruction back through the decoder's DC transform and residual
/// add. The shape of `code_chroma` in `src/encode/h264_intra.rs`, with the
/// inter scaling lists (Cb 4, Cr 5) and the inter dead zone.
fn code_inter_chroma(ctx: &MeCtx, rec: &mut [Recon], refp: &[Recon], cx: usize, cy: usize, mv: Mv, src: [&[u8]; 2], src_stride: usize, out: &mut InterDecision) {
    let h = ctx.chroma_h;
    if h == 0 {
        return;
    }
    let blocks = h / 4 * 2;
    let mut any_ac = false;
    let mut any_dc = false;
    let mut scratch = [0u8; 16 * PRED_STRIDE];
    for comp in 0..2 {
        chroma_pred_into(ctx, &refp[comp + 1], cx as i32, cy as i32, mv, h, &mut scratch);
        let plane = &mut rec[comp + 1];
        let off = plane.offset(cx as isize, cy as isize);
        (ctx.dsp.copy)(&mut plane.data[off..], plane.stride, &scratch, 8, h);

        let qp = ctx.qpc[comp];
        let list = 4 + comp; // Cb inter, Cr inter
        let mut dcs = [0i32; 8];
        let mut levels = [[0i16; 16]; 8];
        let mut nz = [0u8; 8];
        for blk in 0..blocks {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = off + by * 4 * plane.stride + bx * 4;
            let soff = by * 4 * src_stride + bx * 4;
            let (lv, n, dc) = code_inter_4x4(ctx, plane, boff, &src[comp][soff..], src_stride, list, qp, false);
            levels[blk] = lv;
            nz[blk] = n as u8;
            dcs[blk] = dc;
        }

        // Chroma DC: the Hadamard, quantised at twice the shift with the
        // position-0 multiplier — what 8.5.11 inverts.
        let m = (qp % 6) as usize;
        let qbits = qbits4(qp) + 1;
        let offset = quant_offset(qbits, false);
        let mut dc_levels = [0i16; 8];
        if blocks == 4 {
            let mut d = [dcs[0], dcs[1], dcs[2], dcs[3]];
            (ctx.enc.hadamard2x2)(&mut d);
            for i in 0..4 {
                let mf = ctx.quant.mf4[list][m][0] as i64;
                let v = ((d[i].unsigned_abs() as i64 * mf + offset as i64) >> qbits) as i32;
                dc_levels[i] = if d[i] < 0 { -v as i16 } else { v as i16 };
            }
        } else {
            let mut d = dcs;
            (ctx.enc.hadamard2x4)(&mut d);
            let mf = ctx.quant.mf4[list][((qp + 3) % 6) as usize][0] as i64;
            for i in 0..8 {
                let v = ((d[i].unsigned_abs() as i64 * mf + offset as i64) >> qbits) as i32;
                dc_levels[i] = if d[i] < 0 { -v as i16 } else { v as i16 };
            }
        }

        // Reconstruct through the decoder's DC transform and residual add.
        let mut dc_rec = [0i32; 8];
        for i in 0..blocks {
            dc_rec[i] = dc_levels[i] as i32;
        }
        if blocks == 4 {
            let mut d4 = [dc_rec[0], dc_rec[1], dc_rec[2], dc_rec[3]];
            chroma_dc_transform_420(&mut d4, ctx.dequant.scale4[list][m][0], qp);
            dc_rec[..4].copy_from_slice(&d4);
        } else {
            chroma_dc_transform_422(&mut dc_rec, ctx.dequant.scale4[list][((qp + 3) % 6) as usize][0], qp);
        }
        for blk in 0..blocks {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = off + by * 4 * plane.stride + bx * 4;
            add_residual_4x4(ctx, plane, boff, &levels[blk], list, qp, Some(dc_rec[blk]));
        }

        any_ac |= nz.iter().any(|&n| n != 0);
        any_dc |= dc_levels.iter().any(|&v| v != 0);
        out.chroma_dc[comp] = dc_levels;
        out.chroma_ac[comp] = levels;
        out.nz_chroma[comp] = nz;
    }
    out.cbp_chroma = if any_ac {
        2
    } else if any_dc {
        1
    } else {
        0
    };
}

// ---------------------------------------------------------------------------
// The macroblock decision
// ---------------------------------------------------------------------------

/// PLACEHOLDER — the inter-versus-intra macroblock choice, and the one
/// deliberately crude decision in this module.
///
/// It compares the searched inter SATD against the SATD of the source
/// block against its own flat mean — a proxy for the *cheapest possible*
/// intra prediction, costing one SATD and no reconstruction state. It
/// knows nothing about the real intra modes, carries no rate term and no
/// lambda, and will therefore hand flat-but-shifted content to intra too
/// eagerly and complex content too late. Replace it when the picture loop
/// can afford to run the real intra decision and compare rate-distortion
/// costs; it lives in this one named function so that replacement touches
/// nothing else.
pub fn placeholder_inter_or_intra(dist: &DistortionDsp<u8>, inter_satd: u32, src: &[u8], src_stride: usize) -> bool {
    let mut sum = 0u32;
    for y in 0..16 {
        for x in 0..16 {
            sum += src[y * src_stride + x] as u32;
        }
    }
    let flat = [((sum + 128) >> 8) as u8; 16 * PRED_STRIDE];
    let intra_proxy = (dist.satd)(src, src_stride, &flat, PRED_STRIDE, 16, 16);
    intra_proxy < inter_satd
}

/// Decide and code one P macroblock, leaving its reconstruction in `rec`.
///
/// `rec` and `refp` are the current and reference pictures' planes in the
/// same layout the intra module uses (luma, then Cb, Cr when present);
/// the reference must have been through [`prepare_reference`]. `nb` is the
/// neighbouring motion (see [`MotionNeighbours`]); the caller feeds the
/// returned `mv` back into later macroblocks' neighbours whatever the
/// kind, because even a skipped macroblock has motion.
///
/// On [`InterMbKind::UseIntra`] the reconstruction planes are untouched:
/// the caller runs the intra decision, which writes them itself.
#[allow(clippy::too_many_arguments)]
pub fn code_macroblock_p16(
    ctx: &MeCtx,
    rec: &mut [Recon],
    refp: &[Recon],
    mb_x: usize,
    mb_y: usize,
    src_luma: &[u8],
    luma_stride: usize,
    src_chroma: [&[u8]; 2],
    chroma_stride: usize,
    nb: &MotionNeighbours,
) -> InterDecision {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let soff = py * luma_stride + px;
    let mut out = InterDecision::default();

    let pred = mv_predictor_16x16(nb);
    let (mv, satd) = search_16x16(ctx, &refp[0], px as i32, py as i32, &src_luma[soff..], luma_stride, pred);
    out.mv = mv;
    out.mvd = Mv::new(mv.x - pred.x, mv.y - pred.y);

    if placeholder_inter_or_intra(ctx.dist, satd, &src_luma[soff..], luma_stride) {
        out.kind = InterMbKind::UseIntra;
        return out;
    }

    // Luma: the chosen vector's prediction into the reconstruction plane
    // (through the decoder's kernel), then each 4x4 coded and reconstructed
    // in place. Raster order — inter blocks predict from the reference,
    // not from each other, so unlike Intra_4x4 nothing here needs the
    // decode scan.
    let off = rec[0].offset(px as isize, py as isize);
    let mut scratch = [0u8; 16 * PRED_STRIDE];
    luma_pred_into(ctx, &refp[0], px as i32, py as i32, mv, &mut scratch);
    (ctx.dsp.copy)(&mut rec[0].data[off..], rec[0].stride, &scratch, 16, 16);
    let mut nz = [0u8; 16];
    for blk in 0..16 {
        let (bx, by) = (blk % 4, blk / 4);
        let boff = off + by * 4 * rec[0].stride + bx * 4;
        let bsoff = soff + by * 4 * luma_stride + bx * 4;
        let (lv, n, _) = code_inter_4x4(ctx, &rec[0], boff, &src_luma[bsoff..], luma_stride, 3, ctx.qp, true);
        out.luma[blk] = lv;
        nz[blk] = n as u8;
        add_residual_4x4(ctx, &mut rec[0], boff, &lv, 3, ctx.qp, None);
    }
    out.nz_luma = nz;
    let mut cbp = 0u8;
    for blk8 in 0..4 {
        let (ox, oy) = ((blk8 % 2) * 2, (blk8 / 2) * 2);
        if (0..4).any(|k| nz[(oy + k / 2) * 4 + ox + k % 2] != 0) {
            cbp |= 1 << blk8;
        }
    }
    out.cbp_luma = cbp;

    if ctx.chroma_h != 0 {
        let (cx, cy) = (mb_x * 8, mb_y * ctx.chroma_h);
        let coff = cy * chroma_stride + cx;
        code_inter_chroma(
            ctx,
            rec,
            refp,
            cx,
            cy,
            mv,
            [&src_chroma[0][coff..], &src_chroma[1][coff..]],
            chroma_stride,
            &mut out,
        );
    }

    // P_Skip needs both legs: the vector the decoder would derive, and no
    // surviving residual. Either alone is a different macroblock — a zero
    // residual at another vector is P_16x16 with cbp 0, and the skip
    // vector with residual is P_16x16 with mvd 0.
    if out.cbp_luma == 0 && out.cbp_chroma == 0 && mv == skip_mv_16x16(nb) {
        out.kind = InterMbKind::PSkip;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::dsp::h264::H264Dsp;
    use crate::dsp::h264_enc::{H264EncDsp, Quant};
    use crate::encode::h264_syntax::recon_plane;
    use crate::h264::frame::{BlockMotion, Frame, LUMA_PAD, CHROMA_PAD, PARITY_FRAME};
    use crate::h264::mb::{MbKind as DecKind, MbNeighbours, MotionCache, PicInfo, p_skip_mv, predict_mv};
    use crate::h264::sps::ScalingLists;
    use crate::h264::transform::{Dequant, dequant4x4, idct4x4};
    use crate::picture::ChromaFormat;

    fn flat() -> ScalingLists {
        ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] }
    }

    struct Tables {
        dsp: H264Dsp<u8>,
        enc: H264EncDsp,
        dist: DistortionDsp<u8>,
        quant: Quant,
        dequant: Dequant,
    }

    impl Tables {
        fn new() -> Self {
            Tables {
                dsp: H264Dsp::<u8>::new(Cpu::SCALAR),
                enc: H264EncDsp::SCALAR,
                dist: DistortionDsp::<u8>::scalar(),
                quant: Quant::new(&flat()),
                dequant: Dequant::new(&flat()),
            }
        }
        fn ctx(&self, qp: i32) -> MeCtx<'_> {
            MeCtx {
                dsp: &self.dsp,
                enc: &self.enc,
                dist: &self.dist,
                quant: &self.quant,
                dequant: &self.dequant,
                qp,
                qpc: [qp, qp],
                chroma_h: 8,
            }
        }
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// A plane holding two triangle gratings, one per axis, periods 25
    /// and 27: structure at the scale of a macroblock window, so a search
    /// window always sees gradients of both signs in both axes.
    ///
    /// What content a greedy small diamond converges on is narrower than
    /// it looks, and three failures were walked into building this test
    /// before the constraints were understood. A convex bowl fails: far
    /// from its apex the window's gradients all point one way, a diagonal
    /// error rides the level line for free, and the walk stalls one
    /// diagonal short. A plain two-slope ramp fails harder: the SAD is
    /// `Σ|a·ex + b·ey|`, zero along a whole line of wrong vectors. And
    /// long-period gratings fail at the range corners: the *error* can
    /// reach 32 (a ±16 target plus the ±16 box), so any period up to 64
    /// puts a reachable alias in the landscape. Short periods dodge the
    /// ridge problems but bring the aliases to |e| = 25 — which is why
    /// the zero-seeded assertions below stay at |d| ≤ 8: the walk starts
    /// inside the true basin (half-period ≈ 12) and, descending strictly,
    /// can never climb over the ridge into an alias. The full ±16 range
    /// is asserted with the predictor seeded at the truth, which is how
    /// large motion is actually found in a picture walk — by neighbour
    /// propagation, not by heroic walks from zero.
    fn grating_plane(w: usize, h: usize, pad: usize) -> Recon {
        let mut p = recon_plane(w as u32, h as u32, pad);
        let o = p.origin();
        for y in 0..h {
            for x in 0..w {
                let tx = (x as i32 % 25 - 12).abs();
                let ty = (y as i32 % 27 - 13).abs();
                p.data[o + y * p.stride + x] = (40 + 4 * tx + 3 * ty) as u8;
            }
        }
        p.extend_edges(false);
        p
    }

    /// Reference planes (luma + 2 chroma) for a 3x3-macroblock picture.
    /// Chroma is never scored by the search, so any structured content
    /// does; only the luma needs [`grating_plane`]'s convergence property.
    fn reference() -> Vec<Recon> {
        let y = grating_plane(48, 48, LUMA_PAD);
        let mut cb = recon_plane(24, 24, CHROMA_PAD);
        let mut cr = recon_plane(24, 24, CHROMA_PAD);
        for (k, p) in [&mut cb, &mut cr].into_iter().enumerate() {
            let o = p.origin();
            for y in 0..24 {
                for x in 0..24 {
                    let r2 = (x as i32 - 11 - k as i32).pow(2) + (y as i32 - 12).pow(2);
                    p.data[o + y * p.stride + x] = (200 - r2.min(160)) as u8;
                }
            }
            p.extend_edges(false);
        }
        vec![y, cb, cr]
    }

    /// The luma source picture that is `refp` translated by `mv` (quarter
    /// samples), generated through the decoder's own kernels so that the
    /// prediction at `mv` is bit-identical to the source by construction.
    fn translated_luma(refp: &Recon, mv: Mv) -> Vec<u8> {
        let (w, h) = (refp.width, refp.height);
        let mut src = vec![0u8; w * h];
        let dsp = H264Dsp::<u8>::new(Cpu::SCALAR);
        let pos = ((mv.y & 3) as usize) * 4 + (mv.x & 3) as usize;
        let k = dsp.qpel[pos];
        let mut block = [0u8; 16 * PRED_STRIDE];
        for by in (0..h).step_by(16) {
            for bx in (0..w).step_by(16) {
                let xi = bx as i32 + (mv.x as i32 >> 2);
                let yi = by as i32 + (mv.y as i32 >> 2);
                k(&mut block, &refp.data[refp.offset((xi - 2) as isize, (yi - 2) as isize)..], refp.stride, 16, 16, 255);
                for y in 0..16 {
                    src[(by + y) * w + bx..(by + y) * w + bx + 16].copy_from_slice(&block[y * PRED_STRIDE..y * PRED_STRIDE + 16]);
                }
            }
        }
        src
    }

    /// The chroma source that is `refp` translated by the luma vector
    /// `mv`, generated straight from clause 8.4.2.2.2's position formula
    /// and the decoder's bilinear kernel — independently of
    /// `chroma_pred_into`, whose derivation it therefore checks.
    fn translated_chroma(refp: &Recon, mv: Mv) -> Vec<u8> {
        let (w, h) = (refp.width, refp.height);
        let mut src = vec![0u8; w * h];
        let dsp = H264Dsp::<u8>::new(Cpu::SCALAR);
        let (xf, yf) = ((mv.x & 7) as i32, (mv.y & 7) as i32);
        let mut block = [0u8; 16 * PRED_STRIDE];
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                let xi = bx as i32 + (mv.x as i32 >> 3);
                let yi = by as i32 + (mv.y as i32 >> 3);
                (dsp.chroma)(&mut block, &refp.data[refp.offset(xi as isize, yi as isize)..], refp.stride, 8, 8, xf, yf);
                for y in 0..8 {
                    src[(by + y) * w + bx..(by + y) * w + bx + 8].copy_from_slice(&block[y * PRED_STRIDE..y * PRED_STRIDE + 8]);
                }
            }
        }
        src
    }

    /// Fresh (zeroed) reconstruction planes matching [`reference`].
    fn fresh_rec() -> Vec<Recon> {
        vec![
            recon_plane(48, 48, LUMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
            recon_plane(24, 24, CHROMA_PAD),
        ]
    }

    fn absent_nb() -> MotionNeighbours {
        MotionNeighbours { a: nb_absent(), b: nb_absent(), c: nb_absent(), d: nb_absent() }
    }

    /// Code the centre macroblock of the 3x3 test picture.
    fn code_centre(ctx: &MeCtx, rec: &mut [Recon], refp: &[Recon], srcy: &[u8], srcc: [&[u8]; 2], nb: &MotionNeighbours) -> InterDecision {
        code_macroblock_p16(ctx, rec, refp, 1, 1, srcy, 48, srcc, 24, nb)
    }

    /// A reference translated by whole samples is found exactly: the
    /// vector is the translation in quarter units, no residual survives,
    /// and P_Skip is chosen exactly when the derived skip vector matches.
    ///
    /// The full ±16 range runs with the predictor carrying the truth (the
    /// picture-walk case); the zero-seeded walks stay at |d| ≤ 8 — see
    /// [`grating_plane`] for why that boundary is where it is.
    #[test]
    fn an_integral_translation_is_found_exactly_and_skips_when_legal() {
        let t = Tables::new();
        let ctx = t.ctx(26);
        let refp = reference();
        for (dx, dy) in [(0i16, 0i16), (1, 0), (0, -1), (5, 3), (-7, 2), (3, -8), (8, 8), (-8, -8), (12, -9), (16, 16), (-16, -16), (16, -16)] {
            let mv = Mv::new(dx * 4, dy * 4);
            let srcy = translated_luma(&refp[0], mv);
            let srcb = translated_chroma(&refp[1], mv);
            let srcr = translated_chroma(&refp[2], mv);

            // All neighbours carrying the true motion: the predictor seeds
            // the search at the answer and the skip vector equals it, so
            // this must skip.
            let nb = MotionNeighbours { a: nb_inter(mv), b: nb_inter(mv), c: nb_inter(mv), d: nb_inter(mv) };
            let mut rec = fresh_rec();
            let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &nb);
            assert_eq!(d.mv, mv, "({dx},{dy}) with agreeing neighbours");
            assert_eq!(d.cbp_luma, 0, "({dx},{dy}) residual should vanish");
            assert_eq!(d.cbp_chroma, 0, "({dx},{dy})");
            assert!(d.nz_luma.iter().all(|&n| n == 0));
            assert_eq!(d.kind, InterMbKind::PSkip, "({dx},{dy})");
            assert_eq!(d.mvd, Mv::ZERO);

            // A zero-motion reference-0 left neighbour forces the skip
            // vector to zero (8.4.1.1), so a moved macroblock must not
            // skip even with an otherwise perfect prediction. The median
            // of (zero, truth, truth) is still the truth, so the search
            // is still seeded at the answer.
            if mv != Mv::ZERO {
                let nb = MotionNeighbours { a: nb_inter(Mv::ZERO), b: nb_inter(mv), c: nb_inter(mv), d: nb_inter(mv) };
                let mut rec = fresh_rec();
                let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &nb);
                assert_eq!(d.mv, mv, "({dx},{dy})");
                assert_eq!(d.kind, InterMbKind::P16x16, "({dx},{dy}) skip vector is zero, must not skip");
            }

            // No neighbours at all: the predictor is zero, so the diamond
            // must walk to the answer on its own; and the skip vector is
            // zero, so only the untranslated case may skip.
            if dx.abs() <= 8 && dy.abs() <= 8 {
                let mut rec = fresh_rec();
                let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &absent_nb());
                assert_eq!(d.mv, mv, "({dx},{dy}) from a zero seed");
                assert_eq!(d.cbp_luma, 0);
                assert_eq!(d.cbp_chroma, 0);
                let want = if mv == Mv::ZERO { InterMbKind::PSkip } else { InterMbKind::P16x16 };
                assert_eq!(d.kind, want, "({dx},{dy}) skip legality");
                assert_eq!(d.mvd, mv, "mvd against a zero predictor is the vector itself");
            }
        }
    }

    /// A residual that survives quantisation forbids P_Skip even at the
    /// skip vector.
    #[test]
    fn a_surviving_residual_forbids_skip() {
        let t = Tables::new();
        let ctx = t.ctx(26);
        let refp = reference();
        let mut srcy = translated_luma(&refp[0], Mv::ZERO);
        let srcb = translated_chroma(&refp[1], Mv::ZERO);
        let srcr = translated_chroma(&refp[2], Mv::ZERO);
        // Noise on the centre macroblock, ±15: enough that levels survive
        // the QP 26 dead zone across sixteen blocks, small enough beside
        // the reference's structure that inter still beats the intra
        // proxy. (A flat DC step would not do here: its inter SATD is so
        // small that the placeholder rightly hands it to intra.)
        let mut seed = 4u64;
        for y in 16..32 {
            for x in 16..32 {
                let n = (lcg(&mut seed) % 31) as i32 - 15;
                srcy[y * 48 + x] = (srcy[y * 48 + x] as i32 + n).clamp(0, 255) as u8;
            }
        }
        let mut rec = fresh_rec();
        let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &absent_nb());
        assert_ne!(d.kind, InterMbKind::PSkip, "residual survived, skip is illegal");
        assert_eq!(d.kind, InterMbKind::P16x16);
        assert_ne!(d.cbp_luma, 0, "±15 noise must leave levels at QP 26");
    }

    /// A half-sample translation is found by the refinement rings, with a
    /// residual that is exactly zero because the source was generated by
    /// the same conformance-proven kernel the prediction runs.
    #[test]
    fn a_half_pel_translation_is_found_with_zero_residual() {
        let t = Tables::new();
        let ctx = t.ctx(26);
        let refp = reference();
        for (qx, qy) in [(2i16, 0i16), (0, 2), (2, 2), (6, 0), (-2, 4), (10, -6), (1, 0), (3, 2), (-1, -1)] {
            let mv = Mv::new(qx, qy);
            let srcy = translated_luma(&refp[0], mv);
            let srcb = translated_chroma(&refp[1], mv);
            let srcr = translated_chroma(&refp[2], mv);
            let mut rec = fresh_rec();
            let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &absent_nb());
            assert_eq!(d.mv, mv, "quarter vector ({qx},{qy})");
            assert_eq!(d.cbp_luma, 0, "({qx},{qy}) the residual is zero by construction");
            assert_eq!(d.cbp_chroma, 0, "({qx},{qy})");
            assert!(d.nz_luma.iter().all(|&n| n == 0));
        }
    }

    /// The mirrored predictor and skip derivations agree with the
    /// decoder's own — `predict_mv` and `p_skip_mv` run over a real
    /// `MotionCache` gathered from a real frame — across every
    /// present/absent combination of A, B, C, D, with inter and intra
    /// neighbours and varied vectors.
    #[test]
    fn the_predictor_and_skip_mv_agree_with_the_decoders_derivation() {
        let mut frame = Frame::<u8>::new(3, 3, ChromaFormat::Yuv420, 8, false);
        let mut seed = 7u64;
        // Current macroblock: address 4 (centre). A=3, B=1, C=2, D=0.
        let cur_addr = 4usize;
        let nbs = [(3usize, 0usize), (1, 1), (2, 2), (0, 3)]; // (addr, slot a/b/c/d)
        for mask in 0..16u32 {
            for intra_mask in 0..16u32 {
                for _ in 0..4 {
                    let mut info = PicInfo::new(3, 3);
                    let mut mine = absent_nb();
                    for &(addr, slot) in &nbs {
                        let present = mask & (1 << slot) != 0;
                        if !present {
                            continue;
                        }
                        let intra = intra_mask & (1 << slot) != 0;
                        info.mbs[addr].decoded = true;
                        info.mbs[addr].slice = 0;
                        info.mbs[addr].kind = if intra { DecKind::I16x16 } else { DecKind::Inter16x16 };
                        let mv = Mv::new((lcg(&mut seed) % 65) as i16 - 32, (lcg(&mut seed) % 65) as i16 - 32);
                        if !intra {
                            let bm = BlockMotion { mv, ref_idx: 0, ref_parity: PARITY_FRAME, ref_id: 1 };
                            for blk in 0..16 {
                                frame.motion[0][addr * 16 + blk] = bm;
                            }
                        }
                        let n = if intra { nb_intra() } else { nb_inter(mv) };
                        match slot {
                            0 => mine.a = n,
                            1 => mine.b = n,
                            2 => mine.c = n,
                            _ => mine.d = n,
                        }
                    }
                    info.mbs[cur_addr].decoded = false;
                    let mut dnb = MbNeighbours::default();
                    dnb.derive_into(&info, cur_addr, 0);
                    let mut cache = MotionCache::default();
                    cache.gather(&dnb, &frame, &info);
                    let cur = [[BlockMotion::default(); 16]; 2];
                    let want_mvp = predict_mv(&cache, &cur, 0, 0, 0, 0, 0, 16, 16);
                    let want_skip = p_skip_mv(&cache, &cur);
                    assert_eq!(mv_predictor_16x16(&mine), want_mvp, "mask={mask:04b} intra={intra_mask:04b}");
                    assert_eq!(skip_mv_16x16(&mine), want_skip, "mask={mask:04b} intra={intra_mask:04b}");
                }
            }
        }
    }

    /// A noisy source over a structured reference: the reconstruction this
    /// module wrote must equal prediction plus the decoder's inverse of
    /// the quantised residual, recomputed here through the *test-only*
    /// scalar transforms (`dequant4x4` / `idct4x4`) — a different path
    /// from the production `residual4` kernel, so a mistake on either side
    /// has nothing to cancel against.
    #[test]
    fn the_reconstruction_equals_the_decoders_own_inverse_path() {
        let t = Tables::new();
        let refp = reference();
        let mut seed = 33u64;
        for qp in [14i32, 26, 38] {
            let ctx = t.ctx(qp);
            // Source: the reference shifted a little, plus noise the
            // quantiser cannot fully absorb — a real residual everywhere.
            let mv_true = Mv::new(6, -3);
            let mut srcy = translated_luma(&refp[0], mv_true);
            let mut srcb = translated_chroma(&refp[1], mv_true);
            let mut srcr = translated_chroma(&refp[2], mv_true);
            for v in srcy.iter_mut().chain(srcb.iter_mut()).chain(srcr.iter_mut()) {
                *v = (*v as i32 + (lcg(&mut seed) % 17) as i32 - 8).clamp(0, 255) as u8;
            }
            let mut rec = fresh_rec();
            let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &absent_nb());
            assert_eq!(d.kind, InterMbKind::P16x16, "qp={qp} inter must win on matched content");

            // Luma: prediction through the same decoder kernel, residual
            // through the test-only inverse pair.
            let mut pred = [0u8; 16 * PRED_STRIDE];
            luma_pred_into(&ctx, &refp[0], 16, 16, d.mv, &mut pred);
            let off = rec[0].offset(16, 16);
            for blk in 0..16 {
                let (bx, by) = (blk % 4, blk / 4);
                let mut co = [0i32; 16];
                for i in 0..16 {
                    co[i] = d.luma[blk][i] as i32;
                }
                dequant4x4(&mut co, &t.dequant.scale4[3][(qp % 6) as usize], qp, false);
                idct4x4(&mut co);
                for y in 0..4 {
                    for x in 0..4 {
                        let p = pred[(by * 4 + y) * PRED_STRIDE + bx * 4 + x] as i32;
                        let want = (p + co[y * 4 + x]).clamp(0, 255) as u8;
                        let got = rec[0].data[off + (by * 4 + y) * rec[0].stride + bx * 4 + x];
                        assert_eq!(got, want, "qp={qp} luma blk={blk} ({x},{y})");
                    }
                }
            }

            // Chroma: the same, with the DC coming back through the
            // decoder's 2x2 DC transform.
            for comp in 0..2 {
                let plane = &rec[comp + 1];
                let mut cpred = [0u8; 16 * PRED_STRIDE];
                chroma_pred_into(&ctx, &refp[comp + 1], 8, 8, d.mv, 8, &mut cpred);
                let coff = plane.offset(8, 8);
                let m = (qp % 6) as usize;
                let mut dc = [0i32; 4];
                for i in 0..4 {
                    dc[i] = d.chroma_dc[comp][i] as i32;
                }
                chroma_dc_transform_420(&mut dc, t.dequant.scale4[4 + comp][m][0], qp);
                for blk in 0..4 {
                    let (bx, by) = (blk % 2, blk / 2);
                    let mut co = [0i32; 16];
                    for i in 0..16 {
                        co[i] = d.chroma_ac[comp][blk][i] as i32;
                    }
                    dequant4x4(&mut co, &t.dequant.scale4[4 + comp][m], qp, false);
                    co[0] = dc[blk];
                    idct4x4(&mut co);
                    for y in 0..4 {
                        for x in 0..4 {
                            let p = cpred[(by * 4 + y) * PRED_STRIDE + bx * 4 + x] as i32;
                            let want = (p + co[y * 4 + x]).clamp(0, 255) as u8;
                            let got = plane.data[coff + (by * 4 + y) * plane.stride + bx * 4 + x];
                            assert_eq!(got, want, "qp={qp} comp={comp} blk={blk} ({x},{y})");
                        }
                    }
                }
            }
        }
    }

    /// The coded block pattern and the nonzero counts are restatements of
    /// the levels, and must agree with them exactly.
    #[test]
    fn cbp_and_nz_counts_agree_with_the_levels() {
        let t = Tables::new();
        let refp = reference();
        let mut seed = 91u64;
        for qp in [10i32, 22, 30, 40] {
            let ctx = t.ctx(qp);
            let mv_true = Mv::new(-5, 7);
            let mut srcy = translated_luma(&refp[0], mv_true);
            let mut srcb = translated_chroma(&refp[1], mv_true);
            let mut srcr = translated_chroma(&refp[2], mv_true);
            for v in srcy.iter_mut().chain(srcb.iter_mut()).chain(srcr.iter_mut()) {
                *v = (*v as i32 + (lcg(&mut seed) % 21) as i32 - 10).clamp(0, 255) as u8;
            }
            let mut rec = fresh_rec();
            let d = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &absent_nb());
            assert_eq!(d.kind, InterMbKind::P16x16, "qp={qp}");

            for blk in 0..16 {
                let count = d.luma[blk].iter().filter(|&&v| v != 0).count() as u8;
                assert_eq!(d.nz_luma[blk], count, "qp={qp} luma blk={blk}");
            }
            for blk8 in 0..4 {
                let (ox, oy) = ((blk8 % 2) * 2, (blk8 / 2) * 2);
                let any = (0..4).any(|k| d.nz_luma[(oy + k / 2) * 4 + ox + k % 2] != 0);
                assert_eq!(d.cbp_luma >> blk8 & 1 != 0, any, "qp={qp} blk8={blk8}");
            }
            let mut any_ac = false;
            let mut any_dc = false;
            for comp in 0..2 {
                for blk in 0..4 {
                    let count = d.chroma_ac[comp][blk].iter().filter(|&&v| v != 0).count() as u8;
                    assert_eq!(d.nz_chroma[comp][blk], count, "qp={qp} comp={comp} blk={blk}");
                    assert_eq!(d.chroma_ac[comp][blk][0], 0, "AC position 0 lives in chroma_dc");
                    any_ac |= count != 0;
                }
                any_dc |= d.chroma_dc[comp][..4].iter().any(|&v| v != 0);
            }
            let want = if any_ac { 2 } else if any_dc { 1 } else { 0 };
            assert_eq!(d.cbp_chroma, want, "qp={qp}");
        }
    }
}
