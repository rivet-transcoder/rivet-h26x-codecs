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
use crate::encode::h264_pic::PicMotion;
use crate::encode::h264_intra::{IntraCtx, code_block_8x8, lambda, quad_rasters, reconstruct_8x8};
use crate::h264::cavlc::sub_block_counts_8x8;
use crate::encode::h264_syntax::Recon;
use crate::h264::frame::{BlockMotion, Frame, Mv, PARITY_FRAME};
use crate::h264::cavlc::mb_partitions;
use crate::h264::mb::{
    MbKind as DecMbKind, MbMotion, MbNeighbours, MotionCache, PicInfo, colocated_block,
    fill_motion, p_skip_mv, predict_mv,
};
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
    /// `transform_size_8x8_flag`. The syntax carries it only when some
    /// luma block is coded (`cbp_luma != 0`) — a decoder reading no flag
    /// infers 0 — so it is false whenever `cbp_luma` is, and the writers
    /// assert as much.
    pub transform_8x8: bool,
    /// `mvd_l0` per partition, in [`InterMbKind::parts`] order — one
    /// entry for `P16x16`, two for `P16x8` and `P8x16`. Meaningless for
    /// `PSkip` (the syntax carries nothing), and not necessarily zero
    /// there, because the skip vector can be the zero vector while the
    /// median predictor is not (8.4.1.1).
    ///
    /// The *vectors* themselves are not here: they live in the
    /// [`MbMotionState`] the decision derived them into, in the decoder's
    /// per-4x4 layout, which is what the picture walk commits and what
    /// later partitions and macroblocks predict from.
    pub mvd: [Mv; 4],
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
    ///
    /// Under the 8x8 transform the same storage holds four blocks of
    /// sixty-four at flat offset `blk8 * 64` — see
    /// [`MbDecision::luma`](crate::encode::h264_intra::MbDecision::luma).
    pub luma: [[i16; 16]; 16],
    /// Chroma DC levels per component: four entries used in 4:2:0, eight
    /// in 4:2:2. Same layout as `MbDecision::chroma_dc`.
    pub chroma_dc: [[i16; 16]; 2],
    /// Chroma AC levels per component, per 4x4 block, position 0 zeroed
    /// (the DC lives in `chroma_dc`). Same layout as
    /// `MbDecision::chroma_ac`.
    pub chroma_ac: [[[i16; 16]; 16]; 2],
    /// Nonzero count per luma 4x4 block (raster), which CAVLC's `nC` needs
    /// from the neighbours and which is free to count while quantising.
    /// Under the 8x8 transform these are the four sub-scan counts — see
    /// [`MbDecision::nz_luma`](crate::encode::h264_intra::MbDecision::nz_luma).
    pub nz_luma: [u8; 16],
    /// The same per chroma 4x4 block.
    pub nz_chroma: [[u8; 16]; 2],
}

impl Default for InterDecision {
    fn default() -> Self {
        InterDecision {
            kind: InterMbKind::P16x16,
            transform_8x8: false,
            mvd: [Mv::ZERO; 4],
            ref_idx: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
            qp_delta: 0,
            luma: [[0; 16]; 16],
            chroma_dc: [[0; 16]; 2],
            chroma_ac: [[[0; 16]; 16]; 2],
            nz_luma: [0; 16],
            nz_chroma: [[0; 16]; 2],
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
    /// `P_L0_16x8`: two partitions, upper and lower.
    P16x8,
    /// `P_L0_8x16`: two partitions, left and right.
    P8x16,
    /// Inter lost: code this macroblock with the intra decision instead.
    /// See [`placeholder_inter_or_intra`] for how little that choice
    /// currently knows.
    UseIntra,
}

impl InterMbKind {
    /// The decoder's own name for this macroblock.
    pub fn dec_kind(self) -> DecMbKind {
        match self {
            InterMbKind::PSkip => DecMbKind::PSkip,
            InterMbKind::P16x16 => DecMbKind::Inter16x16,
            InterMbKind::P16x8 => DecMbKind::Inter16x8,
            InterMbKind::P8x16 => DecMbKind::Inter8x16,
            InterMbKind::UseIntra => DecMbKind::I16x16,
        }
    }

    /// The partition rectangles `(x, y, w, h)` in luma samples, in the
    /// order the syntax carries them — the decoder's own `mb_partitions`
    /// (src/h264/cavlc.rs), so the writers and the walk cannot disagree
    /// about how many there are or where they sit.
    pub fn parts(self) -> &'static [(usize, usize, usize, usize)] {
        mb_partitions(self.dec_kind())
    }

    /// `mb_type` in Table 7-13's numbering: 0 for 16x16, 1 for 16x8, 2
    /// for 8x16. (`P_8x8` is 3 and `P_8x8ref0` 4; neither is produced.)
    pub fn p_mb_type(self) -> u32 {
        match self {
            InterMbKind::P16x16 => 0,
            InterMbKind::P16x8 => 1,
            InterMbKind::P8x16 => 2,
            _ => unreachable!("only a coded P macroblock has an mb_type here"),
        }
    }
}

/// The motion one macroblock's partitions predict from: the neighbours
/// gathered from the picture, the blocks of this macroblock derived so
/// far, and a mask of which those are.
///
/// This is the decoder's own working set — `MotionCache`, `MbMotion`
/// and the `done` bitmask that `derive_motion` (src/h264/recon.rs)
/// carries through a macroblock — held so that every prediction here is a
/// *call* to `predict_mv` rather than a mirror of it. That matters most
/// for partitions smaller than the macroblock, whose A / B / C neighbours
/// are frequently blocks of this same macroblock: `done` is what makes an
/// already-derived one readable and a not-yet-derived one absent, and
/// there is no way to say that with a per-macroblock summary.
pub struct MbMotionState {
    cache: MotionCache,
    cur: MbMotion,
    done: u16,
}

impl Default for MbMotionState {
    fn default() -> Self {
        Self::new()
    }
}

impl MbMotionState {
    /// Empty state; `start` fills it per macroblock.
    pub fn new() -> Self {
        MbMotionState {
            cache: MotionCache::default(),
            cur: [[BlockMotion::default(); 16]; 2],
            done: 0,
        }
    }

    /// Begin macroblock `addr`: gather its neighbours from the picture
    /// coded so far, and clear the per-macroblock part. `nb` is scratch.
    pub fn start(
        &mut self,
        frame: &Frame<u8>,
        info: &PicInfo,
        addr: usize,
        nb: &mut MbNeighbours,
    ) {
        nb.derive_into(info, addr, 0);
        self.cache.gather(nb, frame, info);
        self.cur = [[BlockMotion::default(); 16]; 2];
        self.done = 0;
    }

    /// Clear the macroblock's own derived motion, keeping the gathered
    /// neighbours. Trying one partition shape and then another means
    /// deriving over the same neighbours twice, and the second trial must
    /// not see the first's partitions as available.
    pub fn reset_mb(&mut self) {
        self.cur = [[BlockMotion::default(); 16]; 2];
        self.done = 0;
    }

    /// 8.4.1.3 for a partition at `(x, y)` of size `w` by `h` samples —
    /// the decoder's own `predict_mv`, directional cases and C-to-D
    /// fallback included.
    pub fn predict(&self, list: usize, ref_idx: i8, x: usize, y: usize, w: usize, h: usize) -> Mv {
        predict_mv(&self.cache, &self.cur, self.done, list, ref_idx, x, y, w, h)
    }

    /// 8.4.1.1's `P_Skip` vector, through the decoder's own `p_skip_mv`.
    pub fn skip_mv(&self) -> Mv {
        p_skip_mv(&self.cache, &self.cur)
    }

    /// Record a partition's derived motion, which later partitions of the
    /// same macroblock predict from. Both lists are written before any
    /// `done` bit is set, exactly as `derive_motion` orders it — an
    /// unused list stores the default, which is what a decoder holds for
    /// one.
    pub fn commit_part(&mut self, x: usize, y: usize, w: usize, h: usize, per_list: [Option<Mv>; 2]) {
        for (list, mv) in per_list.iter().enumerate() {
            let m = match mv {
                Some(mv) => BlockMotion {
                    mv: *mv,
                    ref_idx: 0,
                    ref_parity: PARITY_FRAME,
                    // An identity the derivations only compare for
                    // equality; one reference per list makes 1 and 2
                    // distinct names for "the list-0 picture" and "the
                    // list-1 picture".
                    ref_id: 1 + list as u16,
                },
                None => BlockMotion::default(),
            };
            fill_motion(&mut self.cur, list, x, y, w, h, m);
        }
        for by in y / 4..(y + h) / 4 {
            for bx in x / 4..(x + w) / 4 {
                self.done |= 1 << (by * 4 + bx);
            }
        }
    }

    /// The macroblock's motion so far, in the decoder's raster layout.
    pub fn motion(&self) -> &MbMotion {
        &self.cur
    }

    /// Spatial direct's reference indices (8.4.1.2.2), through the
    /// decoder's own `spatial_direct_ref_idx`.
    pub fn direct_ref_idx(&self) -> [i8; 2] {
        crate::h264::mb::spatial_direct_ref_idx(&self.cache, &self.cur)
    }
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

/// Interpolate a `w` by `h` luma prediction for `mv` at position
/// `(x, y)` into a [`PRED_STRIDE`]-strided scratch block, through the
/// decoder's own quarter-sample kernel — the addressing is
/// `predict_partition`'s (src/h264/inter.rs): six-tap window two left of
/// and two above the integer position, kernel picked by the two fraction
/// bits of each component.
///
/// `(x, y)` is the *partition's* position in the picture, not the
/// macroblock's: a partition interpolates from its own corner, so a 16x8
/// lower half reads eight rows further down and its window moves with it.
#[allow(clippy::too_many_arguments)]
fn luma_pred_into(
    ctx: &MeCtx,
    r: &Recon,
    x: i32,
    y: i32,
    mv: Mv,
    w: usize,
    h: usize,
    dst: &mut [u8; 16 * PRED_STRIDE],
) {
    let xi = x + (mv.x as i32 >> 2);
    let yi = y + (mv.y as i32 >> 2);
    let pos = ((mv.y & 3) as usize) * 4 + (mv.x & 3) as usize;
    let k = ctx.dsp.qpel[pos];
    interp(r, xi - 2, yi - 2, w + 5, h + 5, dst, |o, s, st| k(o, s, st, w, h, 255));
}

/// Interpolate one chroma component's prediction (8 wide, `ch` high) for
/// luma vector `mv` at chroma position `(cx, cy)` into a scratch block,
/// through the decoder's bilinear kernel. The vector conversion is
/// `predict_partition`'s (src/h264/inter.rs, 8.4.1.4), progressive frames
/// only: eighth-sample fractions in 4:2:0; in 4:2:2 the vertical component
/// is in quarter *chroma* samples, so the fraction doubles.
#[allow(clippy::too_many_arguments)]
fn chroma_pred_into(
    ctx: &MeCtx,
    r: &Recon,
    cx: i32,
    cy: i32,
    mv: Mv,
    cw: usize,
    ch_h: usize,
    ch: usize,
    dst: &mut [u8; 16 * PRED_STRIDE],
) {
    let xci = cx + (mv.x as i32 >> 3);
    let (yci, yf) = if ch == 8 {
        (cy + (mv.y as i32 >> 3), (mv.y & 7) as i32)
    } else {
        (cy + (mv.y as i32 >> 2), ((mv.y & 3) << 1) as i32)
    };
    let xf = (mv.x & 7) as i32;
    let kc = ctx.dsp.chroma;
    interp(r, xci, yci, cw + 1, ch_h + 1, dst, |o, s, st| kc(o, s, st, cw, ch_h, xf, yf));
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

/// Search range: full samples either side of the (clamped) predictor.
const RANGE: i32 = 16;

/// Find the vector for the `w` by `h` partition at luma position
/// `(x, y)`, returning it with its SATD. See the module docs for the
/// algorithm, its limits and its cost.
///
/// `(x, y)` and the size are the *partition's*: a 16x8 lower half
/// searches from its own corner over its own eight rows, which is what
/// makes two halves able to find different motion.
#[allow(clippy::too_many_arguments)]
fn search_rect(
    ctx: &MeCtx,
    r: &Recon,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    src: &[u8],
    src_stride: usize,
    pred: Mv,
) -> (Mv, u32) {
    // The searchable full-sample window: every position whose own six-tap
    // window is a direct read, shrunk by one on the low side so quarter
    // refinement (which can lower the floor by one) stays a direct read
    // too. See "Window legality" in the module docs.
    let pad = r.pad as i32;
    let (lo_x, hi_x) = (3 - pad - x, r.width as i32 + pad - (w as i32 + 3) - x);
    let (lo_y, hi_y) = (3 - pad - y, r.height as i32 + pad - (h as i32 + 3) - y);
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
        (ctx.dist.sad)(src, src_stride, &r.data[off..], r.stride, w, h)
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
        luma_pred_into(ctx, r, x, y, mv, w, h, &mut scratch);
        (ctx.dist.satd)(src, src_stride, &scratch, PRED_STRIDE, w, h)
    };
    let mut mv = Mv::new((best.1.0 * 4) as i16, (best.1.1 * 4) as i16);
    let mut cost = satd_of(mv);
    for step in [2i16, 1] {
        let base = mv;
        for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
            let cand = Mv::new(base.x + dx * step, base.y + dy * step);
            if !window_ok(
                r,
                x + (cand.x as i32 >> 2) - 2,
                y + (cand.y as i32 >> 2) - 2,
                w as i32 + 5,
                h as i32 + 5,
            ) {
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

/// Write one partition's inter prediction for one or two references into
/// the reconstruction planes, through the decoder's kernels and — when both
/// lists predict — the decoder's default bi-predictive combine
/// (`(a + b + 1) >> 1`, the `dsp.avg` kernel `predict_partition` runs for
/// `Weighting::Default`). What stands in `rec` afterwards is bit-identical
/// to what a decoder derives for the same vectors.
#[allow(clippy::too_many_arguments)]
fn predict_inter_rect(
    ctx: &MeCtx,
    rec: &mut [Recon],
    refs: [&[Recon]; 2],
    px: usize,
    py: usize,
    pw: usize,
    ph: usize,
    used: [bool; 2],
    mv: [Mv; 2],
) {
    debug_assert!(used[0] || used[1]);
    let mut a = [0u8; 16 * PRED_STRIDE];
    let mut b = [0u8; 16 * PRED_STRIDE];
    // Luma.
    let off = rec[0].offset(px as isize, py as isize);
    let stride = rec[0].stride;
    if used[0] && used[1] {
        luma_pred_into(ctx, &refs[0][0], px as i32, py as i32, mv[0], pw, ph, &mut a);
        luma_pred_into(ctx, &refs[1][0], px as i32, py as i32, mv[1], pw, ph, &mut b);
        (ctx.dsp.avg)(&mut rec[0].data[off..], stride, &a, &b, pw, ph);
    } else {
        let l = if used[0] { 0 } else { 1 };
        luma_pred_into(ctx, &refs[l][0], px as i32, py as i32, mv[l], pw, ph, &mut a);
        (ctx.dsp.copy)(&mut rec[0].data[off..], stride, &a, pw, ph);
    }
    // Chroma. 4:4:4 interpolates its chroma with the luma six-tap kernel
    // at the unscaled vector (8.4.2.2: mvCLX = mvLX) — the `c444` branch
    // of `predict_partition` in src/h264/inter.rs — so the planes go
    // through the same helper the luma did.
    let h = ctx.chroma_h;
    if h == 0 {
        return;
    }
    if ctx.c444 {
        for comp in 0..2 {
            let plane = &mut rec[comp + 1];
            let off = plane.offset(px as isize, py as isize);
            let stride = plane.stride;
            if used[0] && used[1] {
                luma_pred_into(ctx, &refs[0][comp + 1], px as i32, py as i32, mv[0], pw, ph, &mut a);
                luma_pred_into(ctx, &refs[1][comp + 1], px as i32, py as i32, mv[1], pw, ph, &mut b);
                (ctx.dsp.avg)(&mut plane.data[off..], stride, &a, &b, pw, ph);
            } else {
                let l = if used[0] { 0 } else { 1 };
                luma_pred_into(ctx, &refs[l][comp + 1], px as i32, py as i32, mv[l], pw, ph, &mut a);
                (ctx.dsp.copy)(&mut plane.data[off..], stride, &a, pw, ph);
            }
        }
        return;
    }
    // The partition's chroma rectangle: horizontally always halved, and
    // vertically halved in 4:2:0 but not in 4:2:2 — `chroma_h` carries
    // exactly that ratio (8 or 16 chroma rows per macroblock).
    let (cx, cy) = (px / 2, py * h / 16);
    let (cw, crh) = (pw / 2, ph * h / 16);
    for comp in 0..2 {
        let plane = &mut rec[comp + 1];
        let off = plane.offset(cx as isize, cy as isize);
        let stride = plane.stride;
        if used[0] && used[1] {
            chroma_pred_into(ctx, &refs[0][comp + 1], cx as i32, cy as i32, mv[0], cw, crh, h, &mut a);
            chroma_pred_into(ctx, &refs[1][comp + 1], cx as i32, cy as i32, mv[1], cw, crh, h, &mut b);
            (ctx.dsp.avg)(&mut plane.data[off..], stride, &a, &b, cw, crh);
        } else {
            let l = if used[0] { 0 } else { 1 };
            chroma_pred_into(ctx, &refs[l][comp + 1], cx as i32, cy as i32, mv[l], cw, crh, h, &mut a);
            (ctx.dsp.copy)(&mut plane.data[off..], stride, &a, cw, crh);
        }
    }
}

/// The coded residual of one inter macroblock, in the layout
/// [`InterDecision`] (and the B decision) carry — produced by
/// `code_inter_mb_residual` once the prediction stands in `rec`.
struct InterResidual {
    /// `transform_size_8x8_flag`. Forced false when `cbp_luma` is zero:
    /// the syntax then carries no flag and a decoder infers zero, so
    /// recording anything else would leave the encoder's loop filter and
    /// the next macroblock's contexts disagreeing with a decoder's over a
    /// bit nobody wrote.
    transform_8x8: bool,
    /// `CodedBlockPatternLuma`.
    cbp_luma: u8,
    /// `CodedBlockPatternChroma`.
    cbp_chroma: u8,
    /// Luma levels per 4x4 (raster), DC in place.
    luma: [[i16; 16]; 16],
    /// Nonzero count per luma block.
    nz_luma: [u8; 16],
    /// Chroma DC levels per component.
    chroma_dc: [[i16; 16]; 2],
    /// Chroma AC levels per component and block, position 0 zeroed.
    chroma_ac: [[[i16; 16]; 16]; 2],
    /// Nonzero count per chroma block.
    nz_chroma: [[u8; 16]; 2],
}

/// A macroblock's worth of luma out of a plane, packed 16 by 16.
fn gather16(p: &Recon, off: usize, out: &mut [u8; 256]) {
    for y in 0..16 {
        out[y * 16..y * 16 + 16].copy_from_slice(&p.data[off + y * p.stride..off + y * p.stride + 16]);
    }
}

/// The inverse of [`gather16`]: put a saved macroblock back.
fn scatter16(p: &mut Recon, off: usize, src: &[u8; 256]) {
    for y in 0..16 {
        p.data[off + y * p.stride..off + y * p.stride + 16]
            .copy_from_slice(&src[y * 16..y * 16 + 16]);
    }
}

/// Code an inter macroblock's luma as sixteen 4x4 blocks, reconstructing
/// in place, and fill the levels, counts and coded block pattern.
fn code_luma_4x4(
    ctx: &MeCtx,
    rec: &mut Recon,
    off: usize,
    src: &[u8],
    src_stride: usize,
    out: &mut InterResidual,
) {
    for blk in 0..16 {
        let (bx, by) = (blk % 4, blk / 4);
        let boff = off + by * 4 * rec.stride + bx * 4;
        let bsoff = by * 4 * src_stride + bx * 4;
        let (lv, n, _) = code_inter_4x4(ctx, rec, boff, &src[bsoff..], src_stride, 3, ctx.qp, true);
        out.luma[blk] = lv;
        out.nz_luma[blk] = n as u8;
        add_residual_4x4(ctx, rec, boff, &lv, 3, ctx.qp, None);
    }
    for blk8 in 0..4 {
        if quad_rasters(blk8).iter().any(|&r| out.nz_luma[r] != 0) {
            out.cbp_luma |= 1 << blk8;
        }
    }
}

/// The same as four 8x8 blocks: the shared block coder at the *inter*
/// dead zone and 8x8 scaling list 1 (`2 * plane + inter`, so luma inter
/// is 1 — not the 4x4 order's 3), and the counts kept per CAVLC sub-scan
/// as [`InterDecision::nz_luma`] promises.
fn code_luma_8x8(
    ctx: &MeCtx,
    rec: &mut Recon,
    off: usize,
    src: &[u8],
    src_stride: usize,
    out: &mut InterResidual,
) {
    for blk8 in 0..4 {
        let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        let boff = off + by * 4 * rec.stride + bx * 4;
        let bsoff = by * 4 * src_stride + bx * 4;
        let (lv, counts) =
            code_block_8x8(ctx, rec, boff, &src[bsoff..], src_stride, 1, ctx.qp, false);
        reconstruct_8x8(ctx, rec, boff, &lv, 1, ctx.qp);
        out.luma.as_flattened_mut()[blk8 * 64..blk8 * 64 + 64].copy_from_slice(&lv);
        for (sub, &raster) in quad_rasters(blk8).iter().enumerate() {
            out.nz_luma[raster] = counts[sub];
        }
        if counts.iter().any(|&n| n != 0) {
            out.cbp_luma |= 1 << blk8;
        }
    }
    debug_assert_eq!(
        out.luma.as_flattened().iter().filter(|&&v| v != 0).count(),
        (0..4)
            .map(|b| sub_block_counts_8x8(&out.luma.as_flattened()[b * 64..b * 64 + 64])
                .iter()
                .map(|&n| n as usize)
                .sum::<usize>())
            .sum::<usize>(),
        "the four sub-scans have to partition an 8x8's sixty-four positions"
    );
}

/// PLACEHOLDER — which transform size an inter macroblock's luma residual
/// uses.
///
/// Both are coded and reconstructed from the same prediction, and this
/// compares the sum of squared errors of each reconstruction against the
/// source plus `lambda` times the one bit the flag costs.
///
/// Squared error, for the reason `intra_distortion` in
/// src/encode/h264_intra.rs sets out at length: both candidates here are
/// fully coded, so what separates them is how close the reconstruction
/// landed, and SATD measures what a *prediction* would cost to code
/// instead. This decision is new, so unlike the intra ladder it has
/// nothing pinning it to the older measure and simply uses the right one.
///
/// It still has no rate term for the residual, which is where the 8x8
/// transform mostly earns its keep, so it takes 8x8 less often than a
/// real rate-distortion decision would. One function, one comparison.
fn placeholder_inter_transform_size(ssd_4x4: u64, ssd_8x8: u64, qp: i32) -> bool {
    ssd_8x8 as f32 + lambda(qp) < ssd_4x4 as f32
}

/// Code the residual of a 16x16 inter macroblock whose *prediction*
/// already stands in every plane of `rec`: forward transform and
/// quantisation with the inter tables, reconstruction back through the
/// decoder's inverse path in place, coefficients and counts out. Shared
/// by the P and B paths — the prediction differs between them, the
/// residual machinery must not.
fn code_inter_mb_residual(
    ctx: &MeCtx,
    rec: &mut [Recon],
    mb_x: usize,
    mb_y: usize,
    src_luma: &[u8],
    luma_stride: usize,
    src_chroma: [&[u8]; 2],
    chroma_stride: usize,
) -> InterResidual {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let soff = py * luma_stride + px;
    let mut out = InterResidual {
        transform_8x8: false,
        cbp_luma: 0,
        cbp_chroma: 0,
        luma: [[0; 16]; 16],
        nz_luma: [0; 16],
        chroma_dc: [[0; 16]; 2],
        chroma_ac: [[[0; 16]; 16]; 2],
        nz_chroma: [[0; 16]; 2],
    };

    // Luma. The 4x4 candidate first — each block coded and reconstructed
    // in place, in raster order, because inter blocks predict from the
    // reference rather than from each other and so, unlike Intra_4x4,
    // nothing here needs the decode scan.
    let off = rec[0].offset(px as isize, py as isize);
    // The prediction, before any residual is added to it: the 8x8
    // candidate has to start from the same samples, and whichever loses
    // has to be undone.
    let mut pred = [0u8; 256];
    gather16(&rec[0], off, &mut pred);
    code_luma_4x4(ctx, &mut rec[0], off, &src_luma[soff..], luma_stride, &mut out);

    if ctx.t8x8 {
        // Score what the 4x4 reconstructed, keep it, and try the 8x8 from
        // the same prediction.
        let mut recon4 = [0u8; 256];
        gather16(&rec[0], off, &mut recon4);
        let cost4 = (ctx.dist.ssd)(&src_luma[soff..], luma_stride, &recon4, 16, 16, 16);
        let r4 = (out.cbp_luma, out.luma, out.nz_luma);

        scatter16(&mut rec[0], off, &pred);
        out.cbp_luma = 0;
        out.luma = [[0; 16]; 16];
        out.nz_luma = [0; 16];
        code_luma_8x8(ctx, &mut rec[0], off, &src_luma[soff..], luma_stride, &mut out);
        let mut recon8 = [0u8; 256];
        gather16(&rec[0], off, &mut recon8);
        let cost8 = (ctx.dist.ssd)(&src_luma[soff..], luma_stride, &recon8, 16, 16, 16);

        // A macroblock with no coded luma block carries no flag at all,
        // so the 8x8 candidate cannot be *recorded* even when it wins:
        // both spell the same empty residual, and 4x4 is what a decoder
        // will infer.
        if placeholder_inter_transform_size(cost4, cost8, ctx.qp) {
            out.transform_8x8 = true;
        } else {
            scatter16(&mut rec[0], off, &recon4);
            (out.cbp_luma, out.luma, out.nz_luma) = r4;
        }
    }

    let h = ctx.chroma_h;
    if h == 0 {
        return out.gate_transform_size();
    }
    if ctx.c444 {
        // 4:4:4: the chroma planes code luma-style — each 4x4 keeps its
        // own DC (no Hadamard split outside Intra_16x16), inter scaling
        // lists 4 and 5 at the chroma QP — and their coefficients ride
        // the *same* coded-block-pattern bits as luma, so the planes'
        // contributions are ORed in.
        for comp in 0..2 {
            let src = &src_chroma[comp][soff..];
            let plane = &mut rec[comp + 1];
            let off = plane.offset(px as isize, py as isize);
            let qp = ctx.qpc[comp];
            if out.transform_8x8 {
                // The transform size is the macroblock's, not the plane's:
                // one `transform_size_8x8_flag` governs all three
                // luma-style planes, exactly as one coded block pattern
                // does. The 8x8 scaling lists run `2 * plane + inter`, so
                // Cb inter is 3 and Cr inter 5 — not the 4x4 order's 4
                // and 5.
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let boff = off + by * 4 * plane.stride + bx * 4;
                    let bsoff = by * 4 * luma_stride + bx * 4;
                    let (lv, counts) = code_block_8x8(
                        ctx, plane, boff, &src[bsoff..], luma_stride, 3 + 2 * comp, qp, false,
                    );
                    reconstruct_8x8(ctx, plane, boff, &lv, 3 + 2 * comp, qp);
                    out.chroma_ac[comp].as_flattened_mut()[blk8 * 64..blk8 * 64 + 64]
                        .copy_from_slice(&lv);
                    for (sub, &raster) in quad_rasters(blk8).iter().enumerate() {
                        out.nz_chroma[comp][raster] = counts[sub];
                    }
                }
            } else {
                for blk in 0..16 {
                    let (bx, by) = (blk % 4, blk / 4);
                    let boff = off + by * 4 * plane.stride + bx * 4;
                    let bsoff = by * 4 * luma_stride + bx * 4;
                    let (lv, n, _) = code_inter_4x4(
                        ctx, plane, boff, &src[bsoff..], luma_stride, 4 + comp, qp, true,
                    );
                    out.chroma_ac[comp][blk] = lv;
                    out.nz_chroma[comp][blk] = n as u8;
                    add_residual_4x4(ctx, plane, boff, &lv, 4 + comp, qp, None);
                }
            }
            for blk8 in 0..4 {
                if quad_rasters(blk8).iter().any(|&r| out.nz_chroma[comp][r] != 0) {
                    out.cbp_luma |= 1 << blk8;
                }
            }
        }
        return out.gate_transform_size();
    }
    // Chroma, per component: per-4x4 AC with the DC pulled into the 2x2
    // (4:2:0) or 2x4 (4:2:2) Hadamard, reconstruction back through the
    // decoder's DC transform and residual add.
    let (cx, cy) = (mb_x * 8, mb_y * h);
    let coff = cy * chroma_stride + cx;
    let blocks = h / 4 * 2;
    let mut any_ac = false;
    let mut any_dc = false;
    for comp in 0..2 {
        let src = &src_chroma[comp][coff..];
        let plane = &mut rec[comp + 1];
        let off = plane.offset(cx as isize, cy as isize);
        let qp = ctx.qpc[comp];
        let list = 4 + comp; // Cb inter, Cr inter
        let mut dcs = [0i32; 8];
        let mut levels = [[0i16; 16]; 16];
        let mut nz = [0u8; 16];
        for blk in 0..blocks {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = off + by * 4 * plane.stride + bx * 4;
            let bsoff = by * 4 * chroma_stride + bx * 4;
            let (lv, n, dc) = code_inter_4x4(ctx, plane, boff, &src[bsoff..], chroma_stride, list, qp, false);
            levels[blk] = lv;
            nz[blk] = n as u8;
            dcs[blk] = dc;
        }

        // Chroma DC: the Hadamard, quantised at twice the shift with the
        // position-0 multiplier — what 8.5.11 inverts.
        let m = (qp % 6) as usize;
        let qbits = qbits4(qp) + 1;
        let offset = quant_offset(qbits, false);
        let mut dc_levels = [0i16; 16];
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
    out.gate_transform_size()
}

impl InterResidual {
    /// Drop a recorded 8x8 transform when nothing is left to carry it.
    ///
    /// The reader takes `transform_size_8x8_flag` only when
    /// `layer.cbp & 15 != 0` (`parse_mb_cavlc` / `parse_mb_cabac`), and
    /// that cbp is the *final* one — which in 4:4:4 the chroma planes
    /// contribute to, so the test cannot be made before they are coded.
    /// With no coded luma-style block there is no flag on the wire, a
    /// decoder infers zero, and both transform sizes reconstruct the bare
    /// prediction anyway; recording anything else would leave the loop
    /// filter and the next macroblock's contexts disagreeing with a
    /// decoder over a bit nobody wrote.
    fn gate_transform_size(mut self) -> Self {
        if self.cbp_luma == 0 {
            self.transform_8x8 = false;
        }
        self
    }
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
/// motion state ([`MbMotionState`], gathered from the picture coded so
/// far); the caller commits the returned `mv` into that state whatever
/// the kind, because even a skipped macroblock has motion.
///
/// On [`InterMbKind::UseIntra`] the reconstruction planes are untouched:
/// the caller runs the intra decision, which writes them itself.
#[allow(clippy::too_many_arguments)]
pub fn code_macroblock_p(
    ctx: &MeCtx,
    rec: &mut [Recon],
    refp: &[Recon],
    mb_x: usize,
    mb_y: usize,
    src_luma: &[u8],
    luma_stride: usize,
    src_chroma: [&[u8]; 2],
    chroma_stride: usize,
    st: &mut MbMotionState,
) -> InterDecision {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let soff = py * luma_stride + px;
    let mut out = InterDecision::default();

    // Every shape on offer is searched, partition by partition, over the
    // state a decoder would hold at that point — which for the second
    // partition of a 16x8 or 8x16 includes the first, already derived.
    // That is why the trials run through `MbMotionState` and reset it
    // between them rather than predicting from a fixed neighbour set.
    let mut best: Option<(InterMbKind, [Mv; 4], [Mv; 4], u32, f32)> = None;
    for kind in [InterMbKind::P16x16, InterMbKind::P16x8, InterMbKind::P8x16] {
        if kind != InterMbKind::P16x16 && !ctx.subparts {
            continue;
        }
        st.reset_mb();
        let mut mvs = [Mv::ZERO; 4];
        let mut mvds = [Mv::ZERO; 4];
        let mut satd = 0u32;
        for (i, &(x, y, w, h)) in kind.parts().iter().enumerate() {
            let (ax, ay) = (px + x, py + y);
            let pred = st.predict(0, 0, x, y, w, h);
            let (mv, cost) = search_rect(
                ctx,
                &refp[0],
                ax as i32,
                ay as i32,
                w,
                h,
                &src_luma[ay * luma_stride + ax..],
                luma_stride,
                pred,
            );
            mvs[i] = mv;
            mvds[i] = Mv::new(mv.x - pred.x, mv.y - pred.y);
            satd += cost;
            st.commit_part(x, y, w, h, [Some(mv), None]);
        }
        let cost = placeholder_partition_cost(kind, satd, &mvds, ctx.qp);
        if best.as_ref().is_none_or(|b| cost < b.4) {
            best = Some((kind, mvs, mvds, satd, cost));
        }
    }
    let (kind, mvs, mvds, satd, _) = best.expect("16x16 is always a candidate");
    out.kind = kind;
    out.mvd = mvds;

    if placeholder_inter_or_intra(ctx.dist, satd, &src_luma[soff..], luma_stride) {
        out.kind = InterMbKind::UseIntra;
        return out;
    }

    // Replay the winner into the state — the losing trial overwrote it —
    // and predict each partition into the reconstruction planes through
    // the decoder's kernels, then code the residual over the whole
    // macroblock and reconstruct in place.
    st.reset_mb();
    for (i, &(x, y, w, h)) in kind.parts().iter().enumerate() {
        st.commit_part(x, y, w, h, [Some(mvs[i]), None]);
        predict_inter_rect(
            ctx,
            rec,
            [refp, refp],
            px + x,
            py + y,
            w,
            h,
            [true, false],
            [mvs[i], Mv::ZERO],
        );
    }
    let r = code_inter_mb_residual(
        ctx, rec, mb_x, mb_y, src_luma, luma_stride, src_chroma, chroma_stride,
    );
    out.transform_8x8 = r.transform_8x8;
    out.cbp_luma = r.cbp_luma;
    out.cbp_chroma = r.cbp_chroma;
    out.luma = r.luma;
    out.nz_luma = r.nz_luma;
    out.chroma_dc = r.chroma_dc;
    out.chroma_ac = r.chroma_ac;
    out.nz_chroma = r.nz_chroma;

    // P_Skip needs both legs: the vector the decoder would derive, and no
    // surviving residual. Either alone is a different macroblock — a zero
    // residual at another vector is P_16x16 with cbp 0, and the skip
    // vector with residual is P_16x16 with mvd 0.
    if out.kind == InterMbKind::P16x16
        && out.cbp_luma == 0
        && out.cbp_chroma == 0
        && mvs[0] == st.skip_mv()
    {
        out.kind = InterMbKind::PSkip;
    }
    out
}

/// The bits an `se(v)` of this value costs — the exact CAVLC length, and
/// close enough to what CABAC spends on the same element for a decision
/// that is choosing between two of them.
fn se_bits(v: i16) -> f32 {
    let code = if v > 0 { 2 * v as u32 - 1 } else { (-2 * v as i32) as u32 };
    (2 * (32 - (code + 1).leading_zeros()) - 1) as f32
}

/// PLACEHOLDER — which partition shape a P macroblock is coded as.
///
/// Each shape is fully searched and this scores it as the sum of its
/// partitions' SATDs plus `lambda` times an estimate of the motion
/// syntax: the `mb_type`, and each partition's mvd priced by its actual
/// magnitude through [`se_bits`].
///
/// Pricing the real vectors rather than charging a flat per-mvd constant
/// is what makes this usable at all. A constant is wrong in the direction
/// that matters: `lambda` grows by a factor of two every three quantiser
/// steps, so a flat sixteen bits an mvd costs ~9800 SATD at QP 40 against
/// ~390 at QP 26, and the encoder simply stopped splitting at high
/// quantisers — where a second vector is usually *close* to the first and
/// its mvd is three or four bits, not sixteen.
///
/// What it still does not have — as with every other decision here — is a
/// rate term for the *residual*, which is the other half of what
/// splitting buys. One function, one comparison.
fn placeholder_partition_cost(kind: InterMbKind, satd: u32, mvds: &[Mv; 4], qp: i32) -> f32 {
    // `mb_type`: one bin for 16x16, three for the two-partition shapes.
    let mut bits = match kind {
        InterMbKind::P16x16 => 1.0,
        InterMbKind::P16x8 | InterMbKind::P8x16 => 3.0,
        _ => unreachable!("only coded shapes are priced"),
    };
    for i in 0..kind.parts().len() {
        bits += se_bits(mvds[i].x) + se_bits(mvds[i].y);
    }
    satd as f32 + lambda(qp) * bits
}

// ---------------------------------------------------------------------------
// B macroblocks
// ---------------------------------------------------------------------------

/// The macroblock types the B decision chooses between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BMbKind {
    /// `B_Skip`: direct motion, no residual, no syntax at all.
    BSkip,
    /// `B_Direct_16x16`: direct motion with a residual (`mb_type` 0).
    BDirect16,
    /// One explicit 16x16 partition — `B_L0_16x16`, `B_L1_16x16` or
    /// `B_Bi_16x16` by [`BDecision::used`].
    B16,
    /// Inter lost: code this macroblock with the intra decision instead.
    UseIntra,
}

/// How a B macroblock was coded, in the form an entropy coder needs — the
/// two-list sibling of [`InterDecision`], with the same residual layout.
#[derive(Clone)]
pub struct BDecision {
    /// The macroblock-level choice. For [`BMbKind::UseIntra`] every other
    /// field except the motion is meaningless, exactly as with
    /// [`InterMbKind::UseIntra`].
    pub kind: BMbKind,
    /// `transform_size_8x8_flag`, as in
    /// [`InterDecision::transform_8x8`]: present in the syntax only when
    /// some luma block is coded, and therefore false whenever `cbp_luma`
    /// is. `B_Direct_16x16` may carry it because the SPS this encoder
    /// writes sets `direct_8x8_inference_flag` (7.3.5).
    pub transform_8x8: bool,
    /// Which lists predict. `B16`: the searched direction (`[true,false]`
    /// L0, `[false,true]` L1, `[true,true]` bi). `BSkip` / `BDirect16`:
    /// the derived direct references — which is `[true,true]` whenever any
    /// neighbour predicts from a list, and *always* both when no
    /// neighbour does (8.4.1.2.2's both-negative rule).
    pub used: [bool; 2],
    /// The vector per 8x8 partition per list, quarter luma samples,
    /// meaningful where `used`. Filled for every kind including the
    /// skips, because later macroblocks' prediction and the loop filter
    /// need it.
    ///
    /// Per 8x8 and not per macroblock because spatial direct derives
    /// colZeroFlag from each 8x8's own colocated corner (8.4.1.2.1), and
    /// those four answers differ the moment the colocated macroblock has
    /// more than one partition. An explicit `B16` fills all four alike.
    pub mv: [[Mv; 2]; 4],
    /// `mv` minus that list's median predictor: what `mvd_lX` carries for
    /// `B16`. Zero for the direct kinds (their syntax carries none).
    pub mvd: [Mv; 2],
    /// Reference index per list: 0 where `used`, -1 where not — the
    /// values a decoder stores.
    pub ref_idx: [i8; 2],
    /// `CodedBlockPatternLuma`, as in [`InterDecision::cbp_luma`].
    pub cbp_luma: u8,
    /// `CodedBlockPatternChroma`.
    pub cbp_chroma: u8,
    /// `mb_qp_delta`. Zero — constant-QP coding.
    pub qp_delta: i8,
    /// Luma levels per 4x4 (raster), DC in place; the scan is the
    /// entropy writer's. Identical layout to [`InterDecision::luma`].
    pub luma: [[i16; 16]; 16],
    /// Chroma DC levels per component.
    pub chroma_dc: [[i16; 16]; 2],
    /// Chroma AC levels per component and block, position 0 zeroed.
    pub chroma_ac: [[[i16; 16]; 16]; 2],
    /// Nonzero count per luma block.
    pub nz_luma: [u8; 16],
    /// The same per chroma block.
    pub nz_chroma: [[u8; 16]; 2],
}

impl Default for BDecision {
    fn default() -> Self {
        BDecision {
            kind: BMbKind::B16,
            transform_8x8: false,
            used: [true, false],
            mv: [[Mv::ZERO; 2]; 4],
            mvd: [Mv::ZERO; 2],
            ref_idx: [0, -1],
            cbp_luma: 0,
            cbp_chroma: 0,
            qp_delta: 0,
            luma: [[0; 16]; 16],
            chroma_dc: [[0; 16]; 2],
            chroma_ac: [[[0; 16]; 16]; 2],
            nz_luma: [0; 16],
            nz_chroma: [[0; 16]; 2],
        }
    }
}

/// The spatial direct motion of a B macroblock (8.4.1.2.2), returning
/// the reference index per list and the vector per 8x8 partition per
/// list.
///
/// The reference indices and the median predictions are derived once for
/// the whole macroblock — the decoder does the same, at the 16x16
/// position with an empty `done` (`direct_partitions`, src/h264/recon.rs)
/// — and only colZeroFlag varies per 8x8.
///
/// Mirrors the decoder's derivation in `src/h264/recon.rs`
/// (`derive_motion`'s `direct_spatial` branch): the reference indices are
/// the `MinPositive` over the A / B / C neighbours per list
/// (`spatial_direct_ref_idx`, with `prediction_neighbours`' C-to-D
/// fallback); both negative means both lists at reference 0 with zero
/// vectors; otherwise each used list takes the 16x16 median prediction —
/// through the decoder's own `median_mvp` — unless the colocated block is
/// effectively still (`colZeroFlag`: not intra, reference index 0, both
/// vector components within plus-or-minus one quarter sample; the list
/// preference of `colocated_motion` in `src/h264/mb.rs` — list 0, else
/// list 1), in which case that list's vector is zero.
///
/// `col` is the list-1 reference's own motion, and `addr` this
/// macroblock's address in it. colZeroFlag is read through the decoder's
/// `colocated_motion` at the partition's colocated corner block, which
/// for a 16x16 partition is block 0 (8.4.1.2.1). Long-term references do
/// not exist here, so the long-term guard on colZeroFlag is vacuously
/// satisfied.
pub fn spatial_direct(
    st: &MbMotionState,
    col: &PicMotion,
    addr: usize,
) -> ([i8; 2], [[Mv; 2]; 4]) {
    let mut ref_idx = st.direct_ref_idx();
    let mut mvp = [Mv::ZERO; 2];
    if ref_idx[0] < 0 && ref_idx[1] < 0 {
        ref_idx = [0, 0];
    } else {
        for l in 0..2 {
            if ref_idx[l] >= 0 {
                // The whole-macroblock median at this list's derived
                // reference index — `predict_mv`'s 16x16 case, which is
                // the plain median with no directional rule.
                mvp[l] = st.predict(l, ref_idx[l], 0, 0, 16, 16);
            }
        }
    }
    // colZeroFlag is derived **per 8x8 partition**, from that partition's
    // own colocated corner block — 0, 3, 12 and 15 in raster under
    // `direct_8x8_inference`, which this encoder's SPS always sets
    // (`colocated_block`, src/h264/mb.rs, 8.4.1.2.1). The four answers
    // agree only when the colocated macroblock has one motion; a
    // colocated 16x8 or 8x16 makes them differ, and reading block 0 for
    // all four was the last place INVARIANT(16x16-only) was still load
    // bearing. It cost a SELF failure on the first B picture over a
    // partitioned P one, which is exactly where the tag said to look.
    let mut mv = [[Mv::ZERO; 2]; 4];
    for part in 0..4 {
        let blk = colocated_block(true, part, 0);
        let (col_mv, col_ref) = col.colocated(addr, blk);
        let col_zero =
            col_ref == 0 && (-1..=1).contains(&col_mv.x) && (-1..=1).contains(&col_mv.y);
        for l in 0..2 {
            if ref_idx[l] >= 0 && !(ref_idx[l] == 0 && col_zero) {
                mv[part][l] = mvp[l];
            }
        }
    }
    (ref_idx, mv)
}

/// SATD of the source against the luma prediction for the given lists
/// and per-8x8 vectors, without touching `rec` — how the B candidates are
/// priced before one of them is committed.
///
/// Scored one 8x8 at a time because direct prediction can give the four
/// partitions different vectors. For a candidate whose four are equal the
/// sum is the same number the whole-macroblock SATD gave: the metric
/// tiles by 4x4, so a 16x16 SATD *is* the sum of its four 8x8 SATDs.
#[allow(clippy::too_many_arguments)]
fn b_luma_satd(
    ctx: &MeCtx,
    refs: [&[Recon]; 2],
    px: i32,
    py: i32,
    src: &[u8],
    src_stride: usize,
    used: [bool; 2],
    mv: &[[Mv; 2]; 4],
) -> u32 {
    let mut a = [0u8; 16 * PRED_STRIDE];
    let mut total = 0u32;
    for part in 0..4 {
        let (ox, oy) = (((part & 1) * 8) as i32, ((part >> 1) * 8) as i32);
        let s = &src[oy as usize * src_stride + ox as usize..];
        if used[0] && used[1] {
            let mut b = [0u8; 16 * PRED_STRIDE];
            let mut c = [0u8; 16 * PRED_STRIDE];
            luma_pred_into(ctx, &refs[0][0], px + ox, py + oy, mv[part][0], 8, 8, &mut a);
            luma_pred_into(ctx, &refs[1][0], px + ox, py + oy, mv[part][1], 8, 8, &mut b);
            (ctx.dsp.avg)(&mut c, PRED_STRIDE, &a, &b, 8, 8);
            total += (ctx.dist.satd)(s, src_stride, &c, PRED_STRIDE, 8, 8);
        } else {
            let l = if used[0] { 0 } else { 1 };
            luma_pred_into(ctx, &refs[l][0], px + ox, py + oy, mv[part][l], 8, 8, &mut a);
            total += (ctx.dist.satd)(s, src_stride, &a, PRED_STRIDE, 8, 8);
        }
    }
    total
}

/// Decide and code one B macroblock, leaving its reconstruction in `rec`.
///
/// `refs` are the list-0 (past) and list-1 (future) reference pictures'
/// planes, borders replicated; `nb` the neighbouring motion per list;
/// `col` the list-1 reference's motion and `addr` this macroblock's
/// address in it (see [`spatial_direct`]). The candidates are direct, L0, L1 and
/// bi-predictive 16x16 — direct keeps ties, because its syntax is
/// cheapest — with the intra fallback consulted last, exactly as in the
/// P path. `B_Skip` is chosen when direct won and no residual survived;
/// unlike `P_Skip` there is no vector-equality leg, because direct motion
/// *is* the derived motion.
///
/// On [`BMbKind::UseIntra`] the reconstruction planes are untouched: the
/// caller runs the intra decision, which writes them itself.
#[allow(clippy::too_many_arguments)]
pub fn code_macroblock_b16(
    ctx: &MeCtx,
    rec: &mut [Recon],
    refs: [&[Recon]; 2],
    mb_x: usize,
    mb_y: usize,
    src_luma: &[u8],
    luma_stride: usize,
    src_chroma: [&[u8]; 2],
    chroma_stride: usize,
    st: &mut MbMotionState,
    col: &PicMotion,
    addr: usize,
) -> BDecision {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let soff = py * luma_stride + px;
    let src = &src_luma[soff..];
    let mut out = BDecision::default();

    // Direct first: it wins ties, because it costs no motion syntax.
    let (dref, dmv) = spatial_direct(st, col, addr);
    let dused = [dref[0] >= 0, dref[1] >= 0];
    let mut best_cost = b_luma_satd(ctx, refs, px as i32, py as i32, src, luma_stride, dused, &dmv);
    out.kind = BMbKind::BDirect16;
    out.used = dused;
    out.mv = dmv;
    out.ref_idx = dref;

    // Explicit candidates: one search per list around that list's median
    // predictor, then the bi combination of the two winners.
    let pred = [st.predict(0, 0, 0, 0, 16, 16), st.predict(1, 0, 0, 0, 16, 16)];
    let (mv0, _) = search_rect(ctx, &refs[0][0], px as i32, py as i32, 16, 16, src, luma_stride, pred[0]);
    let (mv1, _) = search_rect(ctx, &refs[1][0], px as i32, py as i32, 16, 16, src, luma_stride, pred[1]);
    for (used, label_mv) in [
        ([true, false], [mv0, Mv::ZERO]),
        ([false, true], [Mv::ZERO, mv1]),
        ([true, true], [mv0, mv1]),
    ] {
        // One vector per list over the whole macroblock: the same four
        // 8x8 entries, which is what an explicit 16x16 partition means.
        let uniform = [label_mv; 4];
        let cost =
            b_luma_satd(ctx, refs, px as i32, py as i32, src, luma_stride, used, &uniform);
        if cost < best_cost {
            best_cost = cost;
            out.kind = BMbKind::B16;
            out.used = used;
            out.mv = uniform;
            out.ref_idx = [if used[0] { 0 } else { -1 }, if used[1] { 0 } else { -1 }];
        }
    }
    if out.kind == BMbKind::B16 {
        for l in 0..2 {
            if out.used[l] {
                out.mvd[l] = Mv::new(out.mv[0][l].x - pred[l].x, out.mv[0][l].y - pred[l].y);
            }
        }
    }

    if placeholder_inter_or_intra(ctx.dist, best_cost, src, luma_stride) {
        out.kind = BMbKind::UseIntra;
        return out;
    }

    // Commit: the winner's motion into the state and its prediction into
    // the reconstruction planes, one 8x8 at a time — direct's four can
    // differ, and an explicit macroblock's four agree, so the same loop
    // serves both.
    st.reset_mb();
    for part in 0..4 {
        let (ox, oy) = ((part & 1) * 8, (part >> 1) * 8);
        st.commit_part(
            ox,
            oy,
            8,
            8,
            [
                out.used[0].then_some(out.mv[part][0]),
                out.used[1].then_some(out.mv[part][1]),
            ],
        );
        predict_inter_rect(ctx, rec, refs, px + ox, py + oy, 8, 8, out.used, out.mv[part]);
    }
    let r = code_inter_mb_residual(ctx, rec, mb_x, mb_y, src_luma, luma_stride, src_chroma, chroma_stride);
    out.transform_8x8 = r.transform_8x8;
    out.cbp_luma = r.cbp_luma;
    out.cbp_chroma = r.cbp_chroma;
    out.luma = r.luma;
    out.nz_luma = r.nz_luma;
    out.chroma_dc = r.chroma_dc;
    out.chroma_ac = r.chroma_ac;
    out.nz_chroma = r.nz_chroma;

    // B_Skip is direct with nothing left to say.
    if out.kind == BMbKind::BDirect16 && out.cbp_luma == 0 && out.cbp_chroma == 0 {
        out.kind = BMbKind::BSkip;
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
    use crate::h264::mb::{MbKind as DecKind, MbNeighbours, MotionCache, PicInfo};
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
                c444: false,
                t8x8: false,
                subparts: false,
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

    /// A macroblock with no neighbours at all: every cache entry absent,
    /// nothing of the macroblock itself derived. Which is exactly what a
    /// fresh state is, so this names the intent rather than building
    /// anything.
    fn absent_state() -> MbMotionState {
        MbMotionState::new()
    }

    /// A state whose four neighbouring macroblocks all carry `motion`
    /// per list — built the way the picture walk builds one, out of a
    /// real `Frame` and `PicInfo` gathered through the decoder's own
    /// `MotionCache`, so the test exercises the same path the encoder
    /// runs rather than a convenient stand-in.
    ///
    /// The centre macroblock of a 3x3 picture is address 4; A, B, C and D
    /// are 3, 1, 2 and 0.
    fn state_with_neighbours(per_mb: &[(usize, [Option<Mv>; 2])]) -> MbMotionState {
        let mut frame = Frame::<u8>::empty();
        frame.mb_width = 3;
        frame.mb_height = 3;
        frame.motion = [vec![BlockMotion::default(); 9 * 16], vec![BlockMotion::default(); 9 * 16]];
        frame.mb_intra = vec![false; 9];
        let mut info = PicInfo::new(3, 3);
        for &(addr, per_list) in per_mb {
            info.mbs[addr].decoded = true;
            info.mbs[addr].slice = 0;
            info.mbs[addr].kind =
                if per_list.iter().all(|m| m.is_none()) { DecKind::I16x16 } else { DecKind::Inter16x16 };
            for (l, mv) in per_list.iter().enumerate() {
                let Some(mv) = mv else { continue };
                let bm = BlockMotion {
                    mv: *mv,
                    ref_idx: 0,
                    ref_parity: PARITY_FRAME,
                    ref_id: 1 + l as u16,
                };
                frame.motion[l][addr * 16..addr * 16 + 16].fill(bm);
            }
        }
        let mut st = MbMotionState::new();
        let mut nb = MbNeighbours::default();
        st.start(&frame, &info, 4, &mut nb);
        st
    }

    /// All four neighbours of the centre macroblock carrying one list-0
    /// vector each.
    fn state_all(a: Mv, b: Mv, c: Mv, d: Mv) -> MbMotionState {
        state_with_neighbours(&[
            (3, [Some(a), None]),
            (1, [Some(b), None]),
            (2, [Some(c), None]),
            (0, [Some(d), None]),
        ])
    }

    /// Code the centre macroblock of the 3x3 test picture.
    /// Code the centre macroblock, returning the decision and the
    /// list-0 vector it derived for block 0 — which now lives in the
    /// motion state rather than on the decision, because a macroblock
    /// no longer has just the one.
    fn code_centre(
        ctx: &MeCtx,
        rec: &mut [Recon],
        refp: &[Recon],
        srcy: &[u8],
        srcc: [&[u8]; 2],
        st: &mut MbMotionState,
    ) -> (InterDecision, Mv) {
        let d = code_macroblock_p(ctx, rec, refp, 1, 1, srcy, 48, srcc, 24, st);
        let mv = st.motion()[0][0].mv;
        (d, mv)
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
            let mut st = state_all(mv, mv, mv, mv);
            let mut rec = fresh_rec();
            let (d, mv0) = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut st);
            assert_eq!(mv0, mv, "({dx},{dy}) with agreeing neighbours");
            assert_eq!(d.cbp_luma, 0, "({dx},{dy}) residual should vanish");
            assert_eq!(d.cbp_chroma, 0, "({dx},{dy})");
            assert!(d.nz_luma.iter().all(|&n| n == 0));
            assert_eq!(d.kind, InterMbKind::PSkip, "({dx},{dy})");
            assert_eq!(d.mvd[0], Mv::ZERO);

            // A zero-motion reference-0 left neighbour forces the skip
            // vector to zero (8.4.1.1), so a moved macroblock must not
            // skip even with an otherwise perfect prediction. The median
            // of (zero, truth, truth) is still the truth, so the search
            // is still seeded at the answer.
            if mv != Mv::ZERO {
                let mut st = state_all(Mv::ZERO, mv, mv, mv);
                let mut rec = fresh_rec();
                let (d, mv0) = code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut st);
                assert_eq!(mv0, mv, "({dx},{dy})");
                assert_eq!(d.kind, InterMbKind::P16x16, "({dx},{dy}) skip vector is zero, must not skip");
            }

            // No neighbours at all: the predictor is zero, so the diamond
            // must walk to the answer on its own; and the skip vector is
            // zero, so only the untranslated case may skip.
            if dx.abs() <= 8 && dy.abs() <= 8 {
                let mut rec = fresh_rec();
                let (d, mv0) =
                code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut absent_state());
                assert_eq!(mv0, mv, "({dx},{dy}) from a zero seed");
                assert_eq!(d.cbp_luma, 0);
                assert_eq!(d.cbp_chroma, 0);
                let want = if mv == Mv::ZERO { InterMbKind::PSkip } else { InterMbKind::P16x16 };
                assert_eq!(d.kind, want, "({dx},{dy}) skip legality");
                assert_eq!(d.mvd[0], mv, "mvd against a zero predictor is the vector itself");
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
        let (d, _mv0) =
                code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut absent_state());
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
            let (d, mv0) =
                code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut absent_state());
            assert_eq!(mv0, mv, "quarter vector ({qx},{qy})");
            assert_eq!(d.cbp_luma, 0, "({qx},{qy}) the residual is zero by construction");
            assert_eq!(d.cbp_chroma, 0, "({qx},{qy})");
            assert!(d.nz_luma.iter().all(|&n| n == 0));
        }
    }

    /// The spatial direct mirror agrees with the decoder's own pieces:
    /// the per-list reference indices with `spatial_direct_ref_idx` over
    /// a real `MotionCache`, the median vectors with
    /// `prediction_neighbours` + `median_mvp`, and the colocated read
    /// with `colocated_motion` over a real colocated `Frame` — across
    /// present/absent/intra neighbour mixes in both lists and colocated
    /// records that are intra, still, and moving.
    ///
    /// The reference indices and the median vectors are now literal calls
    /// into the decoder, so what this still earns is the *colZeroFlag*
    /// half — and that half is now the decoder's `colocated_motion` over
    /// the colocated picture's real per-4x4 motion, read at the
    /// partition's own corner block. What this still earns is the
    /// *composition*: that the reference indices, the median vectors and
    /// colZeroFlag are combined into 8.4.1.2.2's answer the way the
    /// decoder combines them.
    #[test]
    fn spatial_direct_agrees_with_the_decoders_derivation() {
        use crate::h264::mb::{
            colocated_motion, median_mvp as dec_median, prediction_neighbours,
            spatial_direct_ref_idx,
        };
        let mut seed = 91u64;
        let mut frame = Frame::<u8>::new(3, 3, ChromaFormat::Yuv420, 8, false);
        // A colocated frame whose single macroblock record we vary.
        let mut colf = Frame::<u8>::new(3, 3, ChromaFormat::Yuv420, 8, false);
        let cur_addr = 4usize;
        let nbs = [(3usize, 0usize), (1, 1), (2, 2), (0, 3)];
        for mask in 0..16u32 {
            for draw in 0..6 {
                let mut info = PicInfo::new(3, 3);
                for &(addr, slot) in &nbs {
                    if mask & (1 << slot) == 0 {
                        continue;
                    }
                    // Each present neighbour: per list, one of unused /
                    // used-with-a-vector; intra when both unused.
                    let mut used_any = false;
                    for l in 0..2 {
                        if lcg(&mut seed) % 3 != 0 {
                            let mv = Mv::new(
                                (lcg(&mut seed) % 33) as i16 - 16,
                                (lcg(&mut seed) % 33) as i16 - 16,
                            );
                            frame.motion[l][addr * 16..addr * 16 + 16].fill(BlockMotion {
                                mv,
                                ref_idx: 0,
                                ref_parity: PARITY_FRAME,
                                ref_id: 1 + l as u16,
                            });
                            used_any = true;
                        } else {
                            frame.motion[l][addr * 16..addr * 16 + 16]
                                .fill(BlockMotion::default());
                        }
                    }
                    info.mbs[addr].decoded = true;
                    info.mbs[addr].slice = 0;
                    info.mbs[addr].kind =
                        if used_any { DecKind::Inter16x16 } else { DecKind::I16x16 };
                }
                // The colocated record: rotate through intra, still (a
                // vector inside the +-1 window), and moving; list 1 only
                // on some draws, exercising `colocated_motion`'s list
                // preference.
                let col_case = draw % 3;
                let col_mv = match col_case {
                    0 => Mv::ZERO,
                    1 => Mv::new(1, -1),
                    _ => Mv::new((lcg(&mut seed) % 21) as i16 + 2, 0),
                };
                let col_list1_only = draw % 2 == 1;
                let col_intra = draw == 5;
                colf.mb_intra[cur_addr] = col_intra;
                for l in 0..2 {
                    let uses = !col_intra && (l == 1 || !col_list1_only);
                    colf.motion[l][cur_addr * 16..cur_addr * 16 + 16].fill(if uses {
                        BlockMotion { mv: col_mv, ref_idx: 0, ref_parity: PARITY_FRAME, ref_id: 9 }
                    } else {
                        BlockMotion::default()
                    });
                }
                // The colocated picture as the encoder now stores one:
                // real per-4x4 motion, which is what `colocated_motion`
                // reads.
                let mut col = PicMotion::new(3, 3);
                let mut col_mot = [[BlockMotion::default(); 16]; 2];
                for l in 0..2 {
                    let uses = !col_intra && (l == 1 || !col_list1_only);
                    if uses {
                        col_mot[l] = [BlockMotion {
                            mv: col_mv,
                            ref_idx: 0,
                            ref_parity: PARITY_FRAME,
                            ref_id: 9,
                        }; 16];
                    }
                }
                col.commit(
                    cur_addr,
                    crate::h264::mb::MbInfo {
                        kind: if col_intra { DecKind::I16x16 } else { DecKind::Inter16x16 },
                        decoded: true,
                        slice: 0,
                        ..crate::h264::mb::MbInfo::default()
                    },
                    &col_mot,
                );

                // Decoder side.
                info.mbs[cur_addr].decoded = false;
                let mut dnb = MbNeighbours::default();
                dnb.derive_into(&info, cur_addr, 0);
                let mut cache = MotionCache::default();
                cache.gather(&dnb, &frame, &info);
                let cur = [[BlockMotion::default(); 16]; 2];
                let mut want_ref = spatial_direct_ref_idx(&cache, &cur);
                let mut want_mv = [Mv::ZERO; 2];
                if want_ref[0] < 0 && want_ref[1] < 0 {
                    want_ref = [0, 0];
                } else {
                    for list in 0..2 {
                        if want_ref[list] >= 0 {
                            let (a, b, c) = prediction_neighbours(&cache, &cur, 0, list, 0, 0, 4);
                            want_mv[list] = dec_median(a, b, c, want_ref[list]);
                        }
                    }
                }
                let (mv_col, ref_col, _, _) = colocated_motion(&colf, cur_addr, 0);
                let col_zero = ref_col == 0
                    && (-1..=1).contains(&mv_col.x)
                    && (-1..=1).contains(&mv_col.y);
                for list in 0..2 {
                    if want_ref[list] >= 0 && want_ref[list] == 0 && col_zero {
                        want_mv[list] = Mv::ZERO;
                    }
                    if want_ref[list] < 0 {
                        want_mv[list] = Mv::ZERO;
                    }
                }

                // Mine, over the same state the picture walk would hold.
                let mut st = MbMotionState::new();
                st.start(&frame, &info, cur_addr, &mut dnb);
                let (got_ref, got_mv) = spatial_direct(&st, &col, cur_addr);
                assert_eq!(got_ref, want_ref, "mask {mask:04b} draw {draw} ref");
                // The colocated macroblock here has one motion, so all
                // four 8x8 answers must agree with the whole-macroblock
                // one; `differing_colocated_partitions_reach_direct` is
                // the case where they must not.
                for part in 0..4 {
                    assert_eq!(got_mv[part], want_mv, "mask {mask:04b} draw {draw} part {part}");
                }
            }
        }
    }

    /// A colocated macroblock whose halves move differently must give
    /// direct prediction two different answers.
    ///
    /// This is the case the whole colocated storage exists for. colZeroFlag
    /// is derived per 8x8 from that partition's own colocated corner
    /// (8.4.1.2.1) — raster blocks 0, 3, 12 and 15 — and while every
    /// colocated macroblock had one motion the four corners agreed and
    /// reading block 0 for all of them was exact. A colocated 16x8 breaks
    /// that: here its upper half is still and its lower half moving, so
    /// the upper two partitions take the zero vector and the lower two
    /// the median prediction.
    ///
    /// It cost a SELF failure on `--subparts --bframes 2` to find, which
    /// is the failure INVARIANT(16x16-only) predicted in so many words.
    #[test]
    fn differing_colocated_partitions_reach_direct() {
        // One available neighbour carrying a list-0 vector, so the direct
        // reference indices come out [0, -1] and the median prediction is
        // that vector — a non-zero mvp, without which "zero because
        // colZero" and "zero anyway" would be indistinguishable.
        let nbmv = Mv::new(12, -8);
        let mut st = state_with_neighbours(&[
            (3, [Some(nbmv), None]),
            (1, [Some(nbmv), None]),
            (2, [Some(nbmv), None]),
        ]);
        let _ = &mut st;

        let mut col = PicMotion::new(3, 3);
        let mut mot = [[BlockMotion::default(); 16]; 2];
        // Upper 16x8 still (colZero true), lower 16x8 moving.
        for blk in 0..16 {
            let moving = blk >= 8;
            mot[0][blk] = BlockMotion {
                mv: if moving { Mv::new(40, 40) } else { Mv::ZERO },
                ref_idx: 0,
                ref_parity: PARITY_FRAME,
                ref_id: 1,
            };
        }
        col.commit(
            4,
            crate::h264::mb::MbInfo {
                kind: DecKind::Inter16x8,
                decoded: true,
                slice: 0,
                ..crate::h264::mb::MbInfo::default()
            },
            &mot,
        );

        let (refs, mv) = spatial_direct(&st, &col, 4);
        assert_eq!(refs, [0, -1], "one list-0 neighbour gives reference 0 on list 0 only");
        // Partitions 0 and 1 are the upper half: their colocated corners
        // are blocks 0 and 3, both still, so colZeroFlag holds.
        assert_eq!(mv[0][0], Mv::ZERO, "upper-left takes zero from a still colocated corner");
        assert_eq!(mv[1][0], Mv::ZERO, "upper-right likewise");
        // Partitions 2 and 3 are the lower half: corners 12 and 15, both
        // moving, so the median prediction stands.
        assert_eq!(mv[2][0], nbmv, "lower-left takes the median prediction");
        assert_eq!(mv[3][0], nbmv, "lower-right likewise");
        assert_ne!(mv[0][0], mv[2][0], "the point of the test is that they differ");
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
            let (d, mv0) =
                code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut absent_state());
            assert_eq!(d.kind, InterMbKind::P16x16, "qp={qp} inter must win on matched content");

            // Luma: prediction through the same decoder kernel, residual
            // through the test-only inverse pair.
            let mut pred = [0u8; 16 * PRED_STRIDE];
            luma_pred_into(&ctx, &refp[0], 16, 16, mv0, 16, 16, &mut pred);
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
                chroma_pred_into(&ctx, &refp[comp + 1], 8, 8, mv0, 8, 8, 8, &mut cpred);
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
            let (d, _mv0) =
                code_centre(&ctx, &mut rec, &refp, &srcy, [&srcb, &srcr], &mut absent_state());
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
