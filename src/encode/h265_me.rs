//! Motion estimation and inter mode decision for H.265 P coding units.
//!
//! The inter counterpart of [`super::h265_intra`], and the H.265 sibling of
//! [`super::h264_me`]: the *deciding* half of coding a P CU — which motion,
//! through which signalling (skip, merge, or AMVP plus a difference), what
//! the quantised coefficients are, and what the reconstruction looks like.
//! Turning that into bits belongs to the CABAC coding-tree writer; the seam
//! is [`InterCuDecision`] — data, not a shared file, shaped after
//! [`super::h265_intra::CuDecision`] so the serialisers meet a familiar
//! layout.
//!
//! The two rules of the other decision modules hold unchanged:
//!
//! **The prediction is the decoder's own.** The chosen candidate's
//! prediction runs through `crate::hevc::inter::predict_block` — the very
//! function the decoder's `prediction_unit` calls, fused kernels, window
//! clamping and all — reading the *reconstructed* reference frame. An
//! encoder that predicted from source samples, or interpolated one rounding
//! differently, would desync, and inter desync compounds picture over
//! picture.
//!
//! **So is the signalling derivation — not mirrored, called.** Choosing a
//! `merge_idx` or an `mvp_l0_flag` requires knowing the candidate lists the
//! decoder will build, and this module gets them by calling the decoder's
//! own `crate::hevc::mvpred::merge_candidate` and
//! `crate::hevc::mvpred::amvp` — the functions `prediction_unit` in
//! `src/hevc/ctu.rs` calls, conformance-proven against the JCT-VC suites.
//! Drift between the two derivations is therefore impossible rather than
//! merely tested against. What that costs is state: those functions read a
//! decoder-grade `PicInfo` (z-scan availability, 6.4.2) and the frame's
//! per-4x4 motion grid, so [`InterPicture`] maintains both exactly as the
//! decoder does — the motion fill mirrors the "Store motion" block of
//! `prediction_unit` (`ref_delta` as POC differences, long-term flags), and
//! the CTB / pred-mode marks mirror what `available_at` consults. The
//! replay test at the bottom re-derives every decision on an independently
//! maintained state to prove the maintenance, not the derivation.
//!
//! **TMVP is foreclosed by the bitstream, not skipped by this module.**
//! `write_sps` writes `sps_temporal_mvp_enabled_flag` = 0, so the slice
//! header never carries the slice-level flag and the decoder derives
//! temporal MVP off. The `RefCtx` here says the same (`tmvp: false`,
//! no collocated picture), which makes the spatial + zero candidate set
//! the *complete* derivation for these streams, not a subset.
//!
//! # Scope (v1) — the same deliberately fixed geometry as the intra module
//!
//! - **P slices, one reference** (list 0, `ref_idx` 0), `PART_2Nx2N`
//!   whole-CTU CUs, one CU-sized TU (no transform split — the SPS's
//!   maximum transform size equals the CTB size precisely so this shape is
//!   representable). 4:2:0, fixed QP, no weighted prediction
//!   (`Weighting::Default`).
//! - **`MaxNumMergeCand` = 5**: the slice header this decision assumes
//!   writes `five_minus_max_num_merge_cand` = 0. The PPS writes
//!   `log2_parallel_merge_level_minus2` = 0 (level 2: no merge-list
//!   sharing), and `RefCtx` says both.
//!
//! # The search, named
//!
//! Full-sample: a greedy small-diamond descent on SAD over the reference
//! luma plane, seeded at the two AMVP predictors, the zero vector and every
//! merge candidate, confined to ±`SEARCH_RANGE` full samples around the
//! best seed and to the padded plane (HEVC clamps out-of-picture reads, so
//! any vector is *legal* — the confinement merely keeps the SAD reads
//! direct). Then two SATD refinement rings — the eight half-sample
//! neighbours, then the eight quarter-sample neighbours of the winner —
//! scored on a luma-only prediction through the decoder's own kernels
//! (the addressing mirrors `interp` / `source` in `src/hevc/inter.rs`).
//! Merge candidates are scored at their exact vectors. The one rate
//! heuristic is `lambda` times an approximate bin count per signalling
//! shape — the same single-function placeholder policy as the intra
//! module's `mode_signalling_cost`, replaced wholesale when a real bit
//! count exists.
//!
//! # What the writer serialises (the seam contract)
//!
//! For a P-slice 2Nx2N inter CU, in the reader's order (`coding_unit` /
//! `prediction_unit` in `src/hevc/ctu.rs`):
//!
//! - [`InterCuKind::Skip`]: `cu_skip_flag` 1, `merge_idx` (TR, cMax
//!   `MaxNumMergeCand` − 1). Nothing else — `rqt_root_cbf` is inferred 0.
//! - [`InterCuKind::Merge`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` 2Nx2N, `merge_flag` 1, `merge_idx`. `rqt_root_cbf` is
//!   **not coded** — the reader infers it true for a non-skip 2Nx2N merge
//!   CU (ctu.rs's `!(part_mode == P2Nx2N && last_pu_merged)` gate), which
//!   is why this module only produces `Merge` when residual survived (a
//!   zero-residual merge becomes `Skip`; an invariant a test holds).
//!   Then the transform tree: `split_transform_flag` 0, `cbf_cb`,
//!   `cbf_cr`, `cbf_luma`, and the coded residuals.
//! - [`InterCuKind::Amvp`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` 2Nx2N, `merge_flag` 0; no `inter_pred_idc` (P slice), no
//!   `ref_idx_l0` (one active reference); `mvd_l0` (as
//!   `abs_mvd_greater0_flag`, `abs_mvd_greater1_flag`, `abs_mvd_minus2`,
//!   `mvd_sign_flag` per component), `mvp_l0_flag`; then `rqt_root_cbf`,
//!   and when set the same unsplit transform tree.
//! - [`InterCuKind::UseIntra`]: code the CU with the intra decision
//!   instead; this module's reconstruction and coefficients are
//!   meaningless and its planes untouched. See [`prefer_intra`].
//!
//! Coefficients are raster within each TU, exactly as in `CuDecision`; the
//! writer derives the scan (always diagonal for inter TUs — the
//! mode-dependent scans are intra-only, 7.4.9.11).
//!
//! # Duplication, flagged
//!
//! `code_residual_inter` is `h265_intra::code_residual` minus the intra
//! DST case and with the inter quantisation offset — copied, not shared,
//! because sharing would mean editing that module's private function into a
//! public one under a parallel delivery. Fold the two together when the
//! files next change hands. `lambda` duplicates the intra module's
//! Lagrangian constant for the same reason.

use crate::dsp::hevc_enc::{qbits, quant_offset, quant_scale};
use crate::encode::h265_intra::IntraCtx;
use crate::hevc::ctu::chroma_qp;
use crate::hevc::frame::{Frame, MotionInfo, Mv, Plane16, fill_motion};
use crate::hevc::inter::{McScratch, Weighting, predict_block};
use crate::hevc::mvpred::{Cand, PuPos, RefCtx, amvp, merge_candidate};
use crate::hevc::pic::{Geometry, PicInfo};
use crate::hevc::pps::Pps;
use crate::hevc::residual::{ScalingSource, scale_coefficients};
use crate::hevc::sps::Sps;
use crate::picture::ChromaFormat;
use crate::sample::Sample;

/// Everything inter coding needs that does not change per CU — the *same*
/// struct the intra module takes, aliased rather than repeated, for the
/// H.264 side's reason: one context serving both modules means the picture
/// loop cannot hand the two halves of a CU decision different QPs.
pub type MeCtx<'a, S> = IntraCtx<'a, S>;

/// `MaxNumMergeCand` this decision assumes the slice header will declare
/// (`five_minus_max_num_merge_cand` = 0).
pub const MAX_MERGE_CAND: usize = 5;

/// Full-sample search confinement around the best seed, in luma samples.
/// Kept well inside the frame's `LUMA_PAD` (80) so every full-sample SAD
/// reads the padded plane directly.
const SEARCH_RANGE: i32 = 48;

/// The CU-level choices this module decides between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterCuKind {
    /// `cu_skip_flag` 1: merge motion, no residual syntax at all. Chosen
    /// exactly when a merge candidate wins the cost comparison *and* no
    /// level survived quantisation.
    Skip {
        /// `merge_idx`.
        merge_idx: u8,
    },
    /// Non-skip 2Nx2N merge: merge motion with a coded residual
    /// (`rqt_root_cbf` inferred 1 by the reader — see the module header).
    Merge {
        /// `merge_idx`.
        merge_idx: u8,
    },
    /// AMVP: `mvp_l0_flag` picks the predictor, `mvd_l0` carries the rest.
    Amvp {
        /// `mvp_l0_flag`.
        mvp_flag: u8,
        /// `mv` minus the chosen predictor, wrapping i16 (the reader's
        /// `uLX` sum is a wrapping add, ctu.rs).
        mvd: Mv,
    },
    /// Inter lost to the flatness proxy: code this CU with the intra
    /// decision instead. Only `mv` is meaningful (callers feeding motion
    /// state must *not* use it — an intra CU's motion is
    /// `MotionInfo::INTRA`, which `InterPicture::code_ctu` has already
    /// stored in that case).
    UseIntra,
}

/// How one P CU was coded, in the form the coding-tree writer needs.
/// Produced once per CTU and meant to be consumed immediately.
#[derive(Clone)]
pub struct InterCuDecision {
    /// log2 of the CU (== CTB) size, 4 or 5.
    pub log2_cu: u32,
    /// The choice, with its signalling payload.
    pub kind: InterCuKind,
    /// The chosen motion vector, quarter luma samples, list 0. Filled for
    /// every kind including `Skip` (the writer carries no vector, but
    /// callers and tests want the motion the CU actually has).
    pub mv: Mv,
    /// Reference index into list 0. Always 0: this module searches exactly
    /// one reference. The field exists so the serialiser's contract does
    /// not change when more arrive.
    pub ref_idx: i8,
    /// `rqt_root_cbf` as the *reader* resolves it: false for `Skip`, true
    /// for `Merge` (inferred, never coded), the explicit value for `Amvp`
    /// (coded). When false, no transform tree follows and every
    /// coefficient below is zero.
    pub rqt_root_cbf: bool,
    /// `cbf_luma` of the single CU-sized luma TU.
    pub cbf_luma: bool,
    /// `cbf_cb` / `cbf_cr` of the single chroma TU per component.
    pub cbf_chroma: [bool; 2],
    /// Quantised luma levels of the one `n x n` TU (`n = 1 << log2_cu`),
    /// raster, at `[0..n*n]`. Entries beyond are zero and meaningless.
    pub luma: [i16; 1024],
    /// Quantised chroma levels per component (`[0]` Cb, `[1]` Cr), one
    /// `nc x nc` TU (`nc = n/2`), raster, at `[0..nc*nc]`.
    pub chroma: [[i16; 256]; 2],
}

impl Default for InterCuDecision {
    fn default() -> Self {
        InterCuDecision {
            log2_cu: 0,
            kind: InterCuKind::Skip { merge_idx: 0 },
            mv: Mv::ZERO,
            ref_idx: 0,
            rqt_root_cbf: false,
            cbf_luma: false,
            cbf_chroma: [false; 2],
            luma: [0; 1024],
            chroma: [[0; 256]; 2],
        }
    }
}

/// Per-picture state of the P-picture walk: the reconstruction the next
/// CUs and the next picture predict from, and the decoder-grade side state
/// the candidate derivation reads. The caller walks CTUs in raster order
/// and calls `InterPicture::code_ctu` for each.
pub struct InterPicture<S: Sample> {
    /// Decoder-grade per-picture arrays: `merge_candidate` / `amvp` read
    /// z-scan availability (6.4.1) out of these, so this module maintains
    /// what the decoder maintains — `ctb_slice_addr` / `ctb_slice` marked
    /// as the walk reaches each CTB, `pred_mode` filled per CU.
    pub info: PicInfo,
    /// The reconstruction, decoder-identical by construction, with the
    /// per-4x4 motion grid filled exactly as `prediction_unit` fills it
    /// ("Store motion", src/hevc/ctu.rs) — `merge_candidate` / `amvp`
    /// read neighbour motion from here, and the next picture's TMVP would
    /// too if the SPS ever enables it.
    pub recon: Frame<S>,
    /// log2 of the fixed CU size, 4 or 5.
    pub log2_cu: u32,
    /// The current picture's POC (the motion grid stores POC differences).
    pub cur_poc: i32,
    /// MC scratch, as the decoder allocates per slice.
    scratch: McScratch<S>,
    /// Luma-only prediction scratch for candidate scoring: the clamp
    /// window, the two-stage filter intermediate, the 14-bit prediction
    /// and the sample-domain prediction.
    swin: Vec<S>,
    stmp: Vec<i16>,
    spred14: Vec<i16>,
    spred: Vec<S>,
}

impl<S: Sample> InterPicture<S> {
    /// State for one P picture, from the *parsed* parameter sets — the
    /// caller round-trips the bytes `write_sps` / `write_pps` produced
    /// through the decoder's own parsers, the same proof the encoder
    /// applies to everything it writes. Fixed geometry as in the intra
    /// module: whole CTUs, 4:2:0, `log2_cu` 4 or 5.
    pub fn new(sps: &Sps, pps: &Pps, cur_poc: i32) -> Self {
        assert!((4..=5).contains(&sps.log2_ctb_size), "log2_ctb {} outside 4..=5", sps.log2_ctb_size);
        assert_eq!(sps.chroma_array_type(), 1, "inter decision models 4:2:0 only");
        let (w, h) = (sps.width as usize, sps.height as usize);
        let n = 1usize << sps.log2_ctb_size;
        assert!(w.is_multiple_of(n) && h.is_multiple_of(n), "{w}x{h} is not a whole number of {n}x{n} CTUs");
        let geo = std::sync::Arc::new(Geometry::new(sps, pps));
        let info = PicInfo::new(geo);
        let mut recon = Frame::new(w, h, ChromaFormat::Yuv420, sps.bit_depth_luma);
        recon.poc = cur_poc;
        let nmax = 1usize << (2 * sps.log2_ctb_size);
        InterPicture {
            info,
            recon,
            log2_cu: sps.log2_ctb_size,
            cur_poc,
            scratch: McScratch::new(),
            swin: vec![S::default(); (64 + 7) * (64 + 7)],
            stmp: vec![0; crate::dsp::hevc::MC_TMP_LEN],
            spred14: vec![0; nmax],
            spred: vec![S::default(); nmax],
        }
    }

    /// The reference-list context the decoder's candidate derivation
    /// takes, for a P slice referencing exactly `ref_poc`. `tmvp` false
    /// and no collocated picture: the SPS this stream carries disables
    /// temporal MVP (see the module header).
    fn ref_ctx<'a>(&self, ref_poc: i32) -> RefCtx<'a, S> {
        RefCtx {
            pocs: [vec![ref_poc], Vec::new()],
            long_term: [vec![false], Vec::new()],
            col: None,
            cur_poc: self.cur_poc,
            no_backward_pred: true,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: false,
            num_ref_idx: [1, 0],
            col_from_l0: true,
        }
    }

    /// Decide and code one CTU (== one 2Nx2N CU) against reference
    /// `refp` (whose borders must be extended — `Frame::extend_rows` —
    /// exactly as the decoder pads references before MC reads them).
    ///
    /// On [`InterCuKind::UseIntra`] the reconstruction planes are
    /// untouched and the motion grid holds `MotionInfo::INTRA`: the
    /// caller runs the intra decision, which writes the planes itself.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refp: &Frame<S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << self.log2_cu;
        let (x0, y0) = (cu_x * n, cu_y * n);
        let mut out = InterCuDecision { log2_cu: self.log2_cu, ..InterCuDecision::default() };

        // Mark the CTB as this (single) slice's, as the decoder does at CTB
        // start: `avail_ctx` reads the current CTB's slice address, and
        // `available_at` compares neighbours against it.
        let ctb = self.info.ctb_of(x0, y0);
        self.info.ctb_slice_addr[ctb] = 0;
        self.info.ctb_slice[ctb] = 0;

        let refs = self.ref_ctx(refp.poc);
        let pu = PuPos {
            x_cb: x0 as i32,
            y_cb: y0 as i32,
            n_cb: n as i32,
            x_pb: x0 as i32,
            y_pb: y0 as i32,
            w: n as i32,
            h: n as i32,
            part_idx: 0,
        };

        // The decoder's own candidate lists (see the module header).
        let merge: Vec<Cand> = (0..MAX_MERGE_CAND).map(|i| merge_candidate(&self.info, &self.recon, &refs, &pu, i)).collect();
        let mvp = [amvp(&self.info, &self.recon, &refs, &pu, 0, 0, 0), amvp(&self.info, &self.recon, &refs, &pu, 0, 0, 1)];

        let soff = y0 * y_stride + x0;
        let src = &src_y[soff..];

        // Full-sample descent from every distinct seed's best, then the
        // two sub-sample rings.
        let mut seeds: Vec<Mv> = vec![Mv::ZERO, mvp[0], mvp[1]];
        seeds.extend(merge.iter().filter(|c| c.ref_idx[0] == 0).map(|c| c.mv[0]));
        let full = self.search_full(ctx, &refp.y, x0, y0, n, src, y_stride, &seeds);
        let (mv_me, satd_me) = self.refine_subpel(ctx, &refp.y, x0, y0, n, src, y_stride, full);

        // Cost the shapes. Merge candidates are scored at their exact
        // vectors; only the first occurrence of a vector matters (a later
        // duplicate signals strictly more bins for the same prediction).
        let lam = lambda(ctx.qp);
        let mut best_merge: Option<(usize, u32)> = None; // (idx, satd)
        let mut best_merge_cost = f32::INFINITY;
        let mut seen: Vec<Mv> = Vec::with_capacity(MAX_MERGE_CAND);
        for (idx, cand) in merge.iter().enumerate() {
            if cand.ref_idx[0] != 0 || seen.contains(&cand.mv[0]) {
                continue;
            }
            seen.push(cand.mv[0]);
            let satd = self.satd_at(ctx, &refp.y, x0, y0, n, src, y_stride, cand.mv[0]);
            // cu_skip_flag or merge_flag, plus the TR-coded index.
            let bins = 2 + tr_bins(idx, MAX_MERGE_CAND - 1);
            let cost = satd as f32 + lam * bins as f32;
            if cost < best_merge_cost {
                best_merge_cost = cost;
                best_merge = Some((idx, satd));
            }
        }
        // AMVP: the searched vector against the cheaper of the two
        // predictors (the reader's uLX sum is a wrapping add, so the mvd
        // is a wrapping difference).
        let mvd_for = |p: Mv| Mv::new(mv_me.x.wrapping_sub(p.x), mv_me.y.wrapping_sub(p.y));
        let amvp_flag = if mvd_cost(mvd_for(mvp[1])) < mvd_cost(mvd_for(mvp[0])) { 1u8 } else { 0 };
        let mvd = mvd_for(mvp[amvp_flag as usize]);
        // merge_flag 0, mvp_l0_flag, rqt_root_cbf, plus the mvd bins
        // (cu_skip_flag and pred_mode/part_mode surround both shapes).
        let amvp_cost = satd_me as f32 + lam * (3 + mvd_cost(mvd)) as f32;

        let merge_wins = best_merge.is_some_and(|_| best_merge_cost <= amvp_cost);
        let (mv, inter_satd) = if merge_wins {
            let (idx, satd) = best_merge.expect("merge_wins implies a candidate");
            (merge[idx].mv[0], satd)
        } else {
            (mv_me, satd_me)
        };
        out.mv = mv;

        if prefer_intra(ctx, inter_satd, src, y_stride, n) {
            out.kind = InterCuKind::UseIntra;
            // An intra CU's motion, stored now so later candidate
            // derivations see what the decoder will see.
            fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, MotionInfo::INTRA);
            let w4 = self.info.w4;
            PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
            return out;
        }

        // The chosen prediction, through the decoder's own MC.
        predict_block(ctx.dsp, &mut self.scratch, &mut self.recon, x0, y0, n, n, Some((refp, mv)), None, [Weighting::Default; 3]);

        // Residual: one CU-sized luma TU, one chroma TU per component,
        // each reconstructed in place through the decoder's inverse path.
        let qp_l = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let nz_l = code_residual_inter(ctx, &mut self.recon.y, x0, y0, self.log2_cu, qp_l, src, y_stride, &mut out.luma);
        let bd_off = 6 * (ctx.bit_depth as i32 - 8);
        let qp_c = chroma_qp(1, ctx.qp.clamp(-bd_off, 57)) + bd_off;
        let (cx, cy, nc_log2) = (x0 / 2, y0 / 2, self.log2_cu - 1);
        let coff = cy * c_stride + cx;
        let nz_cb = code_residual_inter(ctx, &mut self.recon.cb, cx, cy, nc_log2, qp_c, &src_cb[coff..], c_stride, &mut out.chroma[0]);
        let nz_cr = code_residual_inter(ctx, &mut self.recon.cr, cx, cy, nc_log2, qp_c, &src_cr[coff..], c_stride, &mut out.chroma[1]);

        out.cbf_luma = nz_l != 0;
        out.cbf_chroma = [nz_cb != 0, nz_cr != 0];
        let any = nz_l + nz_cb + nz_cr != 0;
        out.rqt_root_cbf = any;
        out.kind = match (merge_wins, any) {
            (true, false) => InterCuKind::Skip { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (true, true) => InterCuKind::Merge { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (false, _) => InterCuKind::Amvp { mvp_flag: amvp_flag, mvd },
        };

        // Store motion and marks exactly as the decoder's `prediction_unit`
        // does after parsing ("Store motion", src/hevc/ctu.rs): the next
        // CUs' candidate lists read them.
        let mut mi = MotionInfo { mv: [mv, Mv::ZERO], ref_delta: [0; 2], ref_idx: [0, -1], flags: 0, pad: 0 };
        mi.ref_delta[0] = (self.cur_poc - refp.poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, mi);
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, matches!(out.kind, InterCuKind::Skip { .. }) as u8);
        out
    }

    /// Greedy small-diamond SAD descent at full-sample positions, seeded
    /// at each of `seeds` (rounded toward zero to full samples, as the
    /// decoder's `>> 2` addresses them), returning the best vector in
    /// quarter units.
    #[allow(clippy::too_many_arguments)]
    fn search_full(&self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, seeds: &[Mv]) -> Mv {
        let clamp_pos = |fx: i32, fy: i32| -> (i32, i32) {
            let pad = refp.pad as i32;
            let xi = (x as i32 + fx).clamp(-pad, refp.width as i32 + pad - n as i32);
            let yi = (y as i32 + fy).clamp(-pad, refp.height as i32 + pad - n as i32);
            (xi - x as i32, yi - y as i32)
        };
        let sad_of = |fx: i32, fy: i32| -> u32 {
            let off = refp.offset((x as i32 + fx) as isize, (y as i32 + fy) as isize);
            (ctx.dist.sad)(src, src_stride, &refp.data[off..], refp.stride, n, n)
        };
        let mut best = (0i32, 0i32);
        let mut best_sad = u32::MAX;
        for s in seeds {
            let (fx, fy) = clamp_pos(s.x as i32 >> 2, s.y as i32 >> 2);
            let sad = sad_of(fx, fy);
            if sad < best_sad {
                best_sad = sad;
                best = (fx, fy);
            }
        }
        let centre = best;
        // ±1 diamond, confined to SEARCH_RANGE around the seeded best and
        // to the padded plane.
        for _ in 0..(2 * SEARCH_RANGE) {
            let mut improved = false;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let cand = (best.0 + dx, best.1 + dy);
                if (cand.0 - centre.0).abs() > SEARCH_RANGE || (cand.1 - centre.1).abs() > SEARCH_RANGE {
                    continue;
                }
                if clamp_pos(cand.0, cand.1) != cand {
                    continue;
                }
                let sad = sad_of(cand.0, cand.1);
                if sad < best_sad {
                    best_sad = sad;
                    best = cand;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16)
    }

    /// SATD refinement: the eight half-sample neighbours of `start`, then
    /// the eight quarter-sample neighbours of that winner.
    #[allow(clippy::too_many_arguments)]
    fn refine_subpel(&mut self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, start: Mv) -> (Mv, u32) {
        let mut best = start;
        let mut best_satd = self.satd_at(ctx, refp, x, y, n, src, src_stride, start);
        for step in [2i16, 1] {
            let centre = best;
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let mv = Mv::new(centre.x.wrapping_add(dx), centre.y.wrapping_add(dy));
                    let satd = self.satd_at(ctx, refp, x, y, n, src, src_stride, mv);
                    if satd < best_satd {
                        best_satd = satd;
                        best = mv;
                    }
                }
            }
        }
        (best, best_satd)
    }

    /// Luma SATD of the prediction `mv` produces, through the decoder's
    /// own interpolation kernels — the addressing mirrors `interp` and
    /// `source` in `src/hevc/inter.rs`, and the sample-domain stage is the
    /// default uni-prediction the decoder applies.
    #[allow(clippy::too_many_arguments)]
    fn satd_at(&mut self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, mv: Mv) -> u32 {
        let xi = x as i32 + (mv.x as i32 >> 2);
        let yi = y as i32 + (mv.y as i32 >> 2);
        let (fx, fy) = ((mv.x & 3) as usize, (mv.y & 3) as usize);
        let bd = ctx.bit_depth;
        let shift1 = bd.min(12) as i32 - 8;
        let shift3 = 14 - bd as i32;
        // The filter window (3 samples before, 4 after), gathered with
        // clamping when it leaves the padded plane — `source`'s rule.
        let (x0, y0) = (xi - 3, yi - 3);
        let (ww, hh) = (n + 7, n + 7);
        let pad = refp.pad as i32;
        let inside = x0 >= -pad && y0 >= -pad && x0 + ww as i32 <= refp.width as i32 + pad && y0 + hh as i32 <= refp.height as i32 + pad;
        let (win, stride) = if inside {
            (&refp.data[refp.offset(x0 as isize, y0 as isize)..], refp.stride)
        } else {
            for yy in 0..hh {
                for xx in 0..ww {
                    self.swin[yy * ww + xx] = refp.at_clamped(x0 + xx as i32, y0 + yy as i32);
                }
            }
            (&self.swin[..], ww)
        };
        let at_block = 3 * stride + 3;
        match (fx, fy) {
            (0, 0) => (ctx.dsp.qpel_copy)(&mut self.spred14, &win[at_block..], stride, n, n, shift3),
            (_, 0) => (ctx.dsp.qpel_h)(&mut self.spred14, &win[3 * stride..], stride, n, n, fx, shift1),
            (0, _) => (ctx.dsp.qpel_v)(&mut self.spred14, &win[3..], stride, n, n, fy, shift1),
            _ => {
                (ctx.dsp.qpel_h)(&mut self.stmp, win, stride, n, n + 7, fx, shift1);
                (ctx.dsp.qpel_v2)(&mut self.spred14, &self.stmp, n, n, n, fy);
            }
        }
        let max = (1i32 << bd) - 1;
        (ctx.dsp.uni)(&mut self.spred, n, &self.spred14, n, n, 14 - bd as i32, max);
        (ctx.dist.satd)(src, src_stride, &self.spred, n, n, n)
    }
}

/// The conventional Lagrangian `0.85 * 2^((QP − 12) / 3)`, in SATD units —
/// the intra module's constant, duplicated (see the module header).
fn lambda(qp: i32) -> f32 {
    0.85f32 * ((qp - 12) as f32 / 3.0).exp2()
}

/// Bins of a truncated-Rice-coded index with the given `c_max` (how the
/// reader parses `merge_idx`).
fn tr_bins(idx: usize, c_max: usize) -> u32 {
    (idx + 1).min(c_max) as u32
}

/// Approximate bins of one `mvd_l0` (both components): the greater0 /
/// greater1 flags, the EG1-coded remainder's rough width, and the sign.
/// A placeholder in the intra module's single-function tradition.
fn mvd_cost(mvd: Mv) -> u32 {
    let comp = |v: i16| -> u32 {
        let a = v.unsigned_abs() as u32;
        match a {
            0 => 1,
            1 => 3,
            _ => 5 + 2 * (32 - (a - 1).leading_zeros()),
        }
    };
    comp(mvd.x) + comp(mvd.y)
}

/// Whether to hand this CU to the intra decision: the H.264 module's
/// flatness proxy (`super::h264_me::placeholder_inter_or_intra`) at CU
/// size and generic sample width — a DC prediction costing one SATD and no
/// reconstruction state, with all of that function's stated limits.
pub fn prefer_intra<S: Sample>(ctx: &MeCtx<'_, S>, inter_satd: u32, src: &[S], src_stride: usize, n: usize) -> bool {
    let mut sum = 0u64;
    for y in 0..n {
        for x in 0..n {
            sum += src[y * src_stride + x].to_i32() as u64;
        }
    }
    let dc = S::from_i32(((sum + (n * n / 2) as u64) / (n * n) as u64) as i32);
    let flat = vec![dc; n * n];
    let intra_proxy = (ctx.dist.satd)(src, src_stride, &flat, n, n, n);
    intra_proxy < inter_satd
}

/// Forward-code and reconstruct one inter transform block whose
/// *prediction is already in the plane*: residual against `src`, forward
/// DCT (inter TUs never take the DST, 8.6.4.2) and quantisation with the
/// inter rounding offset, then reconstruction through the decoder's own
/// `scale_coefficients`, inverse transform and `add_residual`. This is
/// `h265_intra::code_residual` minus its intra-only branches — copied, and
/// flagged in the module header.
#[allow(clippy::too_many_arguments)]
fn code_residual_inter<S: Sample>(
    ctx: &MeCtx<'_, S>,
    plane: &mut Plane16<S>,
    x: usize,
    y: usize,
    log2: u32,
    qp: i32,
    src: &[S],
    src_stride: usize,
    levels: &mut [i16],
) -> u32 {
    let n = 1usize << log2;
    let off = plane.offset(x as isize, y as isize);
    let stride = plane.stride;
    let max = (1i32 << ctx.bit_depth) - 1;

    let mut work = [0i16; 1024];
    for yy in 0..n {
        for xx in 0..n {
            work[yy * n + xx] = (src[yy * src_stride + xx].to_i32() - plane.data[off + yy * stride + xx].to_i32()) as i16;
        }
    }
    (ctx.enc.fdct[(log2 - 2) as usize])(&mut work, log2, ctx.bit_depth);
    let qb = qbits(qp, log2, ctx.bit_depth);
    let nz = (ctx.enc.quant)(&work, levels, n, quant_scale((qp % 6) as usize), qb, quant_offset(qb, false));

    work[..n * n].copy_from_slice(&levels[..n * n]);
    scale_coefficients(&mut work, log2, qp, ctx.bit_depth, ScalingSource::Flat, false, n - 1, n - 1);
    let bd_shift = 20 - ctx.bit_depth as i32;
    (ctx.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
    (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
    nz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::dsp::distortion::DistortionDsp;
    use crate::dsp::hevc::HevcDsp;
    use crate::dsp::hevc_enc::HevcEncDsp;
    use crate::encode::Config;
    use crate::encode::h265_syntax::{Geometry as SynGeometry, write_pps, write_sps};

    fn lcg(s: &mut u64) -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as u32
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
        fn ctx(&self, qp: i32) -> MeCtx<'_, u8> {
            IntraCtx { dsp: &self.dsp, enc: &self.enc, dist: &self.dist, qp, bit_depth: 8, strong_smoothing: false, bypass: false }
        }
    }

    /// The parameter sets this stream would carry, through the decoder's
    /// own parsers — the same round trip the encoder applies to everything
    /// it writes.
    fn parsed_sets(w: u32, h: u32) -> (Sps, Pps) {
        let cfg = Config { width: w, height: h, gop: 8, ..Config::default() };
        let syn = SynGeometry::new(&cfg);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &syn, 16))).unwrap();
        let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false, false))).unwrap();
        pps.resolve_tiles(&sps).unwrap();
        (sps, pps)
    }

    /// A reference picture whose luma is the H.264 module's triangle
    /// grating (periods 25 and 27) — see `grating_plane` in
    /// `h264_me.rs`'s tests for the three failure modes that shaped it.
    /// The constraint carries over unchanged: a greedy diamond seeded at
    /// zero converges to the true offset only when the walk starts inside
    /// the true basin (half-period, ≈ 12) and no alias is nearer, so the
    /// zero-seeded assertions below keep |d| small; larger motion is
    /// found the way a picture walk finds it, by neighbour propagation
    /// through the merge candidates. Chroma is never scored by the
    /// search, so radial bowls do. Borders extended as the decoder
    /// extends them before MC reads.
    fn reference(w: usize, h: usize, seed: u64) -> Frame<u8> {
        let mut f = Frame::new(w, h, ChromaFormat::Yuv420, 8);
        f.poc = 0;
        let _ = seed;
        for y in 0..h {
            for x in 0..w {
                let tx = (x as i32 % 25 - 12).abs();
                let ty = (y as i32 % 27 - 13).abs();
                let off = f.y.offset(x as isize, y as isize);
                f.y.data[off] = (40 + 4 * tx + 3 * ty) as u8;
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                let r2 = (x as i32 - (w / 4) as i32).pow(2) + (y as i32 - (h / 4) as i32).pow(2);
                let off = f.cb.offset(x as isize, y as isize);
                f.cb.data[off] = (200 - r2.min(160)) as u8;
                let off = f.cr.offset(x as isize, y as isize);
                f.cr.data[off] = (60 + r2.min(160)) as u8;
            }
        }
        f.extend_rows(0, h);
        f
    }

    /// Source planes translated by `(dx, dy)` full luma samples relative
    /// to `refp` (a block at `p` in the source equals the reference at
    /// `p + (dx, dy)`), reads outside the picture clamped as the padded
    /// plane clamps them.
    fn translated(refp: &Frame<u8>, dx: i32, dy: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (refp.width, refp.height);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = refp.y.at_clamped(xx as i32 + dx, yy as i32 + dy);
            }
        }
        let (cw, ch) = (w / 2, h / 2);
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..ch {
            for xx in 0..cw {
                cb[yy * cw + xx] = refp.cb.at_clamped(xx as i32 + dx / 2, yy as i32 + dy / 2);
                cr[yy * cw + xx] = refp.cr.at_clamped(xx as i32 + dx / 2, yy as i32 + dy / 2);
            }
        }
        (y, cb, cr)
    }

    fn code_picture(
        ctx: &MeCtx<'_, u8>,
        sps: &Sps,
        pps: &Pps,
        refp: &Frame<u8>,
        src_y: &[u8],
        src_cb: &[u8],
        src_cr: &[u8],
    ) -> (InterPicture<u8>, Vec<InterCuDecision>) {
        let (w, h) = (sps.width as usize, sps.height as usize);
        let n = 1usize << sps.log2_ctb_size;
        let mut pic = InterPicture::new(sps, pps, 1);
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu(ctx, refp, cx, cy, src_y, w, src_cb, src_cr, w / 2));
            }
        }
        (pic, decisions)
    }

    /// Every decision's internal consistency: `Skip` carries nothing,
    /// `Merge` always carries residual (the inferred `rqt_root_cbf`), the
    /// cbf flags agree with the coefficient arrays.
    fn assert_invariants(decisions: &[InterCuDecision]) {
        for (i, d) in decisions.iter().enumerate() {
            let n = 1usize << d.log2_cu;
            let nz_l = d.luma[..n * n].iter().any(|&v| v != 0);
            let nz_cb = d.chroma[0][..n * n / 4].iter().any(|&v| v != 0);
            let nz_cr = d.chroma[1][..n * n / 4].iter().any(|&v| v != 0);
            assert_eq!(d.cbf_luma, nz_l, "cu {i}: cbf_luma disagrees with the levels");
            assert_eq!(d.cbf_chroma, [nz_cb, nz_cr], "cu {i}: cbf_chroma disagrees");
            assert_eq!(d.rqt_root_cbf, nz_l || nz_cb || nz_cr, "cu {i}: rqt_root_cbf disagrees");
            match d.kind {
                InterCuKind::Skip { .. } => assert!(!d.rqt_root_cbf, "cu {i}: a skip CU carries residual"),
                InterCuKind::Merge { .. } => {
                    // The reader infers rqt_root_cbf 1 for a non-skip
                    // 2Nx2N merge CU: producing one without residual
                    // would desync — such a CU must have been Skip.
                    assert!(d.rqt_root_cbf, "cu {i}: a zero-residual merge CU escaped becoming Skip")
                }
                InterCuKind::Amvp { .. } | InterCuKind::UseIntra => {}
            }
        }
    }

    #[test]
    fn an_integral_translation_is_found_exactly_and_skips_when_it_can() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 7);
        // Even components, so the chroma offset (dx/2, dy/2) is integral
        // too and the prediction is a plain copy in all three planes — an
        // odd luma translation is a *half-sample* chroma one, which the
        // integer-shifted chroma source of `translated` cannot match.
        let (dx, dy) = (-6i32, 4i32);
        let (sy, scb, scr) = translated(&refp, dx, dy);
        let (pic, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions);
        let want = Mv::new((dx * 4) as i16, (dy * 4) as i16);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d.mv, want, "cu {i} missed the translation: {:?}", d.kind);
            assert!(!d.rqt_root_cbf, "cu {i}: an exact translation left residual");
        }
        // The first CU has no motion neighbours, so its candidates are the
        // zero-vector pads: it signals AMVP with no residual. Every later
        // CU sees the translation in a spatial candidate and skips.
        assert!(matches!(decisions[0].kind, InterCuKind::Amvp { .. }), "first CU: {:?}", decisions[0].kind);
        for (i, d) in decisions.iter().enumerate().skip(1) {
            assert!(matches!(d.kind, InterCuKind::Skip { .. }), "cu {i}: {:?}", d.kind);
        }
        // And the reconstruction is the translated reference, exactly: the
        // prediction was the decoder's own and no residual was added.
        for y in 0..32usize {
            for x in 0..64usize {
                let off = pic.recon.y.offset(x as isize, y as isize);
                assert_eq!(pic.recon.y.data[off], sy[y * 64 + x], "luma ({x},{y})");
            }
        }
        for y in 0..16usize {
            for x in 0..32usize {
                let off = pic.recon.cb.offset(x as isize, y as isize);
                assert_eq!(pic.recon.cb.data[off], scb[y * 32 + x], "cb ({x},{y})");
                let off = pic.recon.cr.offset(x as isize, y as isize);
                assert_eq!(pic.recon.cr.data[off], scr[y * 32 + x], "cr ({x},{y})");
            }
        }
    }

    #[test]
    fn a_surviving_residual_forbids_skip() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 11);
        let (mut sy, scb, scr) = translated(&refp, 2, -1);
        // Structured luma damage well above what QP 26 quantises away.
        let mut s = 99u64;
        for v in sy.iter_mut() {
            let d = (lcg(&mut s) % 64) as i32 - 32;
            *v = (*v as i32 + d).clamp(0, 255) as u8;
        }
        let (_, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions);
        assert!(
            decisions.iter().any(|d| d.rqt_root_cbf),
            "no CU carried residual — the damage was supposed to survive quantisation"
        );
        for (i, d) in decisions.iter().enumerate() {
            if d.rqt_root_cbf {
                assert!(!matches!(d.kind, InterCuKind::Skip { .. }), "cu {i} skipped with residual");
            }
        }
    }

    #[test]
    fn a_half_sample_translation_is_found_by_the_refinement() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 13);
        // The source is the decoder's own half-sample interpolation of the
        // reference: predict every CTU of a scratch frame at mv (2, 0).
        let want = Mv::new(2, 0);
        let mut interp = Frame::<u8>::new(64, 32, ChromaFormat::Yuv420, 8);
        let mut scratch = McScratch::new();
        for cy in 0..2usize {
            for cx in 0..4usize {
                predict_block(&kit.dsp, &mut scratch, &mut interp, cx * 16, cy * 16, 16, 16, Some((&refp, want)), None, [Weighting::Default; 3]);
            }
        }
        let flat = |p: &Plane16<u8>, w: usize, h: usize| -> Vec<u8> {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = p.data[p.offset(x as isize, y as isize)];
                }
            }
            v
        };
        let sy = flat(&interp.y, 64, 32);
        let scb = flat(&interp.cb, 32, 16);
        let scr = flat(&interp.cr, 32, 16);
        let (_, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d.mv, want, "cu {i}: {:?}", d.kind);
            assert!(!d.rqt_root_cbf, "cu {i}: the exact interpolation left residual");
        }
    }

    /// The anchor: every decision, replayed through the decoder's own
    /// candidate derivation over an *independently maintained* state —
    /// fresh `PicInfo`, fresh motion grid, fresh reconstruction — the way
    /// a writer plus a decoder would consume it. This proves the state
    /// maintenance (motion fills, availability marks), which is the half
    /// of the signalling contract that direct reuse of `merge_candidate` /
    /// `amvp` does not already make unbreakable; the derivation itself is
    /// held by the decoder's conformance suites.
    #[test]
    fn every_decision_replays_through_an_independent_decoder_state() {
        let kit = Kit::new();
        let ctx = kit.ctx(30);
        let (sps, pps) = parsed_sets(64, 64);
        let refp = reference(64, 64, 17);
        // Mixed content: a translation with damage in some regions, so
        // skip, merge-with-residual and AMVP all appear.
        let (mut sy, scb, scr) = translated(&refp, 3, 1);
        let mut s = 5u64;
        for yy in 0..64usize {
            for xx in 0..64usize {
                if (xx / 16 + yy / 16) % 3 == 0 {
                    let d = (lcg(&mut s) % 48) as i32 - 24;
                    let v = &mut sy[yy * 64 + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        let (pic, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions);
        let kinds: Vec<_> = decisions.iter().map(|d| std::mem::discriminant(&d.kind)).collect();
        assert!(kinds.iter().collect::<std::collections::HashSet<_>>().len() >= 2, "one-note content: the replay would prove less");

        // The independent state.
        let geo = std::sync::Arc::new(Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let mut frame = Frame::<u8>::new(64, 64, ChromaFormat::Yuv420, 8);
        frame.poc = 1;
        let mut scratch = McScratch::new();
        let refs = RefCtx::<u8> {
            pocs: [vec![refp.poc], Vec::new()],
            long_term: [vec![false], Vec::new()],
            col: None,
            cur_poc: 1,
            no_backward_pred: true,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: false,
            num_ref_idx: [1, 0],
            col_from_l0: true,
        };
        let n = 1usize << sps.log2_ctb_size;
        for (i, d) in decisions.iter().enumerate() {
            let (cx, cy) = (i % (64 / n), i / (64 / n));
            let (x0, y0) = (cx * n, cy * n);
            let ctb = info.ctb_of(x0, y0);
            info.ctb_slice_addr[ctb] = 0;
            info.ctb_slice[ctb] = 0;
            let pu = PuPos {
                x_cb: x0 as i32,
                y_cb: y0 as i32,
                n_cb: n as i32,
                x_pb: x0 as i32,
                y_pb: y0 as i32,
                w: n as i32,
                h: n as i32,
                part_idx: 0,
            };
            let w4 = info.w4;
            let mv = match d.kind {
                InterCuKind::Skip { merge_idx } | InterCuKind::Merge { merge_idx } => {
                    let cand = merge_candidate(&info, &frame, &refs, &pu, merge_idx as usize);
                    assert_eq!(cand.ref_idx, [0, -1], "cu {i}: replayed merge candidate references differently");
                    cand.mv[0]
                }
                InterCuKind::Amvp { mvp_flag, mvd } => {
                    let mvp = amvp(&info, &frame, &refs, &pu, 0, 0, mvp_flag as u32);
                    Mv::new(mvp.x.wrapping_add(mvd.x), mvp.y.wrapping_add(mvd.y))
                }
                InterCuKind::UseIntra => {
                    fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, MotionInfo::INTRA);
                    PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 1);
                    continue;
                }
            };
            assert_eq!(mv, d.mv, "cu {i}: the signalling does not replay to the chosen vector ({:?})", d.kind);

            // Reconstruct as a decoder would: the prediction, then the
            // carried coefficients through the inverse path.
            predict_block(&kit.dsp, &mut scratch, &mut frame, x0, y0, n, n, Some((&refp, mv)), None, [Weighting::Default; 3]);
            if d.rqt_root_cbf {
                let bd_shift = 20 - 8i32;
                let mut work = [0i16; 1024];
                if d.cbf_luma {
                    work[..n * n].copy_from_slice(&d.luma[..n * n]);
                    let log2 = d.log2_cu;
                    scale_coefficients(&mut work, log2, ctx.qp, 8, ScalingSource::Flat, false, n - 1, n - 1);
                    (kit.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
                    let off = frame.y.offset(x0 as isize, y0 as isize);
                    (kit.dsp.add_residual)(&mut frame.y.data[off..], frame.y.stride, &work, n, 255);
                }
                let qp_c = chroma_qp(1, ctx.qp.clamp(0, 57));
                let nc = n / 2;
                for comp in 0..2 {
                    if !d.cbf_chroma[comp] {
                        continue;
                    }
                    work[..nc * nc].copy_from_slice(&d.chroma[comp][..nc * nc]);
                    let log2c = d.log2_cu - 1;
                    scale_coefficients(&mut work, log2c, qp_c, 8, ScalingSource::Flat, false, nc - 1, nc - 1);
                    (kit.dsp.idct[(log2c - 2) as usize])(&mut work, bd_shift, nc - 1, nc - 1);
                    let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                    let off = plane.offset((x0 / 2) as isize, (y0 / 2) as isize);
                    (kit.dsp.add_residual)(&mut plane.data[off..], plane.stride, &work, nc, 255);
                }
            }

            // The decoder's own motion store, on the replay's state.
            let mut mi = MotionInfo { mv: [mv, Mv::ZERO], ref_delta: [0; 2], ref_idx: [0, -1], flags: 0, pad: 0 };
            mi.ref_delta[0] = (1 - refp.poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, mi);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 0);
        }
        // The replayed reconstruction is the encoder's, byte for byte.
        assert_eq!(frame.y.data, pic.recon.y.data, "luma reconstruction differs from the replay");
        assert_eq!(frame.cb.data, pic.recon.cb.data, "cb reconstruction differs from the replay");
        assert_eq!(frame.cr.data, pic.recon.cr.data, "cr reconstruction differs from the replay");
        // And the two sides' motion state agrees, which is what the next
        // picture would predict TMVP from if the SPS ever enables it.
        assert_eq!(frame.motion, pic.recon.motion, "motion grids diverged");
    }
}
