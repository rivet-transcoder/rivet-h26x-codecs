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
//! **And so is chroma, in every format.** All four chroma formats are
//! coded, and none of the three derivations that differ between them is
//! written out here:
//!
//! - The **chroma motion vector** comes from `predict_block` itself, which
//!   reads `(SubWidthC, SubHeightC)` off the frame it is writing into and
//!   scales by `mv * 2 / SubWidthC` per axis (its `mvc` closure,
//!   8.5.3.2.10). Building [`InterPicture::recon`] in the SPS's own format
//!   is therefore the entire chroma-MC change: 4:2:2's horizontally
//!   unscaled, vertically doubled vector is the decoder's derivation, not
//!   a second copy of it. This is the one rule self-consistency cannot
//!   check — an encoder and a decoder sharing a wrong vector agree with
//!   each other — so it is CROSS against libavcodec that arbitrates it.
//! - The **chroma transform blocks** are placed by
//!   `h265_intra::chroma_tbs`, which is `transform_unit`'s `here`
//!   expression plus its `yct = yc + t * nc` stacked-pair loop.
//! - The **chroma QP** is `hevc::ctu::chroma_qp` told this stream's
//!   `ChromaArrayType`: Table 8-10 for 4:2:0 and `Min(qPi, 51)` for
//!   everything else. Note for anyone testing this: the two agree for
//!   every `qPi` below 30, so a fixed-QP-26 stream cannot tell them apart.
//!
//! **TMVP is foreclosed by the bitstream, not skipped by this module.**
//! `write_sps` writes `sps_temporal_mvp_enabled_flag` = 0, so the slice
//! header never carries the slice-level flag and the decoder derives
//! temporal MVP off. The `RefCtx` here says the same (`tmvp: false`,
//! no collocated picture), which makes the spatial + zero candidate set
//! the *complete* derivation for these streams, not a subset.
//!
//! # Priced, not built (2026-09-13)
//!
//! Three of the standard's inter tools were costed against this corpus
//! before deciding not to build them. The numbers are here so the next
//! reader decides against a bigger corpus rather than re-deriving them.
//!
//! - **Prediction-unit partitions, symmetric and AMP.** Measured while the
//!   encoder coded one `PART_2Nx2N` unit per CTB; the coding quadtree now
//!   splits CTBs into smaller `PART_2Nx2N` units, which takes part of the
//!   same opportunity by another road. `tools/partition_opportunity.py`
//!   (integer-sample SAD, ±4, a split must beat the whole block by 5%)
//!   over the seven 8-bit 4:2:0 clips: 16.2% of blocks would take a
//!   symmetric split and 13.3% an AMP shape beyond it — but 7.1% and
//!   less once the smooth-gradient clip is set aside (an integer search
//!   approximates a fractional pan with two offsets, which quarter-sample
//!   refinement does without a split), and the AMP count lives in the
//!   fractal half of the cut clip, where a 32x8 strip overfits. The
//!   clips with real uniform motion and the held frame come out at 0.0%.
//!   Against that: a second vector and `part_mode` on every split unit,
//!   and under this SPS's `max_transform_hierarchy_depth_inter` an
//!   inferred transform split (`interSplitFlag`) the inter writer would
//!   have to learn to spell. The symmetric shapes are the prerequisite
//!   for AMP, and AMP's own value sits inside the probe's bias.
//! - **Temporal merge / AMVP candidates (TMVP).** The SPS disables it;
//!   the decoder's derivation is complete and the motion grid the
//!   collocated picture would need is already kept in `Frame`. What a
//!   temporal candidate can buy is turning an AMVP unit into a merge
//!   (its `mvd` and `mvp_l0_flag` for a `merge_idx`), so the ceiling is
//!   bounded by the AMVP share: on the corpus's census at QP 26 that is
//!   56 of 336 P CUs on the cut clip and 0–1 of 28–60 on every other,
//!   about 12% corpus-wide and nearly all on one clip. At ten to twenty
//!   bits per unit that is under half a percent of the cut stream and
//!   nothing elsewhere, for `slice_temporal_mvp_enabled_flag`,
//!   `collocated_ref_idx` and a second reader-derivation to mirror.
//! - **WPP / tile-parallel coding.** A speed tool, not a compression
//!   one, and the corpus cannot measure it: every clip is two CTB rows.
//!   The ceiling is rows over two (entropy sync lets a row start when
//!   the row above is two CTBs ahead) — nothing at 64x64, up to ~17x on
//!   1080p at CTB 32 in the limit — bought with `entry_point_offset`s in
//!   the slice header, a context save after the second CTB of each row,
//!   and a decision walk that runs per row behind the same lag. rivet
//!   already parallelises across pictures and chunks, so this would buy
//!   latency rather than throughput there; the timing that would justify
//!   it needs a clip with seconds and rows, which the corpus lacks.
//!
//! # Scope (v1) — the same deliberately fixed geometry as the intra module
//!
//! - **P slices, one reference** (list 0, `ref_idx` 0), `PART_2Nx2N` CUs
//!   — a whole CTB, or a quadtree leaf down to 8x8 through
//!   `InterPicture::code_ctu_tree` — each with one CU-sized TU (no transform
//!   split — the SPS's
//!   maximum transform size equals the CTB size precisely so this shape is
//!   representable). The quantiser and the weighting are the caller's:
//!   per-unit quantisers through `MeCtx::qp`, and list 0's explicit
//!   weighting through [`InterPicture::wp`] when the slice carries a
//!   `pred_weight_table` (`Weighting::Default` otherwise), and a B
//!   slice's list-0, list-1 and bi weighting through
//!   [`InterPicture::wp_b`].
//! - **All four chroma formats.** Monochrome omits every chroma element,
//!   mirroring the reader's uniform `chroma_array_type != 0` gates in
//!   `transform_tree` and `transform_unit`; 4:2:0 carries one half-size
//!   chroma TB per component; 4:2:2 the stacked pair of half-size squares,
//!   each with its own cbf; 4:4:4 one chroma TB at the luma TB's own size.
//!   The unsplit depth-0 node is above 4x4 in every geometry this module
//!   produces, so `transform_unit`'s chroma-at-the-parent case (its
//!   `blk_idx == 3` arm, for 4x4 luma TBs) never arises here.
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
//!   Then the transform tree, whose shape is the format's (below).
//! - [`InterCuKind::Amvp`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` 2Nx2N, `merge_flag` 0; no `inter_pred_idc` (P slice), no
//!   `ref_idx_l0` (one active reference); `mvd_l0` (as
//!   `abs_mvd_greater0_flag`, `abs_mvd_greater1_flag`, `abs_mvd_minus2`,
//!   `mvd_sign_flag` per component), `mvp_l0_flag`; then `rqt_root_cbf`,
//!   and when set the same unsplit transform tree.
//! - [`InterCuKind::UseIntra`]: not an inter CU at all. This decision's
//!   coefficients are meaningless and the planes are untouched; the
//!   caller calls [`InterPicture::code_ctu_intra`], which runs the intra
//!   decision over this same picture's reconstruction, and serialises
//!   the [`CuDecision`] that returns instead. See [`prefer_intra`], and
//!   [`PCuDecision`] for the seam that carries either kind.
//!
//! The transform tree, at `split_transform_flag` 0 and depth 0, spells in
//! the reader's order (`transform_tree`, then `transform_unit`):
//!
//! 1. `split_transform_flag`, 0.
//! 2. The chroma cbfs, **only when `chroma_array_type != 0`**: per
//!    component `cbf_c[c][0]`, and at 4:2:2 `cbf_c[c][1]` immediately
//!    after it (`transform_tree`'s `cat == 2 && (!split || log2 == 3)`
//!    arm — an unsplit node always codes both halves of the pair). The
//!    fields are [`InterCuDecision::cbf_chroma`] and
//!    [`InterCuDecision::cbf_chroma_bot`].
//! 3. `cbf_luma` — **but only if some chroma cbf above was set.** At an
//!    inter leaf of depth 0 with every chroma cbf clear the reader codes
//!    no bin and infers `cbf_luma` 1, so writing one desyncs. Monochrome
//!    has no chroma cbf to set and therefore never carries the bin at
//!    all; such a CU must genuinely have luma coefficients, which this
//!    module guarantees by spelling a residual-free CU as a skip or as
//!    `rqt_root_cbf` 0.
//! 4. The luma residual, if `cbf_luma`.
//! 5. The chroma residuals, components outermost and the 4:2:2 pair
//!    within (`transform_unit`'s `for c` around its `for t`), each at the
//!    TB size and slot [`InterCuDecision::chroma`] documents.
//!
//! Coefficients are raster within each TB, exactly as in `CuDecision`; the
//! writer derives the scan (always diagonal for inter TBs — the
//! mode-dependent scans are intra-only, and `residual_scan_idx` returns 0
//! for every non-intra block, 7.4.9.11).
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
use crate::encode::h265_intra::{
    CuDecision, Geo, IntraCtx, MIN_CB_LOG2, RegionSave, Srcs, TreeCu, chroma_tbs, code_cu_2nx2n_intra, code_cu_nxn_intra, cu_ssd, restore4, satd_lambda_scale,
    save4, ssd_lambda, sub_wh,
};
use crate::cabac_enc::CabacEncoder;
use crate::hevc::ctu::{
    PartMode, SplitCuNb, chroma_qp, explicit_weighting, write_cu_skip_flag, write_inter_pred_idc, write_merge_flag,
    write_merge_idx, write_mvd, write_mvp_flag, write_part_mode_inter, write_pred_mode_flag,
    write_ref_idx, write_rqt_root_cbf, write_split_cu_flag,
};
use crate::hevc::ctx::Contexts;
use crate::hevc::frame::{Frame, MotionInfo, Mv, Plane16, fill_motion};
use crate::hevc::inter::{McScratch, Weighting, predict_block};
use crate::hevc::intra::IntraScratch;
use crate::hevc::mvpred::{Cand, PuPos, RefCtx, amvp, merge_candidate};
use crate::hevc::pic::{Geometry, PicInfo};
use crate::hevc::pps::Pps;
use crate::hevc::residual::{ScalingSource, scale_coefficients};
use crate::hevc::slice::PredWeightTable;
use crate::hevc::sps::Sps;
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
    /// B: AMVP in one or both lists. `idc` is `inter_pred_idc` in the
    /// reader's own encoding — 0 `PRED_L0`, 1 `PRED_L1`, 2 `PRED_BI` — and
    /// only the entries of `mvd` / `mvp_flag` whose list the `idc` uses
    /// are meaningful; the other is zero and must not be written.
    ///
    /// What the writer spells, in `prediction_unit`'s order: `merge_flag`
    /// 0, then `inter_pred_idc` through
    /// `hevc::ctu::write_inter_pred_idc` (whose docblock carries the
    /// `w + h != 12` reading), then **per list, interleaved** — L0's
    /// `ref_idx` / `mvd` / `mvp_flag`, then L1's, not both `ref_idx`
    /// followed by both `mvd`. `ref_idx` is absent in both lists because
    /// each declares exactly one active reference.
    BAmvp {
        /// `inter_pred_idc`: 0 L0, 1 L1, 2 BI.
        idc: u8,
        /// `mvd_l0` / `mvd_l1`, wrapping i16 differences from the chosen
        /// predictor of that list.
        mvd: [Mv; 2],
        /// `mvp_l0_flag` / `mvp_l1_flag`.
        mvp_flag: [u8; 2],
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
    /// `cu_transquant_bypass_flag`. When set, `luma` and `chroma` carry
    /// the **raw residual** rather than quantised levels: the decoder
    /// skips dequantisation and the inverse transform for such a CU and
    /// adds what it reads straight to the prediction, so prediction plus
    /// residual is the source exactly.
    ///
    /// Two consequences the writer and the deblocker must respect.
    /// `coding_unit` reads this flag as the CU's **very first** bin —
    /// before `cu_skip_flag`, so even a skipped CU spells it — and only
    /// when the PPS sets `transquant_bypass_enabled_flag`. And a bypass
    /// CU is exempt from the in-loop filters sample for sample, which is
    /// what keeps a lossless stream lossless once deblocking is on.
    pub bypass: bool,
    /// `QpY` of this CU as a decoder will hold it: the context quantiser
    /// when the CU carries residual, the *predicted* one when it does not
    /// — a skip or a root-cbf-0 CU codes no `cu_qp_delta`. Filled with
    /// the context quantiser here; the encoder's quantiser chain
    /// overwrites it when the picture varies the quantiser per CTB. Read
    /// by the deblocker. See `CuDecision::qp_y`.
    pub qp_y: i32,
    /// The choice, with its signalling payload.
    pub kind: InterCuKind,
    /// The chosen motion vector, quarter luma samples, list 0. Filled for
    /// every kind including `Skip` (the writer carries no vector, but
    /// callers and tests want the motion the CU actually has).
    pub mv: Mv,
    /// Reference index into list 0, or -1 when this CU does not use
    /// list 0 (only a B CU can say that). Otherwise always 0: this module
    /// searches exactly one reference per list. The field exists so the
    /// serialiser's contract does not change when more arrive.
    pub ref_idx: i8,
    /// The list-1 motion vector, quarter luma samples. Meaningful only
    /// when [`InterCuDecision::ref_idx_l1`] is not -1, which only a B CU
    /// can arrange; a P decision leaves this zero.
    pub mv_l1: Mv,
    /// Reference index into list 1, -1 when the CU does not use list 1.
    /// Always -1 for P (a P slice has no list 1 at all) and for a B CU
    /// whose `inter_pred_idc` is `PRED_L0`.
    pub ref_idx_l1: i8,
    /// `rqt_root_cbf` as the *reader* resolves it: false for `Skip`, true
    /// for `Merge` (inferred, never coded), the explicit value for `Amvp`
    /// (coded). When false, no transform tree follows and every
    /// coefficient below is zero.
    pub rqt_root_cbf: bool,
    /// `cbf_luma` of the single CU-sized luma TU.
    pub cbf_luma: bool,
    /// `cbf_cb` / `cbf_cr` of this CU's *first* (or only) chroma TB per
    /// component — the `cbf_c[c][0]` bin `transform_tree` codes. At 4:2:2
    /// this is the top square of the stacked pair; see
    /// [`InterCuDecision::cbf_chroma_bot`] for the bottom one. Always
    /// false when `chroma_array_type` is 0, where no chroma bin exists.
    pub cbf_chroma: [bool; 2],
    /// 4:2:2 only: `cbf_c[c][1]`, the stacked pair's *bottom* square.
    /// `transform_tree` codes it right after `cbf_c[c][0]` on an unsplit
    /// node (its `cat == 2 && (!split || log2 == 3)` arm), so the writer
    /// spells the two bins adjacently, per component. False in every other
    /// format. Named as in [`super::h265_intra::CuDecision`], whose
    /// serialiser has the same shape.
    pub cbf_chroma_bot: [bool; 2],
    /// Quantised luma levels of the one `n x n` TU (`n = 1 << log2_cu`),
    /// raster, at `[0..n*n]`. Entries beyond are zero and meaningless.
    pub luma: [i16; 1024],
    /// Quantised chroma levels per component (`[0]` Cb, `[1]` Cr),
    /// raster within each chroma TB, packed one TB per slot of
    /// `nc * nc` where `nc = 1 << log2c` — the chroma TB edge
    /// `transform_unit` derives (`here`'s `if cat == 3 { log2 } else
    /// { log2 - 1 }`). Slot `t` occupies `[t*nc*nc .. (t+1)*nc*nc]`, in
    /// the reader's own coding order:
    ///
    /// | `chroma_array_type` | `log2c` | slots | where each TB sits |
    /// |---|---|---|---|
    /// | 0 (4:0:0) | — | none | no chroma syntax exists at all |
    /// | 1 (4:2:0) | `log2_cu - 1` | 1 | `(x0/2, y0/2)`, one half-size square |
    /// | 2 (4:2:2) | `log2_cu - 1` | 2 | `(x0/2, y0)` then `(x0/2, y0 + nc)` — the stacked pair, top slot 0, bottom slot 1 |
    /// | 3 (4:4:4) | `log2_cu` | 1 | `(x0, y0)`, at the luma TB's own size |
    ///
    /// Positions are chroma-plane coordinates; the placement is
    /// `h265_intra::chroma_tbs`, which mirrors `transform_unit`'s
    /// `here` and its `yct = yc + t * nc` loop. Entries beyond the slots a
    /// format uses are zero and meaningless. Sized for the largest case,
    /// a 4:4:4 32x32 chroma TB.
    pub chroma: [[i16; 1024]; 2],
}

impl Default for InterCuDecision {
    fn default() -> Self {
        InterCuDecision {
            log2_cu: 0,
            bypass: false,
            qp_y: 26,
            kind: InterCuKind::Skip { merge_idx: 0 },
            mv: Mv::ZERO,
            ref_idx: 0,
            mv_l1: Mv::ZERO,
            ref_idx_l1: -1,
            rqt_root_cbf: false,
            cbf_luma: false,
            cbf_chroma: [false; 2],
            cbf_chroma_bot: [false; 2],
            luma: [0; 1024],
            chroma: [[0; 1024]; 2],
        }
    }
}

/// How one CU of a P slice was coded — the seam the picture writer and
/// the deblocker consume, because a P slice holds CUs of both kinds.
///
/// [`InterPicture::code_ctu`] answers [`InterCuKind::UseIntra`] when the
/// flatness proxy says inter has lost; the caller then calls
/// [`InterPicture::code_ctu_intra`] and keeps *that* decision instead.
/// The inter one it displaces described nothing that was coded: its
/// coefficients were never quantised and its reconstruction never
/// written.
///
/// The intra half is boxed because it is the larger of the two by a
/// factor of two (a [`CuDecision`] carries the split shape's sixteen
/// coefficient slots) and it is the rarer by far; an unboxed enum would
/// double the memory of every P picture for the variant that seldom
/// fires.
#[derive(Clone)]
pub enum PCuDecision {
    /// An inter CU: skip, merge or AMVP.
    Inter(InterCuDecision),
    /// An intra CU inside the P slice, decided by the intra module over
    /// this same picture's reconstruction.
    Intra(Box<CuDecision>),
}

impl PCuDecision {
    /// The `QpY` a decoder holds for this CU — see `InterCuDecision::qp_y`.
    pub fn qp_y(&self) -> i32 {
        match self {
            PCuDecision::Inter(d) => d.qp_y,
            PCuDecision::Intra(d) => d.qp_y,
        }
    }

    /// Store the `QpY` the encoder's quantiser chain settled on.
    pub fn set_qp_y(&mut self, qp_y: i32) {
        match self {
            PCuDecision::Inter(d) => d.qp_y = qp_y,
            PCuDecision::Intra(d) => d.qp_y = qp_y,
        }
    }

    /// Whether the CU carries any coded cbf, which is when — and only
    /// when — a decoder reads a `cu_qp_delta` inside it. An inter CU's
    /// tree exists exactly when `rqt_root_cbf` is set, and a tree at
    /// this geometry always carries a cbf (the decision spells a
    /// residual-free tree as a skip or a root cbf of 0).
    pub fn any_cbf(&self) -> bool {
        match self {
            PCuDecision::Inter(d) => d.rqt_root_cbf,
            PCuDecision::Intra(d) => d.any_cbf(),
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
    /// log2 of the CTB size, 4 or 5. A coding unit is this size or, under
    /// the coding quadtree, smaller — down to the 8x8 minimum coding block;
    /// the CU-level calls take their own size.
    pub log2_ctb: u32,
    /// `ChromaArrayType` (0 monochrome, 1 4:2:0, 2 4:2:2, 3 4:4:4), read
    /// off the parsed SPS. Every chroma decision below — whether chroma
    /// exists, where its transform blocks sit, which QP mapping applies —
    /// is gated on this exactly as the reader gates on
    /// `Sps::chroma_array_type`. Chroma *motion compensation* is not
    /// gated here at all: `predict_block` takes the subsampling from the
    /// frame it writes into, so building `recon` in the SPS's own format
    /// is what makes the chroma vector right (see `code_ctu`).
    pub cat: u32,
    /// The current picture's POC (the motion grid stores POC differences).
    pub cur_poc: i32,
    /// How deep [`InterPicture::code_ctu_intra`] may let an intra CU's
    /// transform tree split — the same knob as
    /// [`super::h265_intra::IntraPicture::split_depth`] and the same
    /// caveat: the coding-tree writer must be able to spell whichever
    /// shapes it permits, and a decision the writer cannot serialise
    /// desyncs the arithmetic coder. **1**, matching what the picture
    /// writer sets for I slices and what `write_cu_intra_body` spells.
    pub split_depth: u32,
    /// The picture descriptor the intra decision's availability and MPM
    /// mirrors read, built once from the SPS.
    geo: Geo,
    /// Reference-sample scratch for the intra decision, held per picture
    /// exactly as [`super::h265_intra::IntraPicture`] holds its own.
    intra_scratch: IntraScratch,
    /// MC scratch, as the decoder allocates per slice.
    scratch: McScratch<S>,
    /// Luma-only prediction scratch for candidate scoring: the clamp
    /// window, the two-stage filter intermediate, the 14-bit prediction
    /// and the sample-domain prediction.
    swin: Vec<S>,
    stmp: Vec<i16>,
    spred14: Vec<i16>,
    /// The second list's 14-bit prediction, for the B bi-prediction trial
    /// (`satd_bi_at`). Unused by the P path.
    spred14_b: Vec<i16>,
    spred: Vec<S>,
    /// The explicit weighting this picture's list-0 predictions carry,
    /// one per reference in `RefPicList0` — per component, what
    /// `hevc::ctu::explicit_weighting` derives from the slice's
    /// `pred_weight_table` for that reference. Empty when the slice
    /// carries no table (every reference default). Read by
    /// [`Self::code_ctu`]'s scoring and its final prediction, so a
    /// vector is chosen for the prediction that will actually be made.
    /// The B walk does not read it; see [`Self::wp_b`].
    pub wp: Vec<[Weighting; 3]>,
    /// The explicit weighting a B picture's predictions carry, by
    /// `inter_pred_idc` — `[0]` `PRED_L0`, `[1]` `PRED_L1`, `[2]` `PRED_BI`
    /// — per component: what `hevc::ctu::explicit_weighting` derives from
    /// the slice's `pred_weight_table` for reference indices `[0, -1]`,
    /// `[-1, 0]` and `[0, 0]`, each list's one active reference. All
    /// `Weighting::Default` when the slice carries no table, or one whose
    /// every entry is the default (which predicts the same samples). Read
    /// by [`Self::code_cu_b`]'s search, its merge and bi scoring and its
    /// final prediction. The P walk does not read it.
    pub wp_b: [[Weighting; 3]; 3],
}

impl<S: Sample> InterPicture<S> {
    /// State for one P picture, from the *parsed* parameter sets — the
    /// caller round-trips the bytes `write_sps` / `write_pps` produced
    /// through the decoder's own parsers, the same proof the encoder
    /// applies to everything it writes. Fixed geometry as in the intra
    /// module: whole CTUs, 4:2:0, `log2_cu` 4 or 5.
    pub fn new(sps: &Sps, pps: &Pps, cur_poc: i32) -> Self {
        assert!((4..=5).contains(&sps.log2_ctb_size), "log2_ctb {} outside 4..=5", sps.log2_ctb_size);
        let (w, h) = (sps.width as usize, sps.height as usize);
        let n = 1usize << sps.log2_ctb_size;
        assert!(w.is_multiple_of(n) && h.is_multiple_of(n), "{w}x{h} is not a whole number of {n}x{n} CTUs");
        let geo = std::sync::Arc::new(Geometry::new(sps, pps));
        let info = PicInfo::new(geo);
        // The reconstruction is built in the SPS's own chroma format, and
        // that single choice is what gives chroma motion compensation its
        // per-format behaviour: `predict_block` reads `cur.chroma`, derives
        // (SubWidthC, SubHeightC) from it and scales the vector by
        // `mv * 2 / SubWidthC` per axis (its `mvc` closure, 8.5.3.2.10) —
        // so 4:2:2's unscaled-horizontal, doubled-vertical chroma vector is
        // the decoder's derivation, not one retyped here.
        let mut recon = Frame::new(w, h, sps.chroma_format(), sps.bit_depth_luma);
        recon.poc = cur_poc;
        let nmax = 1usize << (2 * sps.log2_ctb_size);
        InterPicture {
            info,
            recon,
            log2_ctb: sps.log2_ctb_size,
            cat: sps.chroma_array_type(),
            cur_poc,
            split_depth: 1,
            geo: Geo::new(sps.log2_ctb_size, w, h, sps.chroma_array_type()),
            intra_scratch: IntraScratch::default(),
            scratch: McScratch::new(),
            swin: vec![S::default(); (64 + 7) * (64 + 7)],
            stmp: vec![0; crate::dsp::hevc::MC_TMP_LEN],
            spred14: vec![0; nmax],
            spred14_b: vec![0; nmax],
            spred: vec![S::default(); nmax],
            wp: Vec::new(),
            wp_b: [[Weighting::Default; 3]; 3],
        }
    }

    /// The weighting list-0 reference `r` carries: the table's, or the
    /// default when the slice has no table.
    fn wp_for(&self, r: usize) -> [Weighting; 3] {
        self.wp.get(r).copied().unwrap_or([Weighting::Default; 3])
    }

    /// Weight this B picture's predictions by its slice's table `t`: each
    /// `inter_pred_idc`'s entry of [`Self::wp_b`] as the reader derives it
    /// — `explicit_weighting` for reference 0 of the lists that
    /// prediction uses, `[0, -1]`, `[-1, 0]` and `[0, 0]`.
    pub fn set_b_weights(&mut self, t: &PredWeightTable, bit_depth_luma: u32, bit_depth_chroma: u32) {
        self.wp_b = [[0, -1], [-1, 0], [0, 0]].map(|r| explicit_weighting(t, bit_depth_luma, bit_depth_chroma, r));
    }

    /// The weighting a B prediction from the lists `ref_idx` uses carries
    /// (see [`Self::wp_b`]): one list's, or the pair's.
    fn wp_b_for(&self, ref_idx: [i8; 2]) -> [Weighting; 3] {
        match (ref_idx[0] >= 0, ref_idx[1] >= 0) {
            (true, false) => self.wp_b[0],
            (false, true) => self.wp_b[1],
            _ => self.wp_b[2],
        }
    }

    /// The reference-list context the decoder's candidate derivation
    /// takes, for a P slice whose `RefPicList0` holds `pocs0`, nearest
    /// first. `tmvp` false and no collocated picture: the SPS this
    /// stream carries disables temporal MVP (see the module header).
    fn ref_ctx<'a>(&self, pocs0: &[i32]) -> RefCtx<'a, S> {
        RefCtx {
            pocs: [pocs0.to_vec(), Vec::new()],
            long_term: [vec![false; pocs0.len()], Vec::new()],
            col: None,
            cur_poc: self.cur_poc,
            no_backward_pred: true,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: false,
            num_ref_idx: [pocs0.len(), 0],
            col_from_l0: true,
        }
    }

    /// The reference-list context for a B slice referencing `poc0` in
    /// list 0 and `poc1` in list 1, one active entry each — what the
    /// header this stream carries declares (`num_ref_idx_active_override`
    /// 0 over PPS defaults of 0, which the reader resolves as [1, 1]).
    ///
    /// `no_backward_pred` is **derived** exactly as `decoder.rs` derives
    /// `NoBackwardPredFlag` — every reference POC at or before the current
    /// picture — rather than pasted from [`Self::ref_ctx`], which
    /// hardcodes true because a P slice's single reference is always in
    /// the past. For a B between two anchors it comes out false, and it
    /// reaches `merge_candidate` through `RefCtx`.
    fn ref_ctx_b<'a>(&self, poc0: i32, poc1: i32) -> RefCtx<'a, S> {
        let no_backward_pred = [poc0, poc1].iter().all(|&p| p <= self.cur_poc);
        RefCtx {
            pocs: [vec![poc0], vec![poc1]],
            long_term: [vec![false], vec![false]],
            col: None,
            cur_poc: self.cur_poc,
            no_backward_pred,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: true,
            num_ref_idx: [1, 1],
            col_from_l0: true,
        }
    }

    /// Decide and code one CTU as a single 2Nx2N CU against the references
    /// of `RefPicList0`, `refs_l0`, nearest first (borders extended —
    /// `Frame::extend_rows` — exactly as the decoder pads references
    /// before MC reads them).
    ///
    /// On [`InterCuKind::UseIntra`] the reconstruction planes are
    /// untouched and the motion grid holds `MotionInfo::INTRA`: the
    /// caller runs the intra decision, which writes the planes itself.
    ///
    /// `src_cb` / `src_cr` are the chroma planes at `c_stride`, in this
    /// stream's own format — `(width / SubWidthC) x (height / SubHeightC)`.
    /// Under monochrome they are never read and may be empty, exactly as
    /// the reader never reads a chroma element when `chroma_array_type`
    /// is 0.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs_l0: &[&Frame<S>],
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << self.log2_ctb;
        self.code_cu(ctx, refs_l0, cu_x * n, cu_y * n, self.log2_ctb, 0, src_y, y_stride, src_cb, src_cr, c_stride)
    }

    /// [`Self::code_ctu`] for one coding unit of `1 << log2_cu` at luma
    /// `(x0, y0)` and coding-tree depth `depth` — the whole CTB at depth 0,
    /// a quadtree leaf below it. Everything the decision derives from its
    /// neighbours (merge and AMVP candidates, the skip context) comes from
    /// the decoder-grade state the walk maintains, so a neighbour of any
    /// size serves; `depth` reaches the rate model only through
    /// `inter_pred_idc`'s context.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs_l0: &[&Frame<S>],
        x0: usize,
        y0: usize,
        log2_cu: u32,
        depth: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << log2_cu;
        let mut out = InterCuDecision { log2_cu, bypass: ctx.bypass, qp_y: ctx.qp, ..InterCuDecision::default() };

        // Mark the CTB as this (single) slice's, as the decoder does at CTB
        // start: `avail_ctx` reads the current CTB's slice address, and
        // `available_at` compares neighbours against it.
        let ctb = self.info.ctb_of(x0, y0);
        self.info.ctb_slice_addr[ctb] = 0;
        self.info.ctb_slice[ctb] = 0;

        debug_assert!(!refs_l0.is_empty(), "a P CU needs at least one reference");
        let pocs: Vec<i32> = refs_l0.iter().map(|f| f.poc).collect();
        let refs = self.ref_ctx(&pocs);
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

        // The decoder's own candidate lists (see the module header). The
        // AMVP predictors depend on the reference index — the derivation
        // scales a neighbour's vector by the POC-distance ratio between
        // its reference and this one — so they are derived per
        // reference by calling `amvp` once for each.
        let merge: Vec<Cand> = (0..MAX_MERGE_CAND).map(|i| merge_candidate(&self.info, &self.recon, &refs, &pu, i)).collect();
        let mvp: Vec<[Mv; 2]> = (0..refs_l0.len())
            .map(|r| [amvp(&self.info, &self.recon, &refs, &pu, 0, r as i8, 0), amvp(&self.info, &self.recon, &refs, &pu, 0, r as i8, 1)])
            .collect();

        let soff = y0 * y_stride + x0;
        let src = &src_y[soff..];

        // One search per reference. Multi-reference is not a wider search
        // of the same picture but a CHOICE between pictures: a block
        // uncovered by motion, or one whose match is periodic, can be
        // predicted better from an older frame than the newest one, and
        // the only way to know which is to search each and price the
        // answer — `ref_idx` is signalled per prediction unit, so the
        // choice is not free and the cost below counts it.
        //
        // The default is ONE reference, and that is a measured choice.
        // On whole-CTB units (`max_cu_depth` 0) the choice never has a
        // better answer: 32x32 coding units average over enough
        // content that one reference always serves them (0 of 140 blocks
        // across the corpus, where the same probe at 16x16 said 6.9% and
        // was answering about a block size this encoder does not code),
        // and two references measured 0.80% worse on every clip, better
        // on none. `Config::max_refs` opts in. The coding quadtree now codes
        // the 16x16 and 8x8 units that probe answered about; it has not
        // been re-run against them.
        //
        // Full-sample descent from every distinct seed's best, then the
        // two sub-sample rings, per reference.
        let mut per_ref: Vec<(Mv, u32)> = Vec::with_capacity(refs_l0.len());
        for (r, rf) in refs_l0.iter().enumerate() {
            let mut seeds: Vec<Mv> = vec![Mv::ZERO, mvp[r][0], mvp[r][1]];
            seeds.extend(merge.iter().filter(|c| c.ref_idx[0] == r as i8).map(|c| c.mv[0]));
            let full = self.search_full(ctx, &rf.y, x0, y0, n, src, y_stride, &seeds);
            let wp = self.wp_for(r)[0];
            per_ref.push(self.refine_subpel(ctx, &rf.y, wp, 0, x0, y0, n, src, y_stride, full));
        }

        // Cost the shapes. Merge candidates are scored at their exact
        // vectors against the picture they actually name; only the first
        // occurrence of a (vector, reference) matters (a later duplicate
        // signals strictly more bins for the same prediction).
        let lam = lambda(ctx.qp) * satd_lambda_scale(ctx.bit_depth);
        let rate = Rate::new_at(ctx.qp, false, log2_cu, depth);
        let mut best_merge: Option<(usize, u32)> = None; // (idx, satd)
        let mut best_merge_cost = f32::INFINITY;
        let mut seen: Vec<(Mv, i8)> = Vec::with_capacity(MAX_MERGE_CAND);
        for (idx, cand) in merge.iter().enumerate() {
            let ri = cand.ref_idx[0];
            if ri < 0 || ri as usize >= refs_l0.len() || seen.contains(&(cand.mv[0], ri)) {
                continue;
            }
            seen.push((cand.mv[0], ri));
            let satd = self.satd_at(ctx, &refs_l0[ri as usize].y, ri as usize, x0, y0, n, src, y_stride, cand.mv[0]);
            // cu_skip_flag or merge_flag, plus the TR-coded index.
            // Skip and merge differ in signalling, and a zero-residual
            // candidate becomes a skip, so price each at the shape it
            // would actually take.
            let bits = rate.skip(idx as u8).min(rate.merge(idx as u8));
            let cost = satd as f32 + lam * bits;
            if cost < best_merge_cost {
                best_merge_cost = cost;
                best_merge = Some((idx, satd));
            }
        }
        // AMVP, over every reference: its searched vector against the
        // cheaper of its own two predictors (the reader's uLX sum is a
        // wrapping add, so the mvd is a wrapping difference), plus what
        // it costs to say WHICH picture — `ref_idx` is truncated unary
        // over the active count and absent entirely at one reference,
        // so a single-reference stream prices exactly as it always did.
        let nref = refs_l0.len() as u32;
        let mut best_amvp: Option<(usize, u8, Mv, u32)> = None; // (ref, flag, mvd, satd)
        let mut amvp_cost = f32::INFINITY;
        for (r, &(mv_r, satd_r)) in per_ref.iter().enumerate() {
            let mvd_for = |p: Mv| Mv::new(mv_r.x.wrapping_sub(p.x), mv_r.y.wrapping_sub(p.y));
            let (d0, d1) = (mvd_for(mvp[r][0]), mvd_for(mvp[r][1]));
            let c0 = rate.amvp_ref(d0, 0, true, nref, r as u32);
            let c1 = rate.amvp_ref(d1, 1, true, nref, r as u32);
            let (flag, mvd, bits) = if c1 < c0 { (1u8, d1, c1) } else { (0u8, d0, c0) };
            // merge_flag 0, ref_idx, mvp_l0_flag, rqt_root_cbf, plus the
            // mvd bins (cu_skip_flag and pred_mode/part_mode surround
            // both shapes).
            let cost = satd_r as f32 + lam * bits;
            if cost < amvp_cost {
                amvp_cost = cost;
                best_amvp = Some((r, flag, mvd, satd_r));
            }
        }
        let (amvp_ref, amvp_flag, mvd, amvp_satd) = best_amvp.expect("at least one reference");

        let merge_wins = best_merge.is_some_and(|_| best_merge_cost <= amvp_cost);
        let (mv, chosen_ref, inter_satd) = if merge_wins {
            let (idx, satd) = best_merge.expect("merge_wins implies a candidate");
            (merge[idx].mv[0], merge[idx].ref_idx[0].max(0) as usize, satd)
        } else {
            (per_ref[amvp_ref].0, amvp_ref, amvp_satd)
        };
        let refp: &Frame<S> = refs_l0[chosen_ref];
        out.mv = mv;
        out.ref_idx = chosen_ref as i8;

        if prefer_intra(ctx, inter_satd, src, y_stride, n) {
            out.kind = InterCuKind::UseIntra;
            // An intra CU's motion, stored now so later candidate
            // derivations see what the decoder will see.
            fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, MotionInfo::INTRA);
            let w4 = self.info.w4;
            PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
            // `coding_unit` records `cu_skip_flag` for every CU before it
            // knows the pred mode (ctu.rs:419), and the *next* CU's
            // `cu_skip_flag` context counts skipped neighbours out of
            // exactly this array. An intra CU is never skipped.
            PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
            return out;
        }

        // The chosen prediction, through the decoder's own MC, weighted
        // exactly as the slice header says this reference is.
        let wp = self.wp_for(chosen_ref);
        predict_block(ctx.dsp, &mut self.scratch, &mut self.recon, x0, y0, n, n, Some((refp, mv)), None, wp);

        let any = self.code_residual_cu(ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, &mut out);
        out.kind = match (merge_wins, any) {
            (true, false) => InterCuKind::Skip { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (true, true) => InterCuKind::Merge { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (false, _) => InterCuKind::Amvp { mvp_flag: amvp_flag, mvd },
        };

        // Store motion and marks exactly as the decoder's `prediction_unit`
        // does after parsing ("Store motion", src/hevc/ctu.rs): the next
        // CUs' candidate lists read them — the reference index included,
        // which is what makes a neighbour's vector scale by the right
        // POC distance.
        let mut mi = MotionInfo { mv: [mv, Mv::ZERO], ref_delta: [0; 2], ref_idx: [chosen_ref as i8, -1], flags: 0, pad: 0 };
        mi.ref_delta[0] = (self.cur_poc - refp.poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, mi);
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, matches!(out.kind, InterCuKind::Skip { .. }) as u8);
        out
    }

    /// Decide and code one CTU as a single 2Nx2N CU of a **B** picture
    /// against `ref0` (list 0) and `ref1` (list 1), whose borders must be
    /// extended as the decoder pads references before MC reads them.
    ///
    /// The shape of the decision, and what is the decoder's rather than
    /// this module's:
    ///
    /// - **The candidate lists are the decoder's.** `merge_candidate` runs
    ///   with `is_b` true, so it returns both lists per candidate and adds
    ///   the combined bi-predictive pairs and the bi zero candidates that
    ///   only a B slice has. `amvp` runs per list. Neither derivation is
    ///   mirrored here.
    /// - **The search is per list**, each the same greedy full-sample
    ///   descent plus two sub-sample rings the P walk uses, seeded from
    ///   that list's own predictors and every merge candidate that uses
    ///   that list.
    /// - **One bi trial, at the two per-list winners**, scored through the
    ///   decoder's `dsp.bi` (`satd_bi_at`). There is deliberately **no
    ///   iterative bi refinement** in this version: the two vectors are
    ///   not re-searched against each other's prediction, so a bi CU here
    ///   is the best pair of independently searched vectors and not the
    ///   best pair. Named because it is a quality ceiling, not a
    ///   correctness one.
    /// - **The prediction is the decoder's own**, uni or bi, through the
    ///   one `predict_block` call that also gives every chroma format its
    ///   vector — so B costs nothing extra outside 4:2:0.
    ///
    /// On [`InterCuKind::UseIntra`] the reconstruction planes are
    /// untouched, exactly as in [`Self::code_ctu`].
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << self.log2_ctb;
        self.code_cu_b(ctx, ref0, ref1, cu_x * n, cu_y * n, self.log2_ctb, 0, src_y, y_stride, src_cb, src_cr, c_stride)
    }

    /// [`Self::code_ctu_b`] for one coding unit of `1 << log2_cu` at luma
    /// `(x0, y0)` and coding-tree depth `depth`, as [`Self::code_cu`] is
    /// to [`Self::code_ctu`].
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        depth: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << log2_cu;
        let mut out = InterCuDecision { log2_cu, bypass: ctx.bypass, qp_y: ctx.qp, ..InterCuDecision::default() };

        let ctb = self.info.ctb_of(x0, y0);
        self.info.ctb_slice_addr[ctb] = 0;
        self.info.ctb_slice[ctb] = 0;

        let refs = self.ref_ctx_b(ref0.poc, ref1.poc);
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

        let merge: Vec<Cand> = (0..MAX_MERGE_CAND).map(|i| merge_candidate(&self.info, &self.recon, &refs, &pu, i)).collect();
        let mvp: [[Mv; 2]; 2] = [
            [amvp(&self.info, &self.recon, &refs, &pu, 0, 0, 0), amvp(&self.info, &self.recon, &refs, &pu, 0, 0, 1)],
            [amvp(&self.info, &self.recon, &refs, &pu, 1, 0, 0), amvp(&self.info, &self.recon, &refs, &pu, 1, 0, 1)],
        ];

        let soff = y0 * y_stride + x0;
        let src = &src_y[soff..];

        // Per-list search. Each list is seeded from the zero vector, its
        // own two AMVP predictors, and every merge candidate that uses it.
        let mut uni = [(Mv::ZERO, u32::MAX); 2];
        for list in 0..2usize {
            let plane = if list == 0 { &ref0.y } else { &ref1.y };
            let mut seeds: Vec<Mv> = vec![Mv::ZERO, mvp[list][0], mvp[list][1]];
            seeds.extend(merge.iter().filter(|c| c.ref_idx[list] == 0).map(|c| c.mv[list]));
            let full = self.search_full(ctx, plane, x0, y0, n, src, y_stride, &seeds);
            // Scored under the weighting a one-list prediction from this
            // list carries, so the vector is chosen for the prediction
            // that will be made.
            let wp = self.wp_b[list][0];
            uni[list] = self.refine_subpel(ctx, plane, wp, list, x0, y0, n, src, y_stride, full);
        }

        let lam = lambda(ctx.qp) * satd_lambda_scale(ctx.bit_depth);
        let rate = Rate::new_at(ctx.qp, true, log2_cu, depth);

        // The three AMVP shapes. `inter_pred_idc` costs two bins for a uni
        // shape and one for BI (the reader stops after a set first bin),
        // on top of merge_flag, rqt_root_cbf and a mvp_flag per used list
        // — the same approximate-bin-count placeholder policy as the rest
        // of this module.
        let mvd_for = |mv: Mv, p: Mv| Mv::new(mv.x.wrapping_sub(p.x), mv.y.wrapping_sub(p.y));
        let mut best_mvd = [Mv::ZERO; 2];
        let mut best_flag = [0u8; 2];
        for list in 0..2usize {
            let a = mvd_for(uni[list].0, mvp[list][0]);
            let b = mvd_for(uni[list].0, mvp[list][1]);
            let idc_for = |l: usize| if l == 0 { 0u8 } else { 1 };
            let one = |m: Mv, f: u8| {
                let mut mvd = [Mv::ZERO; 2];
                mvd[list] = m;
                let mut fl = [0u8; 2];
                fl[list] = f;
                rate.amvp_b(idc_for(list), mvd, fl, true)
            };
            let pick1 = one(b, 1) < one(a, 0);
            best_flag[list] = u8::from(pick1);
            best_mvd[list] = if pick1 { b } else { a };
        }
        // idc 0 / 1: one list, its mvd and mvp_flag.
        let mut best: Option<(u8, u32)> = None; // (idc, satd)
        let mut best_cost = f32::INFINITY;
        for (list, u) in uni.iter().enumerate() {
            let mut mvd = [Mv::ZERO; 2];
            mvd[list] = best_mvd[list];
            let mut fl = [0u8; 2];
            fl[list] = best_flag[list];
            let bits = rate.amvp_b(list as u8, mvd, fl, true);
            let cost = u.1 as f32 + lam * bits;
            if cost < best_cost {
                best_cost = cost;
                best = Some((list as u8, u.1));
            }
        }
        // idc 2: the bi trial at the two winners.
        let bi_satd = self.satd_bi_at(ctx, &ref0.y, &ref1.y, x0, y0, n, src, y_stride, uni[0].0, uni[1].0, self.wp_b[2][0]);
        {
            let bits = rate.amvp_b(2, best_mvd, best_flag, true);
            let cost = bi_satd as f32 + lam * bits;
            if cost < best_cost {
                best_cost = cost;
                best = Some((2, bi_satd));
            }
        }
        let (best_idc, amvp_satd) = best.expect("three shapes were costed");

        // Merge, over the decoder's own candidates. A candidate may use
        // either list or both, and is scored the way it would be predicted.
        let mut best_merge: Option<(usize, u32)> = None;
        let mut best_merge_cost = f32::INFINITY;
        let mut seen: Vec<([Mv; 2], [i8; 2])> = Vec::with_capacity(MAX_MERGE_CAND);
        for (idx, cand) in merge.iter().enumerate() {
            let key = (cand.mv, cand.ref_idx);
            if cand.ref_idx == [-1, -1] || seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let satd = match (cand.ref_idx[0] >= 0, cand.ref_idx[1] >= 0) {
                (true, true) => self.satd_bi_at(ctx, &ref0.y, &ref1.y, x0, y0, n, src, y_stride, cand.mv[0], cand.mv[1], self.wp_b[2][0]),
                (true, false) => self.satd_at_weighted(ctx, &ref0.y, x0, y0, n, src, y_stride, cand.mv[0], self.wp_b[0][0], 0),
                (false, true) => self.satd_at_weighted(ctx, &ref1.y, x0, y0, n, src, y_stride, cand.mv[1], self.wp_b[1][0], 1),
                (false, false) => unreachable!("filtered above"),
            };
            let bits = rate.skip(idx as u8).min(rate.merge(idx as u8));
            let cost = satd as f32 + lam * bits;
            if cost < best_merge_cost {
                best_merge_cost = cost;
                best_merge = Some((idx, satd));
            }
        }

        let merge_wins = best_merge.is_some_and(|_| best_merge_cost <= best_cost);
        // The motion the winner carries, as `ref_idx` pairs the decoder
        // would store.
        let (mv_pair, ref_pair, inter_satd) = if merge_wins {
            let (idx, satd) = best_merge.expect("merge_wins implies a candidate");
            (merge[idx].mv, merge[idx].ref_idx, satd)
        } else {
            let mv = [uni[0].0, uni[1].0];
            let r: [i8; 2] = match best_idc {
                0 => [0, -1],
                1 => [-1, 0],
                _ => [0, 0],
            };
            (
                [if r[0] >= 0 { mv[0] } else { Mv::ZERO }, if r[1] >= 0 { mv[1] } else { Mv::ZERO }],
                r,
                amvp_satd,
            )
        };
        out.mv = mv_pair[0];
        out.ref_idx = ref_pair[0];
        out.mv_l1 = mv_pair[1];
        out.ref_idx_l1 = ref_pair[1];

        if prefer_intra(ctx, inter_satd, src, y_stride, n) {
            out.kind = InterCuKind::UseIntra;
            fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, MotionInfo::INTRA);
            let w4 = self.info.w4;
            PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
            // Never skipped — and under the quadtree the area may still
            // hold a skip mark from a trial that lost, which the P walk's
            // identical fill already clears.
            PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
            return out;
        }

        // The chosen prediction, through the decoder's own MC — uni or bi
        // by which lists the winner uses, and per-format chroma for free —
        // weighted exactly as the slice header says those lists are.
        let r0 = (ref_pair[0] >= 0).then_some((ref0, mv_pair[0]));
        let r1 = (ref_pair[1] >= 0).then_some((ref1, mv_pair[1]));
        let wp = self.wp_b_for(ref_pair);
        predict_block(ctx.dsp, &mut self.scratch, &mut self.recon, x0, y0, n, n, r0, r1, wp);

        let any = self.code_residual_cu(ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, &mut out);
        out.kind = match (merge_wins, any) {
            (true, false) => InterCuKind::Skip { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (true, true) => InterCuKind::Merge { merge_idx: best_merge.expect("merge_wins").0 as u8 },
            (false, _) => InterCuKind::BAmvp { idc: best_idc, mvd: best_mvd, mvp_flag: best_flag },
        };

        // Store motion and marks exactly as `prediction_unit` does after
        // parsing ("Store motion", src/hevc/ctu.rs), both lists this time.
        let mut mi = MotionInfo { mv: mv_pair, ref_idx: ref_pair, ref_delta: [0; 2], flags: 0, pad: 0 };
        for list in 0..2usize {
            if ref_pair[list] >= 0 {
                let poc = if list == 0 { ref0.poc } else { ref1.poc };
                mi.ref_delta[list] = (self.cur_poc - poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
        fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, mi);
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, matches!(out.kind, InterCuKind::Skip { .. }) as u8);
        out
    }

    /// The residual of one 2Nx2N CU, luma then every chroma TB this
    /// format carries, each reconstructed in place through the decoder's
    /// inverse path. Fills `cbf_luma`, `cbf_chroma`, `cbf_chroma_bot`,
    /// `rqt_root_cbf` and the coefficient arrays; returns `rqt_root_cbf`.
    ///
    /// Shared by the P and B walks precisely so the two cannot drift: the
    /// residual does not depend on how the prediction was signalled, only
    /// on what the prediction left behind, and a second copy of the
    /// chroma-format placement is the drift hazard this crate keeps
    /// finding. `predict_block` must already have written the prediction
    /// into `recon`.
    #[allow(clippy::too_many_arguments)]
    fn code_residual_cu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) -> bool {
        let qp_l = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let nz_l = code_residual_inter(ctx, &mut self.recon.y, x0, y0, log2_cu, qp_l, src, y_stride, &mut out.luma);
        let mut nz_c = 0u32;
        if self.cat != 0 {
            // QpC: Table 8-10 for 4:2:0, `Min(qPi, 51)` otherwise — the
            // decoder's own `hevc::ctu::chroma_qp`, told which format it is
            // rather than the constant 1 this module used while it modelled
            // 4:2:0 alone. The PPS and slice header write zero cb/cr
            // offsets, so `qPi` is the luma QP.
            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(self.cat, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            let (sw, sh) = sub_wh(self.cat);
            // Where this CU's chroma TBs sit and how big they are, from the
            // placement `transform_unit` performs — `here` plus its 4:2:2
            // `yct = yc + t * nc` pair. Anchors come back in *luma*
            // coordinates; dividing by (SubWidthC, SubHeightC) puts them on
            // the chroma plane, which is also how the source is addressed.
            let (tbs, ntb, log2c) = chroma_tbs(self.cat, x0, y0, log2_cu);
            let nc2 = 1usize << (2 * log2c);
            for comp in 0..2usize {
                let plane = if comp == 0 { &mut self.recon.cb } else { &mut self.recon.cr };
                let srcp = if comp == 0 { src_cb } else { src_cr };
                for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                    let (px, py) = (ax / sw, ay / sh);
                    let soff = py * c_stride + px;
                    let levels = &mut out.chroma[comp][t * nc2..(t + 1) * nc2];
                    let nz = code_residual_inter(ctx, plane, px, py, log2c, qp_c, &srcp[soff..], c_stride, levels);
                    if t == 0 {
                        out.cbf_chroma[comp] = nz != 0;
                    } else {
                        out.cbf_chroma_bot[comp] = nz != 0;
                    }
                    nz_c += nz;
                }
            }
        }
        out.cbf_luma = nz_l != 0;
        out.rqt_root_cbf = nz_l + nz_c != 0;
        out.rqt_root_cbf
    }

    /// Code the CTU at `(cu_x, cu_y)` as an **intra** CU inside this P
    /// slice, after [`InterPicture::code_ctu`] answered
    /// [`InterCuKind::UseIntra`] for it. Call it only then, and only
    /// immediately: the walk's ordering invariants are the intra
    /// decision's too.
    ///
    /// This is not a second intra encoder. It is
    /// `code_cu_2nx2n_intra` — the very function
    /// [`super::h265_intra::IntraPicture::code_ctu`] calls for an I
    /// slice — pointed at *this* picture's state:
    ///
    /// - **`self.recon`**, so intra prediction reads reconstructed
    ///   neighbours *including the inter-coded ones*. That is legal and
    ///   deliberate rather than an oversight of constrained intra
    ///   prediction: `write_pps` writes `constrained_intra_pred_flag` 0
    ///   (`h265_syntax.rs:272`), which switches off the second half of
    ///   the reader's own reference check —
    ///   `available_at(..) && (!cip || pred_mode == 1)`, `ctu.rs:1157`.
    ///   With `cip` false the reader takes any decoded neighbour, so the
    ///   encoder must too, or the two predict from different samples.
    /// - **`self.info.intra_mode`**, the decoder's own per-4x4 luma-mode
    ///   grid, as the grid the MPM derivation reads and fills — rather
    ///   than a private copy that would have to be kept in step with it.
    /// - **`self.info.pred_mode`**, so the MPM derivation applies the
    ///   reader's not-intra gate (`ctu.rs:627`): a neighbouring *inter*
    ///   CU contributes `INTRA_DC`, not whatever mode last stood in the
    ///   mode grid at that position.
    ///
    /// `code_ctu` has already stored the motion (`MotionInfo::INTRA`),
    /// `pred_mode` 1 and `skip` 0 over the CU — the marks
    /// `coding_unit` writes before it parses any intra syntax — so the
    /// deblocker and every later candidate derivation see what a decoder
    /// of this stream will see.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_intra(
        &mut self,
        ctx: &MeCtx<'_, S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        let n = 1usize << self.log2_ctb;
        self.code_cu_intra(ctx, cu_x * n, cu_y * n, self.log2_ctb, src_y, y_stride, src_cb, src_cr, c_stride)
    }

    /// [`Self::code_ctu_intra`] for one coding unit of `1 << log2_cu` at
    /// luma `(x0, y0)`: the CU [`Self::code_cu`] or [`Self::code_cu_b`]
    /// answered [`InterCuKind::UseIntra`] for, called immediately after.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_intra(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        // Split the borrows: the mode grid is written, the pred-mode grid
        // is read, and both live in `info` beside each other.
        let PicInfo { intra_mode, pred_mode, .. } = &mut self.info;
        let (intra_mode, pred_mode) = (&mut intra_mode[..], &pred_mode[..]);
        code_cu_2nx2n_intra(
            ctx,
            self.geo,
            &mut self.recon,
            intra_mode,
            Some(pred_mode),
            &mut self.intra_scratch,
            self.split_depth,
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        )
    }

    /// Decide and code the CTB at `(cu_x, cu_y)` (CTB units) of a P or B
    /// picture as a coding quadtree — the walk and the cost of
    /// [`super::h265_intra::IntraPicture::code_ctu_tree`] over this
    /// picture's units: every node coded whole by the inter decision
    /// ([`Self::code_cu`] or [`Self::code_cu_b`], handing over to
    /// [`Self::code_cu_intra`] where it answers [`InterCuKind::UseIntra`])
    /// and then split, the cheaper kept.
    ///
    /// A trial here leaves more behind than an intra one: beside the
    /// samples, the motion grid and the `pred_mode`, `skip` and
    /// `intra_mode` marks the candidate derivations read. All of it is put
    /// back when the whole node wins (`TrialSave`), over exactly the
    /// node's area — and nothing outside it is ever written — so the next
    /// unit's merge and AMVP lists are built from the winners alone, as a
    /// decoder builds them from what it parsed.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_tree(
        &mut self,
        ctx: &MeCtx<'_, S>,
        want: &dyn Fn(usize, usize, u32) -> i32,
        max_depth: u32,
        refs: TreeRefs<'_, S>,
        cu_x: usize,
        cu_y: usize,
        src: &Srcs<'_, S>,
    ) -> Vec<TreeCu<PCuDecision>> {
        let log2 = self.log2_ctb;
        let mut out = Vec::new();
        self.tree_node(ctx, want, max_depth, refs, cu_x << log2, cu_y << log2, log2, 0, src, &mut out);
        out
    }

    /// One node of [`Self::code_ctu_tree`]: returns its cost, having
    /// pushed its winning units onto `out`.
    #[allow(clippy::too_many_arguments)]
    fn tree_node(
        &mut self,
        ctx: &MeCtx<'_, S>,
        want: &dyn Fn(usize, usize, u32) -> i32,
        max_depth: u32,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
        out: &mut Vec<TreeCu<PCuDecision>>,
    ) -> f64 {
        let qp = want(x0, y0, log2);
        let cctx = IntraCtx { qp, ..*ctx };
        let init = if matches!(refs, TreeRefs::B(..)) { 2 } else { 1 };
        let lam = ssd_lambda(qp, ctx.bit_depth);
        let (d, ssd, unit_bits) = self.tree_leaf(&cctx, refs, x0, y0, log2, depth, src);
        let flag = if log2 > MIN_CB_LOG2 { crate::encode::h265::split_flag_bits(init, qp, false) } else { 0.0 };
        let bits = unit_bits + flag;
        let j_whole = ssd as f64 + lam * f64::from(bits);
        if depth >= max_depth || log2 <= MIN_CB_LOG2 {
            out.push(TreeCu { x0, y0, log2, depth, bits, d });
            return j_whole;
        }
        let n = 1usize << log2;
        let saved = TrialSave::take(self, x0, y0, n);
        let mark = out.len();
        let mut j_split = lam * f64::from(crate::encode::h265::split_flag_bits(init, qp, true));
        let half = n / 2;
        for i in 0..4 {
            if j_split > j_whole {
                break;
            }
            j_split += self.tree_node(ctx, want, max_depth, refs, x0 + (i & 1) * half, y0 + (i >> 1) * half, log2 - 1, depth + 1, src, out);
        }
        if j_whole <= j_split {
            saved.put(self, x0, y0, n);
            out.truncate(mark);
            out.push(TreeCu { x0, y0, log2, depth, bits, d });
            j_whole
        } else {
            j_split
        }
    }

    /// Code the unit of `1 << log2` at `(x0, y0)` whole, as a quadtree
    /// leaf: the decision (inter, or the intra one it handed over to), its
    /// reconstruction SSD, and its counted syntax bits without the
    /// `split_cu_flag`.
    #[allow(clippy::too_many_arguments)]
    fn tree_leaf(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
    ) -> (PCuDecision, u64, f32) {
        let (d, nref, is_b) = match refs {
            TreeRefs::P(l0) => {
                (self.code_cu(ctx, l0, x0, y0, log2, depth, src.y, src.y_stride, src.cb, src.cr, src.c_stride), l0.len() as u32, false)
            }
            TreeRefs::B(r0, r1) => (self.code_cu_b(ctx, r0, r1, x0, y0, log2, depth, src.y, src.y_stride, src.cb, src.cr, src.c_stride), 1, true),
        };
        let coded = if matches!(d.kind, InterCuKind::UseIntra) {
            PCuDecision::Intra(Box::new(self.code_cu_intra(ctx, x0, y0, log2, src.y, src.y_stride, src.cb, src.cr, src.c_stride)))
        } else {
            PCuDecision::Inter(d)
        };
        let n = 1usize << log2;
        let ssd = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
        let bits = crate::encode::h265::p_cu_bits(&coded, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth);
        // An intra unit at the minimum coding block has a second shape, as
        // in an I picture: code `PART_NxN` too and keep the cheaper, the
        // loser's state put back from a copy.
        if log2 == MIN_CB_LOG2 && matches!(coded, PCuDecision::Intra(_)) {
            let saved = TrialSave::take(self, x0, y0, n);
            let nxn = PCuDecision::Intra(Box::new(self.code_cu_intra_nxn(ctx, x0, y0, src.y, src.y_stride, src.cb, src.cr, src.c_stride)));
            let ssd_n = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
            let bits_n = crate::encode::h265::p_cu_bits(&nxn, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth);
            let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
            if ssd_n as f64 + lam * f64::from(bits_n) < ssd as f64 + lam * f64::from(bits) {
                return (nxn, ssd_n, bits_n);
            }
            saved.put(self, x0, y0, n);
        }
        (coded, ssd, bits)
    }

    /// The `PART_NxN` alternative to [`Self::code_cu_intra`] for an 8x8 unit
    /// the inter decision handed over: `code_cu_nxn_intra` over this
    /// picture's state, as `code_cu_intra` runs `code_cu_2nx2n_intra`.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_intra_nxn(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        let PicInfo { intra_mode, pred_mode, .. } = &mut self.info;
        let (intra_mode, pred_mode) = (&mut intra_mode[..], &pred_mode[..]);
        code_cu_nxn_intra(ctx, self.geo, &mut self.recon, intra_mode, Some(pred_mode), &mut self.intra_scratch, x0, y0, src_y, y_stride, src_cb, src_cr, c_stride)
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
    /// the eight quarter-sample neighbours of that winner — each scored
    /// under `wp`, the luma weighting a one-list prediction from `list`
    /// carries (see [`Self::satd_at_weighted`]).
    #[allow(clippy::too_many_arguments)]
    fn refine_subpel(&mut self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, wp: Weighting, list: usize, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, start: Mv) -> (Mv, u32) {
        let mut best = start;
        let mut best_satd = self.satd_at_weighted(ctx, refp, x, y, n, src, src_stride, start, wp, list);
        for step in [2i16, 1] {
            let centre = best;
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let mv = Mv::new(centre.x.wrapping_add(dx), centre.y.wrapping_add(dy));
                    let satd = self.satd_at_weighted(ctx, refp, x, y, n, src, src_stride, mv, wp, list);
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
    fn satd_at(&mut self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, r: usize, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, mv: Mv) -> u32 {
        let wp = self.wp_for(r)[0];
        self.satd_at_weighted(ctx, refp, x, y, n, src, src_stride, mv, wp, 0)
    }

    /// [`Self::satd_at`] under a given luma weighting: the default
    /// uni-prediction, or the decoder's `weighted_uni` at the table's
    /// `log2WD` and the weight and offset of `list` — the entry
    /// `predict_block` takes for a one-list prediction from that list, and
    /// the same kernel it will commit the winner through.
    #[allow(clippy::too_many_arguments)]
    fn satd_at_weighted(&mut self, ctx: &MeCtx<'_, S>, refp: &Plane16<S>, x: usize, y: usize, n: usize, src: &[S], src_stride: usize, mv: Mv, wp: Weighting, list: usize) -> u32 {
        let InterPicture { swin, stmp, spred14, spred, .. } = self;
        predict14(ctx, refp, x, y, n, mv, swin, stmp, spred14);
        let bd = ctx.bit_depth;
        let max = (1i32 << bd) - 1;
        match wp {
            Weighting::Default => (ctx.dsp.uni)(spred, n, spred14, n, n, 14 - bd as i32, max),
            Weighting::Explicit { log2_wd, w, o } => (ctx.dsp.weighted_uni)(spred, n, spred14, n, n, log2_wd, w[list], o[list], max),
        }
        (ctx.dist.satd)(src, src_stride, spred, n, n, n)
    }

    /// The model check for weighted prediction: the luma SATD of the CU
    /// at `(x0, y0)` predicted from `refp` at `mv` without any weighting
    /// and with this picture's, so the caller can count whether the
    /// table's fit helped the vectors the search actually chose.
    #[allow(clippy::too_many_arguments)]
    pub fn weighting_gain(&mut self, ctx: &MeCtx<'_, S>, refp: &Frame<S>, r: usize, x0: usize, y0: usize, log2_cu: u32, src_y: &[S], y_stride: usize, mv: Mv) -> (u32, u32) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let wp = self.wp_for(r)[0];
        let plain = self.satd_at_weighted(ctx, &refp.y, x0, y0, n, src, y_stride, mv, Weighting::Default, 0);
        let weighted = self.satd_at_weighted(ctx, &refp.y, x0, y0, n, src, y_stride, mv, wp, 0);
        (plain, weighted)
    }

    /// [`Self::weighting_gain`] for a B CU: the luma SATD at the chosen
    /// vectors `mv` of the lists `ref_idx` uses (reference 0 of each, `ref0`
    /// and `ref1`), without weighting and with this picture's — one list's
    /// prediction, or the pair's.
    #[allow(clippy::too_many_arguments)]
    pub fn weighting_gain_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        mv: [Mv; 2],
        ref_idx: [i8; 2],
    ) -> (u32, u32) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let wp = self.wp_b_for(ref_idx)[0];
        match (ref_idx[0] >= 0, ref_idx[1] >= 0) {
            (true, true) => {
                let plain = self.satd_bi_at(ctx, &ref0.y, &ref1.y, x0, y0, n, src, y_stride, mv[0], mv[1], Weighting::Default);
                (plain, self.satd_bi_at(ctx, &ref0.y, &ref1.y, x0, y0, n, src, y_stride, mv[0], mv[1], wp))
            }
            (one0, _) => {
                let (refp, list) = if one0 { (&ref0.y, 0) } else { (&ref1.y, 1) };
                let plain = self.satd_at_weighted(ctx, refp, x0, y0, n, src, y_stride, mv[list], Weighting::Default, list);
                (plain, self.satd_at_weighted(ctx, refp, x0, y0, n, src, y_stride, mv[list], wp, list))
            }
        }
    }

    /// SATD of the *bi-predicted* luma block at `(mv0, mv1)`: the two
    /// lists' 14-bit predictions combined through the decoder's own
    /// `dsp.bi` at `15 - bit_depth`, which is the shift `predict_block`
    /// uses for default-weighted bi-prediction (8.5.3.3.4.2), or under
    /// `wp` through its `dsp.weighted_bi` at the table's `log2WD` and both
    /// lists' weights and offsets (8.5.3.3.4.3) — the arithmetic
    /// `predict_block` commits the winner through. Scoring the average
    /// rather than either half is what makes the BI trial comparable with
    /// the two uni ones.
    #[allow(clippy::too_many_arguments)]
    fn satd_bi_at(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Plane16<S>,
        ref1: &Plane16<S>,
        x: usize,
        y: usize,
        n: usize,
        src: &[S],
        src_stride: usize,
        mv0: Mv,
        mv1: Mv,
        wp: Weighting,
    ) -> u32 {
        let InterPicture { swin, stmp, spred14, spred14_b, spred, .. } = self;
        predict14(ctx, ref0, x, y, n, mv0, swin, stmp, spred14);
        predict14(ctx, ref1, x, y, n, mv1, swin, stmp, spred14_b);
        let bd = ctx.bit_depth;
        let max = (1i32 << bd) - 1;
        match wp {
            Weighting::Default => (ctx.dsp.bi)(spred, n, spred14, spred14_b, n, n, 15 - bd as i32, max),
            Weighting::Explicit { log2_wd, w, o } => (ctx.dsp.weighted_bi)(spred, n, spred14, spred14_b, n, n, log2_wd, w[0], w[1], o[0], o[1], max),
        }
        (ctx.dist.satd)(src, src_stride, spred, n, n, n)
    }
}

/// The references a coding-quadtree walk predicts from.
pub enum TreeRefs<'r, S: Sample> {
    /// A P slice's `RefPicList0`, nearest first.
    P(&'r [&'r Frame<S>]),
    /// A B slice's list-0 and list-1 anchors.
    B(&'r Frame<S>, &'r Frame<S>),
}

impl<S: Sample> Clone for TreeRefs<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Sample> Copy for TreeRefs<'_, S> {}

/// Everything a coding trial over a square region of an inter picture
/// writes, as the trial left it: the samples, the motion grid, and the
/// `pred_mode`, `skip` and `intra_mode` marks. See
/// [`InterPicture::code_ctu_tree`].
struct TrialSave<S: Sample> {
    planes: RegionSave<S>,
    motion: Vec<MotionInfo>,
    pred_mode: Vec<u8>,
    skip: Vec<u8>,
    intra_mode: Vec<u8>,
}

impl<S: Sample> TrialSave<S> {
    fn take(pic: &InterPicture<S>, x0: usize, y0: usize, n: usize) -> Self {
        let w4 = pic.info.w4;
        TrialSave {
            planes: RegionSave::take(&pic.recon, pic.cat, x0, y0, n),
            motion: save4(&pic.recon.motion, pic.recon.w4, x0, y0, n),
            pred_mode: save4(&pic.info.pred_mode, w4, x0, y0, n),
            skip: save4(&pic.info.skip, w4, x0, y0, n),
            intra_mode: save4(&pic.info.intra_mode, w4, x0, y0, n),
        }
    }

    fn put(&self, pic: &mut InterPicture<S>, x0: usize, y0: usize, n: usize) {
        let w4 = pic.info.w4;
        self.planes.put(&mut pic.recon, pic.cat, x0, y0, n);
        let mw4 = pic.recon.w4;
        restore4(&mut pic.recon.motion, mw4, x0, y0, n, &self.motion);
        restore4(&mut pic.info.pred_mode, w4, x0, y0, n, &self.pred_mode);
        restore4(&mut pic.info.skip, w4, x0, y0, n, &self.skip);
        restore4(&mut pic.info.intra_mode, w4, x0, y0, n, &self.intra_mode);
    }
}

/// One list's 14-bit luma prediction of the `n x n` block at `(x, y)` for
/// vector `mv`, into `out`. The window gathering and the kernel choice are
/// `hevc::inter`'s `source` and `interp` (3 taps before the sample, 4
/// after; `qpel_copy` / `_h` / `_v` / `_h`+`_v2` by fraction), which is
/// what keeps a scored candidate identical to what `predict_block` will
/// later commit for the winner.
///
/// A free function rather than a method because both callers need it to
/// write into one field of `InterPicture` while reading two others, which
/// a `&mut self` method cannot express.
#[allow(clippy::too_many_arguments)]
fn predict14<S: Sample>(
    ctx: &MeCtx<'_, S>,
    refp: &Plane16<S>,
    x: usize,
    y: usize,
    n: usize,
    mv: Mv,
    swin: &mut [S],
    stmp: &mut [i16],
    out: &mut [i16],
) {
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
                swin[yy * ww + xx] = refp.at_clamped(x0 + xx as i32, y0 + yy as i32);
            }
        }
        (&swin[..], ww)
    };
    let at_block = 3 * stride + 3;
    match (fx, fy) {
        (0, 0) => (ctx.dsp.qpel_copy)(out, &win[at_block..], stride, n, n, shift3),
        (_, 0) => (ctx.dsp.qpel_h)(out, &win[3 * stride..], stride, n, n, fx, shift1),
        (0, _) => (ctx.dsp.qpel_v)(out, &win[3..], stride, n, n, fy, shift1),
        _ => {
            (ctx.dsp.qpel_h)(stmp, win, stride, n, n + 7, fx, shift1);
            (ctx.dsp.qpel_v2)(out, stmp, n, n, n, fy);
        }
    }
}

/// The conventional Lagrangian `0.85 * 2^((QP − 12) / 3)`, in 8-bit SATD
/// units — the intra module's constant, duplicated (see the module
/// header). Callers multiply by `satd_lambda_scale` for deeper samples.
fn lambda(qp: i32) -> f32 {
    0.85f32 * ((qp - 12) as f32 / 3.0).exp2()
}

/// What a candidate shape costs to *signal*, in real bits.
///
/// This replaces the hand-rolled bin counts this module used to carry —
/// `tr_bins`, and an `mvd_cost` that approximated `write_mvd`'s
/// exponential-Golomb remainder as `5 + 2 * log2(a - 1)`. The trouble with
/// those was never accuracy: it was that nothing could check them. A wrong
/// cost changes which shape wins, and every check this project has — SELF,
/// CROSS, the replays — passes whatever the decision picks.
///
/// So the bins are no longer guessed at. Each shape is priced by running
/// **the production writers** through a counting [`CabacEncoder`], which
/// tallies exactly the bits it would have written (an equality asserted in
/// `cabac_enc`'s round trip). `write_mvd`'s Golomb coding, `write_merge_idx`'s
/// truncated unary and its `MaxNumMergeCand` cap, the context each bin
/// lands in — all of it is the real thing rather than a model of it.
///
/// # What this is not, stated precisely
///
/// **The probabilities are the slice's initial ones, not the ones in force
/// at this CU.** The decision runs in its own pass, before serialisation —
/// SAO forced that split — so the live context array does not exist yet
/// when a shape is chosen. Pricing therefore starts from a freshly
/// initialised `Contexts` for this slice type and QP. The bin *sequence*
/// is exact; the bit *width* of each bin is priced under the slice's
/// starting model rather than its adapted one.
///
/// **The neighbour-dependent contexts are the neutral ones.**
/// `cu_skip_flag`'s context counts skipped neighbours and
/// `split_cu_flag`'s counts deeper ones; both are serialiser state. They
/// are priced here as if no neighbour were available, which is what the
/// first CU of a slice genuinely sees.
///
/// **Residual bits are not included**, matching the scope of the counts it
/// replaces: at shape-choice time the residual has not been coded. What is
/// compared is signalling against signalling.
///
/// Each of those is a bounded, named offset shared by every candidate at
/// the same CU, which is what a comparison needs — the shapes are ranked
/// against each other, and a common offset cancels.
pub(crate) struct Rate {
    /// The slice's initial contexts, cloned per pricing.
    cx: Contexts,
    /// log2 of the CU size, for `inter_pred_idc`'s block dimensions and
    /// for whether `split_cu_flag` is coded at all (not at the 8x8
    /// minimum coding block).
    log2_cu: u32,
    /// The CU's coding-tree depth, `CtDepth` — the context of
    /// `inter_pred_idc`'s first bin.
    depth: u32,
}

impl Rate {
    /// `init_type` as `code_inter_picture` derives it: 1 for P, 2 for B.
    #[cfg(test)]
    pub(crate) fn new(qp: i32, is_b: bool, log2_cu: u32) -> Self {
        Self::new_at(qp, is_b, log2_cu, 0)
    }

    /// The rate model for a `1 << log2_cu` CU at coding-tree depth
    /// `depth` of a P (`is_b` false) or B slice at quantiser `qp`.
    pub(crate) fn new_at(qp: i32, is_b: bool, log2_cu: u32, depth: u32) -> Self {
        Rate { cx: Contexts::new(if is_b { 2 } else { 1 }, qp), log2_cu, depth }
    }

    /// Run `f` over a counting encoder and a private copy of the contexts.
    /// Fractional bits, not emitted ones: a shape short enough to fit
    /// inside the arithmetic coder's first output byte emits nothing at
    /// all, so `bits_counted` would price several distinct shapes at zero
    /// and delete the rate term from the comparison. See
    /// `CabacEncoder::fractional_bits`.
    fn count(&self, f: impl FnOnce(&mut CabacEncoder<'static>, &mut Contexts)) -> f32 {
        let mut cx = self.cx.clone();
        let mut e = CabacEncoder::counting();
        f(&mut e, &mut cx);
        e.fractional_bits() as f32
    }

    /// The elements every inter CU spells before its shape diverges:
    /// `split_cu_flag` then `cu_skip_flag`. `nb` is the neutral neighbour
    /// context described on [`Rate`].
    fn prefix(&self, e: &mut CabacEncoder<'static>, cx: &mut Contexts, skip: bool) {
        // `split_cu_flag` exists only above the 8x8 minimum coding block;
        // at it the reader infers the leaf and takes no bin.
        if self.log2_cu > 3 {
            let nb = SplitCuNb { left_depth: None, above_depth: None };
            write_split_cu_flag(e, cx, &nb, self.depth, false);
        }
        write_cu_skip_flag(e, cx, None, None, skip);
    }

    /// `cu_skip_flag` 1 and a `merge_idx`; the reader infers the rest.
    pub(crate) fn skip(&self, merge_idx: u8) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, true);
            write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
        })
    }

    /// A non-skip 2Nx2N merge CU. `rqt_root_cbf` is not coded — the reader
    /// infers it — so nothing stands in for it here either.
    pub(crate) fn merge(&self, merge_idx: u8) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, true);
            write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
        })
    }

    /// P-slice AMVP at one reference: one list, its `mvd` and
    /// `mvp_l0_flag`, then `rqt_root_cbf` — whose value is a parameter
    /// because it is a coded bin with a cost, and because it is what lets
    /// a test price exactly the shape `write_cu_inter` emits.
    /// [`Rate::amvp_ref`] with a single active reference, which spells no
    /// `ref_idx` — kept for the test that prices the shape by hand.
    #[cfg(test)]
    pub(crate) fn amvp(&self, mvd: Mv, mvp_flag: u8, root_cbf: bool) -> f32 {
        self.amvp_ref(mvd, mvp_flag, root_cbf, 1, 0)
    }

    /// [`Rate::amvp`] plus what it costs to name the reference: `ref_idx`
    /// is truncated unary over the active count `nref`, and absent
    /// entirely when the list has one entry — so at one reference this
    /// prices identically to a stream that never had the choice, which
    /// is what keeps single-reference streams byte-identical.
    pub(crate) fn amvp_ref(&self, mvd: Mv, mvp_flag: u8, root_cbf: bool, nref: u32, ref_idx: u32) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, false);
            if nref > 1 {
                write_ref_idx(e, cx, nref, ref_idx);
            }
            write_mvd(e, cx, mvd);
            write_mvp_flag(e, cx, mvp_flag != 0);
            write_rqt_root_cbf(e, cx, root_cbf);
        })
    }

    /// B-slice AMVP: `inter_pred_idc`, then per list — interleaved as
    /// `prediction_unit` reads them — the `mvd` and `mvp_lX_flag` of each
    /// list the shape uses.
    pub(crate) fn amvp_b(&self, idc: u8, mvd: [Mv; 2], mvp_flag: [u8; 2], root_cbf: bool) -> f32 {
        let n = 1i32 << self.log2_cu;
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, false);
            write_inter_pred_idc(e, cx, n, n, self.depth, u32::from(idc));
            for list in 0..2usize {
                let uses = match idc {
                    0 => list == 0,
                    1 => list == 1,
                    _ => true,
                };
                if !uses {
                    continue;
                }
                write_mvd(e, cx, mvd[list]);
                write_mvp_flag(e, cx, mvp_flag[list] != 0);
            }
            write_rqt_root_cbf(e, cx, root_cbf);
        })
    }
}

/// Whether to hand this CU to the intra decision: the H.264 module's
/// flatness proxy (`super::h264_me::placeholder_inter_or_intra`) at CU
/// size and generic sample width — a DC prediction costing one SATD and no
/// reconstruction state, with all of that function's stated limits.
///
/// # It carries no rate term, and measurement says that is fine
///
/// Every other decision in this encoder now prices its candidates in real
/// bits. This one does not, and the obvious next step — give it a rate
/// term like the rest — was measured before being written, and would
/// change nothing. Over 808 CU decisions (three clips, two quantisers,
/// the encoder's own CU size rather than a chosen one):
///
/// - intra wins 1.0% of the time;
/// - **0.0% of decisions fall within 10%** of the boundary, and 0.2%
///   within 25%;
/// - the median separation is 1.22, meaning the two sides typically
///   differ by more than the whole inter SATD.
///
/// A Lagrangian rate term moves a comparison by lambda times a bit
/// difference. Nothing that size flips a decision separated by more than
/// 100%, so adding one would cost work and change no output.
///
/// # What that probe does NOT bound, stated because it is the same trap
///
/// It measures how marginal the comparison *as written* is, so it bounds
/// the effect of adding a term to that comparison. It says nothing about
/// replacing the DC-flat distortion proxy, because a bad proxy can sit
/// far from the boundary and still be on the wrong side of it — being
/// unmarginal is not being right. If this decision is ever worth
/// improving, the proxy is the thing to attack, and testing that needs a
/// different probe: code both candidates properly and compare real costs,
/// rather than asking whether the existing numbers are close.
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

    if ctx.bypass {
        // Lossless: the residual IS the coefficients. `residual_block`
        // skips `scale_coefficients` and the inverse transform for a
        // bypassed CU (`hevc::ctu`'s `if !cu.bypass` gate) and adds what
        // it parsed as it stands, so carrying the raw difference makes
        // prediction plus residual equal the source — and the clip inside
        // `add_residual` never bites, because that sum is a real sample
        // value by construction. The same branch `h265_intra`'s
        // `code_residual` takes, and for the same reason.
        levels[..n * n].copy_from_slice(&work[..n * n]);
        (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
        return levels[..n * n].iter().filter(|&&v| v != 0).count() as u32;
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
    use crate::picture::ChromaFormat;
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
            IntraCtx { dsp: &self.dsp, enc: &self.enc, dist: &self.dist, qp, bit_depth: 8, strong_smoothing: false, bypass: false, free_to_trim: false }
        }
    }

    /// The parameter sets this stream would carry, through the decoder's
    /// own parsers — the same round trip the encoder applies to everything
    /// it writes.
    fn parsed_sets(w: u32, h: u32) -> (Sps, Pps) {
        parsed_sets_fmt(w, h, ChromaFormat::Yuv420)
    }

    /// The same, in a chosen chroma format — `chroma_format_idc` travels
    /// through `write_sps` and back out of the decoder's parser, so the
    /// decision module reads the format from a round trip rather than from
    /// the caller's word for it.
    fn parsed_sets_fmt(w: u32, h: u32, chroma: ChromaFormat) -> (Sps, Pps) {
        let cfg = Config { width: w, height: h, gop: 8, chroma, ..Config::default() };
        let syn = SynGeometry::new(&cfg);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &syn, 16, None))).unwrap();
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
        reference_fmt(w, h, seed, ChromaFormat::Yuv420)
    }

    /// The same reference in a chosen chroma format. The chroma bowls are
    /// laid out over whatever plane the format gives, so 4:2:2's
    /// full-height and 4:4:4's full-size chroma each carry real structure
    /// rather than a stretched copy of the 4:2:0 one — content that a
    /// wrongly scaled chroma vector would visibly miss.
    fn reference_fmt(w: usize, h: usize, seed: u64, chroma: ChromaFormat) -> Frame<u8> {
        let mut f = Frame::new(w, h, chroma, 8);
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
        let (cw, ch) = (f.cb.width, f.cb.height);
        for y in 0..ch {
            for x in 0..cw {
                let r2 = (x as i32 - (cw / 2) as i32).pow(2) + (y as i32 - (ch / 2) as i32).pow(2);
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
        // The chroma shift is the luma one divided by (SubWidthC,
        // SubHeightC): a 4:2:2 chroma plane is not subsampled vertically,
        // so its rows shift by the *whole* `dy`, and a caller wanting an
        // integral chroma translation must therefore keep `dx` even but
        // may leave `dy` odd. Taking this from the frame's own format is
        // what makes the fixture agree with `predict_block`'s `mvc`.
        let (sw, sh) = refp.chroma.subsampling();
        let (sw, sh) = (sw as i32, sh as i32);
        let (cw, ch) = (refp.cb.width, refp.cb.height);
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..ch {
            for xx in 0..cw {
                cb[yy * cw + xx] = refp.cb.at_clamped(xx as i32 + dx / sw, yy as i32 + dy / sh);
                cr[yy * cw + xx] = refp.cr.at_clamped(xx as i32 + dx / sw, yy as i32 + dy / sh);
            }
        }
        (y, cb, cr)
    }

    /// The average of two references, each shifted by its own full-sample
    /// vector: the content a B picture between them carries. `d0` is the
    /// list-0 motion and `d1` the list-1 motion, in luma samples; chroma
    /// shifts by the same vector divided by this format's (SubWidthC,
    /// SubHeightC), so an even vector stays integral in every format.
    /// Monochrome returns empty chroma planes.
    fn bi_translated(r0: &Frame<u8>, d0: (i32, i32), r1: &Frame<u8>, d1: (i32, i32)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (r0.width, r0.height);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                let a = r0.y.at_clamped(xx as i32 + d0.0, yy as i32 + d0.1) as i32;
                let b = r1.y.at_clamped(xx as i32 + d1.0, yy as i32 + d1.1) as i32;
                y[yy * w + xx] = ((a + b + 1) >> 1) as u8;
            }
        }
        let (sw, sh) = r0.chroma.subsampling();
        let (sw, sh) = (sw as i32, sh as i32);
        let (cw, ch) = (r0.cb.width, r0.cb.height);
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..ch {
            for xx in 0..cw {
                for (plane0, plane1, dst) in [(&r0.cb, &r1.cb, &mut cb), (&r0.cr, &r1.cr, &mut cr)] {
                    let a = plane0.at_clamped(xx as i32 + d0.0 / sw, yy as i32 + d0.1 / sh) as i32;
                    let b = plane1.at_clamped(xx as i32 + d1.0 / sw, yy as i32 + d1.1 / sh) as i32;
                    dst[yy * cw + xx] = ((a + b + 1) >> 1) as u8;
                }
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
        let (sw, _) = sub_wh(sps.chroma_array_type());
        let c_stride = if sps.chroma_array_type() == 0 { 0 } else { w / sw };
        let mut pic = InterPicture::new(sps, pps, 1);
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu(ctx, &[refp], cx, cy, src_y, w, src_cb, src_cr, c_stride));
            }
        }
        (pic, decisions)
    }

    /// Every decision's internal consistency: `Skip` carries nothing,
    /// `Merge` always carries residual (the inferred `rqt_root_cbf`), the
    /// cbf flags agree with the coefficient arrays.
    fn assert_invariants(decisions: &[InterCuDecision], cat: u32) {
        for (i, d) in decisions.iter().enumerate() {
            let n = 1usize << d.log2_cu;
            let nz_l = d.luma[..n * n].iter().any(|&v| v != 0);
            // The chroma slots this format uses, at this format's TB size.
            let (_, ntb, log2c) = chroma_tbs(cat, 0, 0, d.log2_cu);
            let nc2 = 1usize << (2 * log2c);
            let slot = |comp: usize, t: usize| d.chroma[comp][t * nc2..(t + 1) * nc2].iter().any(|&v| v != 0);
            let mut nz_c = false;
            for comp in 0..2 {
                assert_eq!(
                    d.cbf_chroma[comp],
                    ntb > 0 && slot(comp, 0),
                    "cu {i}: cbf_chroma[{comp}] disagrees with the levels"
                );
                assert_eq!(
                    d.cbf_chroma_bot[comp],
                    ntb > 1 && slot(comp, 1),
                    "cu {i}: cbf_chroma_bot[{comp}] disagrees with the levels"
                );
                nz_c |= d.cbf_chroma[comp] || d.cbf_chroma_bot[comp];
            }
            if cat == 0 {
                assert!(!nz_c, "cu {i}: monochrome carries a chroma cbf");
            }
            if cat != 2 {
                assert_eq!(d.cbf_chroma_bot, [false; 2], "cu {i}: only 4:2:2 has a bottom chroma square");
            }
            // Every slot this format does not use must be untouched: the
            // writer indexes by slot, so a stray level there would be
            // spelled into some other format's stream shape.
            for comp in 0..2 {
                let used = ntb * nc2;
                assert!(d.chroma[comp][used..].iter().all(|&v| v == 0), "cu {i}: levels beyond the format's chroma slots");
            }
            assert_eq!(d.cbf_luma, nz_l, "cu {i}: cbf_luma disagrees with the levels");
            assert_eq!(d.rqt_root_cbf, nz_l || nz_c, "cu {i}: rqt_root_cbf disagrees");
            // The inference trap the writer relies on: at an inter leaf of
            // depth 0 with no chroma cbf the reader reads no cbf_luma bin
            // and infers 1, so a coded tree with neither is unspellable.
            if d.rqt_root_cbf && !nz_c {
                assert!(d.cbf_luma, "cu {i}: a coded tree with no chroma cbf must have cbf_luma 1 (the reader infers it)");
            }
            match d.kind {
                InterCuKind::Skip { .. } => assert!(!d.rqt_root_cbf, "cu {i}: a skip CU carries residual"),
                InterCuKind::Merge { .. } => {
                    // The reader infers rqt_root_cbf 1 for a non-skip
                    // 2Nx2N merge CU: producing one without residual
                    // would desync — such a CU must have been Skip.
                    assert!(d.rqt_root_cbf, "cu {i}: a zero-residual merge CU escaped becoming Skip")
                }
                InterCuKind::Amvp { .. } | InterCuKind::BAmvp { .. } | InterCuKind::UseIntra => {}
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
        assert_invariants(&decisions, 1);
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
        assert_invariants(&decisions, 1);
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
        assert_invariants(&decisions, 1);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d.mv, want, "cu {i}: {:?}", d.kind);
            assert!(!d.rqt_root_cbf, "cu {i}: the exact interpolation left residual");
        }
    }

    /// The B anchor: every B decision replayed through the decoder's own
    /// candidate derivation over an independently maintained state, once
    /// per chroma format.
    ///
    /// This is [`every_decision_replays_through_an_independent_decoder_state`]
    /// for the two-list case, and it proves the same thing plus what only
    /// B has: that `merge_candidate` with `is_b` re-derives the *pair* of
    /// vectors and reference indices the decision recorded, that a
    /// per-list AMVP replays through `amvp(list)` and the wrapping mvd
    /// sum, and that the bi reconstruction the encoder holds is the one
    /// `predict_block` builds from both lists.
    ///
    /// The vacuity guards matter as much as the replay: content that never
    /// chose BI, or never chose merge, would let a broken second list pass
    /// unnoticed, so both are asserted to have occurred.
    #[test]
    fn every_b_decision_replays_through_an_independent_decoder_state() {
        for chroma in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
            // Three sources per format, because ONE source does not reach
            // the shapes. An earlier version of this test used only the
            // bi fixture and every CU came out `Merge` at the zero vector;
            // `BAmvp` never occurred at all, and a mutation sweep showed
            // five of six seeded faults passing. The scenarios exist to
            // drive each `inter_pred_idc` for real.
            let mut seen_idc = [false; 3];
            for scen in [BScenario::Uni0, BScenario::Uni1, BScenario::Bi] {
                replay_b_one_format(chroma, scen, &mut seen_idc, None);
            }
            assert!(seen_idc[0], "{chroma:?}: no CU ever coded PRED_L0 through AMVP");
            assert!(seen_idc[1], "{chroma:?}: no CU ever coded PRED_L1 through AMVP");
            // PRED_BI through AMVP is deliberately NOT asserted, and the
            // reason is a property of the decision rather than a gap in
            // this test. Whenever both lists find good vectors, the merge
            // list already holds an equivalent two-list candidate — from a
            // neighbour, a combined bi-predictive pair, or the bi zero
            // candidate — and merge costs a couple of bins against AMVP-BI's
            // `inter_pred_idc` plus two mvds plus two mvp flags. Merge
            // therefore wins on rate, correctly. Bi prediction and bi
            // reconstruction ARE exercised here, through those merge CUs
            // (the `BScenario::Bi` guard below requires a coded two-list
            // CU); what is not exercised is the AMVP-BI *signalling*.
            //
            // That signalling is covered where it can be driven directly:
            // `hevc::ctu`'s `b_inter_pred_idc_round_trips_by_value` writes
            // all three `inter_pred_idc` values through the production
            // writer and reads them back with the production decoder. If
            // an iterative bi refinement ever lands, AMVP-BI should start
            // winning here and this comment becomes an assertion.
            let _ = seen_idc[2];
        }
    }

    /// The B replay under explicit weighting: the same three scenarios,
    /// with each anchor handed to the decision as the picture its list's
    /// table entry weights back to the content the source was built from
    /// — so the weighted prediction, not the default one, is what matches.
    /// The list-0 and list-1 entries differ in every component, gain and
    /// offset, luma and chroma, so a walk that predicted without the
    /// weights, or weighted one list with the other's entry, diverges from
    /// the replay, which derives its weighting from the table through the
    /// reader's own `explicit_weighting`.
    #[test]
    fn every_weighted_b_decision_replays_through_an_independent_decoder_state() {
        let table = weighted_b_table();
        for chroma in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
            let mut seen_idc = [false; 3];
            for scen in [BScenario::Uni0, BScenario::Uni1, BScenario::Bi] {
                replay_b_one_format(chroma, scen, &mut seen_idc, Some(&table));
            }
            assert!(seen_idc[0], "{chroma:?} weighted: no CU ever coded PRED_L0 through AMVP");
            assert!(seen_idc[1], "{chroma:?} weighted: no CU ever coded PRED_L1 through AMVP");
        }
    }

    /// A B slice's table whose two lists differ in every component: list 0
    /// a gain below the identity with positive offsets, list 1 a gain above
    /// it with negative ones. Denominators 6, 8-bit offsets.
    fn weighted_b_table() -> crate::hevc::slice::PredWeightTable {
        use crate::hevc::slice::{PredWeightTable, WeightEntry};
        PredWeightTable {
            luma_log2_denom: 6,
            chroma_log2_denom: 6,
            lists: [vec![WeightEntry { luma: (56, 12), chroma: [(60, 6), (64, 3)] }], vec![WeightEntry { luma: (72, -10), chroma: [(66, -2), (68, -6)] }]],
        }
    }

    /// The inverse of the weighting `(w, o)` at denominator 6 over the
    /// picture area of `plane`: each sample `p` becomes the `q` whose
    /// `((q * w + 32) >> 6) + o` lands nearest `p`, so a prediction from
    /// the result under that weighting reproduces the plane it was made
    /// from, to a rounding.
    fn unweight(plane: &mut Plane16<u8>, (w, o): (i32, i32)) {
        for y in 0..plane.height {
            for x in 0..plane.width {
                let off = plane.offset(x as isize, y as isize);
                let p = i32::from(plane.data[off]);
                plane.data[off] = (((p - o) * 64 + w / 2) / w).clamp(0, 255) as u8;
            }
        }
    }

    /// What the source of a B replay looks like, and therefore which
    /// signalling the decision should reach for.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BScenario {
        /// An exact translation of the list-0 anchor: `PRED_L0` should win.
        Uni0,
        /// An exact translation of the list-1 anchor: `PRED_L1` should win.
        Uni1,
        /// The average of both anchors, each moved toward this picture —
        /// what default-weighted bi-prediction produces, so `PRED_BI`
        /// should win.
        Bi,
    }

    fn replay_b_one_format(chroma: ChromaFormat, scen: BScenario, seen_idc: &mut [bool; 3], weights: Option<&crate::hevc::slice::PredWeightTable>) {
        let kit = Kit::new();
        let ctx = kit.ctx(30);
        let (sps, pps) = parsed_sets_fmt(64, 64, chroma);
        let cat = sps.chroma_array_type();
        // Two anchors around the current picture: POC 1 in the past, POC 3
        // in the future, current POC 2 — a real B, so NoBackwardPredFlag
        // is false and both lists are live.
        // The two anchors must carry DIFFERENT texture, not the same
        // grating at an offset. With identical structure, a per-list
        // search cannot tell which anchor it is looking at: both lists
        // converge on the same vector, their average equals either half,
        // and PRED_BI can never beat a uni shape on rate — so the bi path
        // would go untested however the source were built. Independent
        // noise per anchor is what makes each list's own vector findable
        // and makes averaging genuinely better than either half, which is
        // the whole premise of bi-prediction.
        let mut ref0 = reference_fmt(64, 64, 17, chroma);
        let mut ref1 = reference_fmt(64, 64, 23, chroma);
        for (f, mut seed) in [(&mut ref0, 0x51ed_u64), (&mut ref1, 0xb0a7_u64)] {
            for y in 0..64usize {
                for x in 0..64usize {
                    let off = f.y.offset(x as isize, y as isize);
                    let d = (lcg(&mut seed) % 24) as i32 - 12;
                    f.y.data[off] = (f.y.data[off] as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        ref0.poc = 1;
        ref0.extend_rows(0, 64);
        ref1.poc = 3;
        ref1.extend_rows(0, 64);

        // The source this scenario asks for. Even vectors throughout, so
        // the chroma translation is integral in every format and the
        // fixture matches what `predict_block` derives.
        let (mut sy, scb, scr) = match scen {
            BScenario::Uni0 => translated(&ref0, 4, 2),
            BScenario::Uni1 => translated(&ref1, -4, -2),
            BScenario::Bi => bi_translated(&ref0, (4, 2), &ref1, (-4, -2)),
        };
        // Light damage, so residual survives and the AMVP shapes are
        // reached rather than everything collapsing to skip. Kept well
        // below the level that would make intra win.
        let mut s = 41u64;
        for yy in 0..64usize {
            for xx in 0..64usize {
                if (xx / 16 + yy / 16) % 3 == 0 {
                    let d = (lcg(&mut s) % 16) as i32 - 8;
                    let v = &mut sy[yy * 64 + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        // Under a table, each anchor becomes the picture its own list's
        // entry weights back to the content the source was built from.
        if let Some(t) = weights {
            for (f, e) in [(&mut ref0, &t.lists[0][0]), (&mut ref1, &t.lists[1][0])] {
                unweight(&mut f.y, e.luma);
                unweight(&mut f.cb, e.chroma[0]);
                unweight(&mut f.cr, e.chroma[1]);
                f.extend_rows(0, 64);
            }
        }
        let weighting = |ref_idx: [i8; 2]| weights.map_or([Weighting::Default; 3], |t| crate::hevc::ctu::explicit_weighting(t, 8, 8, ref_idx));

        let (w, h) = (64usize, 64usize);
        let n = 1usize << sps.log2_ctb_size;
        let (sw, _) = sub_wh(cat);
        let c_stride = if cat == 0 { 0 } else { w / sw };
        let mut pic = InterPicture::new(&sps, &pps, 2);
        // The production mapping from the table to each prediction's
        // weighting — so a mapping that bound a list to the other's entry
        // makes one-list prediction lose everywhere and the PRED_L0 /
        // PRED_L1 guards go red; the replay derives its own weighting from
        // each decision's reference indices.
        if let Some(t) = weights {
            pic.set_b_weights(t, 8, 8);
        }
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu_b(&ctx, &ref0, &ref1, cx, cy, &sy, w, &scb, &scr, c_stride));
            }
        }
        assert_invariants(&decisions, cat);
        let tag = format!("{chroma:?}/{scen:?}{}", if weights.is_some() { " weighted" } else { "" });
        // Record which `inter_pred_idc` values were reached, for the
        // caller's aggregate coverage check. Only a CODED CU counts: a
        // `UseIntra` decision still carries the motion fields this module
        // filled before the intra check, so counting those would make the
        // guard vacuous — which is exactly how the earlier version of this
        // test managed to prove nothing.
        for d in &decisions {
            if let InterCuKind::BAmvp { idc, .. } = d.kind {
                seen_idc[idc as usize] = true;
            }
        }
        assert!(
            decisions.iter().any(|d| !matches!(d.kind, InterCuKind::UseIntra)),
            "{tag}: every CU went intra; the inter path is untested here"
        );
        // The scenario must actually reach the shape it is named for,
        // through some CU: uni scenarios a single-list CU, the bi scenario
        // a two-list one. Merge candidates count here — they carry lists
        // too — but intra decisions do not.
        let coded = decisions.iter().filter(|d| !matches!(d.kind, InterCuKind::UseIntra));
        let hit = match scen {
            BScenario::Uni0 => coded.clone().any(|d| d.ref_idx >= 0 && d.ref_idx_l1 < 0),
            BScenario::Uni1 => coded.clone().any(|d| d.ref_idx < 0 && d.ref_idx_l1 >= 0),
            BScenario::Bi => coded.clone().any(|d| d.ref_idx >= 0 && d.ref_idx_l1 >= 0),
        };
        assert!(hit, "{tag}: the scenario never reached its own prediction shape");

        // The independent state.
        let geo = std::sync::Arc::new(Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let mut frame = Frame::<u8>::new(64, 64, chroma, 8);
        frame.poc = 2;
        let mut scratch = McScratch::new();
        let no_backward_pred = [ref0.poc, ref1.poc].iter().all(|&p| p <= 2);
        assert!(!no_backward_pred, "a future anchor must make NoBackwardPredFlag false");
        let refs = RefCtx::<u8> {
            pocs: [vec![ref0.poc], vec![ref1.poc]],
            long_term: [vec![false], vec![false]],
            col: None,
            cur_poc: 2,
            no_backward_pred,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: true,
            num_ref_idx: [1, 1],
            col_from_l0: true,
        };
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
            let (mv, ref_idx) = match d.kind {
                InterCuKind::Skip { merge_idx } | InterCuKind::Merge { merge_idx } => {
                    let cand = merge_candidate(&info, &frame, &refs, &pu, merge_idx as usize);
                    (cand.mv, cand.ref_idx)
                }
                InterCuKind::BAmvp { idc, mvd, mvp_flag } => {
                    let r: [i8; 2] = match idc {
                        0 => [0, -1],
                        1 => [-1, 0],
                        _ => [0, 0],
                    };
                    let mut mv = [Mv::ZERO; 2];
                    for list in 0..2usize {
                        if r[list] >= 0 {
                            let p = amvp(&info, &frame, &refs, &pu, list, 0, mvp_flag[list] as u32);
                            mv[list] = Mv::new(p.x.wrapping_add(mvd[list].x), p.y.wrapping_add(mvd[list].y));
                        }
                    }
                    (mv, r)
                }
                InterCuKind::Amvp { .. } => unreachable!("the B walk never produces the P shape"),
                InterCuKind::UseIntra => {
                    fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, MotionInfo::INTRA);
                    PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 1);
                    continue;
                }
            };
            assert_eq!(ref_idx, [d.ref_idx, d.ref_idx_l1], "cu {i}: replayed lists differ ({:?})", d.kind);
            assert_eq!([mv[0], mv[1]], [d.mv, d.mv_l1], "cu {i}: signalling does not replay to the chosen vectors ({:?})", d.kind);

            let r0 = (ref_idx[0] >= 0).then_some((&ref0, mv[0]));
            let r1 = (ref_idx[1] >= 0).then_some((&ref1, mv[1]));
            predict_block(&kit.dsp, &mut scratch, &mut frame, x0, y0, n, n, r0, r1, weighting(ref_idx));
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
                if cat != 0 {
                    let qp_c = chroma_qp(cat, ctx.qp.clamp(0, 57));
                    let (sw, sh) = sub_wh(cat);
                    let (tbs, ntb, log2c) = chroma_tbs(cat, x0, y0, d.log2_cu);
                    let nc = 1usize << log2c;
                    let nc2 = nc * nc;
                    for comp in 0..2 {
                        for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                            let cbf = if t == 0 { d.cbf_chroma[comp] } else { d.cbf_chroma_bot[comp] };
                            if !cbf {
                                continue;
                            }
                            work[..nc2].copy_from_slice(&d.chroma[comp][t * nc2..(t + 1) * nc2]);
                            scale_coefficients(&mut work, log2c, qp_c, 8, ScalingSource::Flat, false, nc - 1, nc - 1);
                            (kit.dsp.idct[(log2c - 2) as usize])(&mut work, bd_shift, nc - 1, nc - 1);
                            let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                            let off = plane.offset((ax / sw) as isize, (ay / sh) as isize);
                            (kit.dsp.add_residual)(&mut plane.data[off..], plane.stride, &work, nc, 255);
                        }
                    }
                }
            }
            let mut mi = MotionInfo { mv, ref_idx, ref_delta: [0; 2], flags: 0, pad: 0 };
            for list in 0..2usize {
                if ref_idx[list] >= 0 {
                    let poc = if list == 0 { ref0.poc } else { ref1.poc };
                    mi.ref_delta[list] = (2 - poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }
            fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, mi);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 0);
        }
        assert_eq!(frame.y.data, pic.recon.y.data, "{tag}: luma reconstruction differs from the replay");
        assert_eq!(frame.cb.data, pic.recon.cb.data, "{tag}: cb reconstruction differs from the replay");
        assert_eq!(frame.cr.data, pic.recon.cr.data, "{tag}: cr reconstruction differs from the replay");
        assert_eq!(frame.motion, pic.recon.motion, "{tag}: motion grids diverged");
    }

    /// The anchor: every decision, replayed through the decoder's own
    /// candidate derivation over an *independently maintained* state —
    /// fresh `PicInfo`, fresh motion grid, fresh reconstruction — the way
    /// a writer plus a decoder would consume it. This proves the state
    /// maintenance (motion fills, availability marks), which is the half
    /// of the signalling contract that direct reuse of `merge_candidate` /
    /// `amvp` does not already make unbreakable; the derivation itself is
    /// held by the decoder's conformance suites.
    ///
    /// Run once per chroma format, because the reconstruction the replay
    /// rebuilds is where each format's chroma differs: the vector
    /// `predict_block` derives, the transform blocks the residual lands
    /// in, and the QP mapping that scales it. A 4:2:0-only replay would
    /// pass with every other format's chroma wrong.
    #[test]
    fn every_decision_replays_through_an_independent_decoder_state() {
        for chroma in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
            replay_one_format(chroma);
        }
    }

    fn replay_one_format(chroma: ChromaFormat) {
        let kit = Kit::new();
        let ctx = kit.ctx(30);
        let (sps, pps) = parsed_sets_fmt(64, 64, chroma);
        let cat = sps.chroma_array_type();
        let refp = reference_fmt(64, 64, 17, chroma);
        // Mixed content: a translation with damage in some regions, so
        // skip, merge-with-residual and AMVP all appear.
        let (mut sy, mut scb, mut scr) = translated(&refp, 3, 1);
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
        // Chroma damage as well, in a different stripe, because at 4:4:4
        // the chroma translation is exact (SubWidthC and SubHeightC are 1,
        // so the fixture's integer shift *is* the true chroma vector) and
        // an undamaged source leaves no chroma residual at all — the
        // replay would then compare two copies of a plain prediction and
        // prove nothing about the chroma transform path. 4:2:0 and 4:2:2
        // get residual for free from their fractional chroma vectors; this
        // makes every format carry some.
        let (cw, ch) = (refp.cb.width, refp.cb.height);
        for yy in 0..ch {
            for xx in 0..cw {
                if (xx / 8 + yy / 8) % 3 == 1 {
                    let d = (lcg(&mut s) % 40) as i32 - 20;
                    let v = &mut scb[yy * cw + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                    let d = (lcg(&mut s) % 40) as i32 - 20;
                    let v = &mut scr[yy * cw + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        let (pic, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions, cat);
        let kinds: Vec<_> = decisions.iter().map(|d| std::mem::discriminant(&d.kind)).collect();
        assert!(kinds.iter().collect::<std::collections::HashSet<_>>().len() >= 2, "one-note content: the replay would prove less ({chroma:?})");
        // A format with chroma must actually exercise it, or the replay's
        // chroma comparison below proves nothing about that format.
        if cat != 0 {
            assert!(
                decisions.iter().any(|d| d.cbf_chroma[0] || d.cbf_chroma[1] || d.cbf_chroma_bot[0] || d.cbf_chroma_bot[1]),
                "{chroma:?}: no CU carried a chroma residual"
            );
        }
        if cat == 2 {
            assert!(
                decisions.iter().any(|d| d.cbf_chroma_bot[0] || d.cbf_chroma_bot[1]),
                "4:2:2: the stacked pair's bottom square never carried anything"
            );
        }

        // The independent state.
        let geo = std::sync::Arc::new(Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let mut frame = Frame::<u8>::new(64, 64, chroma, 8);
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
                InterCuKind::BAmvp { .. } => unreachable!("this replay drives the P walk, which never produces a B shape"),
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
                if cat != 0 {
                    let qp_c = chroma_qp(cat, ctx.qp.clamp(0, 57));
                    let (sw, sh) = sub_wh(cat);
                    // Placed by the reader's own derivation, not by this
                    // test's arithmetic: `chroma_tbs` is `transform_unit`'s
                    // `here` plus its stacked-pair loop.
                    let (tbs, ntb, log2c) = chroma_tbs(cat, x0, y0, d.log2_cu);
                    let nc = 1usize << log2c;
                    let nc2 = nc * nc;
                    for comp in 0..2 {
                        for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                            let cbf = if t == 0 { d.cbf_chroma[comp] } else { d.cbf_chroma_bot[comp] };
                            if !cbf {
                                continue;
                            }
                            work[..nc2].copy_from_slice(&d.chroma[comp][t * nc2..(t + 1) * nc2]);
                            scale_coefficients(&mut work, log2c, qp_c, 8, ScalingSource::Flat, false, nc - 1, nc - 1);
                            (kit.dsp.idct[(log2c - 2) as usize])(&mut work, bd_shift, nc - 1, nc - 1);
                            let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                            let off = plane.offset((ax / sw) as isize, (ay / sh) as isize);
                            (kit.dsp.add_residual)(&mut plane.data[off..], plane.stride, &work, nc, 255);
                        }
                    }
                }
            }

            // The decoder's own motion store, on the replay's state.
            let mut mi = MotionInfo { mv: [mv, Mv::ZERO], ref_delta: [0; 2], ref_idx: [0, -1], flags: 0, pad: 0 };
            mi.ref_delta[0] = (1 - refp.poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, mi);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 0);
        }
        // The replayed reconstruction is the encoder's, byte for byte.
        assert_eq!(frame.y.data, pic.recon.y.data, "{chroma:?}: luma reconstruction differs from the replay");
        assert_eq!(frame.cb.data, pic.recon.cb.data, "{chroma:?}: cb reconstruction differs from the replay");
        assert_eq!(frame.cr.data, pic.recon.cr.data, "{chroma:?}: cr reconstruction differs from the replay");
        // And the two sides' motion state agrees, which is what the next
        // picture would predict TMVP from if the SPS ever enables it.
        assert_eq!(frame.motion, pic.recon.motion, "motion grids diverged");
    }
}
