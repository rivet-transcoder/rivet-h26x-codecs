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
//! - **One CU per CTU, no coding quadtree.** The CU size *is* the CTB
//!   size, `log2_cu` in 3..=5. Pictures must be whole multiples of it.
//! - **Partitioning per size, one transform-split level.** `log2_cu` 4 or
//!   5 codes `PART_2Nx2N`, and when [`IntraPicture::split_depth`] allows the
//!   decision tries one level of transform split — four quarter-size TUs
//!   against the single CU-sized one, chroma splitting alongside as the
//!   decoder's `transform_tree` dictates. The split search is **off by
//!   default** because the coding-tree writer does not spell the split
//!   shape yet; producing it into today's serialiser would desync the
//!   stream. One level from a 16/32 CTB bottoms out at 8x8/16x16 luma, so
//!   the 4x4 DST is *still* only reached by `log2_cu` 3's `PART_NxN`
//!   (four 4x4 luma TUs); the second split level, which the SPS's
//!   `max_transform_hierarchy_depth_intra` of 2 already permits, is what
//!   will make it reachable from production geometry — and brings the
//!   chroma-at-the-parent rule (`transform_unit`'s `blk_idx == 3` case)
//!   that one level never triggers.
//! - **All four chroma formats** ([`IntraPicture::new_with_chroma`];
//!   plain `new` stays 4:2:0). Monochrome simply omits every chroma
//!   element, mirroring the reader's `chroma_array_type == 0` gates.
//!   4:2:0 carries one chroma TU per component at half the luma TU size;
//!   4:2:2 the stacked pair of half-size squares `transform_unit` walks
//!   (`yct = yc + t * nc`), each square with its own cbf, the derived
//!   mode passing through the Table 8-3 remap and the chroma QP through
//!   the plain clamp rather than Table 8-10; 4:4:4 one chroma TU at the
//!   luma TU's own size and position, with the reference-smoothing
//!   filter on for chroma too (`intra_predict_block`'s
//!   `c_idx == 0 || cat == 3`). One 4:4:4 corner is refused by name:
//!   `PART_NxN` (8x8 CTUs, test-only geometry) would need four chroma
//!   modes and per-4x4 chroma TBs.
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
    /// one luma TU the size of the CU — or four quarter TUs, see
    /// `split_tu`).
    pub nxn: bool,
    /// `split_transform_flag` at depth 0: the CU's residual is carried by
    /// a transform quadtree instead of one CU-sized TU. Only ever
    /// produced for `PART_2Nx2N` (`log2_cu` 4 or 5), never with `nxn`.
    /// What the writer emits, read off the decoder's `transform_tree`:
    /// `split_transform_flag` 1 at depth 0; `cbf_cb`/`cbf_cr` once at
    /// depth 0 (from `cbf_chroma`); then four depth-1 subtrees in
    /// z-order, each spelling its own `split_transform_flag` (from
    /// `split_child` — always coded in this geometry: the child size is
    /// above the 4x4 minimum and depth 1 is below the SPS's
    /// `max_transform_hierarchy_depth_intra` of 2), its `cbf_cb`/`cbf_cr`
    /// (from `cbf_chroma_tu`, coded only where the depth-0 flag for that
    /// component is set), and — per the child's own shape — `cbf_luma`
    /// and residuals as `split_child` describes.
    pub split_tu: bool,
    /// The depth-1 `split_transform_flag` of each child of a split CU, in
    /// z-order; meaningful only when `split_tu`, all false is exactly the
    /// one-level shape. A set flag subdivides that child into four leaf
    /// TBs at `log2_cu - 2` — 4x4 at a 16 CTB (the DST leaves, scanning
    /// mode-dependently like every 4x4) and 8x8 at a 32 CTB (whose intra
    /// luma also scans mode-dependently, 7.4.9.11's `log2 == 3` arm). The
    /// leaves spell no flag of their own: at a 16 CTB they sit at the
    /// 4x4 minimum, at a 32 CTB at the depth limit — so depth 2 is the
    /// tree's floor either way, and only the 16 CTB ever reaches the
    /// 4x4 DST (a 32 CTB would need depth 3, which the SPS forbids).
    ///
    /// Where the *chroma* of a subdivided child lives follows
    /// `transform_unit` exactly: per luma leaf when the leaf is bigger
    /// than 4x4 luma (a 32 CTB's 8x8 leaves) or the format is 4:4:4
    /// (chroma coded at every TB, the `|| cat == 3` arm); once at the
    /// child for 4x4 leaves in 4:2:0/4:2:2 — the `blk_idx == 3` arm,
    /// chroma coded a single time at the parent's size after the fourth
    /// leaf, exactly the shape `PART_NxN` already uses, stacked pair
    /// included in 4:2:2 (both its depth-1 bins are coded there, the
    /// `log2 == 3` arm of the cbf gate). At 4:0:0 there is, as ever,
    /// nothing.
    pub split_child: [bool; 4],
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
    /// luma. One per CU — 4:4:4 `PART_NxN` would need four. For a
    /// monochrome picture this and every other chroma field is
    /// meaningless: the syntax element does not exist
    /// (`coding_unit` reads it only when `chroma_array_type != 0`) and
    /// the writer emits nothing chroma at all.
    pub chroma_syntax: u8,
    /// The derived `IntraPredModeC` — what `chroma_syntax` decodes to
    /// (8.4.3), stored so the writer's mode-dependent scan for 4x4 chroma
    /// TUs does not re-derive it.
    pub chroma_mode: u8,
    /// `cbf_luma` per luma leaf TB, in **positional** slots: quadrant `i`
    /// of the CU owns `[4*i..4*i + 4]`; a quadrant that is a single leaf
    /// (an unsplit child, or a `PART_NxN` prediction block) uses its
    /// first slot `[4*i]`, a subdivided child fills all four in z-order,
    /// and a CU that is itself one TU uses `[0]`. Slots a shape does not
    /// describe stay false. Positional rather than packed so a slot never
    /// depends on a *sibling's* structure — the price is that the
    /// one-level split and `PART_NxN`, which previously packed their four
    /// flags into `[0..4]`, now sit at `[0], [4], [8], [12]`.
    pub cbf_luma: [bool; 16],
    /// The depth-0 `cbf_cb`, `cbf_cr` bins — `transform_tree`'s
    /// `cbf_c[c][0]`. For an unsplit CU this is the (first, in 4:2:2) TU's
    /// own cbf; for `PART_NxN` it belongs to the parent-size chroma TU
    /// pair. Under `split_tu` it is the gate bin: whether the component
    /// carries any coded residual in any child — the OR of
    /// `cbf_chroma_tu[comp]` and (4:2:2) `cbf_chroma_tu_bot[comp]`, an
    /// invariant a test holds — and the writer emits the per-child bins
    /// only where it is set, exactly the reader's
    /// `depth == 0 || parent_cbf[c][0]` gate.
    pub cbf_chroma: [bool; 2],
    /// 4:2:2 only: the depth-0 `cbf_c[c][1]` bins — the cbf of the
    /// *bottom* square of the stacked chroma pair, coded right after
    /// `cbf_chroma`'s bin for an unsplit CU or a `PART_NxN` one
    /// (`transform_tree` codes it when `cat == 2 && (!split || log2 ==
    /// 3)`). Never coded — and false here — under `split_tu` (the parent
    /// gate is `[c][0]` alone) or in any other format.
    pub cbf_chroma_bot: [bool; 2],
    /// The depth-1 `cbf_cb`/`cbf_cr` per component per child TU in
    /// z-order — each child's `cbf_c[c][0]` bin (its only one in 4:2:0;
    /// the *top* square's in 4:2:2). Meaningful only when `split_tu`; all
    /// false otherwise.
    pub cbf_chroma_tu: [[bool; 4]; 2],
    /// 4:2:2 with `split_tu` only: each child's `cbf_c[c][1]` bin, the
    /// bottom square of that child's stacked pair — which exists exactly
    /// where the child carries parent-level chroma: an unsplit child, or
    /// a child subdivided to 4x4 luma leaves at a 16 CTB (the reader
    /// codes both bins at a split `log2 == 3` node). All false otherwise.
    pub cbf_chroma_tu_bot: [[bool; 4]; 2],
    /// Depth-2 chroma cbfs, in the same positional slots as `cbf_luma`:
    /// where a subdivided child's chroma is coded *per luma leaf* (8x8
    /// leaves at a 32 CTB in any format; every leaf at 4:4:4), slot
    /// `[4*i + j]` holds leaf `j`'s `cbf_c[c][0]` bin, and the child's
    /// `cbf_chroma_tu` entry becomes the depth-1 gate — the OR of its
    /// leaves' bins (an invariant a test holds), gating them in the
    /// reader exactly as depth 0 gates depth 1. All false where chroma
    /// does not subdivide.
    pub cbf_chroma_leaf: [[bool; 16]; 2],
    /// 4:2:2 only: the `cbf_c[c][1]` bins of the per-leaf stacked pairs
    /// of a subdivided child (8x8 luma leaves at a 32 CTB carry a 4x4
    /// chroma pair each). Same slots as `cbf_chroma_leaf`; all false
    /// elsewhere.
    pub cbf_chroma_leaf_bot: [[bool; 16]; 2],
    /// Quantised luma levels (or raw residual when `bypass`), raster
    /// within each TB, laid out **positionally by area** so a region
    /// never depends on a sibling's structure. With `n = 1 << log2_cu`
    /// and `q = (n/2) * (n/2)`:
    /// - `PART_2Nx2N` unsplit: one TU of `n*n` entries at `[0..n*n]`;
    /// - split CU: quadrant `i` owns `[i*q..(i+1)*q]` — an unsplit child
    ///   fills it as one TB; a subdivided child (`split_child[i]`) puts
    ///   leaf `j` (z-order) at `[i*q + j*(q/4)..i*q + (j+1)*(q/4)]`;
    /// - `PART_NxN`: four 4x4 TUs at `[16*i..16*i + 16]` (the same
    ///   quadrant regions, `q = 16`).
    ///
    /// Entries beyond the described TBs are zero and meaningless.
    pub luma: [i16; 1024],
    /// Quantised chroma levels per component (`[0]` Cb, `[1]` Cr), raster
    /// within each TB, raw residual when `bypass`, laid out positionally
    /// by area like `luma`. All-zero at 4:0:0. Let `ac` be the CU's total
    /// chroma area per component (`(n/SubWidthC) * (n/SubHeightC)`):
    /// - unsplit CU (and `PART_NxN`): the parent-shape TBs in coding
    ///   order from `[0]` — one square in 4:2:0/4:4:4, the 4:2:2 stacked
    ///   pair top then bottom, each TB `area` entries;
    /// - split CU: quadrant `i` owns `[i*(ac/4)..(i+1)*(ac/4)]`. A child
    ///   whose chroma is coded at child level — an unsplit child, or 4x4
    ///   luma leaves in 4:2:0/4:2:2 (the `blk_idx == 3` shape) — fills
    ///   its region with the parent-shape TBs in coding order (pair top
    ///   then bottom in 4:2:2). A child whose chroma subdivides per luma
    ///   leaf (8x8 leaves at a 32 CTB; every leaf at 4:4:4) puts leaf
    ///   `j`'s TBs at `[i*(ac/4) + j*(ac/16)..]`, pair-within-leaf in
    ///   4:2:2.
    ///
    /// Worked sizes: 4:2:0 one-level split at a 32 CTB — child region
    /// `ac/4 = 64`, one 8x8 TB each, unchanged from before depth 2
    /// existed; 4:2:2 subdivided child at a 32 CTB — leaf slot
    /// `ac/16 = 32` holding a 4x4 pair. Entries beyond the described TBs
    /// are zero and meaningless. Sized for the largest shape (4:4:4
    /// chroma at a 32 CTB is a 32x32 TU).
    pub chroma: [[i16; 1024]; 2],
}

impl Default for CuDecision {
    fn default() -> Self {
        CuDecision {
            log2_cu: 0,
            nxn: false,
            split_tu: false,
            split_child: [false; 4],
            bypass: false,
            luma_modes: [1; 4],
            luma_syntax: [LumaModeSyntax::default(); 4],
            chroma_syntax: 4,
            chroma_mode: 1,
            cbf_luma: [false; 16],
            cbf_chroma: [false; 2],
            cbf_chroma_bot: [false; 2],
            cbf_chroma_tu: [[false; 4]; 2],
            cbf_chroma_tu_bot: [[false; 4]; 2],
            cbf_chroma_leaf: [[false; 16]; 2],
            cbf_chroma_leaf_bot: [[false; 16]; 2],
            luma: [0; 1024],
            chroma: [[0; 1024]; 2],
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
///
/// `pub(crate)` because an intra CU is not the exclusive property of an
/// intra picture: a P slice may hold one, and
/// [`super::h265_me::InterPicture`] then builds this same descriptor over
/// its own reconstruction and calls [`code_cu_2nx2n_intra`] with it, so
/// that both slice types run one spelling of the intra decision.
#[derive(Clone, Copy)]
pub(crate) struct Geo {
    /// log2 of the CU == CTB size.
    pub(crate) log2_cu: u32,
    /// CTUs per row.
    pub(crate) wc: usize,
    /// 4x4 blocks per row.
    pub(crate) w4: usize,
    /// Luma picture width and height in samples.
    pub(crate) width: usize,
    /// See `width`.
    pub(crate) height: usize,
    /// `chroma_array_type`: 0 monochrome, 1 = 4:2:0 — the discriminator
    /// the reader's chroma gates test, carried in its numeric form so the
    /// mirrors read like the code they mirror.
    pub(crate) cat: u32,
}

impl Geo {
    /// The descriptor for a picture of `width` by `height` luma samples
    /// coded as whole CTUs of `1 << log2_cu`. The two derived fields are
    /// computed here rather than at each call site, because a `wc` or a
    /// `w4` that disagrees with the picture silently corrupts every
    /// availability answer instead of failing.
    pub(crate) fn new(log2_cu: u32, width: usize, height: usize, cat: u32) -> Self {
        Geo { log2_cu, wc: width >> log2_cu, w4: width / 4, width, height, cat }
    }
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
    /// How deep the `PART_2Nx2N` transform-split search may go — what
    /// used to be `try_split`, grown a level:
    /// - `0` — never split (a single CU-sized TU always);
    /// - `1` — try the one-level split, bit-identical to the old
    ///   `try_split = true` (the cost accounting reduces to exactly the
    ///   old formula when no child subdivides);
    /// - `2` — additionally let each child of a split CU try subdividing
    ///   into four leaf TBs, decided greedily child by child in decode
    ///   order (each child's structure is final before the next one
    ///   codes, because the next one predicts from its reconstruction).
    ///
    /// **Defaults to 0**: the coding-tree writer in `encode::h265` must
    /// spell whichever shapes this permits, and a decision it cannot
    /// serialise would desync the arithmetic coder — the wiring flips it
    /// to what the writer supports. Values above 2 are a bug (the SPS's
    /// `max_transform_hierarchy_depth_intra` is 2) and assert.
    pub split_depth: u32,
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
    /// State for a 4:2:0 picture of `width` by `height` luma samples,
    /// both multiples of the CU size — the fixed-geometry simplification
    /// above. Kept as-is so existing callers stay source- and
    /// bit-identical; other chroma formats go through
    /// [`IntraPicture::new_with_chroma`].
    pub fn new(width: usize, height: usize, log2_cu: u32, bit_depth: u32) -> Self {
        Self::new_with_chroma(width, height, log2_cu, bit_depth, ChromaFormat::Yuv420)
    }

    /// [`IntraPicture::new`] with the chroma format spelled out.
    /// Monochrome codes no chroma at all; formats this module does not
    /// model yet are refused by name rather than mis-coded.
    pub fn new_with_chroma(width: usize, height: usize, log2_cu: u32, bit_depth: u32, chroma: ChromaFormat) -> Self {
        assert!((3..=5).contains(&log2_cu), "log2_cu {log2_cu} outside 3..=5");
        let n = 1usize << log2_cu;
        assert!(width.is_multiple_of(n) && height.is_multiple_of(n), "{width}x{height} is not a whole number of {n}x{n} CTUs");
        let cat = match chroma {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        };
        // 4:4:4 PART_NxN carries four chroma modes (one per PB) and a
        // chroma TB inside every 4x4 luma TB — a shape CuDecision's single
        // chroma mode cannot describe. The 8x8-CTB geometry is test-only,
        // so refuse the combination by name rather than mis-code it.
        assert!(
            !(cat == 3 && log2_cu == 3),
            "H.265 intra decision: 4:4:4 with 8x8 CTUs (PART_NxN needs per-PB chroma modes; unimplemented)"
        );
        let w4 = width / 4;
        let h4 = height / 4;
        IntraPicture {
            recon: Frame::new(width, height, chroma, bit_depth),
            log2_cu,
            split_depth: 0,
            // 1 (DC) everywhere, as the decoder initialises intra_mode; the
            // availability test keeps uncoded entries from ever being read.
            modes: vec![1; w4 * h4],
            scratch: IntraScratch::default(),
            geo: Geo::new(log2_cu, width, height, cat),
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
        let split_depth = self.split_depth;
        let n = 1usize << geo.log2_cu;
        let (x0, y0) = (cu_x * n, cu_y * n);
        // Monochrome codes no chroma at all — the reader's chroma work is
        // uniformly gated on `chroma_array_type != 0` (the mode syntax in
        // `coding_unit`, the cbfs in `transform_tree`, prediction and
        // residual in `transform_unit`), and so is every chroma step
        // below. The source slices are never indexed then, so callers may
        // pass empty ones.
        let (scb, scr) = if geo.cat != 0 {
            let (sw, sh) = sub_wh(geo.cat);
            let coff = (y0 / sh) * c_stride + x0 / sw;
            (&src_cb[coff..], &src_cr[coff..])
        } else {
            (&src_cb[..0], &src_cr[..0])
        };
        let mut out = CuDecision { log2_cu: geo.log2_cu, bypass: ctx.bypass, ..CuDecision::default() };

        let IntraPicture { recon, modes, scratch, .. } = self;
        if geo.log2_cu == 3 {
            // PART_NxN: four 4x4 luma PBs/TUs (the DST path) and one 4x4
            // chroma TU pair. No transform-split choice exists here — the
            // tree is forced by IntraSplitFlag.
            out.nxn = true;
            for pb in 0..4 {
                // z-order within the CU, which is decode order: each block
                // predicts from the reconstruction of those before it.
                let (px, py) = (x0 + (pb & 1) * 4, y0 + (pb >> 1) * 4);
                let cands = mpm_candidates(geo, modes, None, px, py);
                let soff = py * y_stride + px;
                let mode = search_luma_mode(ctx, geo, &mut recon.y, scratch, px, py, 2, &src_y[soff..], y_stride, cands);
                let nz = code_luma_tb(
                    ctx,
                    geo,
                    &mut recon.y,
                    scratch,
                    px,
                    py,
                    2,
                    mode,
                    &src_y[soff..],
                    y_stride,
                    &mut out.luma[pb * 16..pb * 16 + 16],
                );
                out.luma_modes[pb] = mode;
                out.luma_syntax[pb] = as_syntax(mode, cands);
                // Positional cbf slot: this prediction block is quadrant
                // `pb`'s single leaf.
                out.cbf_luma[4 * pb] = nz != 0;
                // The decoder records each PU's mode as it derives it, so
                // the next PU's MPM list sees this one; mirror that.
                PicInfo::fill4(modes, geo.w4, px, py, 4, 4, mode);
            }
            if geo.cat != 0 {
                let (csyn, cmode) = search_chroma_mode(
                    ctx,
                    geo,
                    &mut recon.cb,
                    &mut recon.cr,
                    scratch,
                    x0,
                    y0,
                    3,
                    out.luma_modes[0],
                    scb,
                    scr,
                    c_stride,
                );
                out.chroma_syntax = csyn;
                out.chroma_mode = cmode;
                // The parent-size chroma TB (pair, in 4:2:2): an NxN CU's
                // chroma is coded once at the CU, `transform_unit`'s
                // `blk_idx == 3` case, with the depth-0 cbfs — at log2 3
                // the reader codes both 4:2:2 bins at the parent and the
                // 4x4 children inherit.
                let (sw, sh) = sub_wh(geo.cat);
                let (tbs, ntb, log2c) = chroma_tbs(geo.cat, x0, y0, 3);
                let qtb = 1usize << (2 * log2c);
                for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                    let src = if comp == 0 { scb } else { scr };
                    for (k, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                        let soff = (ay - y0) / sh * c_stride + (ax - x0) / sw;
                        let nz = code_chroma_tb(ctx, geo, plane, scratch, ax, ay, log2c, 1 + comp, cmode, &src[soff..], c_stride, &mut out.chroma[comp][k * qtb..(k + 1) * qtb]);
                        if k == 0 {
                            out.cbf_chroma[comp] = nz != 0;
                        } else {
                            out.cbf_chroma_bot[comp] = nz != 0;
                        }
                    }
                }
            }
        } else {
            return code_cu_2nx2n_intra(ctx, geo, recon, modes, None, scratch, split_depth, x0, y0, src_y, y_stride, src_cb, src_cr, c_stride);
        }
        out
    }

    /// The MPM candidate list for the prediction block at luma position
    /// `(xp, yp)` — exposed for a replaying test or writer that needs to
    /// re-derive what [`LumaModeSyntax`] indexes into.
    pub fn mpm_list(&self, xp: usize, yp: usize) -> [u32; 3] {
        mpm_candidates(self.geo, &self.modes, None, xp, yp)
    }
}

/// Decide and code one `PART_2Nx2N` intra CU at luma `(x0, y0)`, leaving
/// its reconstruction in `recon` and its chosen luma mode in `modes`.
///
/// The body of [`IntraPicture::code_ctu`]'s 2Nx2N arm, lifted out whole
/// so a **P slice can code an intra CU through exactly this code**. An
/// intra CU is an intra CU: the syntax the reader parses inside a P slice
/// is the same `coding_unit` tail (`src/hevc/ctu.rs:448` onward) it parses
/// inside an I slice, so having two spellings of the decision would be two
/// things to keep in step, and one of them would eventually drift.
///
/// Everything the decision touches arrives as an argument rather than
/// through `self`, which is what makes that sharing possible — the caller
/// supplies whichever reconstruction and mode grid its picture owns:
///
/// - `recon` is *the picture's* reconstruction, and for a P picture that
///   means one already holding inter-coded neighbours. Reading them is
///   correct and deliberate: `write_pps` writes
///   `constrained_intra_pred_flag` 0 (`h265_syntax.rs:272`), so the
///   reader's own reference-availability check —
///   `available_at(..) && (!cip || pred_mode == 1)`, `ctu.rs:1157` — has
///   its second clause disabled and inter neighbours *are* references.
/// - `modes` is the per-4x4 luma-mode grid the MPM derivation reads and
///   this function fills, `PicInfo::intra_mode`'s twin.
/// - `pred_mode` is the per-4x4 intra/inter grid, `None` in an all-intra
///   picture; see [`mpm_candidates`].
///
/// The luma source is the whole plane; the chroma sources are whole
/// planes too, offset here by the CU's position.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_cu_2nx2n_intra<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    recon: &mut Frame<S>,
    modes: &mut [u8],
    pred_mode: Option<&[u8]>,
    scratch: &mut IntraScratch,
    split_depth: u32,
    x0: usize,
    y0: usize,
    src_y: &[S],
    y_stride: usize,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
) -> CuDecision {
    let n = 1usize << geo.log2_cu;
    // The chroma sources at this CU, or empty slices in monochrome —
    // where every chroma step below is skipped and they are never
    // indexed, mirroring the reader's uniform `chroma_array_type != 0`
    // gates.
    let (scb, scr) = if geo.cat != 0 {
        let (sw, sh) = sub_wh(geo.cat);
        let coff = (y0 / sh) * c_stride + x0 / sw;
        (&src_cb[coff..], &src_cr[coff..])
    } else {
        (&src_cb[..0], &src_cr[..0])
    };
    let mut out = CuDecision { log2_cu: geo.log2_cu, bypass: ctx.bypass, ..CuDecision::default() };
    // PART_2Nx2N. The luma mode is chosen once, by SATD on the
    // unsplit CU-sized prediction, and both transform structures
    // reuse it — a per-structure mode search would be fairer and
    // costs double, a simplification to lift with real RD. The
    // chroma mode likewise, on the parent-size prediction.
    let cands = mpm_candidates(geo, modes, pred_mode, x0, y0);
    let soff = y0 * y_stride + x0;
    let mode = search_luma_mode(ctx, geo, &mut recon.y, scratch, x0, y0, geo.log2_cu, &src_y[soff..], y_stride, cands);
    out.luma_modes = [mode; 4];
    out.luma_syntax[0] = as_syntax(mode, cands);
    PicInfo::fill4(modes, geo.w4, x0, y0, n, n, mode);
    if geo.cat != 0 {
        let (csyn, cmode) = search_chroma_mode(
            ctx,
            geo,
            &mut recon.cb,
            &mut recon.cr,
            scratch,
            x0,
            y0,
            geo.log2_cu,
            mode,
            scb,
            scr,
            c_stride,
        );
        out.chroma_syntax = csyn;
        out.chroma_mode = cmode;
    }
    let cmode = out.chroma_mode;

    // The transform structure: one CU-sized TU, or — when the
    // writer-side knob allows — a split, judged by reconstruction
    // SSD plus the placeholder rate terms. The trials overwrite
    // each other in the plane and the decision; a trial reads
    // only samples outside the CU or samples it wrote itself, so
    // no state needs saving — whichever loses is simply
    // recomputed, the way the H.264 side puts back the I_4x4
    // coding its I_16x16 trials overwrote.
    let (ssd_u, _nz_u) = code_cu_2nx2n(
        ctx, geo, recon, scratch, x0, y0, mode, cmode, false, [false; 4], &src_y[soff..], y_stride, scb, scr, c_stride, &mut out,
    );
    if split_depth >= 1 {
        assert!(split_depth <= 2, "split_depth {split_depth} above the SPS transform depth of 2");
        let lam = 0.85f32 * ((ctx.qp - 12) as f32 / 3.0).exp2();
        let cost_u = ssd_u as f32 + lam * cu_bits(&out, geo.cat, ctx.qp, ctx.bypass);
        // The split trial, children in decode order. With the
        // deeper search on, each child is coded unsplit, re-coded
        // subdivided, and settled — losing structure recomputed —
        // BEFORE the next child codes, because the next child
        // predicts from this one's final reconstruction; a joint
        // search over the sixteen child-shape combinations is a
        // refinement real RD might want, greedy is the
        // simplification taken here. At split_depth 1 this loop
        // is code_cu_2nx2n's own split path verbatim, so the
        // decisions are bit-identical to the one-level search.
        clear_for_trial(&mut out, true);
        let mut ssd_s = 0u64;
        for i in 0..4 {
            let (mut ssd_i, mut nz_i) = code_child(ctx, geo, recon, scratch, x0, y0, i, false, mode, cmode, &src_y[soff..], y_stride, scb, scr, c_stride, &mut out);
            if split_depth >= 2 {
                let cost_a = ssd_i as f32 + lam * cu_bits(&out, geo.cat, ctx.qp, ctx.bypass);
                let (ssd_b, nz_b) = code_child(ctx, geo, recon, scratch, x0, y0, i, true, mode, cmode, &src_y[soff..], y_stride, scb, scr, c_stride, &mut out);
                let cost_b = ssd_b as f32 + lam * cu_bits(&out, geo.cat, ctx.qp, ctx.bypass);
                if cost_a <= cost_b {
                    let (sa, na) = code_child(ctx, geo, recon, scratch, x0, y0, i, false, mode, cmode, &src_y[soff..], y_stride, scb, scr, c_stride, &mut out);
                    ssd_i = sa;
                    nz_i = na;
                } else {
                    ssd_i = ssd_b;
                    nz_i = nz_b;
                }
            }
            ssd_s += ssd_i;
            let _ = nz_i;
        }
        if geo.cat != 0 {
            for comp in 0..2 {
                out.cbf_chroma[comp] = out.cbf_chroma_tu[comp].iter().any(|&f| f) || out.cbf_chroma_tu_bot[comp].iter().any(|&f| f);
            }
        }
        let cost_s = ssd_s as f32 + lam * cu_bits(&out, geo.cat, ctx.qp, ctx.bypass);
        if cost_u <= cost_s {
            let _ = code_cu_2nx2n(
                ctx, geo, recon, scratch, x0, y0, mode, cmode, false, [false; 4], &src_y[soff..], y_stride, scb, scr, c_stride, &mut out,
            );
        }
    }
    out
}

/// Every luma transform block a decision describes, as `(x, y, log2,
/// cbf)` in coding order with absolute positions given the CU's origin —
/// the one enumeration of the transform tree, for consumers that walk a
/// coded CU's blocks after the fact (the deblocking state builder
/// mirrors `transform_unit`'s per-TB bookkeeping over exactly this).
pub(crate) fn luma_tbs(d: &CuDecision, x0: usize, y0: usize) -> Vec<(usize, usize, u32, bool)> {
    let mut out = Vec::with_capacity(16);
    if d.nxn {
        for pb in 0..4 {
            out.push((x0 + (pb & 1) * 4, y0 + (pb >> 1) * 4, 2, d.cbf_luma[4 * pb]));
        }
    } else if !d.split_tu {
        out.push((x0, y0, d.log2_cu, d.cbf_luma[0]));
    } else {
        let h = 1usize << (d.log2_cu - 1);
        for (i, &deeper) in d.split_child.iter().enumerate() {
            let (tx, ty) = (x0 + (i & 1) * h, y0 + (i >> 1) * h);
            if !deeper {
                out.push((tx, ty, d.log2_cu - 1, d.cbf_luma[4 * i]));
            } else {
                let hh = h / 2;
                for j in 0..4 {
                    out.push((tx + (j & 1) * hh, ty + (j >> 1) * hh, d.log2_cu - 2, d.cbf_luma[4 * i + j]));
                }
            }
        }
    }
    out
}

/// z-scan address of the 4x4 block holding luma sample `(x, y)` within its
/// CTB — the within-CTB part of `Geometry::min_tb_addr_zs` (6.5.2),
/// computed by the same bit interleave that table is built from: bit `i`
/// of the 4x4 x-coordinate contributes `4^i`, of the y-coordinate
/// `2 * 4^i`.
pub(crate) fn z_within_ctb(log2_ctb: u32, x: usize, y: usize) -> u32 {
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
/// unavailable, is not intra-coded, or — for an above neighbour — lies
/// outside the current CTB row), expanded to three by the standard's
/// formula. The mirror of `mpm_candidates` in `hevc::ctu` (ctu.rs:620),
/// element for element.
///
/// `pred_mode` is that reader's `pred_mode[i] != 1 => INTRA_DC` gate
/// (ctu.rs:627), per 4x4 as `PicInfo::pred_mode` holds it. `None` says
/// every coded block in this picture is intra — an I slice — where the
/// gate provably cannot fire; a P slice passes the decoder's own array,
/// because there the neighbour to the left may well be an inter CU and
/// its stored mode must not be believed.
fn mpm_candidates(geo: Geo, modes: &[u8], pred_mode: Option<&[u8]>, xp: usize, yp: usize) -> [u32; 3] {
    let cand = |xn: i32, yn: i32, is_above: bool| -> u32 {
        if !decoded_before(geo, xp, yp, xn, yn) {
            return 1;
        }
        let i = (yn as usize >> 2) * geo.w4 + (xn as usize >> 2);
        if pred_mode.is_some_and(|p| p[i] != 1) {
            return 1;
        }
        if is_above && (yn as usize) < (yp >> geo.log2_cu) << geo.log2_cu {
            return 1;
        }
        modes[i] as u32
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

/// The rate half of a candidate's cost — **still a placeholder**, and
/// now a measured one.
///
/// It counts the mode-signalling bins, treating each as a bit, and
/// ignores the residual, weighted by the conventional Lagrangian so it
/// lives in the same units as a SATD.
///
/// # It was replaced by a real bit count, and that made the encoder worse
///
/// The counting encoder (`CabacEncoder::fractional_bits`) can price these
/// shapes exactly, by running `write_prev_intra_luma_pred_flag`,
/// `write_mpm_idx`, `write_rem_intra_luma_pred_mode` and
/// `write_intra_chroma_pred_mode` themselves. Doing that was tried and
/// measured over the whole encode gate, and it LOST: seventeen cells
/// regressed against six that improved, PSNR −0.039 dB on average. The
/// change is not in the tree; this note is, so the next person does not
/// spend the day rediscovering it.
///
/// What the measurement showed, which is the useful part:
///
/// - The luma numbers here are very nearly right. At QP 26 the true costs
///   are 1.92 / 2.92 / 6.09 against this table's 2 / 3 / 6, and the
///   difference that actually drives the decision — escape minus
///   most-probable — is 4.17 against 4.
/// - **The chroma numbers are not.** `ChromaExplicit` costs 3.69 bits at
///   QP 26 and 5.64 at QP 40, against 3 here: understated by 88% at high
///   QP, where these cells regressed. `ChromaDerived` is 0.53 and 0.12
///   against 1.
/// - So the correction is real, and pricing it correctly still lost.
///
/// Two explanations were tested and BOTH are wrong, which is worth more
/// than the guesses would have been:
///
/// - **Not the Lagrangian.** Doubling it made things worse again (size
///   +1.36%), so the imbalance is not a uniform scale error.
/// - **Not the residual charge beside it, though that WAS wrong.** The
///   obvious suspect was the flat three-bins-per-level residual charge
///   these costs were compared against. Sweeping its scale to 6 and 9
///   recovered nothing (13 / 15 / 17 cells regressing at 3 / 6 / 9) — but
///   a scale sweep only ever tested the scale. Its *shape* was indeed
///   wrong, and replacing it with a real count of `write_ctu_intra` was a
///   large win on its own: 35 cells better against 4 worse, size -0.90%,
///   PSNR +0.019 dB. It still did not rescue the counted mode rate, which
///   was then retried on top of it and lost again, 10 cells worse against
///   2. So the residual term was a real bug and a separate one.
///
/// Three explanations tested, three out. What remains is the limit named
/// on the counting itself: these prices are taken at the slice's INITIAL
/// probabilities, while the mode decision is a 35-way exhaustive search
/// whose own choices are exactly what the real contexts adapt to.
/// `intra_chroma_pred_mode` is the sharpest case — a decision that starts
/// preferring "derived" makes "derived" cheaper still, and a static price
/// cannot see that loop at all. It is also consistent with where the
/// damage lands: every regressed cell is a cqp40 row, and the counted
/// chroma prices swing hardest with QP (ChromaExplicit is 3.69 bits at QP
/// 26 and 5.64 at QP 40 against a flat 3 here).
///
/// So: an adapted context array carried through the decision pass is the
/// next thing to try, and the only one left standing. Until then this
/// table stays, not because it is right but because it is measurably
/// less wrong than a correct price under a static model.
fn mode_signalling_cost(qp: i32, signal: ModeSignal) -> f32 {
    let bins = match signal {
        ModeSignal::LumaMpm(0) => 2,
        ModeSignal::LumaMpm(_) => 3,
        ModeSignal::LumaEscape => 6,
        ModeSignal::ChromaDerived => 1,
        ModeSignal::ChromaExplicit => 3,
    };
    lambda_bits(qp, bins, 0)
}

/// The Lagrangian core every structure and mode cost shares — **a
/// placeholder for real RD**, and the one place the heuristic constants
/// live: the conventional `0.85 * 2^((QP - 12) / 3)` multiplier, and a
/// flat three bins per nonzero level standing in for residual bits. Bin
/// counts arrive from the callers, which count exact syntax elements;
/// what is heuristic is pricing them all at one bit and the residual at
/// a constant, which a real bit count replaces in this one function.
fn lambda_bits(qp: i32, bins: u32, nz: u32) -> f32 {
    const LEVEL_BINS: f32 = 3.0;
    let lambda = 0.85f32 * ((qp - 12) as f32 / 3.0).exp2();
    lambda * (bins as f32 + LEVEL_BINS * nz as f32)
}

/// The signalling bins one child of a split CU costs with each
/// structure, for the greedy per-child depth choice: its own split flag,
/// its cbf_luma bins (one, or four leaves), and its chroma bins as this
/// geometry codes them — per leaf where chroma subdivides, at the child
/// otherwise, doubled for 4:2:2 pairs. Exact syntax counts except that
/// the parent-gate conditioning of chroma bins is ignored (counted as if
/// always coded); the pricing itself is [`lambda_bits`]'s business.
#[allow(dead_code)] // superseded by `cu_bits`; kept for the accounting it documents
fn child_structure_bins(cat: u32, log2_cu: u32, deeper: bool) -> u32 {
    let halves = if cat == 2 { 2 } else { 1 };
    let mut bins = 1; // the child's split_transform_flag
    if !deeper {
        bins += 1; // cbf_luma
        if cat != 0 {
            bins += halves;
        }
    } else {
        bins += 4; // cbf_luma per leaf
        if cat != 0 {
            let per_leaf = log2_cu - 2 > 2 || cat == 3;
            bins += if per_leaf { 4 * halves } else { halves };
        }
    }
    bins
}

/// Forward-code and reconstruct one transform block whose *prediction is
/// already in the plane*: residual against `src`, forward transform (DST
/// for 4x4 intra luma, DCT otherwise) and quantisation from `hevc_enc`,
/// then reconstruction through the decoder's own `scale_coefficients`,
/// inverse transform and `add_residual` — the path `residual_block` in
/// `hevc::ctu` takes, so the two cannot disagree. Under `bypass` the
/// residual is carried and added raw, which is what the decoder does with
/// a `cu_transquant_bypass` block. Returns the count of nonzero levels;
/// the TU's cbf is that count being nonzero, and the count itself feeds
/// the structure decision's rate placeholder.
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
) -> u32 {
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
        return levels[..n * n].iter().filter(|&&v| v != 0).count() as u32;
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
    nz
}

/// Choose the luma mode for one prediction block by SATD over all 35
/// candidate predictions against the reconstruction plane. Trial
/// predictions are written into the plane and scored before the next
/// overwrites them; the block's own reference samples lie outside it, so
/// the trials never disturb what they read. The plane is left holding the
/// *last* trial, not the winner — coding re-predicts.
#[allow(clippy::too_many_arguments)]
fn search_luma_mode<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    x: usize,
    y: usize,
    log2: u32,
    src: &[S],
    src_stride: usize,
    cands: [u32; 3],
) -> u8 {
    let n = 1usize << log2;
    let off = plane.offset(x as isize, y as isize);
    fill_ref_avail(geo, &mut sc.avail, x, y, n, 1, 1);
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
    best.1
}

/// Predict one luma transform block with the CU's chosen mode and code
/// its residual — the per-transform-block behaviour of the decoder's
/// `transform_unit`, which predicts every TB from the reconstruction as
/// it stands when that TB is reached, so under a transform split the
/// later TBs of a CU predict from the reconstructed earlier ones. An
/// encoder that predicted the whole CU once and split only the residual
/// would desync — and both sides of a private round trip would agree
/// about it, which is why the replay test predicts per-TB too. Returns
/// the nonzero-level count.
#[allow(clippy::too_many_arguments)]
fn code_luma_tb<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    x: usize,
    y: usize,
    log2: u32,
    mode: u8,
    src: &[S],
    src_stride: usize,
    levels: &mut [i16],
) -> u32 {
    let n = 1usize << log2;
    fill_ref_avail(geo, &mut sc.avail, x, y, n, 1, 1);
    predict(plane, sc, x, y, n, mode as u32, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
    let qp = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
    code_residual(ctx, plane, x, y, log2, 0, qp, src, src_stride, levels)
}

/// The chroma subsampling factors for a `chroma_array_type`, exactly as
/// `Sps::sub_wh` derives them. Monochrome never asks.
///
/// `pub(crate)` for the inter decision, which needs the same derivation
/// and must not carry a second copy of it — the drift hazard this module
/// exists to avoid applies across modules as much as within one.
pub(crate) fn sub_wh(cat: u32) -> (usize, usize) {
    match cat {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    }
}

/// The chroma transform blocks a luma leaf TB at `(xl, yl)` of size
/// `log2` carries — `transform_unit`'s `here` placement plus its 4:2:2
/// stacked-pair loop (`yct = yc + t * nc`), reported as *luma-anchor*
/// positions in coding order with the chroma TB size. Monochrome carries
/// none; 4:2:0 one half-size square; 4:2:2 two half-size squares stacked
/// vertically, the second one `nc` luma rows down (no vertical
/// subsampling, so component rows are luma rows); 4:4:4 one square at
/// the luma size itself (`here`'s `if cat == 3 { log2 }` arm).
///
/// `pub(crate)` for the same reason as [`sub_wh`]: the inter decision and
/// the coding-tree writers place chroma TBs by this one derivation.
pub(crate) fn chroma_tbs(cat: u32, xl: usize, yl: usize, log2: u32) -> ([(usize, usize); 2], usize, u32) {
    let log2c = if cat == 3 { log2 } else { log2 - 1 };
    let nc = 1usize << log2c;
    match cat {
        0 => ([(0, 0); 2], 0, log2c),
        2 => ([(xl, yl), (xl, yl + nc)], 2, log2c),
        _ => ([(xl, yl), (0, 0)], 1, log2c),
    }
}

/// The 4:2:2 chroma intra mode mapping (Table 8-3), by `modeIdc` — the
/// decoder's own table, not a copy. It was copied here while
/// `hevc::ctu` was frozen under a concurrent merge, with a note to
/// unify when the file thawed; this is that unification. Two copies of
/// one table is the drift hazard this module exists to avoid.
use crate::hevc::ctu::MODE_422;

/// The derived chroma mode (`IntraPredModeC`, 8.4.3) for a syntax value
/// against the luma mode — the mapping `hevc::ctu::coding_unit` applies:
/// 0..=3 pick planar/26/10/1 with 34 substituted where the pick equals
/// luma, 4 is luma itself; then, for 4:2:2 only, the Table 8-3 remap of
/// the *substituted* mode, in that order exactly as the reader has it.
fn chroma_mode_for(cat: u32, syntax: u8, luma: u8) -> u8 {
    let m = match syntax {
        0 => 0,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => luma,
    };
    let m = if syntax < 4 && m == luma { 34 } else { m };
    if cat == 2 { MODE_422[m as usize] as u8 } else { m }
}

/// Choose the chroma mode over the five codable candidates by SATD
/// summed across both components and over every chroma TB the parent
/// (unsplit) shape carries — one square in 4:2:0, the stacked pair in
/// 4:2:2 — which is also what a split CU's chroma mode search uses,
/// since one mode serves all its child TUs. `(xl, yl)` are luma
/// coordinates of the parent leaf and `log2_luma` its luma size; the
/// chroma placement comes from [`chroma_tbs`]. Availability is derived
/// per TB and serves both planes, their geometry being identical.
/// Returns `(intra_chroma_pred_mode, IntraPredModeC)`.
#[allow(clippy::too_many_arguments)]
fn search_chroma_mode<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    cb: &mut Plane16<S>,
    cr: &mut Plane16<S>,
    sc: &mut IntraScratch,
    xl: usize,
    yl: usize,
    log2_luma: u32,
    luma0: u8,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
) -> (u8, u8) {
    let (tbs, ntb, log2c) = chroma_tbs(geo.cat, xl, yl, log2_luma);
    let nc = 1usize << log2c;
    let (sw, sh) = sub_wh(geo.cat);
    let mut best = (f32::MAX, 4u8);
    for syntax in 0..5u8 {
        let mode = chroma_mode_for(geo.cat, syntax, luma0) as u32;
        let mut satd = 0u32;
        for &(ax, ay) in &tbs[..ntb] {
            let (cx, cy) = (ax / sw, ay / sh);
            let soff = (cy - yl / sh) * c_stride + (cx - xl / sw);
            fill_ref_avail(geo, &mut sc.avail, ax, ay, nc, sw, sh);
            for (plane, src) in [(&mut *cb, src_cb), (&mut *cr, src_cr)] {
                // The decoder's flags for a subsampled chroma block: no
                // reference smoothing (that is 4:4:4's privilege), no
                // boundary filter (luma's alone).
                predict(plane, sc, cx, cy, nc, mode, 1, geo.cat == 3, false, ctx.bit_depth, ctx.strong_smoothing);
                let off = plane.offset(cx as isize, cy as isize);
                satd += (ctx.dist.satd)(&src[soff..], c_stride, &plane.data[off..], plane.stride, nc, nc);
            }
        }
        let signal = if syntax == 4 { ModeSignal::ChromaDerived } else { ModeSignal::ChromaExplicit };
        let cost = satd as f32 + mode_signalling_cost(ctx.qp, signal);
        if cost < best.0 {
            best = (cost, syntax);
        }
    }
    (best.1, chroma_mode_for(geo.cat, best.1, luma0))
}

/// Predict and code one chroma transform block of component `c_idx` at
/// the block whose *luma* anchor is `(xl, yl)` — availability is a luma
/// question, exactly as `intra_predict_block` poses it, and the
/// component position falls out of the subsampling. Per-TB like its luma
/// counterpart: under a split, later chroma children predict from the
/// reconstructed earlier ones, and in 4:2:2 the bottom square of a pair
/// predicts from the reconstructed top one. Returns the nonzero-level
/// count.
#[allow(clippy::too_many_arguments)]
fn code_chroma_tb<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    xl: usize,
    yl: usize,
    log2c: u32,
    c_idx: usize,
    mode: u8,
    src: &[S],
    c_stride: usize,
    levels: &mut [i16],
) -> u32 {
    let nc = 1usize << log2c;
    let (sw, sh) = sub_wh(geo.cat);
    let (cx, cy) = (xl / sw, yl / sh);
    fill_ref_avail(geo, &mut sc.avail, xl, yl, nc, sw, sh);
    predict(plane, sc, cx, cy, nc, mode as u32, c_idx, geo.cat == 3, false, ctx.bit_depth, ctx.strong_smoothing);
    // QP for chroma as the decoder derives it: the bit-depth offset comes
    // off, the `chroma_array_type`-aware mapping applies (Table 8-10 for
    // 4:2:0, a plain clamp to 51 otherwise), and it goes back on. No PPS
    // or slice offsets.
    let bd_off = 6 * (ctx.bit_depth as i32 - 8);
    let qp_c = chroma_qp(geo.cat, ctx.qp.clamp(-bd_off, 57)) + bd_off;
    code_residual(ctx, plane, cx, cy, log2c, c_idx, qp_c, src, c_stride, levels)
}

/// A fresh slate for a structure trial: the previous trial may have
/// filled a different shape, and the layout promises zeros beyond the
/// described TBs.
fn clear_for_trial(out: &mut CuDecision, split: bool) {
    out.split_tu = split;
    out.split_child = [false; 4];
    out.cbf_luma = [false; 16];
    out.cbf_chroma = [false; 2];
    out.cbf_chroma_bot = [false; 2];
    out.cbf_chroma_tu = [[false; 4]; 2];
    out.cbf_chroma_tu_bot = [[false; 4]; 2];
    out.cbf_chroma_leaf = [[false; 16]; 2];
    out.cbf_chroma_leaf_bot = [[false; 16]; 2];
    out.luma.fill(0);
    out.chroma[0].fill(0);
    out.chroma[1].fill(0);
}

/// Code one depth-1 child of a split CU: its luma as a single quarter TB
/// or — when `deeper` — as four leaf TBs at `log2_cu - 2` in z-order,
/// and its chroma in whichever shape `transform_unit` gives this
/// geometry: per luma leaf where the leaf is larger than 4x4 luma or the
/// format is 4:4:4, else once at the child's own size — the
/// `blk_idx == 3` arm, where four 4x4 luma leaves share one parent-size
/// chroma coding, exactly the shape `PART_NxN` uses (4:2:2 keeps its
/// stacked pair there, with both depth-1 bins). Levels and cbf flags go
/// into the child's positional slots on [`CuDecision`]; this child's
/// slots are cleared first, so a child can be re-coded with either
/// structure over the same state — everything it reads is outside the
/// child or written by itself, the same argument as the CU-level trials.
/// Every TB is predicted from the reconstruction as it stands, at both
/// depths. (The decoder interleaves luma and chroma; coding a child's
/// luma then its chroma is identical, because the planes are disjoint.)
/// Returns the child's reconstruction SSD over its own luma and chroma
/// regions and its nonzero-level count — the per-child structure
/// comparison's inputs.
#[allow(clippy::too_many_arguments)]
fn code_child<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    recon: &mut Frame<S>,
    sc: &mut IntraScratch,
    x0: usize,
    y0: usize,
    i: usize,
    deeper: bool,
    mode: u8,
    chroma_mode: u8,
    src_y: &[S],
    y_stride: usize,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
    out: &mut CuDecision,
) -> (u64, u32) {
    let n = 1usize << geo.log2_cu;
    let h = n / 2;
    let q = h * h;
    let (tx, ty) = (x0 + (i & 1) * h, y0 + (i >> 1) * h);
    let mut nz_total = 0u32;
    out.split_child[i] = deeper;

    // Luma: one TB, or four leaves in z-order.
    out.luma[i * q..(i + 1) * q].fill(0);
    for s in 4 * i..4 * i + 4 {
        out.cbf_luma[s] = false;
    }
    if !deeper {
        let soff = (ty - y0) * y_stride + (tx - x0);
        let nz = code_luma_tb(ctx, geo, &mut recon.y, sc, tx, ty, geo.log2_cu - 1, mode, &src_y[soff..], y_stride, &mut out.luma[i * q..(i + 1) * q]);
        out.cbf_luma[4 * i] = nz != 0;
        nz_total += nz;
    } else {
        let hh = h / 2;
        let qq = q / 4;
        for j in 0..4 {
            let (lx, ly) = (tx + (j & 1) * hh, ty + (j >> 1) * hh);
            let soff = (ly - y0) * y_stride + (lx - x0);
            let base = i * q + j * qq;
            let nz = code_luma_tb(ctx, geo, &mut recon.y, sc, lx, ly, geo.log2_cu - 2, mode, &src_y[soff..], y_stride, &mut out.luma[base..base + qq]);
            out.cbf_luma[4 * i + j] = nz != 0;
            nz_total += nz;
        }
    }

    // Chroma, in the shape this geometry dictates (see the docs above).
    if geo.cat != 0 {
        let (sw, sh) = sub_wh(geo.cat);
        let ac4 = (n / sw) * (n / sh) / 4;
        let per_leaf = deeper && (geo.log2_cu - 2 > 2 || geo.cat == 3);
        for comp in 0..2 {
            out.chroma[comp][i * ac4..(i + 1) * ac4].fill(0);
            out.cbf_chroma_tu[comp][i] = false;
            out.cbf_chroma_tu_bot[comp][i] = false;
            for s in 4 * i..4 * i + 4 {
                out.cbf_chroma_leaf[comp][s] = false;
                out.cbf_chroma_leaf_bot[comp][s] = false;
            }
        }
        if !per_leaf {
            // Chroma once at the child's size: an unsplit child, or the
            // blk_idx == 3 shape over 4x4 luma leaves.
            let (tbs, ntb, log2c) = chroma_tbs(geo.cat, tx, ty, geo.log2_cu - 1);
            let qtb = 1usize << (2 * log2c);
            for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                let src = if comp == 0 { src_cb } else { src_cr };
                for (k, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                    let soff = (ay - y0) / sh * c_stride + (ax - x0) / sw;
                    let base = i * ac4 + k * qtb;
                    let nz = code_chroma_tb(ctx, geo, plane, sc, ax, ay, log2c, 1 + comp, chroma_mode, &src[soff..], c_stride, &mut out.chroma[comp][base..base + qtb]);
                    if k == 0 {
                        out.cbf_chroma_tu[comp][i] = nz != 0;
                    } else {
                        out.cbf_chroma_tu_bot[comp][i] = nz != 0;
                    }
                    nz_total += nz;
                }
            }
        } else {
            // Chroma per luma leaf; the child's own bin becomes the
            // depth-1 gate over its leaves' bins.
            let hh = h / 2;
            let ac16 = ac4 / 4;
            for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                let src = if comp == 0 { src_cb } else { src_cr };
                for j in 0..4 {
                    let (lx, ly) = (tx + (j & 1) * hh, ty + (j >> 1) * hh);
                    let (tbs, ntb, log2c) = chroma_tbs(geo.cat, lx, ly, geo.log2_cu - 2);
                    let qtb = 1usize << (2 * log2c);
                    for (k, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                        let soff = (ay - y0) / sh * c_stride + (ax - x0) / sw;
                        let base = i * ac4 + j * ac16 + k * qtb;
                        let nz = code_chroma_tb(ctx, geo, plane, sc, ax, ay, log2c, 1 + comp, chroma_mode, &src[soff..], c_stride, &mut out.chroma[comp][base..base + qtb]);
                        if k == 0 {
                            out.cbf_chroma_leaf[comp][4 * i + j] = nz != 0;
                        } else {
                            out.cbf_chroma_leaf_bot[comp][4 * i + j] = nz != 0;
                        }
                        nz_total += nz;
                    }
                }
                out.cbf_chroma_tu[comp][i] = (4 * i..4 * i + 4)
                    .any(|s| out.cbf_chroma_leaf[comp][s] || out.cbf_chroma_leaf_bot[comp][s]);
            }
        }
    }

    // The child's own distortion, over its luma quadrant and chroma
    // region.
    let ysoff = (ty - y0) * y_stride + (tx - x0);
    let yoff = recon.y.offset(tx as isize, ty as isize);
    let mut ssd = (ctx.dist.ssd)(&src_y[ysoff..], y_stride, &recon.y.data[yoff..], recon.y.stride, h, h);
    if geo.cat != 0 {
        let (sw, sh) = sub_wh(geo.cat);
        for (plane, src) in [(&recon.cb, src_cb), (&recon.cr, src_cr)] {
            let soff = (ty - y0) / sh * c_stride + (tx - x0) / sw;
            let off = plane.offset((tx / sw) as isize, (ty / sh) as isize);
            ssd += (ctx.dist.ssd)(&src[soff..], c_stride, &plane.data[off..], plane.stride, h / sw, h / sh);
        }
    }
    (ssd, nz_total)
}

/// Code the residual of one `PART_2Nx2N` CU with the given modes and
/// transform structure: one CU-sized TU, or a split with each child's
/// own shape from `split_child` (ignored when `split` is false), every
/// TB coded in the decoder's `transform_tree` order and predicted from
/// the reconstruction as it stands. Returns the CU's reconstruction SSD
/// against the source over all components the format has, and the total
/// nonzero-level count — the structure comparison's inputs. Callable
/// repeatedly with any structure over the same state: everything a trial
/// reads is either outside the CU or written by that trial before it
/// reads it, so trials simply overwrite one another.
#[allow(clippy::too_many_arguments)]
fn code_cu_2nx2n<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    geo: Geo,
    recon: &mut Frame<S>,
    sc: &mut IntraScratch,
    x0: usize,
    y0: usize,
    mode: u8,
    chroma_mode: u8,
    split: bool,
    split_child: [bool; 4],
    src_y: &[S],
    y_stride: usize,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
    out: &mut CuDecision,
) -> (u64, u32) {
    let n = 1usize << geo.log2_cu;
    clear_for_trial(out, split);
    let mut nz_total = 0u32;

    if split {
        for (i, &deeper) in split_child.iter().enumerate() {
            let (_, nz) = code_child(ctx, geo, recon, sc, x0, y0, i, deeper, mode, chroma_mode, src_y, y_stride, src_cb, src_cr, c_stride, out);
            nz_total += nz;
        }
        if geo.cat != 0 {
            for comp in 0..2 {
                // The depth-0 bin is "any child coded" — over every square
                // of every child, both 4:2:2 halves included — which is
                // what gates the per-child bins in the reader. (A child
                // whose chroma subdivided already folded its leaves into
                // its `cbf_chroma_tu` gate.)
                out.cbf_chroma[comp] = out.cbf_chroma_tu[comp].iter().any(|&f| f) || out.cbf_chroma_tu_bot[comp].iter().any(|&f| f);
            }
        }
    } else {
        let nz = code_luma_tb(ctx, geo, &mut recon.y, sc, x0, y0, geo.log2_cu, mode, src_y, y_stride, &mut out.luma[..n * n]);
        out.cbf_luma[0] = nz != 0;
        nz_total += nz;
        if geo.cat != 0 {
            let (sw, sh) = sub_wh(geo.cat);
            let (tbs, ntb, log2c) = chroma_tbs(geo.cat, x0, y0, geo.log2_cu);
            let qtb = 1usize << (2 * log2c);
            for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                let src = if comp == 0 { src_cb } else { src_cr };
                for (k, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                    let soff = (ay - y0) / sh * c_stride + (ax - x0) / sw;
                    let nz = code_chroma_tb(ctx, geo, plane, sc, ax, ay, log2c, 1 + comp, chroma_mode, &src[soff..], c_stride, &mut out.chroma[comp][k * qtb..(k + 1) * qtb]);
                    if k == 0 {
                        out.cbf_chroma[comp] = nz != 0;
                    } else {
                        out.cbf_chroma_bot[comp] = nz != 0;
                    }
                    nz_total += nz;
                }
            }
        }
    }

    // The trial's distortion: SSD of the reconstruction against the
    // source over the whole CU, all components the format has.
    let yoff = recon.y.offset(x0 as isize, y0 as isize);
    let mut ssd = (ctx.dist.ssd)(src_y, y_stride, &recon.y.data[yoff..], recon.y.stride, n, n);
    if geo.cat != 0 {
        let (sw, sh) = sub_wh(geo.cat);
        for (plane, src) in [(&recon.cb, src_cb), (&recon.cr, src_cr)] {
            let off = plane.offset((x0 / sw) as isize, (y0 / sh) as isize);
            ssd += (ctx.dist.ssd)(src, c_stride, &plane.data[off..], plane.stride, n / sw, n / sh);
        }
    }
    (ssd, nz_total)
}

/// The rate half of the split-vs-unsplit comparison — **a placeholder for
/// real RD**, and the one function that heuristic lives in. It counts the
/// signalling bins the two structures actually differ by, read off the
/// freshly coded decision (the four child split flags, per-TU `cbf_luma`,
/// the parent-gated child chroma cbfs), plus a flat charge per nonzero
/// level standing in for residual bits — the crudest rate model that
/// still sees the real mechanism, which is that a split isolates a busy
/// quadrant so the flat ones code nothing while an unsplit transform
/// smears that quadrant's energy across the whole block's spectrum. The
/// Lagrangian is the same conventional `0.85 * 2^((QP - 12) / 3)` as
/// [`mode_signalling_cost`], here paired with SSD rather than SATD, which
/// a real RD pass would want to revisit along with everything else in
/// this function.
fn cu_bits(d: &CuDecision, cat: u32, qp: i32, bypass: bool) -> f32 {
    use crate::cabac_enc::CabacEncoder;
    use crate::hevc::ctx::Contexts;
    let mut cx = Contexts::new(0, qp);
    let mut e = CabacEncoder::counting();
    // The production serialiser, over the decision as coded: every
    // signalling bin AND every residual block, at their real widths.
    //
    // Counting `write_ctu_intra` rather than re-deriving the transform
    // tree is the whole point. The transform-block layout, the cbf
    // gating, the mode-dependent scan and the coefficient binarisation
    // are the writer's own here, not a copy of them that can drift —
    // and the residual is where nearly all the bits are, so a copy would
    // have been the largest guess in the encoder rather than the
    // smallest.
    //
    // (1, 1) as the CTU position gives the neutral neighbour context the
    // other counted costs use. Two structures of the same CU carry the
    // same mode syntax and the same neighbours, so everything shared
    // cancels in the difference that decides between them.
    crate::encode::h265::write_ctu_intra(&mut e, &mut cx, d, 1, 1, bypass, cat);
    e.fractional_bits() as f32
}

#[allow(dead_code)]
fn tu_structure_cost(cat: u32, qp: i32, d: &CuDecision, nz: u32) -> f32 {
    // split_transform_flag itself is one bin either way.
    let mut bins = 1u32;
    // How many depth-1 chroma bins one component that carried anything
    // costs: one per child, two in 4:2:2 (the stacked pair).
    let halves = if cat == 2 { 2 } else { 1 };
    if d.split_tu {
        // Each child spells its own split flag, then cbf_luma; the
        // parent chroma gate bins are followed by the child bins of every
        // component that carried anything. This arm reduces to exactly
        // the pre-depth-2 accounting when no child subdivides, which is
        // what keeps split_depth 1 decisions bit-identical to the old
        // one-level search.
        bins += 4 + 4;
        if cat != 0 {
            bins += 2;
            for comp in 0..2 {
                if d.cbf_chroma[comp] {
                    bins += 4 * halves;
                }
            }
        }
        // The depth-2 delta: a subdivided child spells three more
        // cbf_luma bins, and where its chroma follows the leaves, three
        // more chroma bin sets per component that carried anything.
        let per_leaf = d.log2_cu - 2 > 2 || cat == 3;
        for &deeper in &d.split_child {
            if deeper {
                bins += 3;
                if cat != 0 && per_leaf {
                    for comp in 0..2 {
                        if d.cbf_chroma[comp] {
                            bins += 3 * halves;
                        }
                    }
                }
            }
        }
    } else {
        // cbf_luma and the depth-0 chroma bins the format has.
        bins += 1 + if cat == 0 { 0 } else { 2 * halves };
    }
    lambda_bits(qp, bins, nz)
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
    #[allow(clippy::too_many_arguments)]
    fn code_picture(
        ctx: &IntraCtx<'_, u8>,
        w: usize,
        h: usize,
        log2_cu: u32,
        split_depth: u32,
        chroma: ChromaFormat,
        src_y: &[u8],
        src_cb: &[u8],
        src_cr: &[u8],
    ) -> (IntraPicture<u8>, Vec<CuDecision>) {
        let mut pic = IntraPicture::new_with_chroma(w, h, log2_cu, 8, chroma);
        pic.split_depth = split_depth;
        let n = 1usize << log2_cu;
        let cs = if chroma == ChromaFormat::Yuv444 { w } else { w / 2 };
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu(ctx, cx, cy, src_y, w, src_cb, src_cr, cs));
            }
        }
        (pic, decisions)
    }

    /// A luma plane built to make the structure decision take both
    /// answers within one picture: CTUs at even raster index are noise
    /// throughout (nothing for a split to isolate), odd ones are flat
    /// with a noisy bottom-right quadrant (everything for a split to
    /// isolate).
    fn mixed_source(w: usize, h: usize, n: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![128u8; w * h];
        let mut s = seed;
        for cy in 0..h / n {
            for cx in 0..w / n {
                let quadrant_only = (cy * (w / n) + cx) % 2 == 1;
                for y in 0..n {
                    for x in 0..n {
                        let in_quadrant = x >= n / 2 && y >= n / 2;
                        if !quadrant_only || in_quadrant {
                            v[(cy * n + y) * w + cx * n + x] = lcg(&mut s) as u8;
                        }
                    }
                }
            }
        }
        v
    }

    /// Three content flavours per CTU by raster index: noise (nothing to
    /// isolate), a busy half-size quadrant (one split level isolates it),
    /// and a busy quarter-size island in the far corner (only the second
    /// level isolates it) — so a depth-2 walk over this picture must
    /// produce all three structures.
    fn mixed_source3(w: usize, h: usize, n: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![128u8; w * h];
        let mut s = seed;
        for cy in 0..h / n {
            for cx in 0..w / n {
                let busy_from = match (cy * (w / n) + cx) % 3 {
                    0 => 0,
                    1 => n / 2,
                    _ => 3 * n / 4,
                };
                for y in 0..n {
                    for x in 0..n {
                        if x >= busy_from && y >= busy_from {
                            v[(cy * n + y) * w + cx * n + x] = lcg(&mut s) as u8;
                        }
                    }
                }
            }
        }
        v
    }

    /// The decoder's mode-from-syntax rule: index the MPM list, or sort
    /// it and bump the remainder past each candidate it reaches.
    fn mode_from_syntax(s: LumaModeSyntax, cands: [u32; 3]) -> u32 {
        if s.prev_flag {
            return cands[s.mpm_idx as usize];
        }
        let mut sorted = cands;
        sorted.sort_unstable();
        let mut m = s.rem as u32;
        for c in sorted {
            if m >= c {
                m += 1;
            }
        }
        m
    }

    /// Every claim of an available neighbour must point at a block that
    /// really is earlier in decode order — checked against the corner
    /// cases the z-scan rules exist to encode, on a 2x2-CTU picture of
    /// 8x8 CTUs (so the within-CTB z-order has two levels to get wrong).
    #[test]
    fn availability_follows_the_z_scan_order() {
        let geo = Geo { log2_cu: 3, wc: 2, w4: 4, width: 16, height: 16, cat: 1 };
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
            let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false, false))).unwrap();
            pps.resolve_tiles(&sps).unwrap();
            let geo_dec = std::sync::Arc::new(PicGeometry::new(&sps, &pps));
            let mut info = PicInfo::new(geo_dec);
            // One decoded slice covering the picture, every block written,
            // as the raster walk guarantees for everything before the
            // current block: the mirror encodes that guarantee, so the
            // decoder's is-it-written check (`pred_mode != 2`) must see
            // written blocks to be comparing the same question.
            info.ctb_slice_addr.fill(0);

            let (pw, ph) = (sps.width as usize, sps.height as usize);
            let geo = Geo {
                log2_cu: sps.log2_ctb_size,
                wc: sps.pic_width_in_ctbs() as usize,
                w4: pw.div_ceil(4),
                width: pw,
                height: ph,
                cat: sps.chroma_array_type(),
            };
            // Two pictures' worth of pred_mode: an I slice, where every
            // coded block is intra; and a P slice, where a checkerboard of
            // CTBs is inter. The reader's own check is `pred_mode != 2` —
            // "written", not "intra" — so the two must give the *same*
            // answers, and that identity is what makes an inter neighbour
            // a legal intra reference under `constrained_intra_pred_flag`
            // 0. If it ever stopped holding, an intra CU in a P slice
            // would predict from samples the decoder refuses.
            for pass in ["I slice", "P slice"] {
                for (i, m) in info.pred_mode.iter_mut().enumerate() {
                    *m = if pass == "I slice" { 1 } else { u8::from((i / 4 + i / 64) % 2 == 0) };
                }
                for yc in (0..ph).step_by(4) {
                    for xc in (0..pw).step_by(4) {
                        let ac = info.avail_ctx(xc as i32, yc as i32, pw as i32, ph as i32);
                        for yn in (-4..ph as i32 + 4).step_by(4) {
                            for xn in (-4..pw as i32 + 4).step_by(4) {
                                assert_eq!(
                                    decoded_before(geo, xc, yc, xn, yn),
                                    info.available_at(&ac, xn, yn),
                                    "{pass} {w}x{h} cur=({xc},{yc}) neighbour=({xn},{yn})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// An **inter** neighbour contributes `INTRA_DC` to the MPM list, not
    /// whatever mode the mode grid happens to hold at its position — the
    /// reader's `pred_mode[i] != 1 => 1` gate (`hevc::ctu::mpm_candidates`,
    /// ctu.rs:627), which only a P slice can exercise.
    ///
    /// The grid is loaded with a mode that is *not* DC precisely so the
    /// gate is observable: an encoder that skipped it would build a
    /// candidate list the decoder never builds, and every `mpm_idx` and
    /// `rem_intra_luma_pred_mode` after it would name a different mode.
    #[test]
    fn an_inter_neighbour_contributes_dc_to_the_mpm_list() {
        // 16x16 CTBs over 64x64: the block at (16, 0) starts CTU 1, so its
        // left neighbour at (15, 0) sits in CTU 0 and is decoded; its
        // above neighbour is outside the picture and is DC either way.
        let geo = Geo::new(4, 64, 64, 1);
        let mut modes = vec![1u8; geo.w4 * (64 / 4)];
        let left = (0 >> 2) * geo.w4 + (15 >> 2);
        modes[left] = 10;
        let mut pred_mode = vec![1u8; modes.len()];

        // Intra neighbour: its mode leads the list.
        assert_eq!(mpm_candidates(geo, &modes, Some(&pred_mode), 16, 0), [10, 1, 0]);
        // Inter neighbour: DC, exactly as if the mode grid had never been
        // written there — which is what `[0, 1, 26]` is, the a == b < 2
        // arm of 8.4.2.
        pred_mode[left] = 0;
        assert_eq!(mpm_candidates(geo, &modes, Some(&pred_mode), 16, 0), [0, 1, 26]);
        // And `None` is the all-intra shorthand: same answer as a grid
        // saying every neighbour is intra.
        pred_mode[left] = 1;
        assert_eq!(
            mpm_candidates(geo, &modes, None, 16, 0),
            mpm_candidates(geo, &modes, Some(&pred_mode), 16, 0),
            "None must mean what an all-intra pred_mode grid means"
        );
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
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 0, ChromaFormat::Yuv420, &y, &c, &c);
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
            // Content that pushes the structure decision both ways within
            // one picture, so the replay covers split and unsplit CUs in
            // the same walk (at log2_cu 3 the split does not exist and
            // this is simply varied content).
            let y = mixed_source(w, h, n, 0x5eed ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w / 2, h / 2, 0xcb);
            let crs = noise(w / 2, h / 2, 0xc7);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv420, &y, &cbs, &crs);
            if log2_cu > 3 {
                assert!(
                    decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu),
                    "log2_cu={log2_cu} qp={qp}: only one structure occurred, the replay is not covering both"
                );
            }
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &decisions);
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
            // The split trial runs. On uniform noise it loses — a spatial
            // residual that is nonzero everywhere gives a split nothing to
            // isolate, so the signalling decides — but that is a property
            // of this content, not of bypass: where prediction zeroes
            // whole quadrants their cbfs vanish under a split, and the
            // encode gate measured lossless splits winning big on the
            // gradient clip. Exactness must survive either choice; the
            // split shape under bypass is pinned content-independently by
            // lossless_bypass_composes_with_the_split_shape.
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv420, &y, &cbs, &crs);
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
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 26);
        }
    }

    /// A cbf flag is a statement about the levels beside it, nothing more:
    /// set exactly when its TU holds a nonzero level.
    #[test]
    fn cbf_flags_state_exactly_which_tus_hold_levels() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 30i32), (4, 32), (5, 40)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = mixed_source(w, h, n, 0xcbf ^ log2_cu as u64);
            let cbs = noise(w / 2, h / 2, 1);
            let crs = noise(w / 2, h / 2, 2);
            let (_, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv420, &y, &cbs, &crs);
            if log2_cu > 3 {
                assert!(
                    decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu),
                    "log2_cu={log2_cu} qp={qp}: only one structure occurred, the split bookkeeping is untested"
                );
            }
            let mut some_set = false;
            let mut some_clear = false;
            for d in &decisions {
                // Luma: the TU slices each shape describes, and zeros
                // beyond them (the layout's promise to the serialiser).
                let q = (n / 2) * (n / 2);
                // (slot, level range) per leaf, positional slots.
                let (tus, end): (&[(usize, usize, usize)], usize) = if d.nxn {
                    (&[(0, 0, 16), (4, 16, 32), (8, 32, 48), (12, 48, 64)], 64)
                } else if d.split_tu {
                    (&[(0, 0, q), (4, q, 2 * q), (8, 2 * q, 3 * q), (12, 3 * q, 4 * q)], 4 * q)
                } else {
                    (&[(0, 0, n * n)], n * n)
                };
                for &(slot, s, e) in tus {
                    let any = d.luma[s..e].iter().any(|&v| v != 0);
                    assert_eq!(d.cbf_luma[slot], any, "luma slot {slot} log2_cu={log2_cu} qp={qp}");
                    some_set |= any;
                    some_clear |= !any;
                }
                for slot in 0..16 {
                    if !tus.iter().any(|&(sl, _, _)| sl == slot) {
                        assert!(!d.cbf_luma[slot], "cbf in a slot the shape does not describe");
                    }
                }
                assert!(d.luma[end..].iter().all(|&v| v == 0), "levels beyond the shape's TUs");

                // Chroma: per-child flags under a split, with the depth-0
                // flag their OR; a single TU's own flag otherwise.
                for comp in 0..2 {
                    if d.split_tu {
                        let qc = (n / 4) * (n / 4);
                        for i in 0..4 {
                            let any = d.chroma[comp][i * qc..(i + 1) * qc].iter().any(|&v| v != 0);
                            assert_eq!(d.cbf_chroma_tu[comp][i], any, "chroma {comp} child {i} log2_cu={log2_cu} qp={qp}");
                        }
                        assert!(d.chroma[comp][4 * qc..].iter().all(|&v| v == 0));
                        assert_eq!(
                            d.cbf_chroma[comp],
                            d.cbf_chroma_tu[comp].iter().any(|&f| f),
                            "the depth-0 chroma cbf is not the OR of its children"
                        );
                    } else {
                        let nc = n / 2;
                        let any = d.chroma[comp][..nc * nc].iter().any(|&v| v != 0);
                        assert_eq!(d.cbf_chroma[comp], any, "chroma {comp} log2_cu={log2_cu} qp={qp}");
                        assert!(d.chroma[comp][nc * nc..].iter().all(|&v| v == 0));
                        assert_eq!(d.cbf_chroma_tu[comp], [false; 4], "child flags outside a split");
                    }
                }
            }
            // The test only means something if both flag values occurred
            // somewhere across the sweep; the mixed content at these QPs
            // produces both.
            assert!(some_set && some_clear, "log2_cu={log2_cu} qp={qp}: cbf never varied, the check is vacuous");
        }
    }

    /// Monochrome end to end: no chroma elements exist, so the decision
    /// codes luma alone (with the split trial live), the replay
    /// reconstructs it from the decisions, and the distortion bound
    /// holds. The chroma sources are empty slices — the contract is that
    /// they are never indexed.
    #[test]
    fn monochrome_codes_luma_alone_and_replays() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 24i32), (4, 30), (5, 34)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = mixed_source(w, h, n, 0x400 ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Monochrome, &y, &[], &[]);
            if log2_cu > 3 {
                assert!(
                    decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu),
                    "log2_cu={log2_cu} qp={qp}: only one structure occurred"
                );
            }
            for d in &decisions {
                assert_eq!(d.cbf_chroma, [false; 2], "monochrome coded chroma");
                assert!(d.chroma.iter().all(|c| c.iter().all(|&v| v == 0)));
            }
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Monochrome, &decisions);
            // Luma-only comparison: the chroma planes are empty.
            let (pa, pb) = (&pic.recon.y, &replayed.y);
            let (oa, ob) = (pa.origin(), pb.origin());
            let step = 1i32 << (qp / 6);
            let mut worst = 0i32;
            for yy in 0..h {
                for xx in 0..w {
                    assert_eq!(
                        pa.data[oa + yy * pa.stride + xx],
                        pb.data[ob + yy * pb.stride + xx],
                        "y ({xx},{yy}) log2_cu={log2_cu} qp={qp}"
                    );
                    let dlt = pa.data[oa + yy * pa.stride + xx] as i32 - y[yy * w + xx] as i32;
                    worst = worst.max(dlt.abs());
                }
            }
            assert!(worst <= 8 * step + 16, "log2_cu={log2_cu} qp={qp} worst={worst}");
        }
    }

    /// Monochrome bypass is exactly lossless, like every other format.
    #[test]
    fn monochrome_bypass_reconstructs_the_source_exactly() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for log2_cu in 3..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let y = noise(w, h, 0x400b + log2_cu as u64);
            let (pic, _) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Monochrome, &y, &[], &[]);
            let off = pic.recon.y.origin();
            for yy in 0..h {
                for xx in 0..w {
                    assert_eq!(pic.recon.y.data[off + yy * pic.recon.y.stride + xx], y[yy * w + xx], "({xx},{yy}) log2_cu={log2_cu}");
                }
            }
        }
    }

    /// 4:2:2 end to end: the stacked chroma pairs, the Table 8-3 mode
    /// remap and the clamped chroma QP all live inside the coding loop,
    /// so a fresh decoder-side replay landing on byte-identical planes —
    /// with the distortion bound riding along per plane — is the same
    /// statement it is for 4:2:0. Mixed content keeps both transform
    /// structures in the walk, which at 4:2:2 also exercises the
    /// split-with-pairs shape (eight chroma TBs per component per CU).
    #[test]
    fn yuv422_replays_and_stays_in_bound() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 24i32), (4, 30), (4, 43), (5, 34)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = mixed_source(w, h, n, 0x422 ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w / 2, h, 0x422cb);
            let crs = noise(w / 2, h, 0x422c7);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv422, &y, &cbs, &crs);
            if log2_cu > 3 {
                assert!(
                    decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu),
                    "log2_cu={log2_cu} qp={qp}: only one structure occurred"
                );
            }
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv422, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);

            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(2, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            for (name, plane, src, pw, ph, pqp) in [
                ("y", &pic.recon.y, &y, w, h, qp),
                ("cb", &pic.recon.cb, &cbs, w / 2, h, qp_c),
                ("cr", &pic.recon.cr, &crs, w / 2, h, qp_c),
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

    /// 4:2:2 transquant bypass is exactly lossless, pairs and all.
    #[test]
    fn yuv422_bypass_reconstructs_the_source_exactly() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for log2_cu in 3..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let y = noise(w, h, 0x422b + log2_cu as u64);
            let cbs = noise(w / 2, h, 0xb1);
            let crs = noise(w / 2, h, 0xb2);
            let (pic, _) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv422, &y, &cbs, &crs);
            for (name, plane, src, pw, ph) in [
                ("y", &pic.recon.y, &y, w, h),
                ("cb", &pic.recon.cb, &cbs, w / 2, h),
                ("cr", &pic.recon.cr, &crs, w / 2, h),
            ] {
                let off = plane.origin();
                for yy in 0..ph {
                    for xx in 0..pw {
                        assert_eq!(plane.data[off + yy * plane.stride + xx], src[yy * pw + xx], "{name} ({xx},{yy}) log2_cu={log2_cu}");
                    }
                }
            }
        }
    }

    /// The two squares of a 4:2:2 pair carry independent cbfs — the
    /// `cbf_c[c][1]` bin exists for exactly this. Flat chroma above noise
    /// makes the top square code nothing (its references substitute or
    /// reconstruct flat) while the bottom one codes, so the pair must
    /// come out (false, true), and the level layout must put every
    /// nonzero in the bottom square's slot.
    #[test]
    fn yuv422_halves_carry_independent_cbfs() {
        let kit = Kit::new();
        let ctx = kit.ctx(30, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let y = vec![128u8; w * h];
            let mut cbs = vec![128u8; w / 2 * h];
            let mut crs = vec![128u8; w / 2 * h];
            let mut s = 0x2b0770u64;
            for yy in h / 2..h {
                for xx in 0..w / 2 {
                    cbs[yy * w / 2 + xx] = lcg(&mut s) as u8;
                    crs[yy * w / 2 + xx] = lcg(&mut s) as u8;
                }
            }
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv422, &y, &cbs, &crs);
            let d = &decisions[0];
            assert!(!d.split_tu, "flat luma split anyway");
            let q = (n / 2) * (n / 2);
            for comp in 0..2 {
                assert!(!d.cbf_chroma[comp], "log2_cu={log2_cu}: the flat top square coded something");
                assert!(d.cbf_chroma_bot[comp], "log2_cu={log2_cu}: the busy bottom square coded nothing");
                assert!(d.chroma[comp][..q].iter().all(|&v| v == 0));
                assert!(d.chroma[comp][q..2 * q].iter().any(|&v| v != 0));
            }
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv422, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 30);
        }
    }

    /// The 4:2:2 cbf bookkeeping across both transform structures: every
    /// flag states exactly whether its square's slot holds a nonzero
    /// level, the depth-0 gate is the OR over all of a component's child
    /// squares under a split, and the bins that are never coded in a
    /// shape stay false.
    #[test]
    fn yuv422_cbf_flags_follow_the_pair_layout() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(3u32, 30i32), (4, 32), (5, 40)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = mixed_source(w, h, n, 0x422cbf ^ log2_cu as u64);
            let cbs = noise(w / 2, h, 11);
            let crs = noise(w / 2, h, 12);
            let (_, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv422, &y, &cbs, &crs);
            if log2_cu > 3 {
                assert!(decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu));
            }
            let mut some_chroma = false;
            for d in &decisions {
                if d.split_tu {
                    let qc = (n / 4) * (n / 4);
                    for comp in 0..2 {
                        for i in 0..4 {
                            for (t, flag) in [d.cbf_chroma_tu[comp][i], d.cbf_chroma_tu_bot[comp][i]].into_iter().enumerate() {
                                let slot = 2 * i + t;
                                let any = d.chroma[comp][slot * qc..(slot + 1) * qc].iter().any(|&v| v != 0);
                                assert_eq!(flag, any, "child {i} half {t} comp {comp}");
                                some_chroma |= any;
                            }
                        }
                        assert_eq!(
                            d.cbf_chroma[comp],
                            d.cbf_chroma_tu[comp].iter().any(|&f| f) || d.cbf_chroma_tu_bot[comp].iter().any(|&f| f),
                            "the depth-0 gate is not the OR of the child squares"
                        );
                        assert!(!d.cbf_chroma_bot[comp], "cbf_c[c][1] is never coded at a split parent");
                        assert!(d.chroma[comp][8 * qc..].iter().all(|&v| v == 0));
                    }
                } else {
                    let q = if d.nxn { 16 } else { (n / 2) * (n / 2) };
                    for comp in 0..2 {
                        for (t, flag) in [d.cbf_chroma[comp], d.cbf_chroma_bot[comp]].into_iter().enumerate() {
                            let any = d.chroma[comp][t * q..(t + 1) * q].iter().any(|&v| v != 0);
                            assert_eq!(flag, any, "half {t} comp {comp}");
                            some_chroma |= any;
                        }
                        assert_eq!(d.cbf_chroma_tu[comp], [false; 4]);
                        assert_eq!(d.cbf_chroma_tu_bot[comp], [false; 4]);
                        assert!(d.chroma[comp][2 * q..].iter().all(|&v| v == 0));
                    }
                }
            }
            assert!(some_chroma, "log2_cu={log2_cu} qp={qp}: no chroma coded, the check is vacuous");
        }
    }

    /// 4:4:4 end to end: chroma TBs at the luma size and position, the
    /// reference-smoothing filter on for chroma, the clamped chroma QP —
    /// all inside the coding loop, so the fresh decoder-side replay plus
    /// the per-plane distortion bound state the same thing they do for
    /// the subsampled formats. Both transform structures occur, which
    /// exercises the 8x8-chroma split children at a 16 CTB.
    #[test]
    fn yuv444_replays_and_stays_in_bound() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(4u32, 30i32), (4, 43), (5, 34)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (4 * n, 2 * n);
            let y = mixed_source(w, h, n, 0x444 ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w, h, 0x444cb);
            let crs = noise(w, h, 0x444c7);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv444, &y, &cbs, &crs);
            assert!(
                decisions.iter().any(|d| d.split_tu) && decisions.iter().any(|d| !d.split_tu),
                "log2_cu={log2_cu} qp={qp}: only one structure occurred"
            );
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv444, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);

            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(3, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            for (name, plane, src, pqp) in [
                ("y", &pic.recon.y, &y, qp),
                ("cb", &pic.recon.cb, &cbs, qp_c),
                ("cr", &pic.recon.cr, &crs, qp_c),
            ] {
                let step = 1i32 << (pqp / 6);
                let off = plane.origin();
                let mut worst = 0i32;
                for yy in 0..h {
                    for xx in 0..w {
                        let d = plane.data[off + yy * plane.stride + xx] as i32 - src[yy * w + xx] as i32;
                        worst = worst.max(d.abs());
                    }
                }
                assert!(worst <= 8 * step + 16, "{name} log2_cu={log2_cu} qp={pqp} worst={worst} step={step}");
            }
        }
    }

    /// 4:4:4 transquant bypass is exactly lossless.
    #[test]
    fn yuv444_bypass_reconstructs_the_source_exactly() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (2 * n, 2 * n);
            let y = noise(w, h, 0x444b + log2_cu as u64);
            let cbs = noise(w, h, 0xc1);
            let crs = noise(w, h, 0xc2);
            let (pic, _) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv444, &y, &cbs, &crs);
            for (name, plane, src) in [("y", &pic.recon.y, &y), ("cb", &pic.recon.cb, &cbs), ("cr", &pic.recon.cr, &crs)] {
                let off = plane.origin();
                for yy in 0..h {
                    for xx in 0..w {
                        assert_eq!(plane.data[off + yy * plane.stride + xx], src[yy * w + xx], "{name} ({xx},{yy}) log2_cu={log2_cu}");
                    }
                }
            }
        }
    }

    /// The construction the split exists for: three flat quadrants and a
    /// busy one. Unsplit, the big transform smears the busy quadrant's
    /// energy across the whole block's spectrum; split, three TUs code
    /// nothing and the levels concentrate in the fourth — so the decision
    /// must split, put every nonzero level in the last-in-z-order TU,
    /// and clear the other three cbfs. The busy quadrant sits bottom-right
    /// so the flat TBs precede it in z-order and predict flat exactly.
    #[test]
    fn a_busy_quadrant_splits_the_transform() {
        let kit = Kit::new();
        let ctx = kit.ctx(30, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let mut y = vec![128u8; w * h];
            let mut s = 0xb1257u64;
            for yy in n / 2..n {
                for xx in n / 2..n {
                    y[yy * w + xx] = lcg(&mut s) as u8;
                }
            }
            let c = vec![128u8; w * h / 4];
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 1, ChromaFormat::Yuv420, &y, &c, &c);
            let d = &decisions[0];
            assert!(d.split_tu, "log2_cu={log2_cu}: the busy quadrant did not force a split");
            // Positional cbf slots: the three flat children's first slots
            // clear, the busy one's set, nothing else touched.
            let mut want = [false; 16];
            want[12] = true;
            assert_eq!(d.cbf_luma, want, "log2_cu={log2_cu}");
            assert_eq!(d.cbf_chroma, [false; 2], "flat chroma coded something");
            let q = (n / 2) * (n / 2);
            assert!(d.luma[..3 * q].iter().all(|&v| v == 0));
            assert!(d.luma[3 * q..4 * q].iter().any(|&v| v != 0));
            // And the shape replays to the same picture, within the bound.
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 30);
        }
    }

    /// The other direction: content with no structure for a split to
    /// exploit. Uniform noise puts comparable levels in either shape, so
    /// the split buys nothing and its extra signalling loses; the CU must
    /// keep the single TU.
    #[test]
    fn uniform_content_keeps_the_single_tu() {
        let kit = Kit::new();
        let ctx = kit.ctx(30, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let y = noise(w, h, 0x0451 + log2_cu as u64);
            let c = vec![128u8; w * h / 4];
            let (_, decisions) = code_picture(&ctx, w, h, log2_cu, 2, ChromaFormat::Yuv420, &y, &c, &c);
            assert!(!decisions[0].split_tu, "log2_cu={log2_cu}: uniform noise split anyway");
            assert_eq!(decisions[0].split_child, [false; 4]);
        }
    }

    /// The per-TB prediction anchor, on content that can actually catch
    /// it. In every split the *decision* produces, the earlier TBs are
    /// flat (that is what makes splitting win), so their reconstruction
    /// matches almost any prediction and a stale-neighbour bug hides; and
    /// under bypass the reconstruction is the source whatever the
    /// prediction was. So: force the split shape on lossy noise through
    /// the same function the trial uses. Now TB1 must predict from TB0's
    /// *reconstructed* (quantised) samples, TB2 from TB0/TB1's, chroma
    /// children likewise — an encoder that predicted the CU in one pass,
    /// or from the source, or in the wrong order, lands on a different
    /// picture than the z-order replay and fails here.
    #[test]
    fn a_forced_split_on_noise_replays_exactly() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(4u32, 20i32), (4, 37), (5, 30)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let y = noise(w, h, 0xf0ced ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w / 2, h / 2, 5);
            let crs = noise(w / 2, h / 2, 6);
            let mut pic = IntraPicture::<u8>::new(w, h, log2_cu, 8);
            let geo = pic.geo;
            let IntraPicture { recon, modes, scratch, .. } = &mut pic;
            let cands = mpm_candidates(geo, &modes, None, 0, 0);
            // An angular mode, so the prediction really propagates
            // neighbour samples rather than averaging them away.
            let mode = 26u8;
            let mut d = CuDecision { log2_cu, ..CuDecision::default() };
            d.luma_modes = [mode; 4];
            d.luma_syntax[0] = as_syntax(mode, cands);
            d.chroma_syntax = 4;
            d.chroma_mode = chroma_mode_for(geo.cat, 4, mode);
            PicInfo::fill4(modes, geo.w4, 0, 0, n, n, mode);
            let _ = code_cu_2nx2n(&ctx, geo, recon, scratch, 0, 0, mode, d.chroma_mode, true, [false; 4], &y, w, &cbs, &crs, w / 2, &mut d);
            assert!(d.split_tu);
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &[d]);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);
        }
    }

    /// The construction the second level exists for: a CU flat but for a
    /// busy quarter-size island in the far corner. One split level puts
    /// the island inside a child that is still three-quarters flat; the
    /// second level isolates it into a single leaf, so the decision must
    /// subdivide exactly that child and put every nonzero level in its
    /// last leaf. At a 16 CTB in 4:2:0 — production geometry — the
    /// subdivided child's leaves are 4x4 luma, so this is also the
    /// assertion that the DST now carries production traffic: the leaf
    /// levels were forward-DST'd, and the replay's inverse-DST plus the
    /// distortion bound would expose a mismatched transform, as the
    /// mutation record shows.
    #[test]
    fn an_isolated_island_earns_the_second_level() {
        let kit = Kit::new();
        let ctx = kit.ctx(30, false);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let mut y = vec![128u8; w * h];
            let mut s = 0x151a4du64;
            for yy in 3 * n / 4..n {
                for xx in 3 * n / 4..n {
                    y[yy * w + xx] = lcg(&mut s) as u8;
                }
            }
            let c = vec![128u8; w * h / 4];
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 2, ChromaFormat::Yuv420, &y, &c, &c);
            let d = &decisions[0];
            assert!(d.split_tu, "log2_cu={log2_cu}: the island did not force a split at all");
            assert_eq!(d.split_child, [false, false, false, true], "log2_cu={log2_cu}: the island's child did not subdivide");
            // Every nonzero level sits in the island's leaf — the last
            // leaf of the last child — and only its cbf slot is set.
            let mut want = [false; 16];
            want[15] = true;
            assert_eq!(d.cbf_luma, want, "log2_cu={log2_cu}");
            let q = (n / 2) * (n / 2);
            assert!(d.luma[..3 * q + 3 * (q / 4)].iter().all(|&v| v == 0));
            assert!(d.luma[3 * q + 3 * (q / 4)..4 * q].iter().any(|&v| v != 0));
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 30);
        }
    }

    /// A depth-2 walk over content with all three flavours must produce
    /// all three structures and still replay byte-identically within the
    /// distortion bound — the decision-produced counterpart of the
    /// forced-shape anchor, and the non-vacuity guard for the deeper
    /// search (a sweep where no child ever subdivided would test
    /// nothing new).
    #[test]
    fn depth2_decisions_replay_across_content() {
        let kit = Kit::new();
        for &(log2_cu, qp) in &[(4u32, 26i32), (4, 40), (5, 30)] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (6 * n, 2 * n);
            let y = mixed_source3(w, h, n, 0xdee9e4 ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(w / 2, h / 2, 41);
            let crs = noise(w / 2, h / 2, 42);
            let (pic, decisions) = code_picture(&ctx, w, h, log2_cu, 2, ChromaFormat::Yuv420, &y, &cbs, &crs);
            assert!(decisions.iter().any(|d| !d.split_tu), "log2_cu={log2_cu} qp={qp}: no unsplit CU");
            assert!(
                decisions.iter().any(|d| d.split_child.iter().any(|&f| f)),
                "log2_cu={log2_cu} qp={qp}: no child ever subdivided, the deeper search is untested"
            );
            // Mixed shapes — a split CU carrying subdivided and plain
            // children side by side — are where the positional layout and
            // the per-child walk earn their keep; require them rather
            // than hope. (Pure one-level CUs need not occur at every QP:
            // a flat child beside a noisy neighbour picks up edge
            // residual, and isolating that into one leaf can genuinely
            // win — the depth-1 regression tests pin the pure shapes.)
            assert!(
                decisions.iter().any(|d| d.split_tu && d.split_child.iter().any(|&f| f) && d.split_child.iter().any(|&f| !f)),
                "log2_cu={log2_cu} qp={qp}: no mixed-shape CU occurred"
            );
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &decisions);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);
            let step = 1i32 << (qp / 6);
            let off = pic.recon.y.origin();
            let mut worst = 0i32;
            for yy in 0..h {
                for xx in 0..w {
                    let dd = pic.recon.y.data[off + yy * pic.recon.y.stride + xx] as i32 - y[yy * w + xx] as i32;
                    worst = worst.max(dd.abs());
                }
            }
            assert!(worst <= 8 * step + 16, "log2_cu={log2_cu} qp={qp} worst={worst}");
        }
    }

    /// The depth-2 anchor, forced shapes on lossy noise — for the same
    /// reason the depth-1 anchor forces: content that makes deeper
    /// splitting win naturally has flat regions that forgive prediction
    /// faults, and bypass reconstructs the source regardless. A mixed
    /// tree (first and last children subdivided, middle two not) makes
    /// the leaves of the last child predict from reconstructed earlier
    /// children at both depths, and the replay must land byte-identical
    /// with the distortion bound holding. Runs the format sweep so the
    /// three depth-2 chroma shapes all occur: parent-level chroma under
    /// 4x4 leaves at a 16 CTB in 4:2:0/4:2:2 (the blk_idx == 3 shape,
    /// pair included), per-leaf chroma at a 32 CTB, per-leaf 4x4 chroma
    /// at 4:4:4. At the 16 CTB the subdivided children's luma leaves are
    /// 4x4 — the DST, through the same path PART_NxN proved.
    #[test]
    fn a_forced_depth2_tree_replays_exactly() {
        let kit = Kit::new();
        for &(log2_cu, chroma, qp) in &[
            (4u32, ChromaFormat::Yuv420, 20i32),
            (4, ChromaFormat::Yuv420, 37),
            (4, ChromaFormat::Yuv422, 30),
            (4, ChromaFormat::Yuv444, 30),
            (5, ChromaFormat::Yuv420, 30),
            (5, ChromaFormat::Yuv422, 34),
            (4, ChromaFormat::Monochrome, 30),
        ] {
            let ctx = kit.ctx(qp, false);
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let (cw, chh) = match chroma {
                ChromaFormat::Monochrome => (0, 0),
                ChromaFormat::Yuv420 => (w / 2, h / 2),
                ChromaFormat::Yuv422 => (w / 2, h),
                ChromaFormat::Yuv444 => (w, h),
            };
            let y = noise(w, h, 0xdee2 ^ ((log2_cu as u64) << 8) ^ qp as u64);
            let cbs = noise(cw, chh, 21);
            let crs = noise(cw, chh, 22);
            let c_stride = cw.max(1);
            let mut pic = IntraPicture::<u8>::new_with_chroma(w, h, log2_cu, 8, chroma);
            let geo = pic.geo;
            let IntraPicture { recon, modes, scratch, .. } = &mut pic;
            let cands = mpm_candidates(geo, &modes, None, 0, 0);
            let mode = 26u8;
            let mut d = CuDecision { log2_cu, ..CuDecision::default() };
            d.luma_modes = [mode; 4];
            d.luma_syntax[0] = as_syntax(mode, cands);
            d.chroma_syntax = 4;
            d.chroma_mode = chroma_mode_for(geo.cat, 4, mode);
            PicInfo::fill4(modes, geo.w4, 0, 0, n, n, mode);
            let shape = [true, false, false, true];
            let _ = code_cu_2nx2n(&ctx, geo, recon, scratch, 0, 0, mode, d.chroma_mode, true, shape, &y, w, &cbs, &crs, c_stride, &mut d);
            assert_eq!(d.split_child, shape);

            // The cbf bookkeeping of the depth-2 shape, against the level
            // slots themselves.
            let q = (n / 2) * (n / 2);
            for (i, &deeper) in shape.iter().enumerate() {
                if deeper {
                    for j in 0..4 {
                        let base = i * q + j * (q / 4);
                        let any = d.luma[base..base + q / 4].iter().any(|&v| v != 0);
                        assert_eq!(d.cbf_luma[4 * i + j], any, "leaf {i}.{j}");
                    }
                } else {
                    let any = d.luma[i * q..(i + 1) * q].iter().any(|&v| v != 0);
                    assert_eq!(d.cbf_luma[4 * i], any, "child {i}");
                    for j in 1..4 {
                        assert!(!d.cbf_luma[4 * i + j]);
                    }
                }
            }
            if geo.cat != 0 {
                let per_leaf = log2_cu - 2 > 2 || geo.cat == 3;
                for comp in 0..2 {
                    for (i, &deeper) in shape.iter().enumerate() {
                        if deeper && per_leaf {
                            let gate = (4 * i..4 * i + 4).any(|s| d.cbf_chroma_leaf[comp][s] || d.cbf_chroma_leaf_bot[comp][s]);
                            assert_eq!(d.cbf_chroma_tu[comp][i], gate, "depth-1 gate is not the OR of its leaves");
                        } else {
                            assert!((4 * i..4 * i + 4).all(|s| !d.cbf_chroma_leaf[comp][s] && !d.cbf_chroma_leaf_bot[comp][s]));
                        }
                    }
                }
            }

            let replayed = replay(&ctx, w, h, log2_cu, chroma, &[d]);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, qp);
            let step = 1i32 << (qp / 6);
            let off = pic.recon.y.origin();
            let mut worst = 0i32;
            for yy in 0..h {
                for xx in 0..w {
                    let dd = pic.recon.y.data[off + yy * pic.recon.y.stride + xx] as i32 - y[yy * w + xx] as i32;
                    worst = worst.max(dd.abs());
                }
            }
            assert!(worst <= 8 * step + 16, "log2_cu={log2_cu} {chroma:?} qp={qp} worst={worst}");
        }
    }

    /// Transquant bypass composed with the full depth-2 tree: exact at
    /// every leaf, and the replay agrees.
    #[test]
    fn depth2_lossless_bypass_stays_exact() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for &(log2_cu, chroma) in &[(4u32, ChromaFormat::Yuv420), (4, ChromaFormat::Yuv422), (5, ChromaFormat::Yuv420)] {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let (cw, chh) = match chroma {
                ChromaFormat::Yuv422 => (w / 2, h),
                _ => (w / 2, h / 2),
            };
            let y = noise(w, h, 0xdee2b ^ log2_cu as u64);
            let cbs = noise(cw, chh, 31);
            let crs = noise(cw, chh, 32);
            let mut pic = IntraPicture::<u8>::new_with_chroma(w, h, log2_cu, 8, chroma);
            let geo = pic.geo;
            let IntraPicture { recon, modes, scratch, .. } = &mut pic;
            let cands = mpm_candidates(geo, &modes, None, 0, 0);
            let mode = 10u8;
            let mut d = CuDecision { log2_cu, bypass: true, ..CuDecision::default() };
            d.luma_modes = [mode; 4];
            d.luma_syntax[0] = as_syntax(mode, cands);
            d.chroma_syntax = 4;
            d.chroma_mode = chroma_mode_for(geo.cat, 4, mode);
            PicInfo::fill4(modes, geo.w4, 0, 0, n, n, mode);
            let (ssd, _) = code_cu_2nx2n(&ctx, geo, recon, scratch, 0, 0, mode, d.chroma_mode, true, [true; 4], &y, w, &cbs, &crs, cw, &mut d);
            assert_eq!(ssd, 0, "log2_cu={log2_cu} {chroma:?}: depth-2 bypass is not exact");
            let replayed = replay(&ctx, w, h, log2_cu, chroma, &[d]);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 26);
        }
    }

    /// Transquant bypass composed with the split shape: the residual is
    /// carried raw per TB and prediction runs per TB over the exact
    /// reconstruction, so a split CU is exactly lossless too. On the
    /// noise this test uses the decision would not pick the split (a
    /// residual that is nonzero everywhere gives it nothing to isolate) —
    /// though on real content bypass splits genuinely win, because
    /// prediction zeroes whole quadrants and their cbfs vanish; the
    /// encode gate measured a 39% smaller lossless gradient stream. So
    /// this codes the split shape directly through the same function the
    /// trial uses, making the shape's coverage independent of what the
    /// decision happens to choose, then replays it fresh.
    #[test]
    fn lossless_bypass_composes_with_the_split_shape() {
        let kit = Kit::new();
        let ctx = kit.ctx(26, true);
        for log2_cu in 4..=5u32 {
            let n = 1usize << log2_cu;
            let (w, h) = (n, n);
            let y = noise(w, h, 0x10551e55 + log2_cu as u64);
            let cbs = noise(w / 2, h / 2, 3);
            let crs = noise(w / 2, h / 2, 4);
            let mut pic = IntraPicture::<u8>::new(w, h, log2_cu, 8);
            let geo = pic.geo;
            let IntraPicture { recon, modes, scratch, .. } = &mut pic;
            let cands = mpm_candidates(geo, &modes, None, 0, 0);
            let mode = 1u8; // DC; any legal mode serves
            let mut d = CuDecision { log2_cu, bypass: true, ..CuDecision::default() };
            d.luma_modes = [mode; 4];
            d.luma_syntax[0] = as_syntax(mode, cands);
            d.chroma_syntax = 4;
            d.chroma_mode = chroma_mode_for(geo.cat, 4, mode);
            PicInfo::fill4(modes, geo.w4, 0, 0, n, n, mode);
            let (ssd, _) = code_cu_2nx2n(&ctx, geo, recon, scratch, 0, 0, mode, d.chroma_mode, true, [false; 4], &y, w, &cbs, &crs, w / 2, &mut d);
            assert!(d.split_tu);
            assert_eq!(ssd, 0, "log2_cu={log2_cu}: bypass with a split is not exact");
            let replayed = replay(&ctx, w, h, log2_cu, ChromaFormat::Yuv420, &[d]);
            assert_planes_equal(&pic.recon, &replayed, log2_cu, 26);
        }
    }

    /// Replay a coded picture the way a decoder would see it: modes from
    /// the stored syntax (asserted against the stored modes), predictions
    /// from the decoder's predictor over the replay's own reconstruction,
    /// residuals from the stored levels through the decoder's inverse
    /// path. Deliberately does not touch the encoder's planes or call
    /// `code_residual`.
    fn replay(ctx: &IntraCtx<'_, u8>, w: usize, h: usize, log2_cu: u32, chroma: ChromaFormat, decisions: &[CuDecision]) -> Frame<u8> {
        let mut pic = IntraPicture::<u8>::new_with_chroma(w, h, log2_cu, 8, chroma);
        let geo = pic.geo;
        let n = 1usize << log2_cu;
        let qp_y = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let bd_off = 6 * (ctx.bit_depth as i32 - 8);
        let qp_c = chroma_qp(geo.cat, ctx.qp.clamp(-bd_off, 57)) + bd_off;
        let mut di = 0;
        for cy in 0..h / n {
            for cx in 0..w / n {
                let d = &decisions[di];
                di += 1;
                let (x0, y0) = (cx * n, cy * n);
                let half = n / 2;
                let IntraPicture { recon, modes, scratch, .. } = &mut pic;
                if d.nxn {
                    // Four prediction blocks, each its own mode; derive
                    // every one from the syntax, the decoder's way, and
                    // hold it against what the encoder said it chose.
                    for pb in 0..4 {
                        let (px, py) = (x0 + (pb & 1) * 4, y0 + (pb >> 1) * 4);
                        let cands = mpm_candidates(geo, modes, None, px, py);
                        let mode = mode_from_syntax(d.luma_syntax[pb], cands);
                        assert_eq!(mode, d.luma_modes[pb] as u32, "syntax and mode disagree at ({px},{py})");
                        fill_ref_avail(geo, &mut scratch.avail, px, py, 4, 1, 1);
                        predict(&mut recon.y, scratch, px, py, 4, mode, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
                        add_tu(ctx, &mut recon.y, px, py, 2, 0, qp_y, d.bypass, &d.luma[pb * 16..pb * 16 + 16]);
                        PicInfo::fill4(modes, geo.w4, px, py, 4, 4, mode as u8);
                    }
                } else {
                    // One prediction block; one or four transform blocks.
                    // With a split, each TB is predicted afresh from the
                    // reconstruction as it stands — the decoder's per-TB
                    // behaviour, and the thing this replay anchors.
                    let cands = mpm_candidates(geo, &modes, None, x0, y0);
                    let mode = mode_from_syntax(d.luma_syntax[0], cands);
                    assert_eq!(mode, d.luma_modes[0] as u32, "syntax and mode disagree at ({x0},{y0})");
                    PicInfo::fill4(modes, geo.w4, x0, y0, n, n, mode as u8);
                    if !d.split_tu {
                        fill_ref_avail(geo, &mut scratch.avail, x0, y0, n, 1, 1);
                        predict(&mut recon.y, scratch, x0, y0, n, mode, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
                        add_tu(ctx, &mut recon.y, x0, y0, log2_cu, 0, qp_y, d.bypass, &d.luma[..n * n]);
                    } else {
                        // The tree walk, leaf by leaf in z-order, each TB
                        // predicted from the reconstruction as it stands.
                        let q = half * half;
                        for i in 0..4 {
                            let (tx, ty) = (x0 + (i & 1) * half, y0 + (i >> 1) * half);
                            if !d.split_child[i] {
                                fill_ref_avail(geo, &mut scratch.avail, tx, ty, half, 1, 1);
                                predict(&mut recon.y, scratch, tx, ty, half, mode, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
                                add_tu(ctx, &mut recon.y, tx, ty, log2_cu - 1, 0, qp_y, d.bypass, &d.luma[i * q..(i + 1) * q]);
                            } else {
                                let hh = half / 2;
                                let qq = q / 4;
                                for j in 0..4 {
                                    let (lx, ly) = (tx + (j & 1) * hh, ty + (j >> 1) * hh);
                                    let base = i * q + j * qq;
                                    fill_ref_avail(geo, &mut scratch.avail, lx, ly, hh, 1, 1);
                                    predict(&mut recon.y, scratch, lx, ly, hh, mode, 0, true, true, ctx.bit_depth, ctx.strong_smoothing);
                                    add_tu(ctx, &mut recon.y, lx, ly, log2_cu - 2, 0, qp_y, d.bypass, &d.luma[base..base + qq]);
                                }
                            }
                        }
                    }
                }
                if geo.cat == 0 {
                    // Monochrome: no chroma elements exist to replay.
                    continue;
                }
                let mode = chroma_mode_for(geo.cat, d.chroma_syntax, d.luma_modes[0]);
                assert_eq!(mode, d.chroma_mode, "chroma syntax and mode disagree");
                let (sw, sh) = sub_wh(geo.cat);
                // One chroma leaf-holder at (lx, ly, luma log2) plus its
                // level base: the parent for an unsplit CU or PART_NxN,
                // per child under a split — where a subdivided child's
                // chroma follows its luma leaves (transform_unit's
                // per-leaf placement) unless the leaves are 4x4 luma in a
                // subsampled format, in which case it stays at the child
                // (the blk_idx == 3 shape). Each TB predicted per-TB.
                let ac4 = (n / sw) * (n / sh) / 4;
                let mut holders: Vec<(usize, usize, u32, usize)> = Vec::new();
                if !d.split_tu {
                    holders.push((x0, y0, if d.nxn { 3 } else { log2_cu }, 0));
                } else {
                    for i in 0..4 {
                        let (tx, ty) = (x0 + (i & 1) * half, y0 + (i >> 1) * half);
                        let per_leaf = d.split_child[i] && (log2_cu - 2 > 2 || geo.cat == 3);
                        if !per_leaf {
                            holders.push((tx, ty, log2_cu - 1, i * ac4));
                        } else {
                            let hh = half / 2;
                            for j in 0..4 {
                                holders.push((tx + (j & 1) * hh, ty + (j >> 1) * hh, log2_cu - 2, i * ac4 + j * (ac4 / 4)));
                            }
                        }
                    }
                }
                for &(lx, ly, llog2, lbase) in &holders {
                    let (tbs, ntb, log2c) = chroma_tbs(geo.cat, lx, ly, llog2);
                    let qtb = 1usize << (2 * log2c);
                    for (comp, plane) in [&mut recon.cb, &mut recon.cr].into_iter().enumerate() {
                        for (k, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                            let base = lbase + k * qtb;
                            fill_ref_avail(geo, &mut scratch.avail, ax, ay, 1 << log2c, sw, sh);
                            predict(plane, scratch, ax / sw, ay / sh, 1 << log2c, mode as u32, 1 + comp, geo.cat == 3, false, ctx.bit_depth, ctx.strong_smoothing);
                            add_tu(ctx, plane, ax / sw, ay / sh, log2c, 1 + comp, qp_c, d.bypass, &d.chroma[comp][base..base + qtb]);
                        }
                    }
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
