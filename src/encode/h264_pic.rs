//! The picture walks: every decision an H.264 transform picture makes,
//! declared once, serialised by whoever calls.
//!
//! Two entropy coders write the same macroblocks, and the classic failure
//! is not a wrong bin — it is the two picture loops drifting apart in the
//! *decisions*: one seeding the motion search differently, one updating a
//! neighbour state the other forgot. So the loops live here, once:
//! `code_intra_picture` and `code_p_picture` own the mode decisions,
//! the neighbour bookkeeping (motion, intra modes), the reconstruction,
//! and the loop filter — and hand each coded macroblock to an `emit`
//! callback that does nothing but spell bits. CAVLC and CABAC pictures
//! are therefore the same decisions by construction, and a third
//! serialisation (B pictures, some day) is a third callback, not a third
//! loop.
//!
//! What stays out here on purpose: the `nC` counts (CAVLC's) and the
//! `WrittenMb` chain (CABAC's, in `crate::h264::cabac_mb`) are *entropy*
//! state — each writer keeps its own beside its bits.

use crate::dsp::Cpu;
use crate::dsp::distortion::DistortionDsp;
use crate::dsp::h264::H264Dsp;
use crate::dsp::h264_enc::{H264EncDsp, Quant};
use crate::encode::aq;
use crate::encode::h264_deblock::{deblock_recon, nz_mask_of};
use crate::encode::h264_intra::{IntraCtx, MbAvail, MbDecision, MbKind, code_macroblock};
use crate::encode::h264_me::{
    BDecision, BMbKind, InterDecision, InterMbKind, MbMotionState, PRef, code_macroblock_b,
    code_macroblock_p, weighted_search_plane, weighting_gain,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::frame::{BlockMotion, Frame, Mv};
use crate::h264::inter::Weighting;
use crate::h264::recon::explicit_weighting;
use crate::h264::slice::PredWeightTable;
use crate::h264::mb::{
    MbInfo, MbKind as DecKind, MbMotion, MbNeighbours, PicInfo, chroma_qp, has_residual, next_qp,
    qp_delta_range,
};
use crate::h264::sps::ScalingLists;
use crate::h264::transform::Dequant;
use crate::picture::ChromaFormat;
use crate::sample::Sample;

/// The kernels and derived tables the transform paths run on, built once
/// per encoder and shared by both entropy coders — and, beside them, the
/// one coding-tool switch that has to reach every decision walk.
///
/// Generic over the sample type, so an 8-bit encoder holds the 8-bit
/// kernel tables (the SIMD tiers) and a deeper one the 16-bit tables —
/// the decoder's own split, made once at construction.
pub struct IntraTools<S: Sample> {
    pub(crate) dsp: H264Dsp<S>,
    pub(crate) enc: H264EncDsp,
    pub(crate) dist: DistortionDsp<S>,
    pub(crate) quant: Quant,
    pub(crate) dequant: Dequant,
    /// Bits per sample, 8 to 14 — what every `IntraCtx` built from these
    /// tools carries to the predictors and the quantiser offsets.
    pub(crate) bit_depth: u32,
    /// `transform_8x8_mode_flag`, as the PPS writes it. A decision may
    /// only produce `transform_size_8x8_flag` when this is true, because
    /// otherwise the element is not in the bitstream at all. It rides
    /// here rather than through six picture-writer signatures because it
    /// is what it looks like: one constant per encoder, shared by every
    /// walk, and impossible to pass to one path and forget on another.
    pub(crate) transform_8x8: bool,
    /// Whether inter partitions smaller than 16x16 are on offer.
    pub(crate) subparts: bool,
    /// Adaptive quantisation strength (`Config::aq_strength`), 0 for off.
    /// Above 0 every picture walk decides each macroblock at the picture
    /// quantiser plus an offset from its own luma variance (`encode::aq`,
    /// over the macroblock's 16x16), and [`QpChain`] settles what a
    /// decoder will hold for it. One constant per encoder, riding here for
    /// the reason the two switches above do.
    pub(crate) aq_strength: f32,
}

impl<S: Sample> IntraTools<S> {
    /// Build for the running CPU, offering the 8x8 transform or not, at
    /// `bit_depth` bits per sample. The scaling lists are flat sixteens
    /// because the parameter sets this encoder writes carry no scaling
    /// matrices, which makes flat the lists a decoder will derive — and
    /// that is as true of the 8x8 lists as of the 4x4 ones, since the
    /// PPS declares `pic_scaling_matrix_present_flag` zero either way.
    pub fn new(transform_8x8: bool, subparts: bool, bit_depth: u32) -> Self {
        let lists = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let cpu = Cpu::detect_honouring_env();
        IntraTools {
            dsp: H264Dsp::new(cpu),
            enc: H264EncDsp::new(cpu),
            dist: DistortionDsp::new(cpu),
            quant: Quant::new(&lists),
            dequant: Dequant::new(&lists),
            bit_depth,
            transform_8x8,
            subparts,
            aq_strength: 0.0,
        }
    }

    /// The same tools with adaptive quantisation at `strength` (0 off).
    pub(crate) fn with_aq(mut self, strength: f32) -> Self {
        self.aq_strength = strength;
        self
    }
}

/// The motion state of the picture being coded, in the *decoder's* own
/// layout — kept so that the decoder's derivations can be **called**
/// rather than mirrored.
///
/// The encoder has always mirrored 8.4.1.3 instead, through
/// `MotionNeighbours`: four macroblock-level neighbours, one motion
/// each. That is expressible only while every partition is the whole
/// macroblock. The neighbours of a smaller partition are 4x4 *blocks*,
/// and for every partition after the first they are blocks of this same
/// macroblock, already derived and gated by a `done` bitmask
/// (`block_available`, src/h264/mb.rs) — which a per-macroblock summary
/// cannot represent at all. So rather than grow the mirror into a second,
/// larger thing to keep in step, the encoder keeps what the decoder
/// keeps: `MbInfo` per macroblock and `BlockMotion` per 4x4, which is
/// precisely what `MbNeighbours::derive_into` and
/// `MotionCache::gather` read.
///
/// The `Frame` is plane-less on purpose: `gather`'s progressive path
/// touches `frame.motion` and `info.mbs[].kind` and nothing else, so
/// carrying the reconstruction here would be a second copy of it for no
/// gain.
pub struct PicMotion {
    /// Per-macroblock info — neighbour availability, the intra test, and
    /// everything the loop filter reads.
    pub(crate) info: PicInfo,
    /// Per-4x4 motion per list, inside a decoder frame so that
    /// `MotionCache::gather` takes it directly.
    pub(crate) frame: Frame<u8>,
    /// What the picture's weighted prediction did ([`WeightCensus`]) — the
    /// default for every picture that carried no table.
    pub(crate) weighting: WeightCensus,
    /// The picture is a field: every macroblock committed is a field
    /// macroblock, as the decoder's `derive()` records one (`m.field` under
    /// `ctx.field_pic`).
    pub(crate) field_pic: bool,
}

/// What a P picture's explicit weighting did, for the census.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightCensus {
    /// The picture's table weights something: some entry is not the default.
    pub on: bool,
    /// Under a luma weighting, inter macroblocks whose luma SATD at the
    /// chosen vectors was lower weighted than plain — the fit's prediction
    /// holding, macroblock by macroblock.
    pub won: u64,
    /// The same, higher weighted than plain — the fit's prediction failing.
    pub lost: u64,
}

impl PicMotion {
    /// Empty state for a picture `mbs_wide` by `mbs_high` macroblocks.
    pub(crate) fn new(mbs_wide: usize, mbs_high: usize) -> Self {
        let n = mbs_wide * mbs_high;
        let mut frame = Frame::<u8>::empty();
        frame.mb_width = mbs_wide;
        frame.mb_height = mbs_high;
        frame.motion = [
            vec![BlockMotion::default(); n * 16],
            vec![BlockMotion::default(); n * 16],
        ];
        frame.mb_intra = vec![false; n];
        PicMotion { info: PicInfo::new(mbs_wide, mbs_high), frame, weighting: WeightCensus::default(), field_pic: false }
    }

    /// Commit one coded macroblock: everything a decoder stores about it
    /// that anything downstream reads — the loop filter, the next
    /// macroblock's predictions, and a later B picture's colocated look-up
    /// — and its per-4x4 motion.
    ///
    /// `mot` is in the decoder's raster layout, one entry per 4x4 per
    /// list; an intra macroblock commits [`BlockMotion::default`]
    /// throughout, as `derive()` does.
    pub(crate) fn commit(&mut self, addr: usize, info: MbInfo, mot: &MbMotion) {
        debug_assert!(info.decoded, "a committed macroblock is decoded");
        self.frame.mb_intra[addr] = info.kind.is_intra();
        self.info.mbs[addr] = MbInfo { field: info.field || self.field_pic, ..info };
        for l in 0..2 {
            self.frame.motion[l][addr * 16..addr * 16 + 16].copy_from_slice(&mot[l]);
        }
    }

}

/// What a B picture's direct prediction reads colocated motion out of —
/// the frame holding `RefPicList1[0]` — and how the current picture maps
/// onto it (8.4.1.2.1, Tables 8-6 and 8-8), read through the decoder's own
/// `colocated_in` and `colocated_motion`: so a field picture over a
/// field-coded anchor, or any other combination the standard spells out,
/// reads the block a decoder reads.
pub struct Colocated<'a> {
    /// The colocated frame's motion in the decoder's frame-row layout: a
    /// field picture's macroblock row `r` at frame row `2r + parity` with
    /// `mb_field` set, and `field_coded`, `mbaff` and `field_poc` as the
    /// decoder records them.
    pub(crate) frame: &'a Frame<u8>,
    /// The current picture's side of the mapping.
    pub(crate) map: crate::h264::recon::ColMap,
}

impl<'a> Colocated<'a> {
    /// A progressive B picture over its progressive list-1 reference: the
    /// same macroblock, the same corner block.
    pub(crate) fn progressive(col: &'a PicMotion) -> Self {
        Colocated {
            frame: &col.frame,
            map: crate::h264::recon::ColMap {
                cur_parity: crate::h264::frame::PARITY_FRAME,
                col_parity: crate::h264::frame::PARITY_FRAME,
                cur_poc: 0,
                cur_mbaff: false,
                mb_width: col.frame.mb_width,
            },
        }
    }

    /// `(mvCol, refIdxCol)` for 8x8 partition `part` of the macroblock at
    /// storage address `addr` (`field_mb` / `mb_parity`: an MBAFF field
    /// macroblock and its parity), under `direct_8x8_inference` — which
    /// every SPS this encoder writes sets.
    pub(crate) fn motion(&self, addr: usize, field_mb: bool, mb_parity: u8, part: usize) -> (Mv, i8) {
        let cb = crate::h264::recon::colocated_in(self.map, self.frame, addr, field_mb, mb_parity, true, part, 0);
        let (mv, ref_idx, _, _) = crate::h264::mb::colocated_motion(self.frame, cb.addr, cb.blk);
        (mv, ref_idx)
    }
}

/// The `MbInfo` a coded macroblock leaves — everything a decoder stores
/// about it that the loop filter and later macroblocks read.
///
/// `part_edges` is `[0, 0]`, which is not a placeholder but a statement:
/// it means "one partition covers this macroblock, so no internal edge
/// can have differing motion across it", and the filter's run-length
/// derivation depends on it meaning exactly that (see the field's own
/// documentation in src/h264/mb.rs). It is true of every shape this
/// encoder codes today and must be derived, the way `derive_motion` does
/// it, the day that changes.
///
/// `q` is what [`QpChain`] settled: the `QP_Y` a decoder holds for the
/// macroblock — the loop filter averages it with each neighbour's — its
/// chroma QP, and whether a non-zero `mb_qp_delta` was coded, which the
/// reader's `derive()` records as the next macroblock's CABAC context.
fn coded_info(kind: DecKind, nz_mask: u16, transform_8x8: bool, q: MbQp, part_edges: [u16; 2]) -> MbInfo {
    MbInfo {
        kind,
        decoded: true,
        slice: 0,
        qp: q.qp_y as i8,
        qpc: [q.qpc as i8; 2],
        qp_delta_nonzero: q.delta != 0,
        transform_8x8,
        nz_mask,
        part_edges,
        ..MbInfo::default()
    }
}

/// What [`QpChain`] settled for one macroblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MbQp {
    /// `QP_Y` as a decoder derives it for the macroblock.
    qp_y: i32,
    /// `QP_C` from it (both chroma QP offsets are zero in this encoder's
    /// PPS, so one value serves both components).
    qpc: i32,
    /// The `mb_qp_delta` to code: 0 where the macroblock carries none.
    delta: i32,
}

/// The encoder's mirror of the reader's quantiser chain (7.4.5), which
/// adaptive quantisation has to run exactly as a decoder will:
///
/// - `QP_Y,PRED` is the slice quantiser for the first macroblock of the
///   slice and the previous macroblock's `QP_Y`, in decoding order, after
///   it — skipped macroblocks included.
/// - A macroblock whose syntax carries a residual — the reader's
///   [`has_residual`]: any coded block, or `Intra_16x16` — codes
///   `mb_qp_delta`, and its `QP_Y` is the prediction plus the delta,
///   wrapped ([`next_qp`], the reader's own arithmetic). The delta is
///   chosen inside [`qp_delta_range`] at the stream's depth, going round
///   the wrap the other way when the plain difference would not fit.
/// - A macroblock without one codes no delta and **holds the prediction**,
///   whatever quantiser the encoder decided it at. That is harmless to the
///   samples — with no coefficient nothing was scaled — but not to the
///   loop filter, which averages that macroblock's `QP_Y` with its
///   neighbours', so what is committed for it is the prediction.
///
/// The decoder's skip paths (`layer.qp = qps.prev_qp`, src/h264/decoder.rs)
/// and its parsers (`qps.prev_qp = layer.qp` after the delta) are what this
/// follows. With adaptive quantisation off every macroblock wants the
/// slice quantiser, so every delta is zero and every `QP_Y` the slice's.
struct QpChain {
    /// `QP_Y,PRED` for the next macroblock.
    prev: i32,
    /// The stream's depth, which widens the delta range and the wrap.
    bit_depth: u32,
}

impl QpChain {
    fn new(slice_qp: i32, bit_depth: u32) -> Self {
        QpChain { prev: slice_qp, bit_depth }
    }

    /// The next macroblock, in decoding order, was decided at `want` and
    /// carries a residual or not: what a decoder will hold for it.
    fn settle(&mut self, want: i32, residual: bool) -> MbQp {
        let bd_off = 6 * (self.bit_depth as i32 - 8);
        let (qp_y, delta) = if residual {
            let range = qp_delta_range(self.bit_depth);
            let span = 52 + bd_off;
            let d = want - self.prev;
            let d = if d > *range.end() {
                d - span
            } else if d < *range.start() {
                d + span
            } else {
                d
            };
            debug_assert!(range.contains(&d), "no mb_qp_delta takes {} to {want} at {} bits", self.prev, self.bit_depth);
            let qp_y = next_qp(self.prev, d, self.bit_depth);
            debug_assert_eq!(qp_y, want, "the coded delta must land the quantiser the macroblock was decided at");
            (qp_y, d)
        } else {
            (self.prev, 0)
        };
        self.prev = qp_y;
        MbQp { qp_y, qpc: chroma_qp(qp_y, 0, bd_off), delta }
    }
}

/// The internal 4x4 edges that are partition boundaries, derived the way
/// `derive_motion` derives them (src/h264/recon.rs): each partition's own
/// left edge and top edge, where those are not the macroblock's.
///
/// The two halves are indexed differently and that is the decoder's
/// layout rather than a slip: `[0]` is keyed by `(x / 4) * 4 + row` — the
/// edge column major — and `[1]` by `(y / 4) * 4 + column`. Getting them
/// the same way round would filter the right edges at the wrong strength
/// on one axis only, which is the kind of thing that shows up as a faint
/// directional artefact rather than as a failure.
fn part_edges_of(parts: &[(usize, usize, usize, usize)]) -> [u16; 2] {
    let mut e = [0u16; 2];
    for &(x, y, w, h) in parts {
        if x > 0 {
            for k in y / 4..(y + h) / 4 {
                e[0] |= 1 << ((x / 4) * 4 + k);
            }
        }
        if y > 0 {
            for k in x / 4..(x + w) / 4 {
                e[1] |= 1 << ((y / 4) * 4 + k);
            }
        }
    }
    e
}

/// The decoder's name for an intra decision's kind — what the loop
/// filter's boundary-strength derivation switches on.
fn filter_kind(kind: MbKind) -> crate::h264::mb::MbKind {
    match kind {
        MbKind::I4x4 => crate::h264::mb::MbKind::I4x4,
        MbKind::I8x8 => crate::h264::mb::MbKind::I8x8,
        MbKind::I16x16 => crate::h264::mb::MbKind::I16x16,
    }
}

/// What an intra macroblock leaves along its right and bottom edges for
/// the next macroblocks' prediction-mode derivation (8.3.1.1), as
/// `(left_modes, top_modes)`.
///
/// `I_NxN` — 4x4 and 8x8 alike — leaves its own modes: `modes` is
/// raster-indexed over the sixteen 4x4 blocks, and an 8x8 macroblock has
/// already replicated each of its four modes over its quad, exactly as
/// the decoder replicates `intra_modes`. So the same four positions
/// answer for both, which is *also* what makes an 8x8 block's own
/// prediction read the right neighbour: 8.3.2.1 picks the neighbouring
/// 8x8's sub-block adjacent to the shared edge, and outside MBAFF that
/// is the very block on the edge.
///
/// Everything else leaves `Some(2)`: an available macroblock that is not
/// `I_NxN` predicts DC.
fn edge_modes(kind: MbKind, modes: &[u8; 16]) -> ([Option<u8>; 4], [Option<u8>; 4]) {
    if kind.is_nxn() {
        (
            [Some(modes[3]), Some(modes[7]), Some(modes[11]), Some(modes[15])],
            [Some(modes[12]), Some(modes[13]), Some(modes[14]), Some(modes[15])],
        )
    } else {
        ([Some(2); 4], [Some(2); 4])
    }
}

/// A source plane grown to the coded size by edge replication — the same
/// fill the PCM path uses, and for the same reason: the cropping
/// rectangle hides these samples, and repeating the edge keeps the coded
/// picture free of an artificial boundary that would cost bits.
fn pad_to<S: Sample>(src: &Plane<'_, S>, w: usize, h: usize) -> Vec<S> {
    let mut out = vec![S::default(); w * h];
    let sw = (src.width as usize).min(w);
    for y in 0..h {
        let sy = y.min(src.height as usize - 1);
        let row = &src.data[sy * src.stride..sy * src.stride + sw];
        let dst = &mut out[y * w..y * w + w];
        dst[..sw].copy_from_slice(row);
        for d in dst[sw..].iter_mut() {
            *d = row[sw - 1];
        }
    }
    out
}

/// The coding context and padded sources a picture walk works from —
/// built one way for intra and inter pictures alike, so the two cannot
/// disagree about geometry or quantisation.
struct PicCoding<'a, S: Sample> {
    /// The per-picture context both mode-decision modules take.
    ctx: IntraCtx<'a, S>,
    /// Picture size in macroblocks.
    mbs_wide: usize,
    /// See `mbs_wide`.
    mbs_high: usize,
    /// Stride of `src_y` (the coded width).
    luma_stride: usize,
    /// Stride of `src_cb` / `src_cr`; 0 for monochrome.
    chroma_stride: usize,
    /// The source planes at coded size, edge-replicated.
    src_y: Vec<S>,
    /// See `src_y` (empty for monochrome).
    src_cb: Vec<S>,
    /// See `src_y`.
    src_cr: Vec<S>,
    /// Adaptive quantisation's offset per macroblock (raster), or `None`
    /// when it is off and every macroblock takes the picture quantiser.
    offsets: Option<Vec<i32>>,
}

impl<'a, S: Sample> PicCoding<'a, S> {
    fn new(g: &Geometry, tools: &'a IntraTools<S>, qp: u8, planes: &[Plane<'_, S>]) -> Self {
        let (cw, ch) = g.chroma_mb();
        let chroma_h = ch as usize;
        debug_assert_eq!(tools.bit_depth, g.bit_depth, "one depth per encoder");
        // The quantisers, in the two forms the reader keeps them
        // (`MbDequant::for_mb`, src/h264/mb.rs): `QP_Y` as the slice
        // header carries it, and `QP'_Y = QP_Y + QpBdOffset_Y` for the
        // scaling tables. The PPS writes both chroma QP offsets as zero,
        // and the chroma map clips at `-QpBdOffset_C` below (8.5.8).
        let bd_off = 6 * (g.bit_depth as i32 - 8);
        let qpc = chroma_qp(qp as i32, 0, bd_off);
        let ctx = IntraCtx {
            dsp: &tools.dsp,
            enc: &tools.enc,
            dist: &tools.dist,
            quant: &tools.quant,
            dequant: &tools.dequant,
            qp: qp as i32,
            qpc: [qpc; 2],
            qp_prime: qp as i32 + bd_off,
            qpc_prime: [qpc + bd_off; 2],
            bit_depth: g.bit_depth,
            max: (1i32 << g.bit_depth) - 1,
            chroma_h,
            c444: g.chroma == ChromaFormat::Yuv444,
            t8x8: tools.transform_8x8,
            subparts: tools.subparts,
            field: g.field_pic,
            chroma_mv_dy: g.chroma_mv_dy,
        };
        let (mbs_wide, mbs_high) = (g.mbs_wide as usize, g.mbs_high as usize);
        let luma_stride = g.coded_width as usize;
        let src_y = pad_to(&planes[0], luma_stride, g.coded_height as usize);
        let (chroma_stride, src_cb, src_cr) = if cw != 0 {
            let stride = mbs_wide * cw as usize;
            let height = mbs_high * chroma_h;
            (
                stride,
                pad_to(&planes[1], stride, height),
                pad_to(&planes[2], stride, height),
            )
        } else {
            (0, Vec::new(), Vec::new())
        };
        // Measured over the source at coded size, edge replication and
        // all — the H.265 side takes its CTB offsets the same way — since
        // the replicated samples are what the edge macroblocks code.
        let offsets = (tools.aq_strength > 0.0).then(|| {
            aq::ctb_offsets(&src_y, luma_stride, luma_stride, g.coded_height as usize, 4, g.bit_depth, tools.aq_strength)
        });
        PicCoding {
            ctx,
            mbs_wide,
            mbs_high,
            luma_stride,
            chroma_stride,
            src_y,
            src_cb,
            src_cr,
            offsets,
        }
    }

    /// The quantiser macroblock `addr` is decided at: the picture's, plus
    /// its offset when adaptive quantisation is on, held to `0..=51` as
    /// the H.265 side holds its CTBs.
    fn mb_qp(&self, addr: usize) -> i32 {
        match &self.offsets {
            Some(o) => (self.ctx.qp + o[addr]).clamp(0, 51),
            None => self.ctx.qp,
        }
    }

    /// The picture's context at quantiser `qp`: `QP_Y`, the chroma QP the
    /// 8.5.8 map gives it, and both primed — the derivation `new` makes for
    /// the picture quantiser, so at that quantiser this *is* `self.ctx`.
    fn ctx_at(&self, qp: i32) -> IntraCtx<'a, S> {
        let bd_off = 6 * (self.ctx.bit_depth as i32 - 8);
        let qpc = chroma_qp(qp, 0, bd_off);
        IntraCtx { qp, qpc: [qpc; 2], qp_prime: qp + bd_off, qpc_prime: [qpc + bd_off; 2], ..self.ctx }
    }
}

/// One coded macroblock of a P picture, as the walk hands it to a
/// serialiser. The decision is borrowed for exactly one call — spell its
/// bits, update your entropy state, return.
pub enum PMb<'a> {
    /// `P_Skip`: the serialiser codes the skip signal and nothing else.
    /// The decision still carries the derived vector (the walk already fed
    /// it to the neighbour state and the loop filter).
    Skip(&'a InterDecision),
    /// `P_L0_16x16` with its vector, cbp and coefficients.
    Coded(&'a InterDecision),
    /// The intra fallback: this macroblock is coded as intra-in-P (the
    /// `mb_type` offset is the serialiser's).
    Intra(&'a MbDecision),
}

/// Decide, reconstruct and filter an all-intra picture, handing each
/// macroblock's decision to `emit` in raster order.
///
/// The walk owns what must be identical whoever serialises: the mode
/// decisions, the neighbouring-mode bookkeeping of 8.3.1.1 (`Some(2)` for
/// an available macroblock that is not `I_NxN` — the DC the reader
/// derives for those), the reconstruction the next macroblock predicts
/// from, and the loop filter, run after the last macroblock so what
/// leaves `rec` is the filtered picture a decoder emits.
pub(crate) fn code_intra_picture<S: Sample>(
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    mut emit: impl FnMut(usize, usize, &MbDecision),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    pm.field_pic = g.field_pic;
    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    let mut chain = QpChain::new(ctx.qp, g.bit_depth);
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let mb = MbAvail {
                left: mb_x > 0,
                top: mb_y > 0,
                top_left: mb_x > 0 && mb_y > 0,
                top_right: mb_y > 0 && mb_x + 1 < mbs_wide,
            };
            let addr = mb_y * mbs_wide + mb_x;
            let mctx = pc.ctx_at(pc.mb_qp(addr));
            let (mut dec, modes) = code_macroblock(
                &mctx,
                rec,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                mb,
                &left_modes,
                &top_modes[mb_x],
            );
            let q = chain.settle(mctx.qp, has_residual(filter_kind(dec.kind), dec.cbp_luma | (dec.cbp_chroma << 4)));
            dec.qp_delta = q.delta as i8;
            emit(mb_x, mb_y, &dec);
            pm.commit(
                addr,
                coded_info(
                    filter_kind(dec.kind),
                    nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                    dec.transform_8x8,
                    q,
                    [0; 2],
                ),
                &[[BlockMotion::default(); 16]; 2],
            );
            (left_modes, top_modes[mb_x]) = edge_modes(dec.kind, &modes);
        }
    }
    // The loop filter, last: the whole picture is reconstructed (intra
    // prediction read its unfiltered neighbours above, as a decoder's
    // does), and what leaves this function — toward the SELF check and
    // the reference list — is the filtered picture a decoder emits. The
    // per-macroblock records go back to the caller: stored beside a
    // reference picture they are what a later B picture's direct
    // derivation reads as colocated motion.
    deblock_recon(&tools.dsp, g, &mut pm, rec);
    pm
}

/// Decide, reconstruct and filter a P picture — motion search, skip, and
/// the intra fallback — handing each macroblock to `emit` in raster order.
///
/// `refp` is the reference picture's reconstruction, borders already
/// replicated ([`crate::encode::h264_me::prepare_reference`]); exactly one
/// reference is active. `weights` is the slice's `pred_weight_table` when
/// the PPS sets `weighted_pred_flag`: every prediction from `refp` then
/// takes the weighting the reader's `explicit_weighting` derives from it,
/// skips included, and the search scores against the weighted luma.
///
/// Three per-macroblock states walk the picture together, each mirroring
/// what the reader derives rather than what would be convenient:
///
/// - **Motion** ([`PicMotion`]): the picture's per-4x4 motion in the
///   decoder's own layout, committed per macroblock — a *skipped* one
///   commits the derived skip vector, because a decoder stores exactly
///   that, and an intra one commits the default throughout. Nothing here
///   summarises: the derivations read it through the decoder's own
///   `MotionCache`, so there is no neighbour bookkeeping left to get
///   subtly wrong.
/// - **Intra modes**: `Some(2)` for every available macroblock that is
///   not `I_NxN` — skip and P_16x16 included — because that is the DC the
///   reader's mode prediction derives for them (8.3.1.1).
/// - **The loop filter's inputs**, which are simply the `MbInfo` a
///   decoder would store, committed as each macroblock is coded and
///   applied after the last one, before the reconstruction becomes a
///   reference.
pub(crate) fn code_p_picture<S: Sample>(
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refp: &[Recon<S>],
    weights: Option<&PredWeightTable>,
    mut emit: impl FnMut(usize, usize, PMb<'_>),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    // The reference as the decisions see it: its planes, the weighting its
    // slice's table gives every prediction — the reader's own derivation,
    // so the encoder predicts exactly what a decoder will — and the luma
    // plane the search scores against, weighted when the luma is.
    let weighting = weights.map_or(Weighting::Default, |t| explicit_weighting(t, g.bit_depth, 0, -1, false));
    let weighted_luma = match weighting {
        Weighting::Weighted { log_wd, w, o } if (w[0][0], o[0][0]) != (1 << log_wd[0], 0) => {
            Some(weighted_search_plane(&refp[0], log_wd[0], w[0][0], o[0][0], ctx.max))
        }
        _ => None,
    };
    let pref = PRef { planes: refp, search: weighted_luma.as_ref().unwrap_or(&refp[0]), weighting };
    let mut wstats = WeightCensus {
        on: weights.is_some_and(|t| t.lists[0].iter().any(|e| e.luma_flag || e.chroma_flag)),
        ..WeightCensus::default()
    };

    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    // The picture's motion in the decoder's own layout, and the
    // per-macroblock working set its derivations read.
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    pm.field_pic = g.field_pic;
    let mut dnb = MbNeighbours::default();
    let mut st = MbMotionState::new();
    let mut chain = QpChain::new(ctx.qp, g.bit_depth);
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let addr = mb_y * mbs_wide + mb_x;
            st.start(&pm.frame, &pm.info, addr, &mut dnb);
            let mctx = pc.ctx_at(pc.mb_qp(addr));
            let ctx = &mctx;
            let mut dec = code_macroblock_p(
                ctx,
                rec,
                &pref,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                &mut st,
            );
            // The weighting's model check, at the vectors this macroblock
            // chose (a skip's is the derived one, over the whole 16x16).
            if weighted_luma.is_some() && dec.kind != InterMbKind::UseIntra {
                let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                let n = if dec.kind == InterMbKind::PSkip {
                    rects[0] = (0, 0, 16, 16);
                    1
                } else {
                    dec.rects(&mut rects)
                };
                let (plain, weighted) =
                    weighting_gain(ctx, &pref, mb_x * 16, mb_y * 16, src_y, pc.luma_stride, &rects[..n], st.motion());
                wstats.won += u64::from(weighted < plain);
                wstats.lost += u64::from(weighted > plain);
            }
            match dec.kind {
                InterMbKind::PSkip => {
                    // A skip carries no delta and holds the prediction.
                    let q = chain.settle(ctx.qp, false);
                    emit(mb_x, mb_y, PMb::Skip(&dec));
                    pm.commit(
                        addr,
                        coded_info(DecKind::PSkip, 0, false, q, [0; 2]),
                        st.motion(),
                    );
                    left_modes = [Some(2); 4];
                    top_modes[mb_x] = [Some(2); 4];
                }
                InterMbKind::P16x16
                | InterMbKind::P16x8
                | InterMbKind::P8x16
                | InterMbKind::P8x8 => {
                    let q = chain.settle(ctx.qp, has_residual(dec.kind.dec_kind(), dec.cbp_luma | (dec.cbp_chroma << 4)));
                    dec.qp_delta = q.delta as i8;
                    emit(mb_x, mb_y, PMb::Coded(&dec));
                    let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                    let n = dec.rects(&mut rects);
                    pm.commit(
                        addr,
                        coded_info(
                            dec.kind.dec_kind(),
                            nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                            dec.transform_8x8,
                            q,
                            part_edges_of(&rects[..n]),
                        ),
                        st.motion(),
                    );
                    left_modes = [Some(2); 4];
                    top_modes[mb_x] = [Some(2); 4];
                }
                InterMbKind::UseIntra => {
                    let mb = MbAvail {
                        left: mb_x > 0,
                        top: mb_y > 0,
                        top_left: mb_x > 0 && mb_y > 0,
                        top_right: mb_y > 0 && mb_x + 1 < mbs_wide,
                    };
                    let (mut idec, modes) = code_macroblock(
                        ctx,
                        rec,
                        mb_x,
                        mb_y,
                        src_y,
                        pc.luma_stride,
                        [src_cb, src_cr],
                        pc.chroma_stride,
                        mb,
                        &left_modes,
                        &top_modes[mb_x],
                    );
                    let q = chain.settle(ctx.qp, has_residual(filter_kind(idec.kind), idec.cbp_luma | (idec.cbp_chroma << 4)));
                    idec.qp_delta = q.delta as i8;
                    emit(mb_x, mb_y, PMb::Intra(&idec));
                    pm.commit(
                        addr,
                        coded_info(
                            filter_kind(idec.kind),
                            nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                            idec.transform_8x8,
                            q,
                            [0; 2],
                        ),
                        &[[BlockMotion::default(); 16]; 2],
                    );
                    (left_modes, top_modes[mb_x]) = edge_modes(idec.kind, &modes);
                }
            }
        }
    }
    // The loop filter, after the whole picture is reconstructed and before
    // the reconstruction becomes the next picture's reference — the
    // decoder's own ordering.
    deblock_recon(&tools.dsp, g, &mut pm, rec);
    pm.weighting = wstats;
    pm
}

/// One coded macroblock of a B picture, as the walk hands it to a
/// serialiser — [`PMb`]'s two-list sibling.
pub enum BMb<'a> {
    /// `B_Skip`: the serialiser codes the skip signal and nothing else.
    Skip(&'a BDecision),
    /// `B_Direct_16x16` with a residual (`mb_type` 0, no motion syntax).
    Direct(&'a BDecision),
    /// An explicitly partitioned macroblock — 16x16, 16x8, 8x16 or the
    /// `B_8x8` tree, directions per partition by [`BDecision::dir`] —
    /// with its mvds, cbp and coefficients.
    Explicit(&'a BDecision),
    /// The intra fallback, coded as intra-in-B (the `mb_type` offset of
    /// 23 is the serialiser's).
    Intra(&'a MbDecision),
}

/// Decide, reconstruct and filter a B picture, handing each macroblock to
/// `emit` in raster order — the two-list sibling of [`code_p_picture`],
/// with one addition: `col` is the *list-1 reference's* per-macroblock
/// motion record (the vec a previous walk returned), which the spatial
/// direct derivation reads as colocated motion. `refs` are the list-0
/// (past) and list-1 (future) reference planes, borders replicated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_b_picture<S: Sample>(
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refs: [&[Recon<S>]; 2],
    col: &Colocated,
    mut emit: impl FnMut(usize, usize, BMb<'_>),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);
    debug_assert_eq!(col.frame.mb_width, mbs_wide, "the colocated picture is the same width");

    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    pm.field_pic = g.field_pic;
    let mut dnb = MbNeighbours::default();
    let mut st = MbMotionState::new();
    let mut chain = QpChain::new(ctx.qp, g.bit_depth);
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let addr = mb_y * mbs_wide + mb_x;
            st.start(&pm.frame, &pm.info, addr, &mut dnb);
            let mctx = pc.ctx_at(pc.mb_qp(addr));
            let ctx = &mctx;
            let mut dec = code_macroblock_b(
                ctx,
                rec,
                refs,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                &mut st,
                col,
                addr,
            );
            if dec.kind == BMbKind::UseIntra {
                let mb = MbAvail {
                    left: mb_x > 0,
                    top: mb_y > 0,
                    top_left: mb_x > 0 && mb_y > 0,
                    top_right: mb_y > 0 && mb_x + 1 < mbs_wide,
                };
                let (mut idec, modes) = code_macroblock(
                    ctx,
                    rec,
                    mb_x,
                    mb_y,
                    src_y,
                    pc.luma_stride,
                    [src_cb, src_cr],
                    pc.chroma_stride,
                    mb,
                    &left_modes,
                    &top_modes[mb_x],
                );
                let q = chain.settle(ctx.qp, has_residual(filter_kind(idec.kind), idec.cbp_luma | (idec.cbp_chroma << 4)));
                idec.qp_delta = q.delta as i8;
                emit(mb_x, mb_y, BMb::Intra(&idec));
                pm.commit(
                    addr,
                    coded_info(
                        filter_kind(idec.kind),
                        nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                        idec.transform_8x8,
                        q,
                        [0; 2],
                    ),
                    &[[BlockMotion::default(); 16]; 2],
                );
                (left_modes, top_modes[mb_x]) = edge_modes(idec.kind, &modes);
                continue;
            }
            // B_Skip carries no delta whatever its record holds; the other
            // shapes carry one exactly when the reader's rule says so.
            let residual = dec.kind != BMbKind::BSkip && has_residual(dec.kind.dec_kind(), dec.cbp_luma | (dec.cbp_chroma << 4));
            let q = chain.settle(ctx.qp, residual);
            dec.qp_delta = q.delta as i8;
            emit(
                mb_x,
                mb_y,
                match dec.kind {
                    BMbKind::BSkip => BMb::Skip(&dec),
                    BMbKind::BDirect16 => BMb::Direct(&dec),
                    BMbKind::B16 | BMbKind::B16x8 | BMbKind::B8x16 | BMbKind::B8x8 => {
                        BMb::Explicit(&dec)
                    }
                    BMbKind::UseIntra => unreachable!(),
                },
            );
            // The rectangles the decoder's motion jobs cover, whose left
            // and top edges are where the loop filter compares motion.
            // A direct macroblock is *four 8x8 partitions*, not one:
            // `direct_partitions` pushes a job per 8x8 under
            // `direct_8x8_inference` (src/h264/recon.rs), so a decoder
            // records the 8x8 cross as partition edges — and
            // `BDecision::rects` says so. Passing [0, 0] for direct was
            // harmless while all four had the same vector, and became a
            // real desync the moment colZeroFlag started varying per
            // 8x8. It cost six cells of `--subparts --t8x8 --bframes 2`.
            let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
            let n = dec.rects(&mut rects);
            // Which 8x8s of a `B_8x8` are direct: what the reader's
            // `is_direct_block` asks of a neighbouring macroblock for
            // its ref_idx and mvd contexts (src/h264/cabac_mb.rs), and
            // what `derive` stores (src/h264/recon.rs).
            let sub_direct = if dec.kind == BMbKind::B8x8 {
                (0..4).map(|p| (dec.is_direct_part(p) as u8) << p).sum()
            } else {
                0
            };
            pm.commit(
                addr,
                MbInfo {
                    sub_direct,
                    ..coded_info(
                        dec.kind.dec_kind(),
                        nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                        dec.transform_8x8,
                        q,
                        part_edges_of(&rects[..n]),
                    )
                },
                st.motion(),
            );
            left_modes = [Some(2); 4];
            top_modes[mb_x] = [Some(2); 4];
        }
    }
    deblock_recon(&tools.dsp, g, &mut pm, rec);
    pm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain lands every quantiser the encoder can want from every
    /// prediction, at 8, 10 and 14 bits, with a delta inside the reader's
    /// range that the reader's own `next_qp` turns back into that
    /// quantiser — including the differences that only fit by going round
    /// the wrap — and a macroblock without a residual codes nothing and
    /// holds the prediction.
    #[test]
    fn the_quantiser_chain_lands_where_the_reader_derives() {
        for bit_depth in [8u32, 10, 14] {
            for prev in 0..=51 {
                for want in 0..=51 {
                    let mut c = QpChain::new(prev, bit_depth);
                    let q = c.settle(want, true);
                    assert!(qp_delta_range(bit_depth).contains(&q.delta), "{prev} -> {want} at {bit_depth} bits: delta {}", q.delta);
                    assert_eq!(next_qp(prev, q.delta, bit_depth), want, "{prev} -> {want} at {bit_depth} bits");
                    assert_eq!(q.qp_y, want);
                    assert_eq!(q.qpc, chroma_qp(want, 0, 6 * (bit_depth as i32 - 8)));
                    let held = c.settle((want + 7) % 52, false);
                    assert_eq!((held.qp_y, held.delta), (want, 0), "a residual-free macroblock holds the prediction");
                }
            }
        }
    }
}
