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
use crate::encode::h264_intra::{IntraCtx, MbAvail, MbDecision, MbKind, code_macroblock, code_macroblock_modes8};
use crate::encode::h264_me::{
    BDecision, BMbKind, BRefs, InterDecision, InterMbKind, MbMotionState, PRef, b_weightings, code_macroblock_b,
    code_macroblock_p, weighted_search_plane, weighting_gain, weighting_gain_b,
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
    /// An MBAFF picture's macroblock pairs, `[frame, field]` — what the
    /// pair decision chose, for the census.
    pub(crate) pairs: [u64; 2],
}

/// What a P or B picture's explicit weighting did, for the census.
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
    /// The picture's fitted table was priced against a table of defaults
    /// (the picture coded both ways).
    pub priced: bool,
    /// The picture's fitted table lost that check, and the picture was
    /// kept coded under the defaults (`on` is then false: the kept table
    /// weights nothing).
    pub rd_default: bool,
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
        PicMotion { info: PicInfo::new(mbs_wide, mbs_high), frame, weighting: WeightCensus::default(), field_pic: false, pairs: [0; 2] }
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
        // An MBAFF frame's per-macroblock field flags, where the deblocking
        // filter and a later picture's colocated derivation read them.
        if let Some(f) = self.frame.mb_field.get_mut(addr) {
            *f = info.field;
        }
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
///
/// `weights` is the slice's `pred_weight_table` when the PPS sets
/// `weighted_bipred_idc` 1: every prediction then takes the weighting the
/// reader's `explicit_weighting` derives from it for the reference pair it
/// uses — one list's entry for a one-list prediction, both for a
/// bi-predicted one, direct and skip at their derived pair — and each
/// weighted list's search scores against its weighted luma, as a P
/// picture's does. A table whose every entry is the default predicts
/// exactly the samples default weighting does (`w = 1 << logWD`, `o = 0`
/// reduce 8.4.2.3.2's formulas to 8.4.2.3.1's, one list and two), so it
/// leaves the walk on default weighting and its average kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_b_picture<S: Sample>(
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refs: [&[Recon<S>]; 2],
    col: &Colocated,
    weights: Option<&PredWeightTable>,
    mut emit: impl FnMut(usize, usize, BMb<'_>),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);
    debug_assert_eq!(col.frame.mb_width, mbs_wide, "the colocated picture is the same width");

    // The references as the decisions see them: each kind of prediction's
    // weighting, the reader's own derivation for its reference pair, and
    // each list's search luma, weighted when that list's luma is.
    let on = weights.is_some_and(|t| t.lists.iter().flatten().any(|e| e.luma_flag || e.chroma_flag));
    let weighting = b_weightings(weights.filter(|_| on), g.bit_depth);
    let weighted_luma: [Option<Recon<S>>; 2] = [0usize, 1].map(|l| match weighting[l] {
        Weighting::Weighted { log_wd, w, o } if (w[0][l], o[0][l]) != (1 << log_wd[0], 0) => {
            Some(weighted_search_plane(&refs[l][0], log_wd[0], w[0][l], o[0][l], ctx.max))
        }
        _ => None,
    });
    let brefs = BRefs {
        planes: refs,
        search: [0usize, 1].map(|l| weighted_luma[l].as_ref().unwrap_or(&refs[l][0])),
        weighting,
    };
    let luma_weighted = weighted_luma.iter().any(Option::is_some);
    let mut wstats = WeightCensus { on, ..WeightCensus::default() };

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
                &brefs,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                &mut st,
                col,
                addr,
                false,
                col.map.cur_parity,
            );
            // The weighting's model check, at the vectors and lists this
            // macroblock chose (direct and skip at their derived ones).
            if luma_weighted && dec.kind != BMbKind::UseIntra {
                let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                let n = dec.rects(&mut rects);
                let (plain, weighted) =
                    weighting_gain_b(ctx, &brefs, mb_x * 16, mb_y * 16, src_y, pc.luma_stride, &rects[..n], st.motion());
                wstats.won += u64::from(weighted < plain);
                wstats.lost += u64::from(weighted > plain);
            }
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
    pm.weighting = wstats;
    pm
}

// ---------------------------------------------------------------------------
// MBAFF pictures
// ---------------------------------------------------------------------------

/// One macroblock of an MBAFF pair as the walk decided it — owned, because
/// a pair is decided (both ways, then once more for the winner) before
/// either of its macroblocks is written.
#[derive(Clone)]
// The variants are all a macroblock's coefficients (one to two kilobytes);
// they differ in size by the B decision's second list, and a walk holds two
// pairs of them at a time, so boxing buys nothing.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PairMb {
    /// A macroblock of an I slice.
    Intra(MbDecision),
    /// A P slice's skipped or coded inter macroblock.
    P(InterDecision),
    /// A P slice's intra macroblock.
    PIntra(MbDecision),
    /// A B slice's skipped, direct or explicit macroblock.
    B(BDecision),
    /// A B slice's intra macroblock.
    BIntra(MbDecision),
}

impl PairMb {
    /// `P_Skip` or `B_Skip`: nothing is coded but the skip.
    pub(crate) fn is_skip(&self) -> bool {
        match self {
            PairMb::P(d) => d.kind == InterMbKind::PSkip,
            PairMb::B(d) => d.kind == BMbKind::BSkip,
            _ => false,
        }
    }
}

/// An MBAFF macroblock pair, decided: the storage address of its top
/// macroblock, whether it is a field pair, and its two macroblocks, top
/// first.
pub(crate) struct CodedPair {
    /// Storage address (frame row raster) of the top macroblock.
    pub(crate) top: usize,
    /// `mb_field_decoding_flag`.
    pub(crate) field: bool,
    /// Top, then bottom.
    pub(crate) mbs: [PairMb; 2],
}

/// The entropy coder an MBAFF walk writes through. The walk codes every
/// pair both ways and prices each candidate with what it would cost to
/// write next — which only the entropy coder can say — before writing the
/// winner.
pub(crate) trait PairWriter {
    /// The bits `pair` would take written now, leaving the writer as it
    /// is. `pm` holds the pair committed, as the decoder's picture state
    /// holds a pair while its syntax is read.
    fn trial_bits(&self, pair: &CodedPair, pm: &PicMotion) -> u64;
    /// Write `pair`.
    fn write_pair(&mut self, pair: &CodedPair, pm: &PicMotion);
}

/// What an MBAFF picture predicts from: each reference as the frame (for
/// frame macroblocks) and as its two fields (for field macroblocks, whose
/// reference index 0 names the field of their own parity).
pub(crate) enum MbaffRefs<'a, S: Sample> {
    /// An I picture.
    Intra,
    /// A P picture's list-0 reference.
    P {
        /// The reference frame.
        frame: &'a [Recon<S>],
        /// Its top and bottom fields.
        fields: [&'a [Recon<S>]; 2],
    },
    /// A B picture's references and colocated view.
    B {
        /// List 0's and list 1's frames.
        frame: [&'a [Recon<S>]; 2],
        /// `[list][parity]`.
        fields: [[&'a [Recon<S>]; 2]; 2],
        /// The colocated frame, mapped for an MBAFF current picture.
        col: Colocated<'a>,
    },
}

/// The neighbouring intra modes an MBAFF macroblock predicts from, as
/// `(left, top, left for 8x8 blocks)`: for each 4x4 row the mode of the
/// block left of it, for each column the block above — through the
/// decoder's Table 6-4 block neighbours — with `None` where no macroblock
/// is there and DC (2) for a neighbour that is not `I_NxN`. The third array
/// is 8.3.2.1's for an 8x8 block (rows 0 and 2): an `I_4x4` left neighbour
/// gives the top-right sub-block of the neighbouring 8x8, or its
/// bottom-right (`n` = 3) for block 2 of a frame macroblock beside a field
/// one — the rule `predicted_intra_mode` applies (src/h264/cavlc.rs), which
/// the test below holds this to. The above neighbour is always a bottom row,
/// where 8.3.2.1's `n` = 2 is the block on the edge.
fn mbaff_edge_modes(nb: &MbNeighbours, info: &PicInfo) -> (EdgeModes, EdgeModes, EdgeModes) {
    let mode = |a: usize, blk: usize| -> u8 {
        if matches!(info.mbs[a].kind, DecKind::I4x4 | DecKind::I8x8) { info.intra_modes[a * 16 + blk] } else { 2 }
    };
    let left: [Option<u8>; 4] = std::array::from_fn(|r| nb.block(-1, r as i32).map(|(a, blk)| mode(a, blk)));
    let top: [Option<u8>; 4] = std::array::from_fn(|c| nb.block(c as i32, -1).map(|(a, blk)| mode(a, blk)));
    let left8: [Option<u8>; 4] = std::array::from_fn(|r| {
        nb.block(-1, r as i32).map(|(a, blk)| {
            if info.mbs[a].kind != DecKind::I4x4 {
                return mode(a, blk);
            }
            let (bx8, by8) = ((blk % 4) / 2 * 2, (blk / 4) / 2 * 2);
            let n3 = nb.mbaff && !nb.cur_field && info.mbs[a].field && r == 2;
            let (sx, sy) = if n3 { (bx8 + 1, by8 + 1) } else { (bx8 + 1, by8) };
            info.intra_modes[a * 16 + sy * 4 + sx]
        })
    });
    (left, top, left8)
}

/// One neighbouring intra mode per 4x4 row or column of a macroblock edge,
/// `None` where no macroblock is there.
type EdgeModes = [Option<u8>; 4];

/// Commit an intra macroblock of an MBAFF pair: its quantiser through the
/// chain, its record (a field macroblock's when `field`), and its modes
/// where a neighbour's prediction reads them.
fn commit_intra(pm: &mut PicMotion, chain: &mut QpChain, addr: usize, field: bool, qp: i32, mut dec: MbDecision, modes: &[u8; 16]) -> MbDecision {
    let q = chain.settle(qp, has_residual(filter_kind(dec.kind), dec.cbp_luma | (dec.cbp_chroma << 4)));
    dec.qp_delta = q.delta as i8;
    pm.commit(
        addr,
        MbInfo {
            field,
            ..coded_info(filter_kind(dec.kind), nz_mask_of(&dec.nz_luma, dec.transform_8x8), dec.transform_8x8, q, [0; 2])
        },
        &[[BlockMotion::default(); 16]; 2],
    );
    let nxn = dec.kind.is_nxn();
    for (m, &c) in pm.info.intra_modes[addr * 16..addr * 16 + 16].iter_mut().zip(modes) {
        *m = if nxn { c } else { 2 };
    }
    dec
}

/// One MBAFF pair's state before it was coded, to undo a candidate.
struct PairSnap<S: Sample> {
    planes: Vec<Vec<S>>,
    info: [MbInfo; 2],
    motion: [[BlockMotion; 32]; 2],
    intra: [bool; 2],
    field: [bool; 2],
    modes: [u8; 32],
    prev_qp: i32,
}

/// The working state of an MBAFF picture walk.
struct MbaffWalk<'a, 'r, S: Sample> {
    g: &'a Geometry,
    pc: PicCoding<'a, S>,
    /// The padded source's fields: `[parity][plane]`, row `r` of field `p`
    /// being row `2r + p` of the frame.
    fsrc: [[Vec<S>; 3]; 2],
    /// The frame view of the reconstruction.
    rec: &'r mut [Recon<S>],
    /// The same reconstruction as its two fields, kept equal to `rec`
    /// macroblock by macroblock: a frame macroblock predicts intra from the
    /// frame, a field macroblock from its own field, and every neighbouring
    /// sample Table 6-4 names lies in the view of the macroblock reading it.
    views: [Vec<Recon<S>>; 2],
    pm: PicMotion,
    chain: QpChain,
    st: MbMotionState,
    nb: MbNeighbours,
    refs: MbaffRefs<'a, S>,
    /// The multiplier bits are priced at against a squared error.
    lam: f64,
}

impl<S: Sample> MbaffWalk<'_, '_, S> {
    /// A macroblock's size in plane `p`, in samples.
    fn mb_dims(&self, p: usize) -> (usize, usize) {
        if p == 0 {
            (16, 16)
        } else {
            let (w, h) = self.g.chroma_mb();
            (w as usize, h as usize)
        }
    }

    /// Copy frame macroblock row `fr`'s samples at column `x` into the
    /// field views.
    fn sync_frame_mb(&mut self, x: usize, fr: usize) {
        for p in 0..self.rec.len() {
            let (bw, bh) = self.mb_dims(p);
            for i in 0..bh {
                let y = fr * bh + i;
                let s = self.rec[p].offset((x * bw) as isize, y as isize);
                let d = self.views[y % 2][p].offset((x * bw) as isize, (y / 2) as isize);
                self.views[y % 2][p].data[d..d + bw].copy_from_slice(&self.rec[p].data[s..s + bw]);
            }
        }
    }

    /// Copy the field macroblock of parity `par` at pair `(x, pr)` from its
    /// field view into the frame.
    fn sync_field_mb(&mut self, x: usize, pr: usize, par: usize) {
        for p in 0..self.rec.len() {
            let (bw, bh) = self.mb_dims(p);
            for i in 0..bh {
                let fy = pr * bh + i;
                let s = self.views[par][p].offset((x * bw) as isize, fy as isize);
                let d = self.rec[p].offset((x * bw) as isize, (2 * fy + par) as isize);
                self.rec[p].data[d..d + bw].copy_from_slice(&self.views[par][p].data[s..s + bw]);
            }
        }
    }

    fn snapshot(&self, x: usize, pr: usize) -> PairSnap<S> {
        let mbw = self.pc.mbs_wide;
        let top = 2 * pr * mbw + x;
        let bot = top + mbw;
        let planes = (0..self.rec.len())
            .map(|p| {
                let (bw, bh) = self.mb_dims(p);
                let plane = &self.rec[p];
                let mut v = Vec::with_capacity(2 * bw * bh);
                for y in 0..2 * bh {
                    let o = plane.offset((x * bw) as isize, (pr * 2 * bh + y) as isize);
                    v.extend_from_slice(&plane.data[o..o + bw]);
                }
                v
            })
            .collect();
        let mut motion = [[BlockMotion::default(); 32]; 2];
        let mut modes = [0u8; 32];
        for (k, a) in [top, bot].into_iter().enumerate() {
            for (l, m) in motion.iter_mut().enumerate() {
                m[k * 16..k * 16 + 16].copy_from_slice(&self.pm.frame.motion[l][a * 16..a * 16 + 16]);
            }
            modes[k * 16..k * 16 + 16].copy_from_slice(&self.pm.info.intra_modes[a * 16..a * 16 + 16]);
        }
        PairSnap {
            planes,
            info: [self.pm.info.mbs[top], self.pm.info.mbs[bot]],
            motion,
            intra: [self.pm.frame.mb_intra[top], self.pm.frame.mb_intra[bot]],
            field: [self.pm.frame.mb_field[top], self.pm.frame.mb_field[bot]],
            modes,
            prev_qp: self.chain.prev,
        }
    }

    fn restore(&mut self, s: &PairSnap<S>, x: usize, pr: usize) {
        let mbw = self.pc.mbs_wide;
        let top = 2 * pr * mbw + x;
        for p in 0..self.rec.len() {
            let (bw, bh) = self.mb_dims(p);
            for y in 0..2 * bh {
                let o = self.rec[p].offset((x * bw) as isize, (pr * 2 * bh + y) as isize);
                self.rec[p].data[o..o + bw].copy_from_slice(&s.planes[p][y * bw..(y + 1) * bw]);
            }
        }
        for (k, a) in [top, top + mbw].into_iter().enumerate() {
            self.pm.info.mbs[a] = s.info[k];
            for l in 0..2 {
                self.pm.frame.motion[l][a * 16..a * 16 + 16].copy_from_slice(&s.motion[l][k * 16..k * 16 + 16]);
            }
            self.pm.frame.mb_intra[a] = s.intra[k];
            self.pm.frame.mb_field[a] = s.field[k];
            self.pm.info.intra_modes[a * 16..a * 16 + 16].copy_from_slice(&s.modes[k * 16..k * 16 + 16]);
        }
        self.chain.prev = s.prev_qp;
        self.sync_frame_mb(x, 2 * pr);
        self.sync_frame_mb(x, 2 * pr + 1);
    }

    /// A coded candidate's price: the squared error of the pair's
    /// reconstruction against the source over every plane, plus `lam`
    /// times what the writer says the pair costs. A pair of skips whose
    /// flag is not the inferred one cannot be written at all — nothing in
    /// the stream would carry the flag — and costs infinitely much.
    fn cost<W: PairWriter>(&self, pair: &CodedPair, inferred: bool, x: usize, pr: usize, writer: &W) -> f64 {
        if pair.mbs[0].is_skip() && pair.mbs[1].is_skip() && pair.field != inferred {
            return f64::INFINITY;
        }
        let mut ssd = 0u64;
        for p in 0..self.rec.len() {
            let (bw, bh) = self.mb_dims(p);
            let (src, stride) = match p {
                0 => (&self.pc.src_y, self.pc.luma_stride),
                1 => (&self.pc.src_cb, self.pc.chroma_stride),
                _ => (&self.pc.src_cr, self.pc.chroma_stride),
            };
            for y in 0..2 * bh {
                let fy = pr * 2 * bh + y;
                let o = self.rec[p].offset((x * bw) as isize, fy as isize);
                for i in 0..bw {
                    let d = i64::from(src[fy * stride + x * bw + i].to_i32()) - i64::from(self.rec[p].data[o + i].to_i32());
                    ssd += (d * d) as u64;
                }
            }
        }
        ssd as f64 + self.lam * writer.trial_bits(pair, &self.pm) as f64
    }

    /// Code both macroblocks of pair `(x, pr)` as frame or field
    /// macroblocks, committing each.
    fn code_pair(&mut self, x: usize, pr: usize, field: bool) -> CodedPair {
        let top = 2 * pr * self.pc.mbs_wide + x;
        let mbs = [self.code_mb(x, pr, 0, field), self.code_mb(x, pr, 1, field)];
        CodedPair { top, field, mbs }
    }

    /// Decide, reconstruct and commit macroblock `b` (0 top, 1 bottom) of
    /// pair `(x, pr)` — in the frame view at frame row `2pr + b`, or as a
    /// field macroblock in field `b`'s view at field row `pr`.
    fn code_mb(&mut self, x: usize, pr: usize, b: usize, field: bool) -> PairMb {
        let mbw = self.pc.mbs_wide;
        let addr = (2 * pr + b) * mbw + x;
        let (vx, vy) = if field { (x, pr) } else { (x, 2 * pr + b) };
        let parity = if field { b as u8 } else { crate::h264::frame::PARITY_FRAME };
        self.nb.derive_mbaff_into(&self.pm.info, addr, 0, field);
        let avail = MbAvail {
            left: self.nb.a.is_some(),
            top: self.nb.b.is_some(),
            top_left: self.nb.d.is_some(),
            top_right: self.nb.c.is_some(),
        };
        let (left4, top4, left8) = mbaff_edge_modes(&self.nb, &self.pm.info);
        let ctx = IntraCtx { field, ..self.pc.ctx_at(self.pc.mb_qp(addr)) };
        let (ls, cs) = (self.pc.luma_stride, self.pc.chroma_stride);
        let (ys, cbs, crs): (&[S], &[S], &[S]) = if field {
            (&self.fsrc[b][0], &self.fsrc[b][1], &self.fsrc[b][2])
        } else {
            (&self.pc.src_y, &self.pc.src_cb, &self.pc.src_cr)
        };
        let rec: &mut [Recon<S>] = if field { &mut self.views[b] } else { &mut *self.rec };
        let out = match &self.refs {
            MbaffRefs::Intra => {
                let (dec, modes) =
                    code_macroblock_modes8(&ctx, rec, vx, vy, ys, ls, [cbs, crs], cs, avail, &left4, &top4, &left8);
                PairMb::Intra(commit_intra(&mut self.pm, &mut self.chain, addr, field, ctx.qp, dec, &modes))
            }
            MbaffRefs::P { frame, fields } => {
                let planes = if field { fields[b] } else { *frame };
                let pref = PRef { planes, search: &planes[0], weighting: Weighting::Default };
                self.st.start_mbaff(&self.pm.frame, &self.pm.info, addr, field, &mut self.nb);
                let mut dec = code_macroblock_p(&ctx, rec, &pref, vx, vy, ys, ls, [cbs, crs], cs, &mut self.st);
                match dec.kind {
                    InterMbKind::UseIntra => {
                        let (idec, modes) =
                            code_macroblock_modes8(&ctx, rec, vx, vy, ys, ls, [cbs, crs], cs, avail, &left4, &top4, &left8);
                        PairMb::PIntra(commit_intra(&mut self.pm, &mut self.chain, addr, field, ctx.qp, idec, &modes))
                    }
                    InterMbKind::PSkip => {
                        let q = self.chain.settle(ctx.qp, false);
                        self.pm.commit(addr, MbInfo { field, ..coded_info(DecKind::PSkip, 0, false, q, [0; 2]) }, self.st.motion());
                        self.pm.info.intra_modes[addr * 16..addr * 16 + 16].fill(2);
                        PairMb::P(dec)
                    }
                    _ => {
                        let q = self.chain.settle(ctx.qp, has_residual(dec.kind.dec_kind(), dec.cbp_luma | (dec.cbp_chroma << 4)));
                        dec.qp_delta = q.delta as i8;
                        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                        let n = dec.rects(&mut rects);
                        self.pm.commit(
                            addr,
                            MbInfo {
                                field,
                                ..coded_info(
                                    dec.kind.dec_kind(),
                                    nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                                    dec.transform_8x8,
                                    q,
                                    part_edges_of(&rects[..n]),
                                )
                            },
                            self.st.motion(),
                        );
                        self.pm.info.intra_modes[addr * 16..addr * 16 + 16].fill(2);
                        PairMb::P(dec)
                    }
                }
            }
            MbaffRefs::B { frame, fields, col } => {
                let refs2 = BRefs::plain(if field { [fields[0][b], fields[1][b]] } else { *frame });
                self.st.start_mbaff(&self.pm.frame, &self.pm.info, addr, field, &mut self.nb);
                let mut dec =
                    code_macroblock_b(&ctx, rec, &refs2, vx, vy, ys, ls, [cbs, crs], cs, &mut self.st, col, addr, field, parity);
                if dec.kind == BMbKind::UseIntra {
                    let (idec, modes) =
                        code_macroblock_modes8(&ctx, rec, vx, vy, ys, ls, [cbs, crs], cs, avail, &left4, &top4, &left8);
                    PairMb::BIntra(commit_intra(&mut self.pm, &mut self.chain, addr, field, ctx.qp, idec, &modes))
                } else {
                    let residual =
                        dec.kind != BMbKind::BSkip && has_residual(dec.kind.dec_kind(), dec.cbp_luma | (dec.cbp_chroma << 4));
                    let q = self.chain.settle(ctx.qp, residual);
                    dec.qp_delta = q.delta as i8;
                    let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                    let n = dec.rects(&mut rects);
                    let sub_direct =
                        if dec.kind == BMbKind::B8x8 { (0..4).map(|p| (dec.is_direct_part(p) as u8) << p).sum() } else { 0 };
                    self.pm.commit(
                        addr,
                        MbInfo {
                            sub_direct,
                            field,
                            ..coded_info(
                                dec.kind.dec_kind(),
                                nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                                dec.transform_8x8,
                                q,
                                part_edges_of(&rects[..n]),
                            )
                        },
                        self.st.motion(),
                    );
                    self.pm.info.intra_modes[addr * 16..addr * 16 + 16].fill(2);
                    PairMb::B(dec)
                }
            }
        };
        if field {
            self.sync_field_mb(x, pr, b);
        } else {
            self.sync_frame_mb(x, 2 * pr + b);
        }
        out
    }
}

/// Decide, reconstruct and filter an MBAFF frame (`mb_adaptive_frame_field_flag`
/// 1, `field_pic_flag` 0), pair by pair in decoding order, handing each pair
/// to `writer`.
///
/// Every pair is coded twice — as two frame macroblocks and as two field
/// macroblocks — each candidate priced by the squared error it leaves plus
/// the multiplier times the bits its writer says it costs, and the cheaper
/// is kept (ties to the frame pair), the loser undone. A frame macroblock
/// is decided in the frame, a field macroblock in its own field: its
/// source, its reconstruction, and its references are that field's, its
/// motion in field units, its residual in the field scans. Neighbours,
/// motion prediction and the skip vector come from the decoder's own MBAFF
/// derivations (`derive_mbaff_into`, `MotionCache::gather`), a B
/// macroblock's colocated motion from its AFRM mapping, and the loop filter
/// runs the decoder's MBAFF pair filter over the frame at the end.
pub(crate) fn code_mbaff_picture<S: Sample, W: PairWriter>(
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refs: MbaffRefs<'_, S>,
    writer: &mut W,
) -> PicMotion {
    debug_assert!(g.mbaff && !g.field_pic && g.mbs_high.is_multiple_of(2));
    let pc = PicCoding::new(g, tools, qp, planes);
    let (mbw, mbh) = (pc.mbs_wide, pc.mbs_high);
    let field_rows = |data: &[S], stride: usize, parity: usize| -> Vec<S> {
        if stride == 0 {
            return Vec::new();
        }
        let rows = data.len() / stride;
        (0..rows / 2).flat_map(|r| data[(2 * r + parity) * stride..(2 * r + parity + 1) * stride].iter().copied()).collect()
    };
    let fsrc = [0usize, 1].map(|p| {
        [
            field_rows(&pc.src_y, pc.luma_stride, p),
            field_rows(&pc.src_cb, pc.chroma_stride, p),
            field_rows(&pc.src_cr, pc.chroma_stride, p),
        ]
    });
    let views = [0usize, 1].map(|_| {
        rec.iter()
            .map(|p| crate::encode::h264_syntax::recon_plane(p.width as u32, (p.height / 2) as u32, p.pad))
            .collect::<Vec<_>>()
    });
    let mut pm = PicMotion::new(mbw, mbh);
    pm.frame.mbaff = true;
    pm.frame.mb_field = vec![false; mbw * mbh];
    let lam = f64::from(crate::encode::h264_intra::lambda(i32::from(qp))) * f64::from(1u32 << (2 * (g.bit_depth - 8)));
    let chain = QpChain::new(pc.ctx.qp, g.bit_depth);
    let mut walk = MbaffWalk { g, pc, fsrc, rec, views, pm, chain, st: MbMotionState::new(), nb: MbNeighbours::default(), refs, lam };
    for pr in 0..mbh / 2 {
        for x in 0..mbw {
            let top = 2 * pr * mbw + x;
            let inferred = crate::h264::decoder::infer_mb_field(&walk.pm.info, top, 0);
            let snap = walk.snapshot(x, pr);
            let frame_pair = walk.code_pair(x, pr, false);
            let frame_cost = walk.cost(&frame_pair, inferred, x, pr, writer);
            walk.restore(&snap, x, pr);
            let field_pair = walk.code_pair(x, pr, true);
            let field_cost = walk.cost(&field_pair, inferred, x, pr, writer);
            let pair = if field_cost < frame_cost {
                field_pair
            } else {
                walk.restore(&snap, x, pr);
                walk.code_pair(x, pr, false)
            };
            walk.pm.pairs[pair.field as usize] += 1;
            writer.write_pair(&pair, &walk.pm);
        }
    }
    let MbaffWalk { mut pm, rec, .. } = walk;
    deblock_recon(&tools.dsp, g, &mut pm, rec);
    pm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MBAFF neighbouring modes are the decoder's: for a macroblock of
    /// every kind (frame or field, top or bottom) beside a left pair of
    /// either kind whose macroblocks are `I_4x4` with a different mode on
    /// every block, `I_8x8` or not `I_NxN`, and under an above pair likewise,
    /// the mode `predicted_intra_mode` derives for a block whose other
    /// neighbour predicts mode 8 is the one `mbaff_edge_modes` hands the
    /// intra decision — for 4x4 blocks down the left column and along the
    /// top row, and for 8x8 blocks 0 and 2 (the `n` = 3 case included).
    #[test]
    fn mbaff_neighbouring_modes_are_the_decoders() {
        use crate::h264::SliceType;
        use crate::h264::cavlc::predicted_intra_mode;
        use crate::h264::mb::{MbLayer, SliceCtx};
        let ctx = SliceCtx {
            slice_type: SliceType::I,
            slice_num: 0,
            num_ref_idx: [0, 0],
            direct_spatial: false,
            transform_8x8_mode: true,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc: 1,
            cabac: true,
            bit_depth: 8,
            transform_bypass: false,
            scaling_plane: 0,
            x264_old_444: false,
            field_pic: false,
            mbaff: true,
            sp: false,
            sp_switch: false,
            sp_qs: 0,
            sp_qsc: [0; 2],
        };
        // Two pairs wide, two pairs high: the current pair at (1, 1), its
        // left pair at (0, 1) and its above pair at (1, 0).
        let (mbw, mbh) = (2usize, 4usize);
        for cur_field in [false, true] {
            for nb_field in [false, true] {
                for kind in [DecKind::I4x4, DecKind::I8x8, DecKind::I16x16] {
                    for bottom in [0usize, 1] {
                        for side in [0usize, 1] {
                            // side 0: vary the left pair, fix the above one at mode 8;
                            // side 1: the reverse.
                            let mut info = PicInfo::new(mbw, mbh);
                            let set = |info: &mut PicInfo, addr: usize, k: DecKind, field: bool, varied: bool| {
                                info.mbs[addr] = MbInfo { kind: k, decoded: true, slice: 0, field, ..MbInfo::default() };
                                for blk in 0..16 {
                                    info.intra_modes[addr * 16 + blk] = if !varied {
                                        8
                                    } else if k == DecKind::I8x8 {
                                        ((blk / 8) * 2 + (blk % 4) / 2) as u8 * 3 % 8
                                    } else {
                                        (blk as u8 * 5 + addr as u8) % 8
                                    };
                                }
                            };
                            for b in 0..2 {
                                set(&mut info, (2 + b) * mbw, if side == 0 { kind } else { DecKind::I4x4 }, nb_field, side == 0);
                                set(&mut info, b * mbw + 1, if side == 1 { kind } else { DecKind::I4x4 }, nb_field, side == 1);
                            }
                            // The current pair's top macroblock, already coded: a
                            // bottom frame macroblock's above neighbour.
                            set(&mut info, 2 * mbw + 1, if side == 1 { kind } else { DecKind::I4x4 }, cur_field, side == 1);
                            let addr = (2 + bottom) * mbw + 1;
                            let mut nb = MbNeighbours::default();
                            nb.derive_mbaff_into(&info, addr, 0, cur_field);
                            let (left, top, left8) = mbaff_edge_modes(&nb, &info);
                            let mut layer = MbLayer::new(DecKind::I4x4);
                            layer.intra_modes = [8; 16];
                            let tag = format!("cur_field {cur_field} nb_field {nb_field} {kind:?} bottom {bottom} side {side}");
                            if side == 0 {
                                for (r, got) in left.iter().enumerate() {
                                    let want = predicted_intra_mode(&info, &layer, &nb, &ctx, 0, r, false);
                                    assert_eq!(Some(want), got.map(|m| m.min(8)), "{tag}: 4x4 row {r}");
                                }
                                for r in [0usize, 2] {
                                    let want = predicted_intra_mode(&info, &layer, &nb, &ctx, 0, r, true);
                                    assert_eq!(Some(want), left8[r].map(|m| m.min(8)), "{tag}: 8x8 row {r}");
                                }
                            } else {
                                for (c, got) in top.iter().enumerate() {
                                    let want = predicted_intra_mode(&info, &layer, &nb, &ctx, c, 0, false);
                                    assert_eq!(Some(want), got.map(|m| m.min(8)), "{tag}: 4x4 column {c}");
                                }
                                for c in [0usize, 2] {
                                    let want = predicted_intra_mode(&info, &layer, &nb, &ctx, c, 0, true);
                                    assert_eq!(Some(want), top[c].map(|m| m.min(8)), "{tag}: 8x8 column {c}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
