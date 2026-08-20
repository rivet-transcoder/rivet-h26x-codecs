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
use crate::encode::h264_deblock::{deblock_recon, nz_mask_of};
use crate::encode::h264_intra::{IntraCtx, MbAvail, MbDecision, MbKind, code_macroblock};
use crate::encode::h264_me::{
    BDecision, BMbKind, InterDecision, InterMbKind, MbMotionState, code_macroblock_b16,
    code_macroblock_p,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::frame::{BlockMotion, Frame, Mv};
use crate::h264::cavlc::mb_partitions;
use crate::h264::mb::{MbInfo, MbKind as DecKind, MbMotion, MbNeighbours, PicInfo, chroma_qp};
use crate::h264::sps::ScalingLists;
use crate::h264::transform::Dequant;
use crate::picture::ChromaFormat;

/// The kernels and derived tables the transform paths run on, built once
/// per encoder and shared by both entropy coders — and, beside them, the
/// one coding-tool switch that has to reach every decision walk.
pub struct IntraTools {
    pub(crate) dsp: H264Dsp<u8>,
    pub(crate) enc: H264EncDsp,
    pub(crate) dist: DistortionDsp<u8>,
    pub(crate) quant: Quant,
    pub(crate) dequant: Dequant,
    /// `transform_8x8_mode_flag`, as the PPS writes it. A decision may
    /// only produce `transform_size_8x8_flag` when this is true, because
    /// otherwise the element is not in the bitstream at all. It rides
    /// here rather than through six picture-writer signatures because it
    /// is what it looks like: one constant per encoder, shared by every
    /// walk, and impossible to pass to one path and forget on another.
    pub(crate) transform_8x8: bool,
    /// Whether inter partitions smaller than 16x16 are on offer.
    pub(crate) subparts: bool,
}

impl IntraTools {
    /// Build for the running CPU, offering the 8x8 transform or not. The
    /// scaling lists are flat sixteens because the parameter sets this
    /// encoder writes carry no scaling matrices, which makes flat the
    /// lists a decoder will derive — and that is as true of the 8x8 lists
    /// as of the 4x4 ones, since the PPS declares
    /// `pic_scaling_matrix_present_flag` zero either way.
    pub fn new(transform_8x8: bool, subparts: bool) -> Self {
        let lists = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let cpu = Cpu::detect_honouring_env();
        IntraTools {
            dsp: H264Dsp::new(cpu),
            enc: H264EncDsp::new(cpu),
            dist: DistortionDsp::new(cpu),
            quant: Quant::new(&lists),
            dequant: Dequant::new(&lists),
            transform_8x8,
            subparts,
        }
    }
}

impl Default for IntraTools {
    fn default() -> Self {
        Self::new(false, false)
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
        PicMotion { info: PicInfo::new(mbs_wide, mbs_high), frame }
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
        self.info.mbs[addr] = info;
        for l in 0..2 {
            self.frame.motion[l][addr * 16..addr * 16 + 16].copy_from_slice(&mot[l]);
        }
    }

    /// The colocated motion of macroblock `addr`, block `blk` (raster
    /// 4x4), through the decoder's own `colocated_motion` — what a B
    /// picture's direct derivation reads out of its list-1 reference.
    pub(crate) fn colocated(&self, addr: usize, blk: usize) -> (Mv, i8) {
        let (mv, ref_idx, _, _) = crate::h264::mb::colocated_motion(&self.frame, addr, blk);
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
fn coded_info(
    kind: DecKind,
    nz_mask: u16,
    transform_8x8: bool,
    qp: i32,
    qpc: [i32; 2],
    part_edges: [u16; 2],
) -> MbInfo {
    MbInfo {
        kind,
        decoded: true,
        slice: 0,
        qp: qp as i8,
        qpc: [qpc[0] as i8, qpc[1] as i8],
        transform_8x8,
        nz_mask,
        part_edges,
        ..MbInfo::default()
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
fn pad_to(src: &Plane<'_>, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
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
struct PicCoding<'a> {
    /// The per-picture context both mode-decision modules take.
    ctx: IntraCtx<'a>,
    /// Picture size in macroblocks.
    mbs_wide: usize,
    /// See `mbs_wide`.
    mbs_high: usize,
    /// Stride of `src_y` (the coded width).
    luma_stride: usize,
    /// Stride of `src_cb` / `src_cr`; 0 for monochrome.
    chroma_stride: usize,
    /// The source planes at coded size, edge-replicated.
    src_y: Vec<u8>,
    /// See `src_y` (empty for monochrome).
    src_cb: Vec<u8>,
    /// See `src_y`.
    src_cr: Vec<u8>,
}

impl<'a> PicCoding<'a> {
    fn new(g: &Geometry, tools: &'a IntraTools, qp: u8, planes: &[Plane<'_>]) -> Self {
        let (cw, ch) = g.chroma_mb();
        let chroma_h = ch as usize;
        // 8-bit only (the encoder refuses deeper at construction), and the
        // PPS writes both chroma QP offsets as zero.
        let qpc = chroma_qp(qp as i32, 0, 0);
        let ctx = IntraCtx {
            dsp: &tools.dsp,
            enc: &tools.enc,
            dist: &tools.dist,
            quant: &tools.quant,
            dequant: &tools.dequant,
            qp: qp as i32,
            qpc: [qpc; 2],
            chroma_h,
            c444: g.chroma == ChromaFormat::Yuv444,
            t8x8: tools.transform_8x8,
            subparts: tools.subparts,
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
        PicCoding {
            ctx,
            mbs_wide,
            mbs_high,
            luma_stride,
            chroma_stride,
            src_y,
            src_cb,
            src_cr,
        }
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
pub(crate) fn code_intra_picture(
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    mut emit: impl FnMut(usize, usize, &MbDecision),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let mb = MbAvail {
                left: mb_x > 0,
                top: mb_y > 0,
                top_left: mb_x > 0 && mb_y > 0,
                top_right: mb_y > 0 && mb_x + 1 < mbs_wide,
            };
            let (dec, modes) = code_macroblock(
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
            emit(mb_x, mb_y, &dec);
            pm.commit(
                mb_y * mbs_wide + mb_x,
                coded_info(
                    filter_kind(dec.kind),
                    nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                    dec.transform_8x8,
                    ctx.qp,
                    ctx.qpc,
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
/// reference is active.
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
pub(crate) fn code_p_picture(
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refp: &[Recon],
    mut emit: impl FnMut(usize, usize, PMb<'_>),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    // The picture's motion in the decoder's own layout, and the
    // per-macroblock working set its derivations read.
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    let mut dnb = MbNeighbours::default();
    let mut st = MbMotionState::new();
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let addr = mb_y * mbs_wide + mb_x;
            st.start(&pm.frame, &pm.info, addr, &mut dnb);
            let dec = code_macroblock_p(
                ctx,
                rec,
                refp,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                &mut st,
            );
            match dec.kind {
                InterMbKind::PSkip => {
                    emit(mb_x, mb_y, PMb::Skip(&dec));
                    pm.commit(
                        addr,
                        coded_info(DecKind::PSkip, 0, false, ctx.qp, ctx.qpc, [0; 2]),
                        st.motion(),
                    );
                    left_modes = [Some(2); 4];
                    top_modes[mb_x] = [Some(2); 4];
                }
                InterMbKind::P16x16
                | InterMbKind::P16x8
                | InterMbKind::P8x16
                | InterMbKind::P8x8 => {
                    emit(mb_x, mb_y, PMb::Coded(&dec));
                    let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
                    let n = dec.rects(&mut rects);
                    pm.commit(
                        addr,
                        coded_info(
                            dec.kind.dec_kind(),
                            nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                            dec.transform_8x8,
                            ctx.qp,
                            ctx.qpc,
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
                    let (idec, modes) = code_macroblock(
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
                    emit(mb_x, mb_y, PMb::Intra(&idec));
                    pm.commit(
                        addr,
                        coded_info(
                            filter_kind(idec.kind),
                            nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                            idec.transform_8x8,
                            ctx.qp,
                            ctx.qpc,
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
    pm
}

/// One coded macroblock of a B picture, as the walk hands it to a
/// serialiser — [`PMb`]'s two-list sibling.
pub enum BMb<'a> {
    /// `B_Skip`: the serialiser codes the skip signal and nothing else.
    Skip(&'a BDecision),
    /// `B_Direct_16x16` with a residual (`mb_type` 0, no motion syntax).
    Direct(&'a BDecision),
    /// An explicit 16x16 — L0, L1 or bi by [`BDecision::used`] — with its
    /// mvds, cbp and coefficients.
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
pub(crate) fn code_b_picture(
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refs: [&[Recon]; 2],
    col: &PicMotion,
    mut emit: impl FnMut(usize, usize, BMb<'_>),
) -> PicMotion {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);
    debug_assert_eq!(
        col.info.mbs.len(),
        mbs_wide * mbs_high,
        "the colocated picture is the same size"
    );

    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    let mut dnb = MbNeighbours::default();
    let mut st = MbMotionState::new();
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        for mb_x in 0..mbs_wide {
            let addr = mb_y * mbs_wide + mb_x;
            st.start(&pm.frame, &pm.info, addr, &mut dnb);
            let dec = code_macroblock_b16(
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
                let (idec, modes) = code_macroblock(
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
                emit(mb_x, mb_y, BMb::Intra(&idec));
                pm.commit(
                    addr,
                    coded_info(
                        filter_kind(idec.kind),
                        nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                        idec.transform_8x8,
                        ctx.qp,
                        ctx.qpc,
                        [0; 2],
                    ),
                    &[[BlockMotion::default(); 16]; 2],
                );
                (left_modes, top_modes[mb_x]) = edge_modes(idec.kind, &modes);
                continue;
            }
            emit(
                mb_x,
                mb_y,
                match dec.kind {
                    BMbKind::BSkip => BMb::Skip(&dec),
                    BMbKind::BDirect16 => BMb::Direct(&dec),
                    BMbKind::B16 => BMb::Explicit(&dec),
                    BMbKind::UseIntra => unreachable!(),
                },
            );
            pm.commit(
                addr,
                coded_info(
                    match dec.kind {
                        BMbKind::BSkip => DecKind::BSkip,
                        BMbKind::BDirect16 => DecKind::BDirect16x16,
                        _ => DecKind::Inter16x16,
                    },
                    nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                    dec.transform_8x8,
                    ctx.qp,
                    ctx.qpc,
                    // A direct macroblock is *four 8x8 partitions*, not
                    // one: `direct_partitions` pushes a job per 8x8 under
                    // `direct_8x8_inference` (src/h264/recon.rs), so a
                    // decoder records the 8x8 cross as partition edges
                    // and compares motion across them. An explicit 16x16
                    // has no internal edge.
                    //
                    // Passing [0, 0] here was harmless while direct gave
                    // all four the same vector — the comparison it
                    // skipped would have come out bS 0 anyway — and
                    // became a real desync the moment colZeroFlag started
                    // varying per 8x8. It cost six cells of
                    // `--subparts --t8x8 --bframes 2`.
                    if dec.kind == BMbKind::B16 {
                        [0; 2]
                    } else {
                        part_edges_of(mb_partitions(DecKind::Inter8x8))
                    },
                ),
                st.motion(),
            );
            left_modes = [Some(2); 4];
            top_modes[mb_x] = [Some(2); 4];
        }
    }
    deblock_recon(&tools.dsp, g, &mut pm, rec);
    pm
}
