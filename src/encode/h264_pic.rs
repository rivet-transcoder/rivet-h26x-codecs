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
use crate::encode::h264_deblock::{FilterMb, deblock_recon, nz_mask_of};
use crate::encode::h264_intra::{IntraCtx, MbAvail, MbDecision, MbKind, code_macroblock};
use crate::encode::h264_me::{
    BDecision, BMbKind, InterDecision, InterMbKind, MotionNeighbours, code_macroblock_b16,
    code_macroblock_p16, mv_predictor_16x16, nb_inter, nb_intra, skip_mv_16x16,
    spatial_direct_ref_idx_mirror,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::frame::{BlockMotion, Frame, Mv};
use crate::h264::mb::{
    MbInfo, MbMotion, MbNeighbours, MotionCache, NbMotion, PicInfo, chroma_qp,
};
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
}

impl IntraTools {
    /// Build for the running CPU, offering the 8x8 transform or not. The
    /// scaling lists are flat sixteens because the parameter sets this
    /// encoder writes carry no scaling matrices, which makes flat the
    /// lists a decoder will derive — and that is as true of the 8x8 lists
    /// as of the 4x4 ones, since the PPS declares
    /// `pic_scaling_matrix_present_flag` zero either way.
    pub fn new(transform_8x8: bool) -> Self {
        let lists = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let cpu = Cpu::detect_honouring_env();
        IntraTools {
            dsp: H264Dsp::new(cpu),
            enc: H264EncDsp::new(cpu),
            dist: DistortionDsp::new(cpu),
            quant: Quant::new(&lists),
            dequant: Dequant::new(&lists),
            transform_8x8,
        }
    }
}

impl Default for IntraTools {
    fn default() -> Self {
        Self::new(false)
    }
}

/// The motion state of the picture being coded, in the *decoder's* own
/// layout — kept so that the decoder's derivations can be **called**
/// rather than mirrored.
///
/// The encoder has always mirrored 8.4.1.3 instead, through
/// [`MotionNeighbours`]: four macroblock-level neighbours, one motion
/// each. That is expressible only while every partition is the whole
/// macroblock. The neighbours of a smaller partition are 4x4 *blocks*,
/// and for every partition after the first they are blocks of this same
/// macroblock, already derived and gated by a `done` bitmask
/// (`block_available`, src/h264/mb.rs) — which a per-macroblock summary
/// cannot represent at all. So rather than grow the mirror into a second,
/// larger thing to keep in step, the encoder keeps what the decoder
/// keeps: [`MbInfo`] per macroblock and [`BlockMotion`] per 4x4, which is
/// precisely what [`MbNeighbours::derive_into`] and
/// [`MotionCache::gather`] read.
///
/// The `Frame` is plane-less on purpose: `gather`'s progressive path
/// touches `frame.motion` and `info.mbs[].kind` and nothing else, so
/// carrying the reconstruction here would be a second copy of it for no
/// gain.
pub(crate) struct PicMotion {
    /// Per-macroblock info — neighbour availability and the intra test.
    pub info: PicInfo,
    /// Per-4x4 motion per list, inside a decoder frame so that
    /// [`MotionCache::gather`] takes it directly.
    pub frame: Frame<u8>,
}

impl PicMotion {
    /// Empty state for a picture `mbs_wide` by `mbs_high` macroblocks.
    pub fn new(mbs_wide: usize, mbs_high: usize) -> Self {
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

    /// The neighbours and gathered motion cache for macroblock `addr`,
    /// derived exactly as the decoder's slice loop derives them. The
    /// macroblock itself is still undecoded at this point, which is what
    /// keeps it out of its own neighbour set.
    pub fn cache_for(&self, addr: usize, nb: &mut MbNeighbours, cache: &mut MotionCache) {
        nb.derive_into(&self.info, addr, 0);
        cache.gather(nb, &self.frame, &self.info);
    }

    /// Commit one coded macroblock: the kind its neighbours will test for
    /// intra-ness, and its per-4x4 motion. `mot` is in the decoder's
    /// raster layout, one entry per 4x4 per list — an intra macroblock
    /// commits [`BlockMotion::default`] throughout, as `derive()` does.
    pub fn commit(&mut self, addr: usize, kind: crate::h264::mb::MbKind, mot: &MbMotion) {
        let m = &mut self.info.mbs[addr];
        *m = MbInfo { kind, decoded: true, slice: 0, ..MbInfo::default() };
        self.frame.mb_intra[addr] = kind.is_intra();
        for l in 0..2 {
            self.frame.motion[l][addr * 16..addr * 16 + 16].copy_from_slice(&mot[l]);
        }
    }
}

/// One macroblock's motion in the decoder's layout, from the single
/// 16x16 record this encoder's inter decisions still produce.
///
/// Transitional: every partition being the whole macroblock is what makes
/// one vector per list a faithful record (INVARIANT(16x16-only) on
/// [`FilterMb`]). It exists so the decoder-shaped state can be filled and
/// cross-checked before the shapes that break that premise land.
pub(crate) fn mb_motion_16x16(l0: Option<Mv>, l1: Option<Mv>) -> MbMotion {
    let mut mot: MbMotion = [[BlockMotion::default(); 16]; 2];
    for (l, mv) in [(0usize, l0), (1usize, l1)] {
        let Some(mv) = mv else { continue };
        // `ref_id` is an identity the derivations only compare for
        // equality; one reference per list makes 1 and 2 distinct names,
        // as `deblock_recon` already spells them.
        mot[l] = [BlockMotion {
            mv,
            ref_idx: 0,
            ref_parity: crate::h264::frame::PARITY_FRAME,
            ref_id: 1 + l as u16,
        }; 16];
    }
    mot
}

/// Assert that the mirrored 16x16 derivations agree with the decoder's
/// own, over the state actually being coded.
///
/// The unit test `the_predictor_and_skip_mv_agree_with_the_decoders_derivation`
/// already holds the mirror against `predict_mv` and `p_skip_mv` over
/// synthetic neighbour sets. This is the same comparison over every
/// macroblock of every clip the gate encodes, and it is here for one
/// reason: it is the evidence that `PicMotion` can *replace* the mirror
/// rather than merely sit beside it. Once it has, this goes, along with
/// [`MotionNeighbours`].
///
/// Debug-only, because it is scaffolding and because the state it reads
/// is maintained whether or not anything checks it.
#[cfg(debug_assertions)]
fn cross_check_16x16(cache: &MotionCache, nbm: &MotionNeighbours, list: usize) {
    let cur: MbMotion = [[BlockMotion::default(); 16]; 2];
    let want = crate::h264::mb::predict_mv(cache, &cur, 0, list, 0, 0, 0, 16, 16);
    assert_eq!(
        mv_predictor_16x16(nbm),
        want,
        "list {list}: the mirrored 16x16 predictor disagrees with the decoder's"
    );
    if list == 0 {
        assert_eq!(
            skip_mv_16x16(nbm),
            crate::h264::mb::p_skip_mv(cache, &cur),
            "the mirrored P_Skip vector disagrees with the decoder's"
        );
    }
}

#[cfg(not(debug_assertions))]
fn cross_check_16x16(_cache: &MotionCache, _nbm: &MotionNeighbours, _list: usize) {}

/// The same for spatial direct's reference indices (8.4.1.2.2's
/// `min_positive` over the three neighbours per list), against the
/// decoder's [`crate::h264::mb::spatial_direct_ref_idx`].
///
/// The vectors are not compared here: the encoder's derivation reads
/// colZeroFlag out of a per-macroblock `FilterMb`, and the decoder's
/// reads it per 8x8 out of the colocated picture's own motion — the two
/// agree only because of INVARIANT(16x16-only), which is the premise
/// this work removes. That comparison arrives with the colocated
/// storage.
#[cfg(debug_assertions)]
fn cross_check_direct_ref_idx(cache: &MotionCache, nbm: &[MotionNeighbours; 2]) {
    let cur: MbMotion = [[BlockMotion::default(); 16]; 2];
    let want = crate::h264::mb::spatial_direct_ref_idx(cache, &cur);
    let mine = spatial_direct_ref_idx_mirror(nbm);
    assert_eq!(mine, want, "the mirrored spatial-direct reference indices disagree");
}

#[cfg(not(debug_assertions))]
fn cross_check_direct_ref_idx(_cache: &MotionCache, _nbm: &[MotionNeighbours; 2]) {}

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
) -> Vec<FilterMb> {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    let mut fmbs: Vec<FilterMb> = Vec::with_capacity(mbs_wide * mbs_high);
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
            fmbs.push(FilterMb {
                kind: filter_kind(dec.kind),
                nz_mask: nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                transform_8x8: dec.transform_8x8,
                l0: None,
                l1: None,
            });
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
    deblock_recon(&tools.dsp, g, qp, &fmbs, rec);
    fmbs
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
/// - **Motion** ([`MotionNeighbours`]): an inter macroblock contributes
///   its vector with `ref_idx` 0 — a *skipped* one contributes the
///   derived skip vector, because a decoder stores exactly that — and an
///   intra macroblock contributes "available, not used for inter
///   prediction". The above-left value is saved before the row entry is
///   overwritten, or every D neighbour would be one picture-row too new.
/// - **Intra modes**: `Some(2)` for every available macroblock that is
///   not `I_NxN` — skip and P_16x16 included — because that is the DC the
///   reader's mode prediction derives for them (8.3.1.1).
/// - **The loop filter's inputs** ([`FilterMb`]), collected as coded and
///   applied after the last macroblock, before the reconstruction becomes
///   a reference.
pub(crate) fn code_p_picture(
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refp: &[Recon],
    mut emit: impl FnMut(usize, usize, PMb<'_>),
) -> Vec<FilterMb> {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);

    let mut fmbs: Vec<FilterMb> = Vec::with_capacity(mbs_wide * mbs_high);
    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    let mut top_motion: Vec<NbMotion> = vec![NbMotion::NONE; mbs_wide];
    // The decoder-shaped motion state, maintained beside the mirrored
    // neighbours and cross-checked against them per macroblock: see
    // `PicMotion`. Nothing reads it yet — this is the proof that it can
    // replace the mirror before anything depends on that.
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    let mut dnb = MbNeighbours::default();
    let mut dcache = MotionCache::default();
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        let mut left_motion = NbMotion::NONE;
        let mut topleft_motion = NbMotion::NONE;
        for mb_x in 0..mbs_wide {
            let nbm = MotionNeighbours {
                a: left_motion,
                b: if mb_y > 0 { top_motion[mb_x] } else { NbMotion::NONE },
                c: if mb_y > 0 && mb_x + 1 < mbs_wide {
                    top_motion[mb_x + 1]
                } else {
                    NbMotion::NONE
                },
                d: topleft_motion,
            };
            let addr = mb_y * mbs_wide + mb_x;
            pm.cache_for(addr, &mut dnb, &mut dcache);
            cross_check_16x16(&dcache, &nbm, 0);
            let dec = code_macroblock_p16(
                ctx,
                rec,
                refp,
                mb_x,
                mb_y,
                src_y,
                pc.luma_stride,
                [src_cb, src_cr],
                pc.chroma_stride,
                &nbm,
            );
            // The above-left of the *next* column is what stands in this
            // column's row entry now, before this macroblock replaces it.
            topleft_motion = if mb_y > 0 { top_motion[mb_x] } else { NbMotion::NONE };
            pm.commit(
                addr,
                match dec.kind {
                    InterMbKind::PSkip => crate::h264::mb::MbKind::PSkip,
                    InterMbKind::P16x16 => crate::h264::mb::MbKind::Inter16x16,
                    InterMbKind::UseIntra => crate::h264::mb::MbKind::I16x16,
                },
                &match dec.kind {
                    InterMbKind::UseIntra => [[BlockMotion::default(); 16]; 2],
                    _ => mb_motion_16x16(Some(dec.mv), None),
                },
            );
            match dec.kind {
                InterMbKind::PSkip => {
                    emit(mb_x, mb_y, PMb::Skip(&dec));
                    fmbs.push(FilterMb {
                        kind: crate::h264::mb::MbKind::PSkip,
                        nz_mask: 0,
                        transform_8x8: false,
                        l0: Some(dec.mv),
                        l1: None,
                    });
                    let m = nb_inter(dec.mv);
                    left_motion = m;
                    top_motion[mb_x] = m;
                    left_modes = [Some(2); 4];
                    top_modes[mb_x] = [Some(2); 4];
                }
                InterMbKind::P16x16 => {
                    emit(mb_x, mb_y, PMb::Coded(&dec));
                    fmbs.push(FilterMb {
                        kind: crate::h264::mb::MbKind::Inter16x16,
                        nz_mask: nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                        transform_8x8: dec.transform_8x8,
                        l0: Some(dec.mv),
                        l1: None,
                    });
                    let m = nb_inter(dec.mv);
                    left_motion = m;
                    top_motion[mb_x] = m;
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
                    fmbs.push(FilterMb {
                        kind: filter_kind(idec.kind),
                        nz_mask: nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                        transform_8x8: idec.transform_8x8,
                        l0: None,
                        l1: None,
                    });
                    left_motion = nb_intra();
                    top_motion[mb_x] = nb_intra();
                    (left_modes, top_modes[mb_x]) = edge_modes(idec.kind, &modes);
                }
            }
        }
    }
    // The loop filter, after the whole picture is reconstructed and before
    // the reconstruction becomes the next picture's reference — the
    // decoder's own ordering.
    deblock_recon(&tools.dsp, g, qp, &fmbs, rec);
    fmbs
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
    col: &[FilterMb],
    mut emit: impl FnMut(usize, usize, BMb<'_>),
) -> Vec<FilterMb> {
    let pc = PicCoding::new(g, tools, qp, planes);
    let ctx = &pc.ctx;
    let (mbs_wide, mbs_high) = (pc.mbs_wide, pc.mbs_high);
    let (src_y, src_cb, src_cr) = (&pc.src_y[..], &pc.src_cb[..], &pc.src_cr[..]);
    debug_assert_eq!(col.len(), mbs_wide * mbs_high, "one colocated record per macroblock");

    let mut fmbs: Vec<FilterMb> = Vec::with_capacity(mbs_wide * mbs_high);
    let mut top_modes: Vec<[Option<u8>; 4]> = vec![[None; 4]; mbs_wide];
    let mut top_motion: [Vec<NbMotion>; 2] =
        [vec![NbMotion::NONE; mbs_wide], vec![NbMotion::NONE; mbs_wide]];
    let mut pm = PicMotion::new(mbs_wide, mbs_high);
    let mut dnb = MbNeighbours::default();
    let mut dcache = MotionCache::default();
    for mb_y in 0..mbs_high {
        let mut left_modes: [Option<u8>; 4] = [None; 4];
        let mut left_motion = [NbMotion::NONE; 2];
        let mut topleft_motion = [NbMotion::NONE; 2];
        for mb_x in 0..mbs_wide {
            let nbm: [MotionNeighbours; 2] = std::array::from_fn(|l| MotionNeighbours {
                a: left_motion[l],
                b: if mb_y > 0 { top_motion[l][mb_x] } else { NbMotion::NONE },
                c: if mb_y > 0 && mb_x + 1 < mbs_wide {
                    top_motion[l][mb_x + 1]
                } else {
                    NbMotion::NONE
                },
                d: topleft_motion[l],
            });
            let addr = mb_y * mbs_wide + mb_x;
            pm.cache_for(addr, &mut dnb, &mut dcache);
            for l in 0..2 {
                cross_check_16x16(&dcache, &nbm[l], l);
            }
            cross_check_direct_ref_idx(&dcache, &nbm);
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
                &nbm,
                &col[mb_y * mbs_wide + mb_x],
            );
            pm.commit(
                addr,
                match dec.kind {
                    BMbKind::BSkip => crate::h264::mb::MbKind::BSkip,
                    BMbKind::BDirect16 => crate::h264::mb::MbKind::BDirect16x16,
                    BMbKind::B16 => crate::h264::mb::MbKind::Inter16x16,
                    BMbKind::UseIntra => crate::h264::mb::MbKind::I16x16,
                },
                &match dec.kind {
                    BMbKind::UseIntra => [[BlockMotion::default(); 16]; 2],
                    _ => mb_motion_16x16(
                        dec.used[0].then_some(dec.mv[0]),
                        dec.used[1].then_some(dec.mv[1]),
                    ),
                },
            );
            for l in 0..2 {
                topleft_motion[l] = if mb_y > 0 { top_motion[l][mb_x] } else { NbMotion::NONE };
            }
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
                fmbs.push(FilterMb {
                    kind: filter_kind(idec.kind),
                    nz_mask: nz_mask_of(&idec.nz_luma, idec.transform_8x8),
                    transform_8x8: idec.transform_8x8,
                    l0: None,
                    l1: None,
                });
                for l in 0..2 {
                    left_motion[l] = nb_intra();
                    top_motion[l][mb_x] = nb_intra();
                }
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
            fmbs.push(FilterMb {
                kind: match dec.kind {
                    BMbKind::BSkip => crate::h264::mb::MbKind::BSkip,
                    BMbKind::BDirect16 => crate::h264::mb::MbKind::BDirect16x16,
                    _ => crate::h264::mb::MbKind::Inter16x16,
                },
                nz_mask: nz_mask_of(&dec.nz_luma, dec.transform_8x8),
                transform_8x8: dec.transform_8x8,
                l0: dec.used[0].then_some(dec.mv[0]),
                l1: dec.used[1].then_some(dec.mv[1]),
            });
            for l in 0..2 {
                // A used list contributes its vector at reference 0; an
                // unused one contributes "available, not used for inter
                // prediction" — the same value the decoder's motion array
                // holds for it, and the same value an intra macroblock
                // leaves (`nb_intra`).
                let m = if dec.used[l] { nb_inter(dec.mv[l]) } else { nb_intra() };
                left_motion[l] = m;
                top_motion[l][mb_x] = m;
            }
            left_modes = [Some(2); 4];
            top_modes[mb_x] = [Some(2); 4];
        }
    }
    deblock_recon(&tools.dsp, g, qp, &fmbs, rec);
    fmbs
}
