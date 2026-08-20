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
    code_macroblock_p16, nb_inter, nb_intra,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::mb::{NbMotion, chroma_qp};
use crate::h264::sps::ScalingLists;
use crate::h264::transform::Dequant;
use crate::picture::ChromaFormat;

/// The kernels and derived tables the transform paths run on, built once
/// per encoder and shared by both entropy coders.
pub struct IntraTools {
    pub(crate) dsp: H264Dsp<u8>,
    pub(crate) enc: H264EncDsp,
    pub(crate) dist: DistortionDsp<u8>,
    pub(crate) quant: Quant,
    pub(crate) dequant: Dequant,
}

impl IntraTools {
    /// Build for the running CPU. The scaling lists are flat sixteens
    /// because the parameter sets this encoder writes carry no scaling
    /// matrices, which makes flat the lists a decoder will derive.
    pub fn new() -> Self {
        let lists = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let cpu = Cpu::detect_honouring_env();
        IntraTools {
            dsp: H264Dsp::new(cpu),
            enc: H264EncDsp::new(cpu),
            dist: DistortionDsp::new(cpu),
            quant: Quant::new(&lists),
            dequant: Dequant::new(&lists),
        }
    }
}

impl Default for IntraTools {
    fn default() -> Self {
        Self::new()
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
        debug_assert!(g.chroma != ChromaFormat::Yuv444, "ChromaArrayType 3 has no path here");
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
                kind: match dec.kind {
                    MbKind::I4x4 => crate::h264::mb::MbKind::I4x4,
                    MbKind::I16x16 => crate::h264::mb::MbKind::I16x16,
                },
                nz_mask: nz_mask_of(&dec.nz_luma),
                l0: None,
                l1: None,
            });
            (left_modes, top_modes[mb_x]) = match dec.kind {
                MbKind::I4x4 => (
                    [Some(modes[3]), Some(modes[7]), Some(modes[11]), Some(modes[15])],
                    [Some(modes[12]), Some(modes[13]), Some(modes[14]), Some(modes[15])],
                ),
                MbKind::I16x16 => ([Some(2); 4], [Some(2); 4]),
            };
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
            match dec.kind {
                InterMbKind::PSkip => {
                    emit(mb_x, mb_y, PMb::Skip(&dec));
                    fmbs.push(FilterMb {
                        kind: crate::h264::mb::MbKind::PSkip,
                        nz_mask: 0,
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
                        nz_mask: nz_mask_of(&dec.nz_luma),
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
                        kind: match idec.kind {
                            MbKind::I4x4 => crate::h264::mb::MbKind::I4x4,
                            MbKind::I16x16 => crate::h264::mb::MbKind::I16x16,
                        },
                        nz_mask: nz_mask_of(&idec.nz_luma),
                        l0: None,
                        l1: None,
                    });
                    left_motion = nb_intra();
                    top_motion[mb_x] = nb_intra();
                    (left_modes, top_modes[mb_x]) = match idec.kind {
                        MbKind::I4x4 => (
                            [Some(modes[3]), Some(modes[7]), Some(modes[11]), Some(modes[15])],
                            [Some(modes[12]), Some(modes[13]), Some(modes[14]), Some(modes[15])],
                        ),
                        MbKind::I16x16 => ([Some(2); 4], [Some(2); 4]),
                    };
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
                    kind: match idec.kind {
                        MbKind::I4x4 => crate::h264::mb::MbKind::I4x4,
                        MbKind::I16x16 => crate::h264::mb::MbKind::I16x16,
                    },
                    nz_mask: nz_mask_of(&idec.nz_luma),
                    l0: None,
                    l1: None,
                });
                for l in 0..2 {
                    left_motion[l] = nb_intra();
                    top_motion[l][mb_x] = nb_intra();
                }
                (left_modes, top_modes[mb_x]) = match idec.kind {
                    MbKind::I4x4 => (
                        [Some(modes[3]), Some(modes[7]), Some(modes[11]), Some(modes[15])],
                        [Some(modes[12]), Some(modes[13]), Some(modes[14]), Some(modes[15])],
                    ),
                    MbKind::I16x16 => ([Some(2); 4], [Some(2); 4]),
                };
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
                nz_mask: nz_mask_of(&dec.nz_luma),
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
