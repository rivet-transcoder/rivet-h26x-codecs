//! The CABAC picture writers: decisions from the shared walks into
//! arithmetic-coded slices.
//!
//! The macroblock-layer primitives live beside their readers in
//! `crate::h264::cabac_mb` — each `write_*_cabac` the exact inverse of a
//! `decode_*`, proven by round trips over the production parsers. This
//! module is the composition: the order the syntax elements go in, the
//! `WrittenMb` neighbour chain their contexts read, and the slice
//! envelope. It mirrors that file's own slice-level test harness
//! (`round_trip_slice`), which was written as the reference for exactly
//! this wiring.
//!
//! Three things shape a CABAC slice that CAVLC has no counterpart for:
//!
//! - **`mb_skip_flag` replaces `mb_skip_run`.** Every macroblock of a P
//!   slice codes the flag against a context counting its non-skipped
//!   neighbours; a skipped macroblock codes nothing else, and the
//!   `mb_qp_delta` carry clears exactly as the decoder's slice loop
//!   clears it — forget that and the *contexts* desync, which surfaces
//!   macroblocks later.
//! - **`end_of_slice_flag` after every macroblock.** A terminate bin of 0
//!   per macroblock, 1 after the last — and the 1 *flushes* the codeword,
//!   which is why the writers here never call `rbsp_trailing_bits`: the
//!   flush's final one is the stop bit, and what remains is zero padding
//!   to the byte.
//! - **One engine spans the slice.** Only I_PCM flushes mid-slice, and
//!   the transform pictures this module writes never code PCM, so a
//!   single `CabacEncoder` carries from the alignment bit to the final
//!   terminate.
//!
//! The slice header's `cabac_init_idc` is 0, matching the
//! `CabacState::new(_, 0, _)` here — the header writer and this module must
//! agree, and both spell the same constant.

use crate::bitwriter::BitWriter;
use crate::cabac_enc::CabacEncoder;
use crate::encode::h264_intra::{MbDecision, MbKind};
use crate::encode::h264_cavlc_mb::sub_mb_type_p;
use crate::encode::h264_me::{BDecision, BMbKind, InterDecision, InterMbKind};
use crate::encode::h264_pic::{
    BMb, CodedPair, Colocated, IntraTools, PMb, PairMb, PicMotion, code_b_picture, code_intra_picture, code_p_picture,
};
use crate::h264::mb::MbNeighbours;
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::SliceType;
use crate::h264::cabac_mb::{
    CabacState, WrittenMb, intra_mb_type_code, write_cbp_cabac, write_intra_pred_modes_cabac,
    write_intra_residual_cabac, write_inter_residual_cabac, write_inter_residual_fields_cabac,
    CurMbMvd, write_mb_field_cabac, write_mb_qp_delta_cabac, write_mb_skip_cabac, write_mb_type_b_cabac,
    write_ref_idx_16x16_cabac,
    write_mb_type_i_cabac, write_mb_type_p_cabac, write_mvd_cabac,
    write_sub_mb_type_b_cabac, write_sub_mb_type_p_cabac, write_transform_8x8_cabac,
};
use crate::h264::cavlc::part_index_of;
use crate::picture::ChromaFormat;
use crate::sample::Sample;

/// What one written macroblock leaves for its neighbours' contexts: the
/// [`WrittenMb`] the primitives read, plus the two facts that live outside
/// it — the same trio the reference harness keeps.
#[derive(Clone, Copy)]
struct Coded {
    nb: WrittenMb,
    /// `mb_type != I_NxN` (the I-slice `mb_type` first-bin context).
    not_nxn: bool,
    /// `intra_chroma_pred_mode != 0` (its context counts neighbours with a
    /// nonzero mode; an inter macroblock stores mode 0, so false).
    chroma_nonzero: bool,
}

/// `chroma_format_idc` as the residual and cbp writers take it.
fn cfi_of(chroma: ChromaFormat) -> u32 {
    match chroma {
        ChromaFormat::Monochrome => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// Everything after `mb_type` for an intra macroblock — shared by the
/// I-slice path and intra-in-P, exactly as the readers share it. `field`:
/// a field macroblock, whose residual takes the field scans and the
/// field-coded significance contexts.
#[allow(clippy::too_many_arguments)]
fn write_intra_body(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    d: &MbDecision,
    left: Option<&Coded>,
    above: Option<&Coded>,
    cfi: u32,
    t8x8_mode: bool,
    bit_depth: u32,
    field: bool,
) {
    let chroma = cfi == 1 || cfi == 2;
    let chroma_nb = chroma.then(|| {
        [
            left.is_some_and(|m| m.chroma_nonzero),
            above.is_some_and(|m| m.chroma_nonzero),
        ]
    });
    let lnb = left.map(|m| &m.nb);
    let anb = above.map(|m| &m.nb);
    // `transform_size_8x8_flag` comes *before* `mb_pred()` for I_NxN — it
    // is what decides whether four 8x8 modes follow or sixteen 4x4 ones —
    // and does not exist for I_16x16 at all (`parse_mb_cabac` reads it
    // under `layer.kind == MbKind::I4x4`, which is I_NxN before the flag
    // has renamed it). The inter placement is the other one, after the
    // coded block pattern.
    if t8x8_mode && d.kind.is_nxn() {
        write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
    }
    debug_assert!(t8x8_mode || !d.transform_8x8, "no PPS flag, no 8x8 transform");
    write_intra_pred_modes_cabac(e, st, d, chroma_nb);
    if d.kind.is_nxn() {
        write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), chroma);
    }
    let has_residual = d.kind == MbKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
    if has_residual {
        write_mb_qp_delta_cabac(e, st, d.qp_delta as i32, bit_depth);
        st.prev_qp_delta_nonzero = d.qp_delta != 0;
        write_intra_residual_cabac(e, st, field, cfi, d, lnb, anb);
    } else {
        st.prev_qp_delta_nonzero = false;
    }
}

/// One coded `P_L0_16x16` macroblock after its skip flag: mb_type, mvd
/// (no `ref_idx` — exactly one reference is active, so the element is
/// absent, as in the CAVLC writer), cbp, then qp_delta and residual when
/// any block coded.
#[allow(clippy::too_many_arguments)]
fn write_p16_body(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    d: &InterDecision,
    left: Option<&Coded>,
    above: Option<&Coded>,
    cfi: u32,
    t8x8_mode: bool,
    bit_depth: u32,
    field: bool,
    ref_idx_zeros: bool,
) {
    debug_assert!(
        !matches!(d.kind, InterMbKind::PSkip | InterMbKind::UseIntra),
        "only a coded P macroblock carries this syntax"
    );
    debug_assert_eq!(d.ref_idx, 0, "more than one reference needs ref_idx writing");
    write_mb_type_p_cabac(e, st, d.kind.p_mb_type());
    let lnb = left.map(|m| &m.nb);
    let anb = above.map(|m| &m.nb);
    // `P_8x8`'s four `sub_mb_type`s, all before any motion.
    if d.kind == InterMbKind::P8x8 {
        for part in 0..4 {
            write_sub_mb_type_p_cabac(e, st, sub_mb_type_p(d.sub_shape[part]));
        }
    }
    // `ref_idx_l0` is absent (one active reference) — except for a field
    // macroblock of an MBAFF frame, whose list is the frame list's fields,
    // two per frame, so the reader takes an index per partition
    // (`parse_mb_cabac`: `layer.field && !ctx.field_pic`). The index is 0,
    // and so is its first bin's context: refIdxZeroFlagN is 1 for every
    // neighbouring index this encoder writes (all 0), leaving no
    // condTermFlag set.
    if ref_idx_zeros {
        let parts = if d.kind == InterMbKind::P8x8 { 4 } else { d.kind.parts().len() };
        for _ in 0..parts {
            write_ref_idx_16x16_cabac(e, st, None, None, 0);
        }
    }
    // Then one mvd per
    // prediction rectangle, each recorded as it is written so the next
    // rectangle's context can read it — which is what the reader does
    // with `layer.mvd`, and the only reason a sub-partitioned macroblock
    // needs a context source inside itself at all.
    let mut cur = CurMbMvd::default();
    let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
    let n = d.rects(&mut rects);
    for &(x, y, w, h) in rects.iter().take(n) {
        let mvd = d.mvd[(y / 4) * 4 + x / 4];
        write_mvd_cabac(e, st, &cur, lnb, anb, 0, x / 4, y / 4, mvd);
        cur.set(0, x, y, w, h, mvd);
    }
    write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), cfi == 1 || cfi == 2);
    // An inter macroblock's flag comes after the coded block pattern,
    // only when some luma block is coded, and only when every
    // sub-macroblock partition is at least 8x8 (7.3.5).
    if t8x8_mode && d.cbp_luma != 0 && d.no_sub_mb_part_less_than_8x8() {
        write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
    }
    debug_assert!(
        !d.transform_8x8 || (t8x8_mode && d.cbp_luma != 0 && d.no_sub_mb_part_less_than_8x8())
    );
    if d.cbp_luma != 0 || d.cbp_chroma != 0 {
        write_mb_qp_delta_cabac(e, st, d.qp_delta as i32, bit_depth);
        st.prev_qp_delta_nonzero = d.qp_delta != 0;
        write_inter_residual_cabac(e, st, field, cfi, d, lnb, anb);
    } else {
        st.prev_qp_delta_nonzero = false;
    }
}

/// One coded B macroblock of any shape after its skip flag: `mb_type`
/// (with `inc` its first-bin increment), `B_8x8`'s four `sub_mb_type`s,
/// the mvds **list-major** — every explicit rectangle's `mvd_l0` in
/// syntax order, then every one's `mvd_l1`, each recorded as it is
/// written so a later rectangle's context can read it, exactly as the
/// reader stores `layer.mvd` — then cbp, the transform flag where 7.3.5
/// allows one, and qp_delta plus residual when any block coded. No
/// `ref_idx`: one reference per list. `B_Direct_16x16` and a
/// `B_Direct_8x8` sub-macroblock carry no motion syntax.
#[allow(clippy::too_many_arguments)]
fn write_b_body(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    d: &BDecision,
    inc: usize,
    left: Option<&Coded>,
    above: Option<&Coded>,
    cfi: u32,
    t8x8_mode: bool,
    bit_depth: u32,
    field: bool,
    ref_idx_zeros: bool,
) {
    debug_assert!(
        !matches!(d.kind, BMbKind::BSkip | BMbKind::UseIntra),
        "only a coded B macroblock carries this syntax"
    );
    debug_assert!(d.ref_idx.iter().flatten().all(|&r| r <= 0), "more than one reference needs ref_idx writing");
    let lnb = left.map(|m| &m.nb);
    let anb = above.map(|m| &m.nb);
    write_mb_type_b_cabac(e, st, inc, d.mb_type());
    if d.kind == BMbKind::B8x8 {
        for part in 0..4 {
            write_sub_mb_type_b_cabac(e, st, d.sub_mb_type(part));
        }
    }
    // An MBAFF field macroblock's reference indices, all 0 (see
    // `write_p16_body`): per list, per partition that uses the list and is
    // not direct, before any mvd.
    if ref_idx_zeros && d.kind != BMbKind::BDirect16 {
        for list in 0..2 {
            if d.kind == BMbKind::B8x8 {
                for part in 0..4 {
                    if !d.is_direct_part(part) && d.used(part)[list] {
                        write_ref_idx_16x16_cabac(e, st, None, None, 0);
                    }
                }
            } else {
                for &(x, y, _, _) in crate::h264::cavlc::mb_partitions(d.kind.dec_kind()) {
                    if d.used(part_index_of(x, y))[list] {
                        write_ref_idx_16x16_cabac(e, st, None, None, 0);
                    }
                }
            }
        }
    }
    if d.kind != BMbKind::BDirect16 {
        let mut cur = CurMbMvd::default();
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = d.rects(&mut rects);
        for list in 0..2 {
            for &(x, y, w, h) in rects.iter().take(n) {
                let part = part_index_of(x, y);
                if d.is_direct_part(part) || !d.used(part)[list] {
                    continue;
                }
                let mvd = d.mvd[list][(y / 4) * 4 + x / 4];
                write_mvd_cabac(e, st, &cur, lnb, anb, list, x / 4, y / 4, mvd);
                cur.set(list, x, y, w, h, mvd);
            }
        }
    }
    write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), cfi == 1 || cfi == 2);
    // After the coded block pattern, only when some luma block is coded
    // and every sub-macroblock partition is at least 8x8 —
    // `B_Direct_16x16` and a direct sub-macroblock count as 8x8, because
    // the SPS this encoder writes sets `direct_8x8_inference_flag`.
    if t8x8_mode && d.cbp_luma != 0 && d.no_sub_mb_part_less_than_8x8() {
        write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
    }
    debug_assert!(
        !d.transform_8x8 || (t8x8_mode && d.cbp_luma != 0 && d.no_sub_mb_part_less_than_8x8())
    );
    if d.cbp_luma != 0 || d.cbp_chroma != 0 {
        write_mb_qp_delta_cabac(e, st, d.qp_delta as i32, bit_depth);
        st.prev_qp_delta_nonzero = d.qp_delta != 0;
        write_inter_residual_fields_cabac(
            e, st, field, cfi, d.transform_8x8, d.cbp_luma, &d.nz_luma, &d.luma, d.cbp_chroma,
            &d.chroma_dc, &d.chroma_ac, &d.nz_chroma, lnb, anb,
        );
    } else {
        st.prev_qp_delta_nonzero = false;
    }
}

/// Write every macroblock of an all-intra CABAC picture: the shared walk
/// makes the decisions, reconstructs and runs the loop filter; this side
/// spells bins and keeps the `WrittenMb` chain. The slice header is
/// already written; the final terminate closes the RBSP (no
/// `rbsp_trailing_bits` — see the module docs).
pub fn write_intra_picture_cabac<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
) -> PicMotion {
    let mbw = g.mbs_wide as usize;
    let total = mbw * g.mbs_high as usize;
    let cfi = cfi_of(g.chroma);
    let t8x8 = tools.transform_8x8;
    w.align_one(); // cabac_alignment_one_bit
    let mut st = CabacState::new(SliceType::I, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    let mut coded: Vec<Coded> = Vec::with_capacity(total);
    let fmbs = code_intra_picture(g, tools, qp, planes, rec, |mb_x, mb_y, dec| {
        let idx = coded.len();
        let left = (mb_x > 0).then(|| &coded[idx - 1]);
        let above = (mb_y > 0).then(|| &coded[idx - mbw]);
        let inc = left.map_or(0, |m| m.not_nxn as usize) + above.map_or(0, |m| m.not_nxn as usize);
        write_mb_type_i_cabac(&mut e, &mut st, inc, intra_mb_type_code(dec));
        write_intra_body(&mut e, &mut st, dec, left, above, cfi, t8x8, g.bit_depth, g.field_pic);
        coded.push(Coded {
            nb: WrittenMb::from_decision(dec, cfi == 3),
            not_nxn: !dec.kind.is_nxn(),
            chroma_nonzero: dec.chroma_mode != 0,
        });
        e.encode_terminate((coded.len() == total) as u32); // end_of_slice_flag
    });
    drop(e);
    w.align_zero();
    fmbs
}

/// Write every macroblock of a P CABAC picture: the shared walk owns the
/// motion search, the skip decision and the intra fallback; this side
/// codes `mb_skip_flag` per macroblock, the macroblock layers, and the
/// `end_of_slice_flag`s. The slice header (with `cabac_init_idc` 0) is
/// already written; the final terminate closes the RBSP.
#[allow(clippy::too_many_arguments)]
pub fn write_p_picture_cabac<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refp: &[Recon<S>],
    weights: Option<&crate::h264::slice::PredWeightTable>,
) -> PicMotion {
    let mbw = g.mbs_wide as usize;
    let total = mbw * g.mbs_high as usize;
    let cfi = cfi_of(g.chroma);
    let t8x8 = tools.transform_8x8;
    w.align_one();
    let mut st = CabacState::new(SliceType::P, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    let mut coded: Vec<Coded> = Vec::with_capacity(total);
    let fmbs = code_p_picture(g, tools, qp, planes, rec, refp, weights, |mb_x, mb_y, mb| {
        let idx = coded.len();
        let left = (mb_x > 0).then(|| &coded[idx - 1]);
        let above = (mb_y > 0).then(|| &coded[idx - mbw]);
        let lnb = left.map(|m| &m.nb);
        let anb = above.map(|m| &m.nb);
        // Every macroblock of a P slice codes mb_skip_flag, whatever it
        // turns out to be.
        write_mb_skip_cabac(&mut e, &mut st, lnb, anb, false, matches!(mb, PMb::Skip(_)));
        let entry = match mb {
            PMb::Skip(dec) => {
                // A skip wrote its flag and writes nothing else; the
                // qp-delta carry clears, as the decoder's slice loop
                // clears it when it takes the skip branch.
                st.prev_qp_delta_nonzero = false;
                Coded {
                    nb: WrittenMb::from_inter_decision(dec, cfi == 3),
                    not_nxn: true,
                    chroma_nonzero: false,
                }
            }
            PMb::Coded(dec) => {
                write_p16_body(&mut e, &mut st, dec, left, above, cfi, t8x8, g.bit_depth, g.field_pic, false);
                Coded {
                    nb: WrittenMb::from_inter_decision(dec, cfi == 3),
                    not_nxn: true,
                    chroma_nonzero: false,
                }
            }
            PMb::Intra(idec) => {
                // Intra in a P slice: the same macroblock, `mb_type`
                // shifted by 5 (Table 7-11's note).
                write_mb_type_p_cabac(&mut e, &mut st, 5 + intra_mb_type_code(idec));
                write_intra_body(&mut e, &mut st, idec, left, above, cfi, t8x8, g.bit_depth, g.field_pic);
                Coded {
                    nb: WrittenMb::from_decision(idec, cfi == 3),
                    not_nxn: !idec.kind.is_nxn(),
                    chroma_nonzero: idec.chroma_mode != 0,
                }
            }
        };
        coded.push(entry);
        e.encode_terminate((coded.len() == total) as u32); // end_of_slice_flag
    });
    drop(e);
    w.align_zero();
    fmbs
}

/// Write every macroblock of a B CABAC picture: the shared walk
/// (`h264_pic::code_b_picture`) owns the searches, the direct derivation and the
/// intra fallback; this side codes `mb_skip_flag` per macroblock against
/// the B contexts, the macroblock layers (`B_Direct_16x16` is `mb_type` 0
/// and no motion syntax; explicit 16x16 carries one mvd per used list;
/// intra rides behind the B prefix at +23), and the `end_of_slice_flag`s.
/// Returns the picture's motion record for the caller's reference
/// bookkeeping.
#[allow(clippy::too_many_arguments)]
pub fn write_b_picture_cabac<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools<S>,
    qp: u8,
    planes: &[Plane<'_, S>],
    rec: &mut [Recon<S>],
    refs: [&[Recon<S>]; 2],
    col: &Colocated,
    weights: Option<&crate::h264::slice::PredWeightTable>,
) -> PicMotion {
    let mbw = g.mbs_wide as usize;
    let total = mbw * g.mbs_high as usize;
    let cfi = cfi_of(g.chroma);
    let t8x8 = tools.transform_8x8;
    w.align_one();
    let mut st = CabacState::new(SliceType::B, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    let mut coded: Vec<Coded> = Vec::with_capacity(total);
    let fmbs = code_b_picture(g, tools, qp, planes, rec, refs, col, weights, |mb_x, mb_y, mb| {
        let idx = coded.len();
        let left = (mb_x > 0).then(|| &coded[idx - 1]);
        let above = (mb_y > 0).then(|| &coded[idx - mbw]);
        let lnb = left.map(|m| &m.nb);
        let anb = above.map(|m| &m.nb);
        write_mb_skip_cabac(&mut e, &mut st, lnb, anb, true, matches!(mb, BMb::Skip(_)));
        // The B `mb_type` first-bin context counts available neighbours
        // that are neither B_Skip nor B_Direct_16x16.
        let cond = |m: Option<&Coded>| -> usize {
            m.map_or(0, |m| !(m.nb.skip || m.nb.direct) as usize)
        };
        let inc = cond(left) + cond(above);
        let entry = match mb {
            BMb::Skip(dec) => {
                st.prev_qp_delta_nonzero = false;
                Coded {
                    nb: WrittenMb::from_b_decision(dec, cfi == 3),
                    not_nxn: true,
                    chroma_nonzero: false,
                }
            }
            BMb::Direct(dec) | BMb::Explicit(dec) => {
                write_b_body(&mut e, &mut st, dec, inc, left, above, cfi, t8x8, g.bit_depth, g.field_pic, false);
                Coded {
                    nb: WrittenMb::from_b_decision(dec, cfi == 3),
                    not_nxn: true,
                    chroma_nonzero: false,
                }
            }
            BMb::Intra(idec) => {
                // Intra in a B slice: the same macroblock behind the B
                // prefix, `mb_type` shifted by 23.
                write_mb_type_b_cabac(&mut e, &mut st, inc, 23 + intra_mb_type_code(idec));
                write_intra_body(&mut e, &mut st, idec, left, above, cfi, t8x8, g.bit_depth, g.field_pic);
                Coded {
                    nb: WrittenMb::from_decision(idec, cfi == 3),
                    not_nxn: !idec.kind.is_nxn(),
                    chroma_nonzero: idec.chroma_mode != 0,
                }
            }
        };
        coded.push(entry);
        e.encode_terminate((coded.len() == total) as u32); // end_of_slice_flag
    });
    drop(e);
    w.align_zero();
    fmbs
}

/// What an MBAFF CABAC slice's macroblocks are written under.
#[derive(Clone, Copy)]
struct MbaffParams {
    slice: SliceType,
    cfi: u32,
    t8x8: bool,
    bit_depth: u32,
    /// Chroma AC block rows per macroblock: 2 (4:2:0), 4 (4:2:2), 0.
    rows: usize,
}

/// The written record of storage address `a`: one of the two macroblocks
/// of the pair being written (`local`, top then bottom), or one written
/// before it.
fn mbaff_lookup(base: &[Option<Coded>], local: &[Option<Coded>; 2], top: usize, bot: usize, a: usize) -> Coded {
    (if a == top {
        local[0]
    } else if a == bot {
        local[1]
    } else {
        base[a]
    })
    .expect("an MBAFF neighbour is written before it is read")
}

/// The left and above neighbours an MBAFF macroblock's contexts read, as
/// the one record per side the progressive primitives take.
///
/// In an MBAFF frame a macroblock's neighbouring *blocks* are not one
/// macroblock's edge: beside a pair of the other kind, the left column of a
/// frame macroblock alternates between the two field macroblocks, and a
/// field macroblock's reads the frame pair's two halves (Table 6-4). The
/// contexts ask two kinds of question, though. The macroblock-level ones —
/// skip, `mb_type`, the transform flag, the chroma mode, the DC blocks'
/// coded_block_flags — read the macroblocks the reader calls A and B, and
/// take those records as they are. The block-level ones read one entry of
/// the neighbour's edge per row or column: nonzero counts (already gated
/// by its coded block pattern), the coded block pattern bit of an 8x8, and
/// the mvd. So each side is that macroblock's record with its edge entries
/// replaced, row by row and column by column, by the entries of the blocks
/// the decoder's own `MbNeighbours::block` / `block_c` name — the mvds
/// scaled across a frame / field boundary as 9.3.3.1.1.7 scales them, and
/// the macroblock-level skip and intra flags cleared, their effect already
/// in the entries (a skipped or intra block's mvd and a skipped
/// macroblock's pattern bits are zero, which is what those flags stood
/// for).
fn mbaff_neighbours(
    nb: &MbNeighbours,
    info: &crate::h264::mb::PicInfo,
    base: &[Option<Coded>],
    local: &[Option<Coded>; 2],
    top: usize,
    bot: usize,
    rows: usize,
) -> (Option<Coded>, Option<Coded>) {
    let get = |a: usize| mbaff_lookup(base, local, top, bot, a);
    let scaled = |a: usize, mv: crate::h264::frame::Mv| -> crate::h264::frame::Mv {
        let nf = info.mbs[a].field;
        let y = i32::from(mv.y).abs();
        let y = if !nb.cur_field && nf {
            y * 2
        } else if nb.cur_field && !nf {
            y / 2
        } else {
            y
        };
        crate::h264::frame::Mv::new(mv.x.saturating_abs(), y.min(i32::from(i16::MAX)) as i16)
    };
    let left = nb.a.map(|a| {
        let mut v = get(a);
        v.nb.skip = false;
        v.nb.intra = false;
        v.nb.cbp &= !0x0a;
        for r in 0..4 {
            let (ma, blk) = nb.block(-1, r as i32).expect("an available pair is whole");
            let m = get(ma).nb;
            v.nb.nz_luma[r * 4 + 3] = m.nz_luma[blk];
            for l in 0..2 {
                v.nb.mvd[l][r * 4 + 3] = scaled(ma, m.mvd[l][blk]);
            }
        }
        for r8 in 0..2usize {
            let (ma, blk) = nb.block(-1, 2 * r8 as i32).expect("an available pair is whole");
            let b8 = (blk / 8) * 2 + (blk % 4) / 2;
            v.nb.cbp |= ((get(ma).nb.cbp >> b8) & 1) << (2 * r8 + 1);
        }
        for r in 0..rows {
            let (ma, cblk) = nb.block_c(-1, r as i32, rows as i32).expect("an available pair is whole");
            let m = get(ma).nb;
            for comp in 0..2 {
                v.nb.nz_chroma[comp][r * 2 + 1] = m.nz_chroma[comp][cblk];
            }
        }
        v
    });
    let above = nb.b.map(|b| {
        let mut v = get(b);
        v.nb.skip = false;
        v.nb.intra = false;
        v.nb.cbp &= !0x0c;
        for c in 0..4 {
            let (ma, blk) = nb.block(c as i32, -1).expect("an available pair is whole");
            let m = get(ma).nb;
            v.nb.nz_luma[12 + c] = m.nz_luma[blk];
            for l in 0..2 {
                v.nb.mvd[l][12 + c] = scaled(ma, m.mvd[l][blk]);
            }
        }
        for c8 in 0..2usize {
            let (ma, blk) = nb.block(2 * c8 as i32, -1).expect("an available pair is whole");
            let b8 = (blk / 8) * 2 + (blk % 4) / 2;
            v.nb.cbp |= ((get(ma).nb.cbp >> b8) & 1) << (2 + c8);
        }
        if rows > 0 {
            for c in 0..2 {
                let (ma, cblk) = nb.block_c(c as i32, -1, rows as i32).expect("an available pair is whole");
                let m = get(ma).nb;
                for comp in 0..2 {
                    v.nb.nz_chroma[comp][(rows - 1) * 2 + c] = m.nz_chroma[comp][cblk];
                }
            }
        }
        v
    });
    (left, above)
}

/// The record a skipped macroblock of a pair leaves.
fn mbaff_skip_record(mb: &PairMb, cfi: u32) -> Coded {
    let nb = match mb {
        PairMb::P(d) => WrittenMb::from_inter_decision(d, cfi == 3),
        PairMb::B(d) => WrittenMb::from_b_decision(d, cfi == 3),
        _ => unreachable!("only an inter macroblock skips"),
    };
    Coded { nb, not_nxn: true, chroma_nonzero: false }
}

/// One coded macroblock of an MBAFF pair — `mb_type` through its residual
/// — its contexts read through the decoder's MBAFF neighbours `nb`
/// ([`mbaff_neighbours`]) and a field macroblock's reference indices
/// written. Returns its record.
#[allow(clippy::too_many_arguments)]
fn write_mbaff_mb(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    nb: &MbNeighbours,
    info: &crate::h264::mb::PicInfo,
    base: &[Option<Coded>],
    local: &[Option<Coded>; 2],
    top: usize,
    bot: usize,
    mb: &PairMb,
    field: bool,
    p: &MbaffParams,
) -> Coded {
    let real = |a: Option<usize>| a.map(|a| mbaff_lookup(base, local, top, bot, a));
    let (ra, rb) = (real(nb.a), real(nb.b));
    let (left, above) = mbaff_neighbours(nb, info, base, local, top, bot, p.rows);
    let (l, a) = (left.as_ref(), above.as_ref());
    let b_inc = |m: Option<Coded>| m.map_or(0, |m| !(m.nb.skip || m.nb.direct) as usize);
    match mb {
        PairMb::Intra(d) => {
            let inc = ra.map_or(0, |m| m.not_nxn as usize) + rb.map_or(0, |m| m.not_nxn as usize);
            write_mb_type_i_cabac(e, st, inc, intra_mb_type_code(d));
            write_intra_body(e, st, d, l, a, p.cfi, p.t8x8, p.bit_depth, field);
            Coded { nb: WrittenMb::from_decision(d, p.cfi == 3), not_nxn: !d.kind.is_nxn(), chroma_nonzero: d.chroma_mode != 0 }
        }
        PairMb::PIntra(d) => {
            write_mb_type_p_cabac(e, st, 5 + intra_mb_type_code(d));
            write_intra_body(e, st, d, l, a, p.cfi, p.t8x8, p.bit_depth, field);
            Coded { nb: WrittenMb::from_decision(d, p.cfi == 3), not_nxn: !d.kind.is_nxn(), chroma_nonzero: d.chroma_mode != 0 }
        }
        PairMb::BIntra(d) => {
            write_mb_type_b_cabac(e, st, b_inc(ra) + b_inc(rb), 23 + intra_mb_type_code(d));
            write_intra_body(e, st, d, l, a, p.cfi, p.t8x8, p.bit_depth, field);
            Coded { nb: WrittenMb::from_decision(d, p.cfi == 3), not_nxn: !d.kind.is_nxn(), chroma_nonzero: d.chroma_mode != 0 }
        }
        PairMb::P(d) => {
            write_p16_body(e, st, d, l, a, p.cfi, p.t8x8, p.bit_depth, field, field);
            Coded { nb: WrittenMb::from_inter_decision(d, p.cfi == 3), not_nxn: true, chroma_nonzero: false }
        }
        PairMb::B(d) => {
            write_b_body(e, st, d, b_inc(ra) + b_inc(rb), l, a, p.cfi, p.t8x8, p.bit_depth, field, field);
            Coded { nb: WrittenMb::from_b_decision(d, p.cfi == 3), not_nxn: true, chroma_nonzero: false }
        }
    }
}

/// Write one MBAFF pair: the decoder's CABAC MBAFF slice loop
/// (src/h264/decoder.rs) mirrored element for element, returning the two
/// macroblocks' records.
///
/// The order is the loop's, and so are the neighbours each element's
/// context is derived under. A top macroblock's `mb_skip_flag` is read
/// before its pair's flag exists, so its neighbours come from the flag the
/// decoder infers (`infer_mb_field`). A skipped top macroblock makes the
/// decoder peek the bottom's skip flag straight away — under the inferred
/// flag too — and when the bottom is coded its `mb_field_decoding_flag`
/// follows that peek; otherwise the flag comes after the top's skip flag.
/// There is no `end_of_slice_flag` between a pair's macroblocks, and a pair
/// of skips has no flag at all (the walk only produces one whose flag is
/// the inferred one).
fn write_pair_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    base: &[Option<Coded>],
    pair: &CodedPair,
    pm: &PicMotion,
    p: &MbaffParams,
    last: bool,
) -> [Coded; 2] {
    let info = &pm.info;
    let mbw = info.mb_width;
    let (top, bot) = (pair.top, pair.top + mbw);
    let mut local: [Option<Coded>; 2] = [None, None];
    let skip = [pair.mbs[0].is_skip(), pair.mbs[1].is_skip()];
    let is_b = p.slice.is_b();
    let rec = |local: &[Option<Coded>; 2], a: Option<usize>| a.map(|a| mbaff_lookup(base, local, top, bot, a).nb);
    let inferred = crate::h264::decoder::infer_mb_field(info, top, 0);
    let mut nb = MbNeighbours::default();
    nb.derive_mbaff_into(info, top, 0, inferred);
    if !p.slice.is_intra() {
        let (la, lb) = (rec(&local, nb.a), rec(&local, nb.b));
        write_mb_skip_cabac(e, st, la.as_ref(), lb.as_ref(), is_b, skip[0]);
        if skip[0] {
            st.prev_qp_delta_nonzero = false;
            local[0] = Some(mbaff_skip_record(&pair.mbs[0], p.cfi));
            let mut nbb = MbNeighbours::default();
            nbb.derive_mbaff_into(info, bot, 0, inferred);
            let (la, lb) = (rec(&local, nbb.a), rec(&local, nbb.b));
            write_mb_skip_cabac(e, st, la.as_ref(), lb.as_ref(), is_b, skip[1]);
            if skip[1] {
                debug_assert_eq!(pair.field, inferred, "a pair of skips carries no flag");
                st.prev_qp_delta_nonzero = false;
                local[1] = Some(mbaff_skip_record(&pair.mbs[1], p.cfi));
            } else {
                write_mb_field_cabac(e, st, &nbb, pair.field);
                nb.derive_mbaff_into(info, bot, 0, pair.field);
                local[1] = Some(write_mbaff_mb(e, st, &nb, info, base, &local, top, bot, &pair.mbs[1], pair.field, p));
            }
            e.encode_terminate(last as u32);
            return [local[0].expect("written"), local[1].expect("written")];
        }
    }
    write_mb_field_cabac(e, st, &nb, pair.field);
    nb.derive_mbaff_into(info, top, 0, pair.field);
    local[0] = Some(write_mbaff_mb(e, st, &nb, info, base, &local, top, bot, &pair.mbs[0], pair.field, p));
    nb.derive_mbaff_into(info, bot, 0, pair.field);
    if !p.slice.is_intra() {
        let (la, lb) = (rec(&local, nb.a), rec(&local, nb.b));
        write_mb_skip_cabac(e, st, la.as_ref(), lb.as_ref(), is_b, skip[1]);
    }
    local[1] = Some(if skip[1] {
        st.prev_qp_delta_nonzero = false;
        mbaff_skip_record(&pair.mbs[1], p.cfi)
    } else {
        write_mbaff_mb(e, st, &nb, info, base, &local, top, bot, &pair.mbs[1], pair.field, p)
    });
    e.encode_terminate(last as u32);
    [local[0].expect("written"), local[1].expect("written")]
}

/// The CABAC slice data of an MBAFF picture, written pair by pair as the
/// walk ([`crate::encode::h264_pic`]'s `code_mbaff_picture`) decides each —
/// one engine from `cabac_alignment_one_bit` to the last pair's terminate,
/// every macroblock's record kept by storage address for the contexts of
/// the macroblocks after it. The caller pads the RBSP to a byte after
/// dropping it.
pub(crate) struct MbaffCabac<'w> {
    e: CabacEncoder<'w>,
    st: CabacState,
    recs: Vec<Option<Coded>>,
    p: MbaffParams,
    pairs: usize,
    written: usize,
}

impl<'w> MbaffCabac<'w> {
    /// Begin the slice data of an MBAFF picture of `slice` type at slice
    /// quantiser `qp`, after its header.
    pub(crate) fn new<S: Sample>(w: &'w mut BitWriter, g: &Geometry, tools: &IntraTools<S>, qp: u8, slice: SliceType) -> Self {
        w.align_one();
        let rows = match g.chroma {
            ChromaFormat::Yuv420 => 2,
            ChromaFormat::Yuv422 => 4,
            _ => 0,
        };
        let n = (g.mbs_wide * g.mbs_high) as usize;
        MbaffCabac {
            e: CabacEncoder::new(w),
            st: CabacState::new(slice, 0, qp as i32),
            recs: vec![None; n],
            p: MbaffParams { slice, cfi: cfi_of(g.chroma), t8x8: tools.transform_8x8, bit_depth: g.bit_depth, rows },
            pairs: n / 2,
            written: 0,
        }
    }
}

impl crate::encode::h264_pic::PairWriter for MbaffCabac<'_> {
    fn trial_bits(&self, pair: &CodedPair, pm: &PicMotion) -> u64 {
        let mut e = CabacEncoder::counting();
        let mut st = self.st.clone();
        let _ = write_pair_cabac(&mut e, &mut st, &self.recs, pair, pm, &self.p, false);
        e.bits_counted()
    }

    fn write_pair(&mut self, pair: &CodedPair, pm: &PicMotion) {
        self.written += 1;
        let last = self.written == self.pairs;
        let recs = write_pair_cabac(&mut self.e, &mut self.st, &self.recs, pair, pm, &self.p, last);
        let mbw = pm.info.mb_width;
        self.recs[pair.top] = Some(recs[0]);
        self.recs[pair.top + mbw] = Some(recs[1]);
    }
}

/// The slice data of an all-skip CABAC inter picture, P or B: one
/// `mb_skip_flag` of 1 and one `end_of_slice_flag` per macroblock,
/// nothing else — the arithmetic-coded spelling of what the CAVLC path
/// says with a single `mb_skip_run`. The reconstruction is the caller's
/// (the reference copied, or the two references averaged for B), exactly
/// as it is for the CAVLC all-skip path.
pub fn write_skip_picture_cabac(w: &mut BitWriter, g: &Geometry, qp: u8, is_b: bool) {
    let total = (g.mbs_wide * g.mbs_high) as usize;
    let mbw = g.mbs_wide as usize;
    w.align_one();
    let slice_type = if is_b { SliceType::B } else { SliceType::P };
    let mut st = CabacState::new(slice_type, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    // The skip-flag context counts *non-skipped* available neighbours, so
    // in an all-skip picture every increment is zero — but the chain is
    // kept anyway, because a shortcut here would be a second spelling of
    // the rule. A skipped B macroblock leaves the same state a skipped P
    // one does: the contexts only ever read `skip`.
    let skipped = WrittenMb::from_inter_decision(
        &InterDecision { kind: InterMbKind::PSkip, ..InterDecision::default() },
        false,
    );
    for idx in 0..total {
        let left = (idx % mbw > 0).then_some(&skipped);
        let above = (idx >= mbw).then_some(&skipped);
        write_mb_skip_cabac(&mut e, &mut st, left, above, is_b, true);
        st.prev_qp_delta_nonzero = false;
        e.encode_terminate((idx + 1 == total) as u32); // end_of_slice_flag
    }
    drop(e);
    w.align_zero();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::Cabac;
    use crate::encode::h264_me::test_b_decision;
    use crate::h264::cabac_mb::{decode_end_of_slice, decode_mb_skip, parse_mb_cabac};
    use crate::h264::frame::BlockMotion;
    use crate::h264::mb::{
        MbKind as DecKind, MbLayer, MbNeighbours, PRED_BI, PRED_L0, PRED_L1, PicInfo, SliceCtx,
        SubMbShape,
    };
    use crate::h264::recon::QpState;
    use crate::h264::sps::ScalingLists;
    use crate::h264::transform::Dequant;

    /// The decoder's per-macroblock bookkeeping (`derive`, src/h264/recon.rs)
    /// for everything the B contexts read back from a neighbour: the
    /// kind, which 8x8s of a `B_8x8` are direct, the cbp and counts, and
    /// the per-list mvds.
    fn commit(info: &mut PicInfo, addr: usize, layer: &MbLayer) {
        let m = &mut info.mbs[addr];
        m.kind = layer.kind;
        m.slice = 0;
        m.decoded = true;
        m.cbp = layer.cbp;
        m.transform_8x8 = layer.transform_8x8;
        m.qp_delta_nonzero = layer.has_residual() && layer.qp_delta != 0;
        m.dc_cbf = layer.dc_cbf;
        m.sub_direct = if layer.kind == DecKind::Inter8x8 {
            (0..4).map(|p| ((layer.sub_shape[p] == SubMbShape::Direct) as u8) << p).sum()
        } else {
            0
        };
        let base = addr * 16;
        info.luma_nz[base..base + 16].copy_from_slice(&layer.nz[0]);
        for comp in 0..2 {
            info.chroma_nz[addr * 32 + comp * 16..addr * 32 + comp * 16 + 8]
                .copy_from_slice(&layer.chroma_nz[comp]);
        }
        info.intra_modes[base..base + 16].fill(2);
        for l in 0..2 {
            for (dst, ent) in info.mvd[l][base..base + 16].iter_mut().zip(&layer.mvd) {
                *dst = ent.mvd[l];
            }
        }
    }

    /// A 2x2 B slice of four differently partitioned macroblocks — a
    /// 16x8, a `B_8x8` mixing direct with every direction, a
    /// `B_Direct_16x16`, an 8x16 — written by [`write_b_body`] over the
    /// `WrittenMb` chain and read back by the production `parse_mb_cabac`
    /// over the decoder's own neighbour machinery, with the whole context
    /// array compared at the end.
    ///
    /// Four macroblocks rather than one because the mvd contexts of a
    /// partition read the blocks left of and above it, and for every
    /// shape below 16x16 at least one of those is *inside* the
    /// macroblock — the `CurMbMvd` half of the writer — while the others
    /// are across an edge from a neighbour whose blocks are direct,
    /// list-0 only, or bi. A single macroblock exercises neither.
    #[test]
    fn b_partition_shapes_round_trip_through_the_cabac_reader() {
        let mbs = [
            test_b_decision(
                BMbKind::B16x8,
                [PRED_L0, PRED_L0, PRED_BI, PRED_BI],
                [SubMbShape::S8x8; 4],
                3,
            ),
            test_b_decision(
                BMbKind::B8x8,
                [PRED_BI, PRED_L1, PRED_BI, PRED_L0],
                [SubMbShape::Direct, SubMbShape::S8x8, SubMbShape::S4x4, SubMbShape::S8x4],
                5,
            ),
            test_b_decision(BMbKind::BDirect16, [PRED_BI; 4], [SubMbShape::S8x8; 4], 1),
            test_b_decision(
                BMbKind::B8x16,
                [PRED_BI, PRED_L1, PRED_BI, PRED_L1],
                [SubMbShape::S8x8; 4],
                4,
            ),
        ];
        let (mbw, total) = (2usize, 4usize);
        let cfi = 1;

        // ---- write ----
        let mut w = BitWriter::new();
        w.align_one();
        let mut enc_st = CabacState::new(SliceType::B, 0, 30);
        let mut coded: Vec<Coded> = Vec::new();
        {
            let mut e = CabacEncoder::new(&mut w);
            for (i, d) in mbs.iter().enumerate() {
                let left = (i % mbw > 0).then(|| &coded[i - 1]);
                let above = (i >= mbw).then(|| &coded[i - mbw]);
                write_mb_skip_cabac(
                    &mut e,
                    &mut enc_st,
                    left.map(|m| &m.nb),
                    above.map(|m| &m.nb),
                    true,
                    false,
                );
                let cond =
                    |m: Option<&Coded>| m.map_or(0, |m| !(m.nb.skip || m.nb.direct) as usize);
                let inc = cond(left) + cond(above);
                write_b_body(&mut e, &mut enc_st, d, inc, left, above, cfi, false, 8, false, false);
                coded.push(Coded {
                    nb: WrittenMb::from_b_decision(d, false),
                    not_nxn: true,
                    chroma_nonzero: false,
                });
                e.encode_terminate((i + 1 == total) as u32);
            }
        }
        w.align_zero();
        let data = w.into_rbsp();

        // ---- read back ----
        let ctx = SliceCtx {
            slice_type: SliceType::B,
            slice_num: 0,
            num_ref_idx: [1, 1],
            direct_spatial: true,
            transform_8x8_mode: false,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc: cfi,
            cabac: true,
            bit_depth: 8,
            transform_bypass: false,
            scaling_plane: 0,
            x264_old_444: false,
            field_pic: false,
            mbaff: false,
            sp: false,
            sp_switch: false,
            sp_qs: 0,
            sp_qsc: [0; 2],
        };
        let lists = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let dq = Dequant::new(&lists);
        let mut qps = QpState { prev_qp: 30, chroma_offset: [0, 0] };
        let mut dec_st = CabacState::new(SliceType::B, 0, 30);
        let mut c = Cabac::new(&data);
        let mut info = PicInfo::new(mbw, total / mbw);
        let mut layer = MbLayer::new(DecKind::I4x4);
        let mut nb = MbNeighbours::default();
        let frame_motion: [Vec<BlockMotion>; 2] = [
            vec![BlockMotion::default(); total * 16],
            vec![BlockMotion::default(); total * 16],
        ];
        for (addr, d) in mbs.iter().enumerate() {
            nb.derive_into(&info, addr, 0);
            nb.gather_nz(&info, 1, 2);
            assert!(!decode_mb_skip(&mut c, &mut dec_st, &info, &nb, true), "mb {addr} skip");
            parse_mb_cabac(
                &mut c,
                &mut dec_st,
                &ctx,
                &info,
                &nb,
                &frame_motion,
                &mut layer,
                &dq,
                &mut qps,
            )
            .unwrap_or_else(|e| panic!("mb {addr}: the reader rejected the writer's bins: {e}"));
            assert_eq!(layer.kind, d.kind.dec_kind(), "mb {addr} kind");
            for part in 0..4 {
                if d.is_direct_part(part) {
                    if d.kind == BMbKind::B8x8 {
                        assert_eq!(
                            layer.sub_shape[part],
                            SubMbShape::Direct,
                            "mb {addr} part {part}"
                        );
                    }
                    continue;
                }
                assert_eq!(layer.pred_dir[part], d.dir[part], "mb {addr} part {part} direction");
                if d.kind == BMbKind::B8x8 {
                    assert_eq!(
                        layer.sub_shape[part],
                        d.sub_shape[part],
                        "mb {addr} part {part} shape"
                    );
                }
            }
            // The CABAC reader stores every mvd over its whole
            // rectangle, exactly as the decision does.
            for blk in 0..16 {
                for l in 0..2 {
                    assert_eq!(
                        layer.mvd[blk].mvd[l],
                        d.mvd[l][blk],
                        "mb {addr} block {blk} list {l} mvd"
                    );
                }
            }
            assert_eq!(layer.cbp, d.cbp_luma | (d.cbp_chroma << 4), "mb {addr} cbp");
            for blk in 0..16 {
                assert_eq!(layer.nz[0][blk], d.nz_luma[blk], "mb {addr} luma nz {blk}");
            }
            // The writer's own record of this macroblock against what
            // the decoder stores of it.
            let wm = &coded[addr].nb;
            for blk in 0..16 {
                assert_eq!(wm.mvd[0][blk], layer.mvd[blk].mvd[0], "mb {addr} WrittenMb l0 {blk}");
                assert_eq!(wm.mvd[1][blk], layer.mvd[blk].mvd[1], "mb {addr} WrittenMb l1 {blk}");
            }
            commit(&mut info, addr, &layer);
            assert_eq!(
                decode_end_of_slice(&mut c),
                addr + 1 == total,
                "end_of_slice after mb {addr}"
            );
        }
        assert!(!c.overrun(), "the reader ran past what the writer produced");
        assert_eq!(enc_st.ctx, dec_st.ctx, "context states diverged");
    }
}
