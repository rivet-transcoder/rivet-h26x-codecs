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
use crate::encode::h264_me::{InterDecision, InterMbKind};
use crate::encode::h264_pic::{
    BMb, IntraTools, PMb, PicMotion, code_b_picture, code_intra_picture, code_p_picture,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::SliceType;
use crate::h264::cabac_mb::{
    CabacState, WrittenMb, intra_mb_type_code, write_cbp_cabac, write_intra_pred_modes_cabac,
    write_intra_residual_cabac, write_inter_residual_cabac, write_inter_residual_fields_cabac,
    write_mb_qp_delta_cabac, write_mb_skip_cabac, write_mb_type_b_cabac, write_mb_type_i_cabac,
    write_mb_type_p_cabac, write_mvd_16x16_cabac, write_transform_8x8_cabac,
};
use crate::picture::ChromaFormat;

/// What one written macroblock leaves for its neighbours' contexts: the
/// [`WrittenMb`] the primitives read, plus the two facts that live outside
/// it — the same trio the reference harness keeps.
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
/// I-slice path and intra-in-P, exactly as the readers share it.
fn write_intra_body(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    d: &MbDecision,
    left: Option<&Coded>,
    above: Option<&Coded>,
    cfi: u32,
    t8x8_mode: bool,
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
        write_mb_qp_delta_cabac(e, st, d.qp_delta as i32);
        st.prev_qp_delta_nonzero = d.qp_delta != 0;
        write_intra_residual_cabac(e, st, false, cfi, d, lnb, anb);
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
) {
    debug_assert_eq!(d.kind, InterMbKind::P16x16);
    debug_assert_eq!(d.ref_idx, 0, "more than one reference needs ref_idx writing");
    write_mb_type_p_cabac(e, st, 0);
    let lnb = left.map(|m| &m.nb);
    let anb = above.map(|m| &m.nb);
    write_mvd_16x16_cabac(e, st, lnb, anb, 0, d.mvd);
    write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), cfi == 1 || cfi == 2);
    // An inter macroblock's flag comes after the coded block pattern and
    // only when some luma block is coded — `no_sub_mb_part_less_than_8x8`
    // holds trivially for a single 16x16 partition.
    if t8x8_mode && d.cbp_luma != 0 {
        write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
    }
    debug_assert!(!d.transform_8x8 || (t8x8_mode && d.cbp_luma != 0));
    if d.cbp_luma != 0 || d.cbp_chroma != 0 {
        write_mb_qp_delta_cabac(e, st, d.qp_delta as i32);
        st.prev_qp_delta_nonzero = d.qp_delta != 0;
        write_inter_residual_cabac(e, st, false, cfi, d, lnb, anb);
    } else {
        st.prev_qp_delta_nonzero = false;
    }
}

/// Write every macroblock of an all-intra CABAC picture: the shared walk
/// makes the decisions, reconstructs and runs the loop filter; this side
/// spells bins and keeps the `WrittenMb` chain. The slice header is
/// already written; the final terminate closes the RBSP (no
/// `rbsp_trailing_bits` — see the module docs).
pub fn write_intra_picture_cabac(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
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
        write_intra_body(&mut e, &mut st, dec, left, above, cfi, t8x8);
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
pub fn write_p_picture_cabac(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refp: &[Recon],
) -> PicMotion {
    let mbw = g.mbs_wide as usize;
    let total = mbw * g.mbs_high as usize;
    let cfi = cfi_of(g.chroma);
    let t8x8 = tools.transform_8x8;
    w.align_one();
    let mut st = CabacState::new(SliceType::P, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    let mut coded: Vec<Coded> = Vec::with_capacity(total);
    let fmbs = code_p_picture(g, tools, qp, planes, rec, refp, |mb_x, mb_y, mb| {
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
                write_p16_body(&mut e, &mut st, dec, left, above, cfi, t8x8);
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
                write_intra_body(&mut e, &mut st, idec, left, above, cfi, t8x8);
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
pub fn write_b_picture_cabac(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refs: [&[Recon]; 2],
    col: &PicMotion,
) -> PicMotion {
    let mbw = g.mbs_wide as usize;
    let total = mbw * g.mbs_high as usize;
    let cfi = cfi_of(g.chroma);
    let t8x8 = tools.transform_8x8;
    w.align_one();
    let mut st = CabacState::new(SliceType::B, 0, qp as i32);
    let mut e = CabacEncoder::new(w);
    let mut coded: Vec<Coded> = Vec::with_capacity(total);
    let fmbs = code_b_picture(g, tools, qp, planes, rec, refs, col, |mb_x, mb_y, mb| {
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
                let direct = matches!(mb, BMb::Direct(_));
                let t = if direct {
                    0
                } else {
                    match dec.used {
                        [true, false] => 1,
                        [false, true] => 2,
                        [true, true] => 3,
                        [false, false] => unreachable!("an explicit B macroblock uses a list"),
                    }
                };
                write_mb_type_b_cabac(&mut e, &mut st, inc, t);
                if !direct {
                    // One reference per list, so no ref_idx; the mvds in
                    // list order for the lists the direction uses.
                    debug_assert!(dec.ref_idx.iter().all(|&r| r <= 0));
                    for l in 0..2 {
                        if dec.used[l] {
                            write_mvd_16x16_cabac(&mut e, &mut st, lnb, anb, l, dec.mvd[l]);
                        }
                    }
                }
                write_cbp_cabac(
                    &mut e,
                    &mut st,
                    lnb,
                    anb,
                    dec.cbp_luma | (dec.cbp_chroma << 4),
                    cfi == 1 || cfi == 2,
                );
                // `B_Direct_16x16` carries the flag too, because the SPS
                // this encoder writes sets `direct_8x8_inference_flag`.
                if t8x8 && dec.cbp_luma != 0 {
                    write_transform_8x8_cabac(&mut e, &mut st, lnb, anb, dec.transform_8x8);
                }
                debug_assert!(!dec.transform_8x8 || (t8x8 && dec.cbp_luma != 0));
                if dec.cbp_luma != 0 || dec.cbp_chroma != 0 {
                    write_mb_qp_delta_cabac(&mut e, &mut st, dec.qp_delta as i32);
                    st.prev_qp_delta_nonzero = dec.qp_delta != 0;
                    write_inter_residual_fields_cabac(
                        &mut e, &mut st, false, cfi, dec.transform_8x8, dec.cbp_luma,
                        &dec.nz_luma, &dec.luma, dec.cbp_chroma, &dec.chroma_dc, &dec.chroma_ac,
                        &dec.nz_chroma, lnb, anb,
                    );
                } else {
                    st.prev_qp_delta_nonzero = false;
                }
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
                write_intra_body(&mut e, &mut st, idec, left, above, cfi, t8x8);
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
