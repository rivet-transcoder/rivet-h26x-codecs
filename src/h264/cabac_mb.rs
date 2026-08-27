//! CABAC-coded macroblock layer (H.264 clause 9.3): context selection
//! (9.3.3.1), binarisations (9.3.2) and the residual block (9.3.3.1.3).
//!
//! The *writers* of the intra macroblock layer live beside their readers:
//! each `write_*` is the exact inverse of the `decode_*` / `parse_*` above
//! it and shares its context constants, so a change to either side is
//! visible from the other (see [`write_residual_block_cabac`] for why that
//! adjacency is the defence worth having).

use crate::cabac::{Cabac, Ctx, init_ctx_h264};
use crate::cabac_enc::CabacEncoder;
use crate::encode::h264_intra::{MbDecision, MbKind as IntraKind};
use crate::encode::h264_me::{BDecision, BMbKind, InterDecision, InterMbKind};
use crate::{Error, Result};

use super::cavlc::{
    b_sub_mb_type, mb_partitions, p_sub_mb_type, part_index_of, predicted_intra_mode,
    sub_partition_rect,
};
use super::frame::Mv;
use super::mb::{MbDequant, 
    MbKind, MbLayer, MbNeighbours, PRED_BI, PRED_L0, PRED_L1, PicInfo, SliceCtx, SubMbShape,
};
use super::slice::SliceType;
use super::tables::*;

/// Number of context variables (ctxIdx 0..=1023).
pub const NUM_CTX: usize = 1024;

/// Initialise all 1024 contexts for a slice (9.3.1.1).
pub fn init_contexts(
    ctxs: &mut [Ctx; NUM_CTX],
    slice_type: SliceType,
    cabac_init_idc: u32,
    qp: i32,
) {
    let table: &[[i8; 2]; 1024] = if slice_type.is_intra() {
        &CABAC_INIT_I
    } else {
        &CABAC_INIT_PB[cabac_init_idc as usize]
    };
    for (i, c) in ctxs.iter_mut().enumerate() {
        let (m, n) = (table[i][0] as i32, table[i][1] as i32);
        *c = init_ctx_h264(m, n, qp);
    }
}

// ctxIdx offsets (Table 9-34).
const CTX_MB_TYPE_SI_PREFIX: usize = 0;
const CTX_MB_TYPE_I: usize = 3;
const CTX_MB_SKIP_P: usize = 11;
const CTX_MB_TYPE_P_PREFIX: usize = 14;
const CTX_MB_TYPE_P_SUFFIX: usize = 17;
const CTX_SUB_MB_TYPE_P: usize = 21;
const CTX_MB_SKIP_B: usize = 24;
const CTX_MB_FIELD: usize = 70;
const CTX_MB_TYPE_B_PREFIX: usize = 27;
const CTX_MB_TYPE_B_SUFFIX: usize = 32;
const CTX_SUB_MB_TYPE_B: usize = 36;
const CTX_MVD_X: usize = 40;
const CTX_MVD_Y: usize = 47;
const CTX_REF_IDX: usize = 54;
const CTX_MB_QP_DELTA: usize = 60;
const CTX_INTRA_CHROMA_PRED_MODE: usize = 64;
const CTX_PREV_INTRA_PRED_MODE_FLAG: usize = 68;
const CTX_REM_INTRA_PRED_MODE: usize = 69;
const CTX_CBP_LUMA: usize = 73;
const CTX_CBP_CHROMA: usize = 77;
const CTX_TRANSFORM_8X8: usize = 399;

/// Per-`ctxBlockCat` context bases (Table 9-34's ctxIdxOffset plus Table
/// 9-40's ctxIdxBlockCatOffset) for the fourteen block categories: luma
/// DC / AC / 4x4 / (chroma DC / AC) / luma 8x8, then the same five luma-style
/// categories for Cb (6..=9) and Cr (10..=13) in 4:4:4.
const CBF_CTX_BASE: [usize; 14] = [
    85, 89, 93, 97, 101, 1012, 460, 464, 468, 1016, 472, 476, 480, 1020,
];
/// significant_coeff_flag / last_significant_coeff_flag bases, frame-coded
/// blocks then field-coded blocks (Table 9-34).
const SIG_CTX_BASE: [[usize; 14]; 2] = [
    [
        105, 120, 134, 149, 152, 402, 484, 499, 513, 660, 528, 543, 557, 718,
    ],
    [
        277, 292, 306, 321, 324, 436, 776, 791, 805, 675, 820, 835, 849, 733,
    ],
];
const LAST_CTX_BASE: [[usize; 14]; 2] = [
    [
        166, 181, 195, 210, 213, 417, 572, 587, 601, 690, 616, 631, 645, 748,
    ],
    [
        338, 353, 367, 382, 385, 451, 864, 879, 893, 699, 908, 923, 937, 757,
    ],
];
const ABS_CTX_BASE: [usize; 14] = [
    227, 237, 247, 257, 266, 426, 952, 962, 972, 708, 982, 992, 1002, 766,
];

/// `ctxBlockCat` values of the chroma blocks (the luma-style ones are
/// [`PLANE_CATS`]).
const CAT_CHROMA_DC: usize = 3;
const CAT_CHROMA_AC: usize = 4;
/// The luma-style categories `[DC, AC, 4x4, 8x8]` of colour plane `p`
/// (luma, and Cb / Cr in 4:4:4).
const PLANE_CATS: [[usize; 4]; 3] = [[0, 1, 2, 5], [6, 7, 8, 9], [10, 11, 12, 13]];

/// The colour plane a luma-style category belongs to (chroma DC / AC: 0).
#[inline]
fn cat_plane(cat: usize) -> usize {
    match cat {
        6..=9 => 1,
        10..=13 => 2,
        _ => 0,
    }
}

/// The CABAC slice decoding state carried across macroblocks.
pub struct CabacState {
    /// The 1024 context variables.
    pub ctx: [Ctx; NUM_CTX],
    /// `mb_qp_delta != 0` for the previous macroblock in decoding order
    /// (false when that macroblock was skipped, PCM, or had no residual).
    pub prev_qp_delta_nonzero: bool,
}

impl CabacState {
    /// Fresh state for a slice.
    pub fn new(slice_type: SliceType, cabac_init_idc: u32, qp: i32) -> Self {
        let mut ctx = [0u8; NUM_CTX];
        init_contexts(&mut ctx, slice_type, cabac_init_idc, qp);
        Self {
            ctx,
            prev_qp_delta_nonzero: false,
        }
    }
}

#[inline]
fn bin(c: &mut Cabac, st: &mut CabacState, ctx: usize) -> u32 {
    c.decision(&mut st.ctx[ctx])
}

/// `mb_skip_flag` (9.3.3.1.1.1).
pub fn decode_mb_skip(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    nb: &MbNeighbours,
    is_b: bool,
) -> bool {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) if !info.mbs[addr].kind.is_skip() => 1,
            _ => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    let base = if is_b { CTX_MB_SKIP_B } else { CTX_MB_SKIP_P };
    bin(c, st, base + inc) != 0
}

/// Write `mb_skip_flag`: the exact inverse of [`decode_mb_skip`]. One
/// decision bin; the ctxIdxInc counts the available neighbours (left,
/// above) that are *not* themselves skipped, read off the [`WrittenMb`]
/// state the caller keeps.
///
/// This replaces CAVLC's `mb_skip_run` counting: every macroblock of a P
/// slice codes the flag, a skipped one codes *nothing else*, and the
/// caller then clears `st.prev_qp_delta_nonzero` exactly as the decoder's
/// slice loop does when it takes the skip branch.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_mb_skip_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    is_b: bool,
    skip: bool,
) {
    let cond = |m: Option<&WrittenMb>| -> usize {
        match m {
            Some(m) if !m.skip => 1,
            _ => 0,
        }
    };
    let inc = cond(left) + cond(above);
    let base = if is_b { CTX_MB_SKIP_B } else { CTX_MB_SKIP_P };
    e.encode_decision(&mut st.ctx[base + inc], skip as u32);
}

/// `mb_field_decoding_flag` (9.3.3.1.1.2): context from whether the left
/// and above pairs (when available) are field pairs.
pub fn decode_mb_field(c: &mut Cabac, st: &mut CabacState, nb: &MbNeighbours) -> bool {
    let inc = (nb.pair[0].is_some() && nb.pair_field[0]) as usize
        + (nb.pair[1].is_some() && nb.pair_field[1]) as usize;
    bin(c, st, CTX_MB_FIELD + inc) != 0
}

/// `end_of_slice_flag`.
pub fn decode_end_of_slice(c: &mut Cabac) -> bool {
    c.terminate() != 0
}

/// Intra `mb_type` binarisation shared by I slices (`base` = 3, with the
/// neighbour-dependent first bin) and the P/B suffixes (`base` = 17 / 32).
fn decode_intra_mb_type(
    c: &mut Cabac,
    st: &mut CabacState,
    base: usize,
    intra_slice: bool,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> u32 {
    if intra_slice {
        let cond = |a: Option<usize>| -> usize {
            match a {
                Some(addr) => match info.mbs[addr].kind {
                    // "mb_type != I_NxN" for available neighbours (SI is not supported).
                    MbKind::I4x4 | MbKind::I8x8 => 0,
                    _ => 1,
                },
                None => 0,
            }
        };
        let inc = cond(nb.a) + cond(nb.b);
        if bin(c, st, base + inc) == 0 {
            return 0; // I_NxN
        }
    } else if bin(c, st, base) == 0 {
        return 0;
    }
    if c.terminate() != 0 {
        return 25; // I_PCM
    }
    // I_16x16: bins for cbp luma, cbp chroma, prediction mode.
    let (b_luma, b_chroma_nz, b_chroma_two, b_pred0, b_pred1) = if intra_slice {
        (base + 3, base + 4, base + 5, base + 6, base + 7)
    } else {
        (base + 1, base + 2, base + 2, base + 3, base + 3)
    };
    let mut t = 1;
    t += 12 * bin(c, st, b_luma);
    if bin(c, st, b_chroma_nz) != 0 {
        t += 4 + 4 * bin(c, st, b_chroma_two);
    }
    t += 2 * bin(c, st, b_pred0);
    t += bin(c, st, b_pred1);
    t
}

/// `mb_type` 25 (I_PCM) in the intra numbering [`decode_intra_mb_type`]
/// returns, [`write_mb_type_i_cabac`] takes and
/// [`super::cavlc::intra_mb_type`] maps to a kind.
pub(crate) const MB_TYPE_I_PCM: u32 = 25;

/// Write an I-slice `mb_type`: the exact inverse of [`decode_intra_mb_type`]
/// with `base = CTX_MB_TYPE_I` (the I-slice binarisation, 9.3.3.1.1.3). `t`
/// is the value in that function's own numbering — 0 `I_NxN`, 1..=24
/// `I_16x16` (see [`intra_mb_type_code`]), [`MB_TYPE_I_PCM`] — so what one
/// returns is exactly what the other takes.
///
/// `inc` is the first bin's ctxIdxInc: how many of the two available
/// neighbours (left, above) are not `I_NxN`. The reader derives it from its
/// picture arrays; it is an argument here because the writer's caller is
/// the one walking the picture — the same seam as the residual writer's
/// `cbf_inc`.
///
/// An I_PCM `mb_type` ends in a terminate bin of 1, which *flushes* the
/// codeword: the caller writes the raw samples byte-aligned and opens a new
/// engine after them, as `write_pcm_slice_data_cabac` (the one caller
/// today, in `encode::h264_syntax`) does.
pub(crate) fn write_mb_type_i_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    inc: usize,
    t: u32,
) {
    write_intra_mb_type_cabac(e, st, CTX_MB_TYPE_I, true, inc, t);
}

/// The intra `mb_type` spelling shared by [`write_mb_type_i_cabac`] and the
/// intra suffix of [`write_mb_type_p_cabac`] — the exact inverse of
/// [`decode_intra_mb_type`], parameterised the same way: `intra_slice`
/// selects the I-slice bin layout (neighbour-dependent first bin at
/// `base + inc`, I_16x16 field bins at `base + 3..=7`) against the P/B
/// suffix layout (fixed first bin at `base`, field bins at `base + 1..=3`,
/// where the chroma pair and the prediction pair each share one context).
fn write_intra_mb_type_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    base: usize,
    intra_slice: bool,
    inc: usize,
    t: u32,
) {
    debug_assert!(t <= MB_TYPE_I_PCM, "mb_type {t} out of the intra range");
    debug_assert!(inc <= 2, "the first bin has three contexts");
    let first = if intra_slice { base + inc } else { base };
    if t == 0 {
        e.encode_decision(&mut st.ctx[first], 0); // I_NxN
        return;
    }
    e.encode_decision(&mut st.ctx[first], 1);
    e.encode_terminate((t == MB_TYPE_I_PCM) as u32);
    if t == MB_TYPE_I_PCM {
        return; // flushed; the PCM samples follow outside the engine
    }
    let (b_luma, b_chroma_nz, b_chroma_two, b_pred0, b_pred1) = if intra_slice {
        (base + 3, base + 4, base + 5, base + 6, base + 7)
    } else {
        (base + 1, base + 2, base + 2, base + 3, base + 3)
    };
    // I_16x16. The reader sums t = 1 + 12·luma + 4·chroma + pred, so the
    // three fields come back out by the inverse arithmetic.
    let t = t - 1;
    e.encode_decision(&mut st.ctx[b_luma], (t >= 12) as u32);
    let chroma = (t / 4) % 3;
    e.encode_decision(&mut st.ctx[b_chroma_nz], (chroma != 0) as u32);
    if chroma != 0 {
        e.encode_decision(&mut st.ctx[b_chroma_two], (chroma == 2) as u32);
    }
    let pred = t % 4;
    e.encode_decision(&mut st.ctx[b_pred0], pred >> 1);
    e.encode_decision(&mut st.ctx[b_pred1], pred & 1);
}

/// The I-slice `mb_type` of an intra `MbDecision`, in the numbering
/// [`write_mb_type_i_cabac`] takes (CAVLC spells the same value as
/// `ue(v)`): the inverse of [`super::cavlc::intra_mb_type`]'s I_16x16
/// arithmetic.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn intra_mb_type_code(d: &MbDecision) -> u32 {
    match d.kind {
        // Both `I_NxN` kinds are `mb_type` 0 — which is the whole reason
        // `transform_size_8x8_flag` exists.
        IntraKind::I4x4 | IntraKind::I8x8 => 0,
        IntraKind::I16x16 => {
            debug_assert!(
                d.cbp_luma == 0 || d.cbp_luma == 15,
                "I_16x16 codes luma all-or-nothing"
            );
            debug_assert!(d.cbp_chroma <= 2 && d.intra16_mode <= 3);
            1 + d.intra16_mode as u32
                + 4 * d.cbp_chroma as u32
                + 12 * (d.cbp_luma != 0) as u32
        }
    }
}

/// `mb_type` for the slice type: returns the value in the same numbering the
/// CAVLC parser uses (P: 0..=4 inter, 5+ intra; B: 0..=22 inter, 23+ intra;
/// I: intra numbering).
fn decode_mb_type(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> Result<u32> {
    match ctx.slice_type {
        SliceType::I => Ok(decode_intra_mb_type(c, st, CTX_MB_TYPE_I, true, info, nb)),
        SliceType::Si => {
            // SI prefix then the I binarisation. SI slices are refused
            // upstream; keep the parse coherent anyway.
            let _ = CTX_MB_TYPE_SI_PREFIX;
            Err(Error::unsupported("SI slices"))
        }
        SliceType::P | SliceType::Sp => {
            if bin(c, st, CTX_MB_TYPE_P_PREFIX) != 0 {
                return Ok(5 + decode_intra_mb_type(c, st, CTX_MB_TYPE_P_SUFFIX, false, info, nb));
            }
            if bin(c, st, CTX_MB_TYPE_P_PREFIX + 1) == 0 {
                // b1 = 0: b2 (ctx 16): 0 -> P_L0_16x16, 1 -> P_8x8
                Ok(3 * bin(c, st, CTX_MB_TYPE_P_PREFIX + 2))
            } else {
                // b1 = 1: b2 (ctx 17): 1 -> 16x8, 0 -> 8x16
                Ok(2 - bin(c, st, CTX_MB_TYPE_P_PREFIX + 3))
            }
        }
        SliceType::B => {
            let cond = |a: Option<usize>| -> usize {
                match a {
                    Some(addr) => match info.mbs[addr].kind {
                        MbKind::BSkip | MbKind::BDirect16x16 => 0,
                        _ => 1,
                    },
                    None => 0,
                }
            };
            let inc = cond(nb.a) + cond(nb.b);
            if bin(c, st, CTX_MB_TYPE_B_PREFIX + inc) == 0 {
                return Ok(0); // B_Direct_16x16
            }
            if bin(c, st, CTX_MB_TYPE_B_PREFIX + 3) == 0 {
                return Ok(1 + bin(c, st, CTX_MB_TYPE_B_PREFIX + 5));
            }
            let mut bits = bin(c, st, CTX_MB_TYPE_B_PREFIX + 4) << 3;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5) << 2;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5) << 1;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5);
            if bits < 8 {
                return Ok(bits + 3);
            }
            match bits {
                13 => Ok(23 + decode_intra_mb_type(c, st, CTX_MB_TYPE_B_SUFFIX, false, info, nb)),
                14 => Ok(11),
                15 => Ok(22),
                _ => {
                    let bits = (bits << 1) | bin(c, st, CTX_MB_TYPE_B_PREFIX + 5);
                    Ok(bits - 4)
                }
            }
        }
    }
}

/// Write a P-slice `mb_type`: the exact inverse of [`decode_mb_type`]'s P
/// branch, taking `t` in the numbering that function returns (and
/// [`super::cavlc::p_mb_type`] consumes): 0 `P_L0_16x16`, 1
/// `P_L0_L0_16x8`, 2 `P_L0_L0_8x16`, and 5+ intra — the I-slice tree
/// behind the P prefix, so an intra decision's value is
/// `5 + intra_mb_type_code(..)`, and I_PCM's terminate flushes the
/// codeword exactly as it does in an I slice.
///
/// 3 is `P_8x8`, whose four `sub_mb_type`s follow through
/// [`write_sub_mb_type_p_cabac`]. 4 (`P_8x8ref0`) does not exist in CABAC
/// at all — Table 9-27 has no binarisation for it; it is a CAVLC-only
/// spelling, and it is refused.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_mb_type_p_cabac(e: &mut CabacEncoder, st: &mut CabacState, t: u32) {
    debug_assert!(t != 4, "P_8x8ref0 has no CABAC binarisation (Table 9-27)");
    if t >= 5 {
        e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX], 1);
        write_intra_mb_type_cabac(e, st, CTX_MB_TYPE_P_SUFFIX, false, 0, t - 5);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX], 0);
    match t {
        // b1 = 0, then b2 (ctx 16): 0 -> P_L0_16x16, 1 -> P_8x8.
        0 | 3 => {
            e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX + 1], 0);
            e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX + 2], (t == 3) as u32);
        }
        // b1 = 1, then b2 (ctx 17): 1 -> 16x8, 0 -> 8x16.
        1 | 2 => {
            e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX + 1], 1);
            e.encode_decision(&mut st.ctx[CTX_MB_TYPE_P_PREFIX + 3], (t == 1) as u32);
        }
        _ => unreachable!("P mb_type {t}"),
    }
}

/// Write a P-slice `sub_mb_type`: the exact inverse of
/// [`decode_sub_mb_type_p`], whose four values are the sub-macroblock
/// shapes of Table 7-17 — 0 is one 8x8, 1 two 8x4, 2 two 4x8, 3 four 4x4.
///
/// A unary tree over three contexts whose first bin is inverted relative
/// to the rest: a 1 there *ends* it at 8x8, where a 1 at the second bin
/// continues.
pub(crate) fn write_sub_mb_type_p_cabac(e: &mut CabacEncoder, st: &mut CabacState, t: u32) {
    debug_assert!(t < 4, "P sub_mb_type {t} out of range");
    if t == 0 {
        e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_P], 1);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_P], 0);
    if t == 1 {
        e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_P + 1], 0);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_P + 1], 1);
    e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_P + 2], (t == 2) as u32);
}

/// Write a B-slice `mb_type`: the exact inverse of [`decode_mb_type`]'s B
/// branch over Table 7-14's whole numbering — 0 `B_Direct_16x16`, 1..=3
/// the explicit 16x16 directions, 4..=21 the 16x8 / 8x16 rows, 22
/// `B_8x8`, and 23+ intra (the I-slice tree behind the B prefix,
/// `23 + intra_mb_type_code(..)`).
///
/// `inc` is the first bin's ctxIdxInc: how many of the two available
/// neighbours are *neither* `B_Skip` nor `B_Direct_16x16` — read off
/// [`WrittenMb::direct`] (a skipped B macroblock sets it too).
///
/// The binarisation (Table 9-37) after the two prefix bins is four fixed
/// bins of a value `bits`, and the reader's arithmetic is the spec: below
/// 8 the type is `bits + 3`; 13 escapes to intra, 14 is type 11, 15 is
/// `B_8x8`; and 8..=12 take one more bin, the type then being the five
/// bins as a number, minus 4. So 12..=21 spell `t + 4` over five bins.
pub(crate) fn write_mb_type_b_cabac(e: &mut CabacEncoder, st: &mut CabacState, inc: usize, t: u32) {
    if t == 0 {
        e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + inc], 0);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + inc], 1);
    if t <= 2 {
        e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 3], 0);
        e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 5], t - 1);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 3], 1);
    // The four fixed bins, and whether a fifth follows.
    let (bits, fifth): (u32, Option<u32>) = match t {
        3..=10 => (t - 3, None),
        11 => (14, None),
        22 => (15, None),
        12..=21 => ((t + 4) >> 1, Some((t + 4) & 1)),
        _ => (13, None),
    };
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 4], (bits >> 3) & 1);
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 5], (bits >> 2) & 1);
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 5], (bits >> 1) & 1);
    e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 5], bits & 1);
    if let Some(b) = fifth {
        e.encode_decision(&mut st.ctx[CTX_MB_TYPE_B_PREFIX + 5], b);
    }
    if t >= 23 {
        write_intra_mb_type_cabac(e, st, CTX_MB_TYPE_B_SUFFIX, false, 0, t - 23);
    }
}

/// Write a B-slice `sub_mb_type` (Table 7-18's 0..=12): the exact inverse
/// of [`decode_sub_mb_type_b`], bin for bin — 0 `B_Direct_8x8` is a single
/// zero; 1 and 2 (`B_L0_8x8`, `B_L1_8x8`) are `1 0 b`; 3..=6 are
/// `1 1 0 b b`; 7..=10 are `1 1 1 0 b b`; 11 and 12 are `1 1 1 1 b`. The
/// contexts are 36..=39 with the reader's own placement: the first three
/// bins take 36, 37 and 38 in turn, everything after them 39.
pub(crate) fn write_sub_mb_type_b_cabac(e: &mut CabacEncoder, st: &mut CabacState, t: u32) {
    debug_assert!(t <= 12, "B sub_mb_type {t} out of range");
    if t == 0 {
        e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B], 0);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B], 1);
    if t <= 2 {
        e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 1], 0);
        e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], t - 1);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 1], 1);
    match t {
        3..=6 => {
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 2], 0);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], (t - 3) >> 1);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], (t - 3) & 1);
        }
        7..=10 => {
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 2], 1);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], 0);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], (t - 7) >> 1);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], (t - 7) & 1);
        }
        _ => {
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 2], 1);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], 1);
            e.encode_decision(&mut st.ctx[CTX_SUB_MB_TYPE_B + 3], t - 11);
        }
    }
}

fn decode_sub_mb_type_p(c: &mut Cabac, st: &mut CabacState) -> u32 {
    if bin(c, st, CTX_SUB_MB_TYPE_P) != 0 {
        return 0;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_P + 1) == 0 {
        return 1;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_P + 2) != 0 {
        2
    } else {
        3
    }
}

fn decode_sub_mb_type_b(c: &mut Cabac, st: &mut CabacState) -> u32 {
    if bin(c, st, CTX_SUB_MB_TYPE_B) == 0 {
        return 0;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_B + 1) == 0 {
        return 1 + bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    }
    let mut t = 3;
    if bin(c, st, CTX_SUB_MB_TYPE_B + 2) != 0 {
        if bin(c, st, CTX_SUB_MB_TYPE_B + 3) != 0 {
            return 11 + bin(c, st, CTX_SUB_MB_TYPE_B + 3);
        }
        t += 4;
    }
    t += 2 * bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    t += bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    t
}

/// Whether the 4x4 block `(mb, blk)` belongs to a direct-predicted
/// partition (B_Skip / B_Direct_16x16 / B_Direct_8x8), for the ref_idx and
/// mvd contexts (`predModeEqualFlag` = 0).
fn is_direct_block(info: &PicInfo, layer: &MbLayer, cur: usize, addr: usize, blk: usize) -> bool {
    if addr == cur {
        return layer.kind.is_direct16x16()
            || layer.sub_shape[(blk / 8) * 2 + (blk % 4) / 2] == SubMbShape::Direct;
    }
    let m = &info.mbs[addr];
    m.kind.is_direct16x16()
        || (m.kind == MbKind::Inter8x8
            && (m.sub_direct >> ((blk / 8) * 2 + (blk % 4) / 2)) & 1 != 0)
}

/// `ref_idx_lX` (9.3.3.1.1.6): unary, contexts 54..=59.
fn decode_ref_idx(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    frame_motion: &[Vec<super::frame::BlockMotion>; 2],
    list: usize,
    bx: i32,
    by: i32,
) -> Result<i8> {
    let cond = |dx: i32, dy: i32| -> usize {
        let Some((addr, blk)) = nb.block(bx + dx, by + dy) else {
            return 0;
        };
        if addr == nb.addr {
            // Inside the current macroblock: the partition's own already-set
            // reference index (partitions are decoded in order, so a left/up
            // neighbour inside the MB is always earlier).
            if is_direct_block(info, layer, nb.addr, addr, blk) {
                return 0;
            }
            let part = (blk / 8) * 2 + (blk % 4) / 2;
            if layer.pred_dir[part] & (1 << list) == 0 {
                return 0;
            }
            return (layer.ref_idx[list][part] > 0) as usize;
        }
        let m = &info.mbs[addr];
        if m.kind.is_skip() || m.kind.is_intra() || is_direct_block(info, layer, nb.addr, addr, blk)
        {
            return 0;
        }
        // refIdxZeroFlagN: a frame macroblock reading a field neighbour in
        // an MBAFF frame counts its index halved (> 1 rather than > 0).
        let thr = if nb.mbaff && !nb.cur_field && m.field {
            1
        } else {
            0
        };
        (frame_motion[list][addr * 16 + blk].ref_idx > thr) as usize
    };
    let inc = cond(-1, 0) + 2 * cond(0, -1);
    let pos0 = c.position();
    let mut v: i8 = 0;
    let mut ctx = CTX_REF_IDX + inc;
    while bin(c, st, ctx) != 0 {
        v += 1;
        ctx = CTX_REF_IDX + if v == 1 { 4 } else { 5 };
        if v > 31 {
            return Err(Error::bitstream("ref_idx runaway"));
        }
    }
    if super::mb::syntax_trace() {
        eprintln!("ref_idx l{list} inc={inc} @{pos0} -> {v}");
    }
    Ok(v)
}

/// Write `ref_idx_l0` for the 16x16 partition of a P macroblock: the exact
/// inverse of [`decode_ref_idx`] over the shapes the encoder produces.
/// Unary — `v` one-bins then a zero — with the first bin against
/// `CTX_REF_IDX + inc`, the second at offset 4 and the rest at 5.
///
/// Present only when the list is longer than one (`num_ref_idx_active >
/// 1`): the caller mirrors [`parse_mb_cabac`]'s condition and does not
/// call this otherwise. The ctxIdxInc reads the 4x4 blocks left of and
/// above the partition, which for a 16x16 partition are always in the
/// neighbouring macroblocks (the left one's block 3, the above one's
/// block 12) — which is why this takes [`WrittenMb`]s rather than
/// coordinates. Partition shapes that put a neighbour inside the current
/// macroblock have no writer yet.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_ref_idx_16x16_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    v: i8,
) {
    debug_assert!((0..32).contains(&v), "ref_idx out of range");
    let cond = |m: Option<&WrittenMb>, blk: usize| -> usize {
        match m {
            None => 0,
            Some(m) => {
                if m.skip || m.intra {
                    0
                } else {
                    (m.ref_idx[blk] > 0) as usize
                }
            }
        }
    };
    let inc = cond(left, 3) + 2 * cond(above, 12);
    let ctx_at = |k: i8| -> usize {
        CTX_REF_IDX
            + match k {
                0 => inc,
                1 => 4,
                _ => 5,
            }
    };
    for k in 0..v {
        e.encode_decision(&mut st.ctx[ctx_at(k)], 1);
    }
    e.encode_decision(&mut st.ctx[ctx_at(v)], 0);
}

/// Both absolute mvd components of one neighbouring 4x4 block, for the
/// ctxIdxInc of 9.3.3.1.1.7 — zero where the neighbour is absent, skipped,
/// intra, or direct-predicted, which have no mvd of their own.
///
/// The two components come back together because the two calls that want
/// them agree on everything but the last step. Finding the block, deciding
/// whether it counts, and scaling across a field boundary is the work; which
/// component is then read is a field access. Deriving the pair once halves
/// what a motion vector difference costs in neighbour lookups — counted on
/// `cabac3.264`, exactly half: 1,557,633 mvd pairs, four neighbour
/// derivations each before and two after, 3,115,266 lookups removed.
///
/// That count is the whole claim. The time saved is below what any
/// instrument here can resolve, and an earlier commit message said 2.4%
/// on the strength of a harness that had never been given a same-binary
/// control; see the retraction in the log. For a bit-exact refactor,
/// count the work removed and say nothing about the clock.
fn mvd_neighbour_abs(
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    list: usize,
    bx: i32,
    by: i32,
) -> (i32, i32) {
    let Some((addr, blk)) = nb.block(bx, by) else {
        return (0, 0);
    };
    if addr == nb.addr {
        if is_direct_block(info, layer, nb.addr, addr, blk) {
            return (0, 0);
        }
        let m = layer.mvd[blk].mvd[list];
        return (m.x.abs() as i32, m.y.abs() as i32);
    }
    let mi = &info.mbs[addr];
    if mi.kind.is_skip() || mi.kind.is_intra() || is_direct_block(info, layer, nb.addr, addr, blk) {
        return (0, 0);
    }
    let m = info.mvd[list][addr * 16 + blk];
    let y = if nb.mbaff && !nb.cur_field && mi.field {
        // 9.3.3.1.1.7: a frame macroblock reads a field neighbour's vertical
        // mvd doubled, a field macroblock a frame one's halved.
        m.y.abs() as i32 * 2
    } else if nb.mbaff && nb.cur_field && !mi.field {
        m.y.abs() as i32 / 2
    } else {
        m.y.abs() as i32
    };
    (m.x.abs() as i32, y)
}

/// `mvd_l0` / `mvd_l1` for one partition: both components, in that order.
fn decode_mvd(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    list: usize,
    bx: i32,
    by: i32,
) -> Result<(i16, i16)> {
    // Nothing between the two components writes the neighbouring mvds — the
    // caller stores this partition's only once both are decoded — so one
    // derivation serves both.
    let left = mvd_neighbour_abs(info, layer, nb, list, bx - 1, by);
    let above = mvd_neighbour_abs(info, layer, nb, list, bx, by - 1);
    let x = decode_mvd_component(c, st, left.0 + above.0, list, 0)?;
    let y = decode_mvd_component(c, st, left.1 + above.1, list, 1)?;
    Ok((x, y))
}

/// Write both `mvd_lX` components of a 16x16 partition for one list:
/// the exact inverse of [`decode_mvd`] over the shapes the encoder
/// produces. The context sums read the 4x4 blocks left of and above the
/// partition — for 16x16 always in the neighbouring macroblocks (left
/// block 3, above block 12) — off the [`WrittenMb`] mvd arrays, exactly
/// as [`mvd_neighbour_abs`]'s frame-picture branches read the decoder's:
/// an absent, skipped or intra neighbour counts zero, and there is no
/// field scaling because the encoder writes frame pictures only.
/// Partition shapes with an in-macroblock neighbour have no writer yet.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_mvd_16x16_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    list: usize,
    mvd: Mv,
) {
    write_mvd_cabac(e, st, &CurMbMvd::default(), left, above, list, 0, 0, mvd);
}

/// The `mvd_lX` of the macroblock being written, per 4x4 block — what
/// this macroblock's *own* later partitions read for their context.
///
/// A 16x16 partition never needs it: both its neighbours are in other
/// macroblocks. Every smaller shape does — the lower half of a 16x8 takes
/// its B neighbour from the upper half, and the right half of an 8x16 its
/// A neighbour from the left — which is the mirror of
/// `mvd_neighbour_abs`'s `addr == nb.addr` branch.
#[derive(Clone, Copy, Default)]
pub(crate) struct CurMbMvd {
    /// Per list, per 4x4 (raster). Blocks not yet written are zero, which
    /// is what the decoder's `layer.reset` leaves them too.
    pub mvd: [[Mv; 16]; 2],
}

impl CurMbMvd {
    /// Record a partition's mvd over the blocks it covers, as the reader
    /// does once it has parsed that partition — and over the whole
    /// rectangle, because that is what the reader stores (its CAVLC twin
    /// keeps only the top-left, having no context to feed).
    pub fn set(&mut self, list: usize, x: usize, y: usize, w: usize, h: usize, mvd: Mv) {
        for by in y / 4..(y + h) / 4 {
            for bx in x / 4..(x + w) / 4 {
                self.mvd[list][by * 4 + bx] = mvd;
            }
        }
    }
}

/// Write one `mvd_lX` for the partition whose top-left 4x4 block is
/// `(bx, by)`: the exact inverse of [`decode_mvd`], its neighbour
/// derivation included.
///
/// The context increment is the sum of the absolute components of the
/// mvds of the blocks left of and above this one (9.3.3.1.1.7) — *4x4
/// blocks*, not macroblocks, which is why this takes coordinates. For a
/// 16x16 partition both land in neighbouring macroblocks, which is all
/// the 16x16-only writer could express; for any smaller shape at least
/// one lands inside this macroblock and comes from `cur`.
///
/// Both neighbours are read before either component is written, exactly
/// as `decode_mvd` reads them before decoding either: the horizontal
/// component's own value must not perturb the vertical component's
/// context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_mvd_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    cur: &CurMbMvd,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    list: usize,
    bx: usize,
    by: usize,
    mvd: Mv,
) {
    debug_assert!(bx < 4 && by < 4);
    // `MbNeighbours::block` restricted to the two lookups this makes,
    // both with 0 <= bx, by < 4: the left neighbour is this macroblock's
    // own block unless bx is 0, when it is the left macroblock's block 3
    // of that row; the above neighbour likewise, or the above
    // macroblock's block 12 + bx.
    let outside = |m: Option<&WrittenMb>, blk: usize| -> (i32, i32) {
        match m {
            None => (0, 0),
            Some(m) => {
                if m.skip || m.intra {
                    (0, 0)
                } else {
                    (m.mvd[list][blk].x.abs() as i32, m.mvd[list][blk].y.abs() as i32)
                }
            }
        }
    };
    let inside = |blk: usize| -> (i32, i32) {
        let m = cur.mvd[list][blk];
        (m.x.abs() as i32, m.y.abs() as i32)
    };
    let a = if bx > 0 { inside(by * 4 + bx - 1) } else { outside(left, by * 4 + 3) };
    let b = if by > 0 { inside((by - 1) * 4 + bx) } else { outside(above, 12 + bx) };
    write_mvd_component_cabac(e, st, a.0 + b.0, 0, mvd.x as i32);
    write_mvd_component_cabac(e, st, a.1 + b.1, 1, mvd.y as i32);
}

/// One `mvd_lX` component (9.3.3.1.1.7), given the sum of its neighbours'
/// absolute values: TU prefix (cMax 9) + UEG3 suffix + sign.
fn decode_mvd_component(
    c: &mut Cabac,
    st: &mut CabacState,
    sum: i32,
    list: usize,
    comp: usize,
) -> Result<i16> {
    let base = if comp == 0 { CTX_MVD_X } else { CTX_MVD_Y };
    let inc = if sum < 3 {
        0
    } else if sum <= 32 {
        1
    } else {
        2
    };
    let trace = super::mb::syntax_trace();
    let pos0 = c.position();
    // Prefix: TU with cMax = 9; bin 0 uses inc, bins 1.. use 3,4,5,6,6,6,6,6.
    let mut prefix = 0u32;
    if bin(c, st, base + inc) != 0 {
        prefix = 1;
        // ctxIdxInc runs 3, 4, 5, 6, 6, 6, 6, 6 over the remaining bins
        // (Table 9-34), which is the count capped at three, plus three —
        // cheaper to work out than to look up.
        while prefix < 9 && bin(c, st, base + 3 + (prefix - 1).min(3) as usize) != 0 {
            prefix += 1;
        }
    }
    let mut abs = prefix as i32;
    if prefix >= 9 {
        // UEG3 suffix (uCoff = 9).
        let mut k = 3u32;
        loop {
            if c.bypass() != 0 {
                abs += 1 << k;
                k += 1;
                if k > 24 {
                    return Err(Error::bitstream("mvd suffix runaway"));
                }
            } else {
                break;
            }
        }
        while k > 0 {
            k -= 1;
            abs += (c.bypass() as i32) << k;
        }
    }
    if abs == 0 {
        if trace {
            eprintln!("mvd l{list} c{comp} sum={sum} @{pos0} -> 0");
        }
        return Ok(0);
    }
    let sign = c.bypass();
    let v = if sign != 0 { -abs } else { abs };
    if !(-32768..=32767).contains(&v) {
        return Err(Error::bitstream("mvd out of range"));
    }
    if trace {
        eprintln!("mvd l{list} c{comp} sum={sum} @{pos0} -> {v}");
    }
    Ok(v as i16)
}

/// Write one `mvd_lX` component: the exact inverse of
/// [`decode_mvd_component`], given the same `sum` of the neighbouring
/// blocks' absolute values that picks the first bin's context. TU prefix
/// (cMax 9) whose later bins run contexts 3, 4, 5, 6, 6, … — the count
/// capped at three, plus three, as the reader works it out — then a UEG3
/// bypass suffix above nine, then a bypass sign for anything nonzero
/// (zero carries no sign bin).
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_mvd_component_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    sum: i32,
    comp: usize,
    v: i32,
) {
    debug_assert!((-32768..=32767).contains(&v), "mvd out of range");
    let base = if comp == 0 { CTX_MVD_X } else { CTX_MVD_Y };
    let inc = if sum < 3 {
        0
    } else if sum <= 32 {
        1
    } else {
        2
    };
    let abs = v.unsigned_abs();
    if abs == 0 {
        e.encode_decision(&mut st.ctx[base + inc], 0);
        return;
    }
    e.encode_decision(&mut st.ctx[base + inc], 1);
    // The reader stops at nine without consuming a terminator, so a capped
    // prefix writes ones only.
    let prefix = abs.min(9);
    for k in 1..prefix {
        e.encode_decision(&mut st.ctx[base + 3 + (k - 1).min(3) as usize], 1);
    }
    if prefix < 9 {
        e.encode_decision(&mut st.ctx[base + 3 + (prefix - 1).min(3) as usize], 0);
    } else {
        // UEG3 escape: ones while the remainder covers the next doubling
        // (starting at 1 << 3), a zero, then the remainder's bits.
        let mut rem = abs - 9;
        let mut k = 3u32;
        while rem >= (1 << k) {
            e.encode_bypass(1);
            rem -= 1 << k;
            k += 1;
            debug_assert!(k <= 24, "mvd too large to binarise");
        }
        e.encode_bypass(0);
        while k > 0 {
            k -= 1;
            e.encode_bypass((rem >> k) & 1);
        }
    }
    e.encode_bypass((v < 0) as u32);
}

/// `intra_chroma_pred_mode` (9.3.3.1.1.8).
fn decode_chroma_pred_mode(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> u8 {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) => {
                let m = &info.mbs[addr];
                if !m.kind.is_intra() || m.kind == MbKind::IPcm || m.chroma_mode == 0 {
                    0
                } else {
                    1
                }
            }
            None => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + inc) == 0 {
        return 0;
    }
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + 3) == 0 {
        return 1;
    }
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + 3) == 0 {
        2
    } else {
        3
    }
}

/// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` → the mode.
fn decode_intra_pred_mode(c: &mut Cabac, st: &mut CabacState, pred: u8) -> u8 {
    let trace = super::mb::syntax_trace();
    let pos0 = c.position();
    if bin(c, st, CTX_PREV_INTRA_PRED_MODE_FLAG) != 0 {
        if trace {
            eprintln!("ipm prev=1 pred={pred} @{pos0}");
        }
        return pred;
    }
    let mut rem = bin(c, st, CTX_REM_INTRA_PRED_MODE);
    rem |= bin(c, st, CTX_REM_INTRA_PRED_MODE) << 1;
    rem |= bin(c, st, CTX_REM_INTRA_PRED_MODE) << 2;
    let rem = rem as u8;
    if trace {
        eprintln!(
            "ipm prev=0 rem={rem} pred={pred} @{pos0} ctx69={:?}",
            st.ctx[CTX_REM_INTRA_PRED_MODE]
        );
    }
    if rem < pred { rem } else { rem + 1 }
}

/// Write the intra prediction modes of a macroblock: the inverse of the
/// reader's mode section — for `I_NxN`, [`decode_intra_pred_mode`]'s two
/// syntax elements per luma 4x4 block in the standard's block order, then
/// (when the slice has 4:2:0 / 4:2:2 chroma)
/// [`decode_chroma_pred_mode`]'s binarisation.
///
/// `d.luma_pred` is raster-indexed; the block-order walk maps through
/// [`super::mb::raster_of_blk`] exactly as the reader does. The `rem` bins
/// go out least significant first, because that is the order the reader
/// assembles them in.
///
/// `I_8x8` sends four modes instead of sixteen, in 8x8 block order, with
/// the *same* two syntax elements and the same two contexts — the reader
/// calls the very same [`decode_intra_pred_mode`] for both
/// (`parse_mb_cabac`'s `MbKind::I8x8` arm). The decision stores each
/// quad's mode on all four of its 4x4s, so the quad's top-left raster is
/// where the writer reads it.
///
/// `chroma_nb` is `None` when `intra_chroma_pred_mode` is not parsed at all
/// (monochrome — and 4:4:4, which has no writer yet), else the two
/// condTermFlags `[left, above]` of 9.3.3.1.1.8: a side is `true` iff that
/// neighbouring macroblock is available, intra but not I_PCM, and its own
/// `intra_chroma_pred_mode` is nonzero. The writer's caller keeps that
/// state, mirroring the reader's `cond` — the same seam as the residual
/// writer's `cbf_inc`.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_intra_pred_modes_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    d: &MbDecision,
    chroma_nb: Option<[bool; 2]>,
) {
    // The blocks whose modes are sent, in the order the reader takes
    // them: the standard's 4x4 scan for `I_4x4`, and for `I_8x8` the four
    // quads in raster order at their top-left 4x4 — which is where the
    // decision stored each quad's mode.
    let blk_order: [usize; 16] = std::array::from_fn(super::mb::raster_of_blk);
    let modes: &[usize] = match d.kind {
        IntraKind::I4x4 => &blk_order[..],
        IntraKind::I8x8 => &[0, 2, 8, 10],
        IntraKind::I16x16 => &[],
    };
    for &raster in modes {
        let m = d.luma_pred[raster];
        e.encode_decision(
            &mut st.ctx[CTX_PREV_INTRA_PRED_MODE_FLAG],
            m.use_predicted as u32,
        );
        if !m.use_predicted {
            debug_assert!(m.rem < 8, "rem_intra4x4_pred_mode is three bits");
            let rem = m.rem as u32;
            e.encode_decision(&mut st.ctx[CTX_REM_INTRA_PRED_MODE], rem & 1);
            e.encode_decision(&mut st.ctx[CTX_REM_INTRA_PRED_MODE], (rem >> 1) & 1);
            e.encode_decision(&mut st.ctx[CTX_REM_INTRA_PRED_MODE], (rem >> 2) & 1);
        }
    }
    if let Some([left, above]) = chroma_nb {
        let mode = d.chroma_mode as u32;
        debug_assert!(mode <= 3, "intra_chroma_pred_mode out of range");
        let inc = left as usize + above as usize;
        // Truncated unary, cMax 3: the bins after the first share a context.
        e.encode_decision(&mut st.ctx[CTX_INTRA_CHROMA_PRED_MODE + inc], (mode != 0) as u32);
        if mode != 0 {
            e.encode_decision(&mut st.ctx[CTX_INTRA_CHROMA_PRED_MODE + 3], (mode != 1) as u32);
            if mode != 1 {
                e.encode_decision(&mut st.ctx[CTX_INTRA_CHROMA_PRED_MODE + 3], (mode == 3) as u32);
            }
        }
    }
}

/// `coded_block_pattern` (9.3.3.1.1.4).
fn decode_cbp(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    nb: &MbNeighbours,
    chroma: bool,
) -> u8 {
    let mut cbp: u32 = 0;
    // Luma: four bins, 8x8 blocks 0..3; each context from the left/above 8x8
    // block (in this MB from earlier bins).
    for b8 in 0..4u32 {
        let (bx8, by8) = (b8 & 1, b8 >> 1);
        let cond = |dx: i32, dy: i32| -> u32 {
            let nx = bx8 as i32 + dx;
            let ny = by8 as i32 + dy;
            if nx >= 0 && ny >= 0 {
                // Inside the current MB: prior bin.
                let nb8 = (ny as u32) * 2 + nx as u32;
                return if (cbp >> nb8) & 1 != 0 { 0 } else { 1 };
            }
            // The 8x8 block holding the sample left of / above this 8x8's
            // top-left (6.4.11.2).
            let Some((addr, blk)) = nb.block(bx8 as i32 * 2 + dx, by8 as i32 * 2 + dy) else {
                return 0;
            };
            let other_b8 = ((blk / 8) * 2 + (blk % 4) / 2) as u32;
            let m = &info.mbs[addr];
            if m.kind == MbKind::IPcm {
                return 0;
            }
            if m.kind.is_skip() {
                return 1;
            }
            if (m.cbp >> other_b8) & 1 != 0 { 0 } else { 1 }
        };
        let inc = cond(-1, 0) + 2 * cond(0, -1);
        cbp |= bin(c, st, CTX_CBP_LUMA + inc as usize) << b8;
    }
    if chroma {
        let cond = |a: Option<usize>, want_two: bool| -> usize {
            match a {
                Some(addr) => {
                    let m = &info.mbs[addr];
                    if m.kind == MbKind::IPcm {
                        return 1;
                    }
                    if m.kind.is_skip() {
                        return 0;
                    }
                    let ch = m.cbp >> 4;
                    if want_two {
                        (ch == 2) as usize
                    } else {
                        (ch != 0) as usize
                    }
                }
                None => 0,
            }
        };
        let inc = cond(nb.a, false) + 2 * cond(nb.b, false);
        if bin(c, st, CTX_CBP_CHROMA + inc) != 0 {
            let inc2 = cond(nb.a, true) + 2 * cond(nb.b, true) + 4;
            let two = bin(c, st, CTX_CBP_CHROMA + inc2);
            cbp |= (1 + two) << 4;
        }
    }
    cbp as u8
}

/// What the CABAC contexts of later macroblocks need to remember about one
/// already-written macroblock — the writer-side mirror of what the
/// decoder stores per macroblock (`MbInfo` plus the picture's nonzero-count
/// arrays), kept by the caller that walks the picture and handed to
/// [`write_cbp_cabac`] and [`write_intra_residual_cabac`] as the left and
/// above neighbours (`None` = not available: outside the picture or the
/// slice).
///
/// Build one with [`WrittenMb::from_decision`] after writing a macroblock,
/// or [`WrittenMb::pcm`] for an I_PCM one; a P macroblock — `P_L0_16x16`
/// or `P_Skip` — comes from [`WrittenMb::from_inter_decision`]. B
/// macroblocks have no representation yet.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WrittenMb {
    /// The macroblock is I_PCM.
    pub pcm: bool,
    /// The macroblock is I_16x16 (the luma-DC coded_block_flag context
    /// reads it).
    pub i16x16: bool,
    /// `transform_size_8x8_flag`, which is what the *next* macroblock's
    /// own flag reads of this one: [`decode_transform_8x8`]'s ctxIdxInc
    /// counts neighbours whose flag was set. In 4:4:4 it is also what the
    /// 8x8 coded_block_flag contexts ask of a neighbour — one transformed
    /// 4x4 contributes zero to those whatever its counts say.
    pub transform_8x8: bool,
    /// `coded_block_pattern` in the decoder's layout: luma bits 0..=3 (0 or
    /// 15 for I_16x16), the chroma value (0 / 1 / 2) in bits 4..
    pub cbp: u8,
    /// coded_block_flags of the DC blocks, as the decoder keeps them: bit 0
    /// luma DC (set iff the macroblock is I_16x16 and its written DC block
    /// had a nonzero coefficient), bits 1 / 2 Cb / Cr DC (set iff chroma
    /// residual was written at all and that component's DC block had one).
    pub dc_cbf: u8,
    /// Nonzero-coefficient count per luma 4x4 block (raster order), exactly
    /// as the residual writer counts them: all 16 coefficients for I_NxN,
    /// the 15 AC coefficients for I_16x16 (the DC lives in `dc_cbf`), 0 for
    /// every block of an uncoded 8x8, 16 for I_PCM.
    pub nz_luma: [u8; 16],
    /// The same for the chroma AC blocks, per component: raster over two
    /// columns and two (4:2:0) or four (4:2:2) rows; all zero when the
    /// chroma cbp is below 2 (no AC coded), 16 for I_PCM.
    pub nz_chroma: [[u8; 16]; 2],
    /// The macroblock was skipped (`P_Skip`): the `mb_skip_flag` context
    /// counts non-skipped neighbours, and the cbp contexts treat a skipped
    /// neighbour specially (luma condTermFlag 1, chroma 0).
    pub skip: bool,
    /// The macroblock is intra (any I kind, I_PCM included): the ref_idx
    /// and mvd contexts count an intra neighbour as zero.
    pub intra: bool,
    /// Reference index into list 0 per 4x4 block (raster), as the
    /// decoder's motion array holds it: the coded index for inter blocks,
    /// 0 for `P_Skip`, -1 for intra. The contexts only ever ask `> 0`
    /// (refIdxZeroFlag), and never of a skipped or intra macroblock.
    pub ref_idx: [i8; 16],
    /// `mvd_lX` per list and 4x4 block (raster), as the decoder's
    /// per-picture mvd arrays hold them: the partition's mvd replicated
    /// over its blocks, zero for intra, skipped and direct macroblocks
    /// (which carry none) and for a list the macroblock does not use.
    pub mvd: [[Mv; 16]; 2],
    /// The macroblock is `B_Skip` or `B_Direct_16x16`: the B `mb_type`
    /// first-bin context counts available neighbours that are *neither*
    /// (9.3.3.1.1.3). False for everything in a P slice.
    pub direct: bool,
}

/// The luma nonzero counts as the decoder will store them: a block of an
/// uncoded 8x8 counts zero whatever the decision array holds.
fn gate_nz_luma(cbp_luma: u8, nz: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (r, o) in out.iter_mut().enumerate() {
        let b8 = (r / 8) * 2 + (r % 4) / 2;
        if cbp_luma & (1 << b8) != 0 {
            *o = nz[r];
        }
    }
    out
}

/// The luma nonzero counts an 8x8-transformed macroblock leaves for its
/// neighbours: one count per 8x8 block, on all four of its 4x4s.
///
/// The decision side counts per 4x4 *sub-scan* — the four interleaved
/// blocks CAVLC codes an 8x8 as — and those four counts sum to the 8x8's
/// total, because the sub-scans partition its sixty-four positions. What
/// the decoder's CABAC parser stores is that total, replicated
/// (`parse_residual_luma_like_cabac`). The difference is invisible in
/// every macroblock but one whose 8x8 has coefficients in some sub-scans
/// and none in others — where a neighbour would read "no coefficients"
/// from a block the decoder calls coded, and the coded_block_flag
/// contexts would part company.
///
/// A 4x4-transformed macroblock passes through untouched.
fn spread8(transform_8x8: bool, nz: &[u8; 16]) -> [u8; 16] {
    if !transform_8x8 {
        return *nz;
    }
    let mut out = [0u8; 16];
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        let idx =
            [by8 * 4 + bx8, by8 * 4 + bx8 + 1, (by8 + 1) * 4 + bx8, (by8 + 1) * 4 + bx8 + 1];
        let total: u32 = idx.iter().map(|&i| nz[i] as u32).sum();
        let total = total.min(64) as u8;
        for i in idx {
            out[i] = total;
        }
    }
    out
}

/// The chroma-DC coded_block_flags as the decoder will store them: the
/// bits exist only when chroma residual was written at all.
fn chroma_dc_cbf(cbp_chroma: u8, chroma_dc: &[[i16; 16]; 2]) -> u8 {
    let mut dc_cbf = 0u8;
    if cbp_chroma != 0 {
        for comp in 0..2 {
            if chroma_dc[comp].iter().any(|&v| v != 0) {
                dc_cbf |= 2 << comp;
            }
        }
    }
    dc_cbf
}

impl WrittenMb {
    /// The state of a macroblock written from `d` — derived here, next to
    /// the contexts that read it, so a caller cannot get the gating subtly
    /// wrong: blocks of an uncoded 8x8 count zero whatever the decision
    /// arrays hold, and a DC flag is set only when that DC block was
    /// actually written.
    #[allow(dead_code)] // the picture loop being built is the caller
    pub(crate) fn from_decision(d: &MbDecision, c444: bool) -> Self {
        let i16x16 = d.kind == IntraKind::I16x16;
        let mut dc_cbf = if c444 {
            // 4:4:4: bits 1 / 2 are the Cb / Cr planes' Intra_16x16 DC
            // flags, as the decoder stores them (`dc_cbf |= 1 << p`).
            let mut b = 0u8;
            if i16x16 {
                for comp in 0..2 {
                    if d.chroma_dc[comp].iter().any(|&v| v != 0) {
                        b |= 2 << comp;
                    }
                }
            }
            b
        } else {
            chroma_dc_cbf(d.cbp_chroma, &d.chroma_dc)
        };
        if i16x16 && d.luma_dc.iter().any(|&v| v != 0) {
            dc_cbf |= 1;
        }
        WrittenMb {
            pcm: false,
            i16x16,
            transform_8x8: d.transform_8x8,
            cbp: (d.cbp_luma & 15) | (d.cbp_chroma << 4),
            dc_cbf,
            // An 8x8-transformed macroblock stores one count per 8x8 over
            // all four of its 4x4s, because that is what the decoder's
            // CABAC parser writes into `layer.nz`
            // (`parse_residual_luma_like_cabac` below). The decision
            // counts per 4x4 sub-scan — CAVLC's view — so the two are
            // reconciled here, once, beside the contexts that read it.
            nz_luma: gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_luma)),
            nz_chroma: if c444 {
                // The planes' luma-style counts, gated by the shared cbp
                // and spread by the shared transform size exactly as
                // plane 0's are — in 4:4:4 Cb and Cr *are* luma-style
                // planes, so every rule that applies to plane 0's counts
                // applies to theirs.
                [
                    gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[0])),
                    gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[1])),
                ]
            } else if d.cbp_chroma == 2 {
                d.nz_chroma
            } else {
                [[0; 16]; 2]
            },
            skip: false,
            intra: true,
            ref_idx: [-1; 16],
            mvd: [[Mv::ZERO; 16]; 2],
            direct: false,
        }
    }

    /// The state of a P macroblock written from `d`, of any coded shape.
    /// `P_Skip` stores what the decoder stores for one — no residual,
    /// reference 0, no mvd. `UseIntra` is refused: the macroblock
    /// actually coded is the intra decision, so build from *that* via
    /// [`WrittenMb::from_decision`].
    #[allow(dead_code)] // the picture loop being built is the caller
    pub(crate) fn from_inter_decision(d: &InterDecision, c444: bool) -> Self {
        match d.kind {
            InterMbKind::UseIntra => {
                unreachable!("UseIntra codes the intra decision; build from that")
            }
            InterMbKind::PSkip => {
                debug_assert!(
                    d.cbp_luma == 0 && d.cbp_chroma == 0,
                    "a skip with residual is not a skip"
                );
                WrittenMb {
                    pcm: false,
                    i16x16: false,
                    transform_8x8: false,
                    cbp: 0,
                    dc_cbf: 0,
                    nz_luma: [0; 16],
                    nz_chroma: [[0; 16]; 2],
                    skip: true,
                    intra: false,
                    ref_idx: [0; 16],
                    mvd: [[Mv::ZERO; 16]; 2],
                    direct: false,
                }
            }
            InterMbKind::P16x16
            | InterMbKind::P16x8
            | InterMbKind::P8x16
            | InterMbKind::P8x8 => WrittenMb {
                pcm: false,
                i16x16: false,
                transform_8x8: d.transform_8x8,
                cbp: (d.cbp_luma & 15) | (d.cbp_chroma << 4),
                // 4:4:4 inter planes have no DC block (each 4x4 keeps its
                // own DC), so the plane DC flags stay clear.
                dc_cbf: if c444 { 0 } else { chroma_dc_cbf(d.cbp_chroma, &d.chroma_dc) },
                nz_luma: gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_luma)),
                nz_chroma: if c444 {
                    [
                        gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[0])),
                        gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[1])),
                    ]
                } else if d.cbp_chroma == 2 {
                    d.nz_chroma
                } else {
                    [[0; 16]; 2]
                },
                skip: false,
                intra: false,
                ref_idx: [d.ref_idx; 16],
                // Each partition's mvd over its own 4x4 blocks, which is
                // what the decoder's CABAC parser stores and what the
                // *next* macroblock's mvd contexts read across the edge
                // (`mvd_neighbour_abs`). One vector for all sixteen was
                // faithful only while there was one partition.
                mvd: [d.mvd, [Mv::ZERO; 16]],
                direct: false,
            },
        }
    }

    /// An I_PCM macroblock: every count is stored as 16, which is how the
    /// decoder's arrays spell it (and what makes every "is the neighbouring
    /// block coded?" question read yes).
    #[allow(dead_code)] // the picture loop being built is the caller
    pub(crate) fn pcm() -> Self {
        WrittenMb {
            pcm: true,
            i16x16: false,
            transform_8x8: false,
            cbp: 0,
            dc_cbf: 0,
            nz_luma: [16; 16],
            nz_chroma: [[16; 16]; 2],
            skip: false,
            intra: true,
            ref_idx: [-1; 16],
            mvd: [[Mv::ZERO; 16]; 2],
            direct: false,
        }
    }

    /// The state of a B macroblock written from `d`, of any coded shape:
    /// the same gating as the P constructor, per list. The direct kinds
    /// store no mvd and no residual-bearing state beyond what their cbp
    /// says; `direct` marks both `B_Skip` and `B_Direct_16x16`, which is
    /// what the B `mb_type` first-bin context counts — a `B_8x8` with
    /// direct sub-macroblocks is *not* marked, exactly as the reader's
    /// `decode_mb_type` counts it, and its direct blocks carry the zero
    /// mvd the reader's `is_direct_block` reads for them.
    pub(crate) fn from_b_decision(d: &BDecision, c444: bool) -> Self {
        // The list-0 reference index per 4x4, from its 8x8 partition's.
        let mut ref_idx = [0i8; 16];
        for (blk, r) in ref_idx.iter_mut().enumerate() {
            *r = d.ref_idx[(blk / 8) * 2 + (blk % 4) / 2][0];
        }
        match d.kind {
            BMbKind::UseIntra => {
                unreachable!("UseIntra codes the intra decision; build from that")
            }
            BMbKind::BSkip => {
                debug_assert!(
                    d.cbp_luma == 0 && d.cbp_chroma == 0,
                    "a skip with residual is not a skip"
                );
                WrittenMb {
                    pcm: false,
                    i16x16: false,
                    transform_8x8: false,
                    cbp: 0,
                    dc_cbf: 0,
                    nz_luma: [0; 16],
                    nz_chroma: [[0; 16]; 2],
                    skip: true,
                    intra: false,
                    ref_idx,
                    mvd: [[Mv::ZERO; 16]; 2],
                    direct: true,
                }
            }
            BMbKind::BDirect16
            | BMbKind::B16
            | BMbKind::B16x8
            | BMbKind::B8x16
            | BMbKind::B8x8 => WrittenMb {
                pcm: false,
                i16x16: false,
                transform_8x8: d.transform_8x8,
                cbp: (d.cbp_luma & 15) | (d.cbp_chroma << 4),
                dc_cbf: if c444 { 0 } else { chroma_dc_cbf(d.cbp_chroma, &d.chroma_dc) },
                nz_luma: gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_luma)),
                nz_chroma: if c444 {
                    [
                        gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[0])),
                        gate_nz_luma(d.cbp_luma, &spread8(d.transform_8x8, &d.nz_chroma[1])),
                    ]
                } else if d.cbp_chroma == 2 {
                    d.nz_chroma
                } else {
                    [[0; 16]; 2]
                },
                skip: false,
                intra: false,
                ref_idx,
                // Each partition's mvd over its own 4x4 blocks per list,
                // as the decoder's CABAC parser stores them; zero over
                // direct-predicted blocks, which carry none.
                mvd: d.mvd,
                direct: d.kind == BMbKind::BDirect16,
            },
        }
    }
}

/// Write `coded_block_pattern`: the exact inverse of [`decode_cbp`]. `cbp`
/// is in the decoder's layout (luma bits 0..=3, the chroma value in bits
/// 4..), and `chroma` says whether the two chroma bins exist at all (true
/// for 4:2:0 / 4:2:2; false for monochrome — 4:4:4 would carry luma-style
/// planes this writer does not spell).
///
/// The luma contexts read the neighbouring 8x8 blocks' cbp bits: earlier
/// bins of this same value inside the macroblock, and `left` / `above`
/// across its edges — which is why the caller hands the neighbour state in
/// rather than an increment: which neighbouring bit matters differs per
/// bin, and that geometry belongs beside the reader's.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_cbp_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    cbp: u8,
    chroma: bool,
) {
    let cbp = cbp as u32;
    for b8 in 0..4u32 {
        let (bx8, by8) = (b8 & 1, b8 >> 1);
        let cond = |dx: i32, dy: i32| -> u32 {
            let nx = bx8 as i32 + dx;
            let ny = by8 as i32 + dy;
            if nx >= 0 && ny >= 0 {
                // Inside this macroblock: an earlier bin of the value being
                // written (the 8x8s left of and above b8 come earlier in
                // the 2x2 raster), which is exactly the bit the reader has
                // decoded by this point.
                let nb8 = (ny as u32) * 2 + nx as u32;
                return if (cbp >> nb8) & 1 != 0 { 0 } else { 1 };
            }
            // The neighbouring macroblock's 8x8 adjacent to this one
            // (6.4.11.2): the left MB's right column, the above MB's
            // bottom row.
            let (m, other_b8) = if nx < 0 { (left, by8 * 2 + 1) } else { (above, 2 + bx8) };
            let Some(m) = m else { return 0 };
            if m.pcm {
                return 0;
            }
            if m.skip {
                return 1;
            }
            if (m.cbp >> other_b8) & 1 != 0 { 0 } else { 1 }
        };
        let inc = cond(-1, 0) + 2 * cond(0, -1);
        e.encode_decision(&mut st.ctx[CTX_CBP_LUMA + inc as usize], (cbp >> b8) & 1);
    }
    if chroma {
        let cond = |m: Option<&WrittenMb>, want_two: bool| -> usize {
            match m {
                Some(m) => {
                    if m.pcm {
                        return 1;
                    }
                    if m.skip {
                        return 0;
                    }
                    let ch = m.cbp >> 4;
                    if want_two { (ch == 2) as usize } else { (ch != 0) as usize }
                }
                None => 0,
            }
        };
        let ch = cbp >> 4;
        let inc = cond(left, false) + 2 * cond(above, false);
        e.encode_decision(&mut st.ctx[CTX_CBP_CHROMA + inc], (ch != 0) as u32);
        if ch != 0 {
            let inc2 = cond(left, true) + 2 * cond(above, true) + 4;
            e.encode_decision(&mut st.ctx[CTX_CBP_CHROMA + inc2], (ch == 2) as u32);
        }
    }
}

/// `mb_qp_delta` (9.3.3.1.1.5): mapped unary.
fn decode_qp_delta(c: &mut Cabac, st: &mut CabacState) -> Result<i32> {
    let inc = st.prev_qp_delta_nonzero as usize;
    if bin(c, st, CTX_MB_QP_DELTA + inc) == 0 {
        return Ok(0);
    }
    let mut k = 1u32;
    let mut ctx = CTX_MB_QP_DELTA + 2;
    while bin(c, st, ctx) != 0 {
        k += 1;
        ctx = CTX_MB_QP_DELTA + 3;
        if k > 52 {
            return Err(Error::bitstream("mb_qp_delta runaway"));
        }
    }
    let v = if k & 1 == 1 {
        ((k + 1) / 2) as i32
    } else {
        -((k / 2) as i32)
    };
    Ok(v)
}

/// Write `mb_qp_delta`: the exact inverse of [`decode_qp_delta`]'s mapped
/// unary — a value `v > 0` is `2v - 1` one-bins, `v < 0` is `-2v` of them,
/// then the stopping zero. The first bin's context is picked by
/// `st.prev_qp_delta_nonzero`, the second uses offset 2 and the rest 3
/// (the stopping zero takes whichever context the next one-bin would
/// have).
///
/// Present iff the macroblock has residual (`cbp != 0` or I_16x16); the
/// caller writes it between the cbp and the residual — and *the caller
/// maintains `st.prev_qp_delta_nonzero`*, exactly as [`parse_mb_cabac`]
/// does around its reader: `qp_delta != 0` after a macroblock with
/// residual, `false` after one without (I_PCM included). This function
/// only reads the flag, because its reader only reads it.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_mb_qp_delta_cabac(e: &mut CabacEncoder, st: &mut CabacState, qp_delta: i32) {
    debug_assert!((-26..=25).contains(&qp_delta), "mb_qp_delta out of range");
    let inc = st.prev_qp_delta_nonzero as usize;
    if qp_delta == 0 {
        e.encode_decision(&mut st.ctx[CTX_MB_QP_DELTA + inc], 0);
        return;
    }
    e.encode_decision(&mut st.ctx[CTX_MB_QP_DELTA + inc], 1);
    let k = if qp_delta > 0 { 2 * qp_delta - 1 } else { -2 * qp_delta } as u32;
    for i in 1..k {
        e.encode_decision(&mut st.ctx[CTX_MB_QP_DELTA + if i == 1 { 2 } else { 3 }], 1);
    }
    e.encode_decision(&mut st.ctx[CTX_MB_QP_DELTA + if k == 1 { 2 } else { 3 }], 0);
}

/// `transform_size_8x8_flag` (9.3.3.1.1.10).
fn decode_transform_8x8(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> bool {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) => info.mbs[addr].transform_8x8 as usize,
            None => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    bin(c, st, CTX_TRANSFORM_8X8 + inc) != 0
}

/// Write `transform_size_8x8_flag`: the exact inverse of
/// [`decode_transform_8x8`], over the writer's neighbour state.
///
/// One bin, one context, and the whole of the derivation is the
/// increment: condTermFlagN is the neighbouring macroblock's own
/// `transform_size_8x8_flag`, and zero where the neighbour is
/// unavailable. Unlike almost every other increment in this file it does
/// *not* fall back to the current macroblock's intra-ness — the reader's
/// `cond` returns 0 for `None` outright — so a writer that reused the
/// coded_block_flag habit would be wrong on the top row and the left
/// column of every picture, which is exactly where nothing else is
/// coded yet to notice.
///
/// The caller writes this where the reader takes it, and the two places
/// differ: for `I_NxN` it comes *before* `mb_pred()` (and decides whether
/// the four 8x8 modes or the sixteen 4x4 ones follow); for an inter
/// macroblock it comes *after* `coded_block_pattern`, and only when some
/// luma block is coded.
pub(crate) fn write_transform_8x8_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    flag: bool,
) {
    let cond = |m: Option<&WrittenMb>| -> usize { m.is_some_and(|m| m.transform_8x8) as usize };
    let inc = cond(left) + cond(above);
    e.encode_decision(&mut st.ctx[CTX_TRANSFORM_8X8 + inc], flag as u32);
}

// ---------------------------------------------------------------------------
// Residual blocks (9.3.3.1.1.9, 9.3.3.1.3)
// ---------------------------------------------------------------------------

/// The `coded_block_flag` context increment for a block of category `cat`
/// at luma 4x4 raster `(bx, by)` (cat 1/2), chroma block `(comp, blk)`
/// (cat 4), or the DC blocks (cat 0/3).
#[allow(clippy::too_many_arguments)]
fn cbf_ctx_inc(
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    x264_old_444: bool,
    cat: usize,
    bx: usize,
    by: usize,
    comp: usize,
    blk: usize,
) -> usize {
    let cur_intra = layer.kind.is_intra();
    let p = cat_plane(cat);
    let cat_8x8 = matches!(cat, 5 | 9 | 13);
    // condTermFlagN for the luma-style 4x4 (or 8x8) block of plane `p` at
    // 4x4 offset `(dx, dy)` from block `(bx, by)`. The block's flag is its
    // nonzero count: an 8x8 block without a coded_block_flag (non-4:4:4,
    // inferred 1) always carries a coefficient, and one with the flag
    // (4:4:4) carries one iff the flag was 1 — so the count stands in for
    // both, and for a neighbouring 8x8-transformed macroblock it is the
    // 8x8 block's flag on all four of its 4x4s. An 8x8 block's own flag
    // (categories 5 / 9 / 13, 4:4:4 only) looks at neighbouring 8x8
    // *transform* blocks: a neighbour transformed 4x4 contributes 0.
    let cond_luma = |dx: i32, dy: i32| -> usize {
        let (nx, ny) = (bx as i32 + dx, by as i32 + dy);
        if nx >= 0 && ny >= 0 {
            // Current MB: block available iff its 8x8 has cbp set (it has,
            // or we would not be decoding it) — the flag is the count so far.
            let nblk = (ny * 4 + nx) as usize;
            let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
            if layer.cbp & (1 << b8) == 0 {
                return 0;
            }
            return (layer.nz[p][nblk] != 0) as usize;
        }
        if !cat_8x8 {
            // A neighbouring macroblock's block, from the gathered counts:
            // skipped 0, I_PCM 16, an uncoded 8x8 0 — the count is the flag.
            let side = if nx < 0 { 0 } else { 1 };
            if !nb.nz_avail[side] {
                return cur_intra as usize;
            }
            let v = if nx < 0 { nb.nz_left[p][ny as usize] } else { nb.nz_top[p][nx as usize] };
            return (v != 0) as usize;
        }
        let Some((addr, nblk)) = nb.block(nx, ny) else {
            return cur_intra as usize;
        };
        let m = &info.mbs[addr];
        if cat_8x8 && !m.transform_8x8 && x264_old_444 {
            // Old x264 (before build 151) coded these as if the neighbour
            // were unavailable.
            return cur_intra as usize;
        }
        if m.kind == MbKind::IPcm {
            return 1;
        }
        if m.kind.is_skip() || (cat_8x8 && !m.transform_8x8) {
            return 0;
        }
        let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
        if m.cbp & (1 << b8) == 0 {
            return 0;
        }
        (info.plane_nz(p, addr, nblk) != 0) as usize
    };
    let cond_dc = |a: Option<usize>, bit: u8, needs_i16: bool| -> usize {
        match a {
            None => cur_intra as usize,
            Some(addr) => {
                let m = &info.mbs[addr];
                if m.kind == MbKind::IPcm {
                    return 1;
                }
                if needs_i16 {
                    if m.kind != MbKind::I16x16 {
                        return 0;
                    }
                } else if m.kind.is_skip() || m.cbp & 0x30 == 0 {
                    return 0;
                }
                ((m.dc_cbf >> bit) & 1) as usize
            }
        }
    };
    let cond_chroma_ac = |dx: i32, dy: i32| -> usize {
        // Chroma 2-column grid neighbours (2 rows for 4:2:0, 4 for 4:2:2).
        let (cx, cy) = ((blk % 2) as i32 + dx, (blk / 2) as i32 + dy);
        if cx >= 0 && cy >= 0 {
            return (layer.chroma_nz[comp][(cy * 2 + cx) as usize] != 0) as usize;
        }
        // A neighbouring macroblock's block, from the gathered counts
        // (skipped or without chroma AC 0, I_PCM 16).
        let side = if cx < 0 { 0 } else { 1 };
        if !nb.nz_avail[side] {
            return cur_intra as usize;
        }
        let v = if cx < 0 { nb.nzc_left[comp][cy as usize] } else { nb.nzc_top[comp][cx as usize] };
        (v != 0) as usize
    };
    match cat {
        // Luma-style DC (Intra_16x16 of the plane): bit p of `dc_cbf`.
        0 | 6 | 10 => cond_dc(nb.a, p as u8, true) + 2 * cond_dc(nb.b, p as u8, true),
        // Luma-style AC / 4x4 / 8x8 (the 8x8's top-left 4x4 is `(bx, by)`).
        1 | 2 | 5 | 7 | 8 | 9 | 11 | 12 | 13 => cond_luma(-1, 0) + 2 * cond_luma(0, -1),
        CAT_CHROMA_DC => {
            cond_dc(nb.a, 1 + comp as u8, false) + 2 * cond_dc(nb.b, 1 + comp as u8, false)
        }
        CAT_CHROMA_AC => cond_chroma_ac(-1, 0) + 2 * cond_chroma_ac(0, -1),
        _ => 0,
    }
}

/// The macroblock currently being written, as its own residual contexts
/// read it: whether it is intra (an unavailable neighbour's condTermFlag
/// in 9.3.3.1.1.9 is the *current* macroblock's intra-ness), its luma cbp
/// bits, and its nonzero counts — usable for the in-macroblock neighbour
/// lookups because every block's left / above neighbour inside a
/// macroblock comes earlier in block order, so its count is what the
/// decoder will have stored by the time it derives the same increment.
struct CurMbResidual {
    /// The macroblock is intra (any I kind).
    intra: bool,
    /// `transform_size_8x8_flag`: the luma-style planes are coded as four
    /// 8x8 blocks rather than sixteen 4x4 ones, which changes the block
    /// categories, the scan, and (outside 4:4:4) whether a
    /// coded_block_flag exists at all.
    transform_8x8: bool,
    /// `CodedBlockPatternLuma` (bits 0..=3).
    cbp_luma: u8,
    /// Nonzero counts per luma 4x4 (raster) *as the decoder will store
    /// them*: full counts for 4x4-coded blocks, AC counts for
    /// Intra_16x16, and one 8x8's total on all four of its 4x4s under the
    /// 8x8 transform ([`spread8`]). Owned rather than borrowed for
    /// exactly that reason — the decision's array is the other view.
    nz_luma: [u8; 16],
    /// Nonzero counts per chroma AC block, per component — in 4:4:4 the
    /// same field holds the Cb / Cr planes' luma-style counts (sixteen
    /// blocks, raster), exactly as the decoder's `chroma_nz` arrays do,
    /// spread with the luma plane's when the 8x8 transform is on.
    nz_chroma: [[u8; 16]; 2],
}

impl CurMbResidual {
    /// Plane `p`'s nonzero counts, luma-style (4:4:4's plane view).
    fn nz_plane(&self, p: usize) -> &[u8; 16] {
        if p == 0 { &self.nz_luma } else { &self.nz_chroma[p - 1] }
    }
}

/// The `coded_block_flag` ctxIdxInc for a residual block of the macroblock
/// being *written*: [`cbf_ctx_inc`]'s mirror over the state an encoder's
/// picture loop keeps ([`WrittenMb`] neighbours plus the current
/// macroblock's own [`CurMbResidual`]) instead of the decoder's picture
/// arrays. It covers the categories a 4:2:0 / 4:2:2 macroblock codes with
/// the 4x4 transform (luma DC / AC / 4x4, chroma DC / AC).
///
/// `rows` is the chroma block-row count (2 for 4:2:0, 4 for 4:2:2): it
/// picks which of the above neighbour's chroma blocks sit on the shared
/// edge.
///
/// The 8x8 categories (5 / 9 / 13) are only ever *coded* in 4:4:4 —
/// elsewhere an 8x8 luma block's coded_block_flag is inferred 1 and the
/// caller passes no increment — and they ask a different question of a
/// neighbouring macroblock: one that was transformed 4x4 contributes 0
/// whatever its counts say, because there is no 8x8 transform block there
/// to have a flag. The reader's `x264_old_444` bug-compatibility arm is
/// deliberately not mirrored: it exists to *decode* streams old x264
/// builds produced, and this encoder writes the standard's answer.
#[allow(clippy::too_many_arguments)]
fn enc_cbf_ctx_inc(
    cur: &CurMbResidual,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    rows: usize,
    cat: usize,
    bx: usize,
    by: usize,
    comp: usize,
    blk: usize,
) -> usize {
    let cur_intra = cur.intra as usize;
    let p = cat_plane(cat);
    let cat_8x8 = matches!(cat, 5 | 9 | 13);
    let cond_luma = |dx: i32, dy: i32| -> usize {
        let (nx, ny) = (bx as i32 + dx, by as i32 + dy);
        if nx >= 0 && ny >= 0 {
            let nblk = (ny * 4 + nx) as usize;
            let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
            if cur.cbp_luma & (1 << b8) == 0 {
                return 0;
            }
            return (cur.nz_plane(p)[nblk] != 0) as usize;
        }
        let (m, edge) = if nx < 0 {
            (left, ny as usize * 4 + 3) // the left MB's right column
        } else {
            (above, 12 + nx as usize) // the above MB's bottom row
        };
        match m {
            None => cur_intra,
            Some(m) => {
                if cat_8x8 {
                    if m.pcm {
                        return 1;
                    }
                    if m.skip || !m.transform_8x8 {
                        return 0;
                    }
                }
                let nz = if p == 0 { &m.nz_luma } else { &m.nz_chroma[p - 1] };
                // The counts are already gated by the neighbour's coded
                // block pattern (`gate_nz_luma`), which is the reader's
                // `m.cbp & (1 << b8) == 0` test.
                (nz[edge] != 0) as usize
            }
        }
    };
    let cond_dc = |m: Option<&WrittenMb>, bit: u8, needs_i16: bool| -> usize {
        match m {
            None => cur_intra,
            Some(m) => {
                if m.pcm {
                    return 1;
                }
                if needs_i16 {
                    if !m.i16x16 {
                        return 0;
                    }
                } else if m.cbp & 0x30 == 0 {
                    return 0;
                }
                ((m.dc_cbf >> bit) & 1) as usize
            }
        }
    };
    let cond_chroma_ac = |dx: i32, dy: i32| -> usize {
        let (cx, cy) = ((blk % 2) as i32 + dx, (blk / 2) as i32 + dy);
        if cx >= 0 && cy >= 0 {
            return (cur.nz_chroma[comp][(cy * 2 + cx) as usize] != 0) as usize;
        }
        let (m, edge) = if cx < 0 {
            (left, cy as usize * 2 + 1)
        } else {
            (above, (rows - 1) * 2 + cx as usize)
        };
        match m {
            None => cur_intra,
            Some(m) => (m.nz_chroma[comp][edge] != 0) as usize,
        }
    };
    match cat {
        // Luma-style DC (Intra_16x16 of the plane): bit p of `dc_cbf` —
        // the mirror of `cbf_ctx_inc`'s 0 | 6 | 10 arm.
        0 | 6 | 10 => cond_dc(left, p as u8, true) + 2 * cond_dc(above, p as u8, true),
        // Luma-style AC / 4x4 / 8x8 of the plane (the 8x8's top-left 4x4
        // is `(bx, by)`) — the mirror of `cbf_ctx_inc`'s combined arm.
        1 | 2 | 5 | 7 | 8 | 9 | 11 | 12 | 13 => cond_luma(-1, 0) + 2 * cond_luma(0, -1),
        CAT_CHROMA_DC => {
            cond_dc(left, 1 + comp as u8, false) + 2 * cond_dc(above, 1 + comp as u8, false)
        }
        CAT_CHROMA_AC => cond_chroma_ac(-1, 0) + 2 * cond_chroma_ac(0, -1),
        _ => unreachable!("category {cat} is not one a macroblock layer codes"),
    }
}

/// Decode one residual block's coefficients: the levels of scan positions
/// `0..max_coeff` are written to `out[scan[start + pos]]` (`out` is the
/// block in raster order, zero where nothing is written — the macroblock
/// layer's reset guarantees it). Returns the number of nonzero
/// coefficients. `cbf_inc` is `None` when coded_block_flag is not present
/// (8x8 luma in non-4:4:4), else its ctxIdxInc.
#[allow(clippy::too_many_arguments)]
fn residual_block_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    field: bool,
    cat: usize,
    cbf_inc: Option<usize>,
    out: &mut [i32],
    scan: &[u8],
    start: usize,
    max_coeff: usize,
    dq: Option<(&[i32], u32)>,
) -> Result<usize> {
    if let Some(inc) = cbf_inc {
        let trace = super::mb::syntax_trace();
        let pos0 = c.position();
        let v = bin(c, st, CBF_CTX_BASE[cat] + inc);
        if trace {
            eprintln!("cbf cat={cat} inc={inc} @{pos0} -> {v}");
        }
        if v == 0 {
            return Ok(0);
        }
    }
    let f = field as usize;
    let (sig_base, last_base, abs_base) = (
        SIG_CTX_BASE[f][cat],
        LAST_CTX_BASE[f][cat],
        ABS_CTX_BASE[cat],
    );
    // Significance map: the context increments per scan position (Table
    // 9-43: the position itself for the 4x4 categories, `Min(i / NumC8x8,
    // 2)` for chroma DC, tables for 8x8), and the significant positions.
    let (sig_off, last_off): (&[u8], &[u8]) = if max_coeff == 64 {
        (&SIG_COEFF_8X8_CTX[f][..], &LAST_COEFF_8X8_CTX[..])
    } else if cat == CAT_CHROMA_DC {
        if max_coeff == 8 { (&CHROMA_DC_422_SIG_OFF[..], &CHROMA_DC_422_SIG_OFF[..]) } else { (&IDENTITY_OFF[..], &IDENTITY_OFF[..]) }
    } else {
        (&IDENTITY_OFF[..], &IDENTITY_OFF[..])
    };
    let mut sig_pos = [0u8; 64];
    let mut n_sig = 0usize;
    let mut i = 0usize;
    let last = max_coeff - 1;
    while i < last {
        if bin(c, st, sig_base + sig_off[i] as usize) != 0 {
            sig_pos[n_sig] = i as u8;
            n_sig += 1;
            if bin(c, st, last_base + last_off[i] as usize) != 0 {
                break;
            }
        }
        i += 1;
    }
    if i == last {
        // Reached the end without a "last": the final coefficient is significant.
        sig_pos[n_sig] = last as u8;
        n_sig += 1;
    }
    // Levels, in reverse scan order (the highest frequency first).
    let mut num_gt1 = 0usize;
    let mut num_eq1 = 0usize;
    let inc1_cap = if cat == CAT_CHROMA_DC { 3 } else { 4 };
    for k in (0..n_sig).rev() {
        let pos = sig_pos[k] as usize;
        let inc0 = if num_gt1 != 0 { 0 } else { (1 + num_eq1).min(4) };
        let mut abs_m1: i32;
        if bin(c, st, abs_base + inc0) == 0 {
            abs_m1 = 0;
        } else {
            let inc1 = 5 + inc1_cap.min(num_gt1);
            let mut prefix = 1;
            while prefix < 14 && bin(c, st, abs_base + inc1) != 0 {
                prefix += 1;
            }
            abs_m1 = prefix;
            if prefix >= 14 {
                // UEG0 suffix.
                let mut kk = 0u32;
                loop {
                    if c.bypass() != 0 {
                        abs_m1 += 1 << kk;
                        kk += 1;
                        if kk > 24 {
                            return Err(Error::bitstream("coeff_abs_level suffix runaway"));
                        }
                    } else {
                        break;
                    }
                }
                while kk > 0 {
                    kk -= 1;
                    abs_m1 += (c.bypass() as i32) << kk;
                }
            }
        }
        let abs = abs_m1 + 1;
        if abs == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
        let sign = c.bypass();
        let level = if sign != 0 { -abs } else { abs };
        let idx = scan[start + pos] as usize;
        out[idx] = match dq {
            Some((table, shift)) => super::mb::dequant_level(level, table[idx], shift),
            None => level,
        };
    }
    if c.overrun() {
        return Err(Error::bitstream("CABAC: slice data truncated"));
    }
    Ok(n_sig)
}


/// Write one residual block's coefficients: the inverse of
/// [`residual_block_cabac`]. `levels` is the block in raster order, the
/// layout that function decodes *into*, and `scan` maps scan position to it
/// the same way. Returns the number of significant coefficients written.
///
/// It lives beside the reader on purpose. The two are exact inverses over a
/// binarisation with a lot of small rules — an inferred last coefficient, a
/// unary prefix that stops one bin early at its cap, an escape whose length
/// is implied — and every one of those rules has to be read off the same
/// lines twice. A change to one that is not made to the other is a desync,
/// and a desync is invisible until a later macroblock decodes as rubbish, so
/// the defence worth having is that the two are on the same screen.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn write_residual_block_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    cat: usize,
    cbf_inc: Option<usize>,
    levels: &[i32],
    scan: &[u8],
    start: usize,
    max_coeff: usize,
) -> usize {
    // The significant scan positions, in scan order.
    let mut sig_pos = [0u8; 64];
    let mut n_sig = 0usize;
    for pos in 0..max_coeff {
        if levels[scan[start + pos] as usize] != 0 {
            sig_pos[n_sig] = pos as u8;
            n_sig += 1;
        }
    }

    if let Some(inc) = cbf_inc {
        e.encode_decision(&mut st.ctx[CBF_CTX_BASE[cat] + inc], (n_sig != 0) as u32);
        if n_sig == 0 {
            return 0;
        }
    } else {
        // No coded_block_flag means it is inferred to be one (8x8 luma
        // outside 4:4:4), so an all-zero block cannot be spelled here at all
        // — the caller has to not code the block.
        debug_assert!(n_sig != 0, "block with an inferred coded_block_flag has no coefficient");
        if n_sig == 0 {
            return 0;
        }
    }

    let f = field as usize;
    let (sig_base, last_base, abs_base) =
        (SIG_CTX_BASE[f][cat], LAST_CTX_BASE[f][cat], ABS_CTX_BASE[cat]);
    let (sig_off, last_off): (&[u8], &[u8]) = if max_coeff == 64 {
        (&SIG_COEFF_8X8_CTX[f][..], &LAST_COEFF_8X8_CTX[..])
    } else if cat == CAT_CHROMA_DC {
        if max_coeff == 8 {
            (&CHROMA_DC_422_SIG_OFF[..], &CHROMA_DC_422_SIG_OFF[..])
        } else {
            (&IDENTITY_OFF[..], &IDENTITY_OFF[..])
        }
    } else {
        (&IDENTITY_OFF[..], &IDENTITY_OFF[..])
    };

    // Significance map. The final position is never coded when it is the last
    // scan position: the reader's loop stops before it and infers it, so
    // writing anything there would be a bin it never reads.
    let last = max_coeff - 1;
    let final_pos = sig_pos[n_sig - 1] as usize;
    let mut k = 0usize;
    let mut i = 0usize;
    while i < last {
        let is_sig = k < n_sig && sig_pos[k] as usize == i;
        e.encode_decision(&mut st.ctx[sig_base + sig_off[i] as usize], is_sig as u32);
        if is_sig {
            let is_last = i == final_pos;
            e.encode_decision(&mut st.ctx[last_base + last_off[i] as usize], is_last as u32);
            if is_last {
                break;
            }
            k += 1;
        }
        i += 1;
    }

    // Levels, highest frequency first.
    let mut num_gt1 = 0usize;
    let mut num_eq1 = 0usize;
    let inc1_cap = if cat == CAT_CHROMA_DC { 3 } else { 4 };
    for k in (0..n_sig).rev() {
        let pos = sig_pos[k] as usize;
        let level = levels[scan[start + pos] as usize];
        let abs = level.unsigned_abs();
        let abs_m1 = abs - 1;
        let inc0 = if num_gt1 != 0 { 0 } else { (1 + num_eq1).min(4) };
        if abs_m1 == 0 {
            e.encode_decision(&mut st.ctx[abs_base + inc0], 0);
        } else {
            e.encode_decision(&mut st.ctx[abs_base + inc0], 1);
            let inc1 = 5 + inc1_cap.min(num_gt1);
            // The reader counts from one and stops at fourteen without
            // consuming a terminator, so a capped prefix writes ones only.
            let prefix = abs_m1.min(14);
            for _ in 1..prefix {
                e.encode_decision(&mut st.ctx[abs_base + inc1], 1);
            }
            if prefix < 14 {
                e.encode_decision(&mut st.ctx[abs_base + inc1], 0);
            } else {
                // UEG0 escape: ones while the remainder covers the next
                // doubling, a zero, then that many bits of what is left.
                let mut rem = abs_m1 - 14;
                let mut kk = 0u32;
                while rem >= (1 << kk) {
                    e.encode_bypass(1);
                    rem -= 1 << kk;
                    kk += 1;
                    debug_assert!(kk <= 24, "coeff_abs_level too large to binarise");
                }
                e.encode_bypass(0);
                while kk > 0 {
                    kk -= 1;
                    e.encode_bypass((rem >> kk) & 1);
                }
            }
        }
        if abs == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
        e.encode_bypass((level < 0) as u32);
    }
    n_sig
}

/// The significance-map context increment for a coefficient of a 4x4-class
/// block: its scan position.
static IDENTITY_OFF: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// The same for 4:2:2 chroma DC (eight coefficients, NumC8x8 = 2):
/// `Min(i / 2, 2)`.
static CHROMA_DC_422_SIG_OFF: [u8; 8] = [0, 0, 1, 1, 2, 2, 2, 2];

/// `residual_luma()` (7.3.5.3.1) for CABAC, for colour plane `p`: the luma
/// plane, or Cb / Cr in 4:4:4 (coded like luma with their own categories,
/// and with a coded_block_flag on the 8x8 blocks too).
fn parse_residual_luma_like_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
    p: usize,
    dq: Option<&MbDequant>,
) -> Result<()> {
    let dq4: Option<(&[i32], u32)> = dq.map(|d| (&d.q4[p].0[..], d.q4[p].1));
    let dq8: Option<(&[i32], u32)> = dq.map(|d| (&d.q8[p].0[..], d.q8[p].1));
    let [cat_dc, cat_ac, cat_4x4, cat_8x8] = PLANE_CATS[p];
    let (scan4, scan8): (&[u8; 16], &[u8; 64]) = if ctx.field_pic || layer.field {
        (&FIELD_SCAN4X4, &FIELD_SCAN8X8)
    } else {
        (&ZIGZAG4X4, &ZIGZAG8X8)
    };
    let field = ctx.field_pic || layer.field;
    if layer.kind == MbKind::I16x16 {
        let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_dc, 0, 0, 0, 0);
        let n = residual_block_cabac(c, st, field, cat_dc, Some(inc), &mut layer.dc[p], scan4, 0, 16, None)?;
        if n > 0 {
            layer.dc_cbf |= 1 << p;
        }
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if layer.cbp & (1 << blk8) == 0 {
            continue;
        }
        if layer.transform_8x8 {
            // The 8x8 block's coded_block_flag is only coded in 4:4:4;
            // otherwise it is inferred 1.
            let inc = if ctx.chroma_format_idc == 3 {
                Some(cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_8x8, bx8, by8, 0, 0))
            } else {
                None
            };
            let base = blk8 * 64;
            let n = residual_block_cabac(c, st, field, cat_8x8, inc, &mut layer.coef[p][base..base + 64], scan8, 0, 64, dq8)?;
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                layer.nz[p][by * 4 + bx] = n as u8;
            }
        } else {
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                let base = raster * 16;
                let n = if layer.kind == MbKind::I16x16 {
                    let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_ac, bx, by, 0, 0);
                    // AC: 15 coefficients at scan positions 1..15.
                    residual_block_cabac(c, st, field, cat_ac, Some(inc), &mut layer.coef[p][base..base + 16], scan4, 1, 15, dq4)?
                } else {
                    let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_4x4, bx, by, 0, 0);
                    residual_block_cabac(c, st, field, cat_4x4, Some(inc), &mut layer.coef[p][base..base + 16], scan4, 0, 16, dq4)?
                };
                layer.nz[p][raster] = n as u8;
            }
        }
    }
    Ok(())
}

/// `residual()` for CABAC, filling the layer's coefficient arrays.
fn parse_residual_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
    dq: Option<&MbDequant>,
) -> Result<()> {
    let scan4: &[u8; 16] = if ctx.field_pic || layer.field {
        &FIELD_SCAN4X4
    } else {
        &ZIGZAG4X4
    };
    let field = ctx.field_pic || layer.field;
    parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 0, dq)?;
    if ctx.chroma_format_idc == 3 {
        parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 1, dq)?;
        parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 2, dq)?;
    }
    if (ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2) && layer.cbp & 0x30 != 0 {
        let c422 = ctx.chroma_format_idc == 2;
        let (n_dc, rows) = if c422 { (8usize, 4usize) } else { (4, 2) };
        let dc_scan: &[u8] = if c422 { &SCAN_CHROMA_DC_422[..] } else { &IDENTITY_OFF[..4] };
        for comp in 0..2 {
            let inc = cbf_ctx_inc(info, layer, nb, false, CAT_CHROMA_DC, 0, 0, comp, 0);
            let n = residual_block_cabac(c, st, field, CAT_CHROMA_DC, Some(inc), &mut layer.chroma_dc[comp], dc_scan, 0, n_dc, None)?;
            if n > 0 {
                layer.dc_cbf |= 2 << comp;
            }
        }
        if layer.cbp & 0x20 != 0 {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    let inc = cbf_ctx_inc(info, layer, nb, false, CAT_CHROMA_AC, 0, 0, comp, blk);
                    let n = residual_block_cabac(c, st, field, CAT_CHROMA_AC, Some(inc), &mut layer.chroma_ac[comp][blk], scan4, 1, 15, dq.map(|d| (&d.q4[1 + comp].0[..], d.q4[1 + comp].1)))?;
                    layer.chroma_nz[comp][blk] = n as u8;
                }
            }
        }
    }
    Ok(())
}

/// Write the residual of one intra macroblock: the inverse of
/// [`parse_residual_cabac`] over the domain the encoder emits — 4x4
/// transform, 4:2:0 / 4:2:2 / monochrome (4:4:4's luma-style chroma
/// planes and the 8x8 categories have no writer yet).
///
/// The caller writes this *after* `mb_qp_delta`, and only when the
/// macroblock has residual at all (`cbp != 0` or I_16x16), matching
/// [`parse_mb_cabac`]. `left` / `above` are the neighbouring macroblocks'
/// [`WrittenMb`] state for the coded_block_flag contexts. `field` selects
/// the field scans and context tables — the current encoder is frame-only
/// and passes false.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_intra_residual_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    chroma_format_idc: u32,
    d: &MbDecision,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
) {
    let cur =
        cur_residual(true, d.transform_8x8, chroma_format_idc, d.cbp_luma, &d.nz_luma, &d.nz_chroma);
    let dc = (d.kind == IntraKind::I16x16).then_some(&d.luma_dc);
    write_residual_walk_cabac(
        e, st, field, chroma_format_idc, &cur, dc, &d.luma, d.cbp_chroma, &d.chroma_dc,
        &d.chroma_ac, left, above,
    );
}

/// Write the residual of one `P_L0_16x16` macroblock: the same inverse of
/// [`parse_residual_cabac`], through the same shared walk — inter differs
/// only in having no DC split (each 4x4 block keeps its DC at position 0)
/// and in the coded_block_flag context selection (an unavailable
/// neighbour's condTermFlag is 0 for an inter macroblock, 1 for intra).
/// Written after `mb_qp_delta`, which for inter is present iff `cbp != 0`.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn write_inter_residual_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    chroma_format_idc: u32,
    d: &InterDecision,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
) {
    debug_assert!(
        !matches!(d.kind, InterMbKind::PSkip | InterMbKind::UseIntra),
        "only a coded inter macroblock carries residual syntax (a skip carries none)"
    );
    write_inter_residual_fields_cabac(
        e, st, field, chroma_format_idc, d.transform_8x8, d.cbp_luma, &d.nz_luma, &d.luma,
        d.cbp_chroma, &d.chroma_dc, &d.chroma_ac, &d.nz_chroma, left, above,
    );
}

/// The inter residual walk over its raw fields — what
/// [`write_inter_residual_cabac`] wraps for an [`InterDecision`], exposed
/// so a B decision (same residual layout, different motion shape) writes
/// through the identical path rather than a second spelling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_inter_residual_fields_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    chroma_format_idc: u32,
    transform_8x8: bool,
    cbp_luma: u8,
    nz_luma: &[u8; 16],
    luma: &[[i16; 16]; 16],
    cbp_chroma: u8,
    chroma_dc: &[[i16; 16]; 2],
    chroma_ac: &[[[i16; 16]; 16]; 2],
    nz_chroma: &[[u8; 16]; 2],
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
) {
    debug_assert!(
        !transform_8x8 || cbp_luma != 0,
        "an inter macroblock with no coded luma block carries no transform_size_8x8_flag, so a decoder infers it zero"
    );
    let cur =
        cur_residual(false, transform_8x8, chroma_format_idc, cbp_luma, nz_luma, nz_chroma);
    write_residual_walk_cabac(
        e, st, field, chroma_format_idc, &cur, None, luma, cbp_chroma, chroma_dc, chroma_ac,
        left, above,
    );
}

/// The current macroblock as its own residual contexts read it, from a
/// decision's fields.
///
/// The one thing it does beyond copying is [`spread8`]: the decision
/// counts an 8x8's four CAVLC sub-scans separately and the decoder stores
/// their total on all four blocks, and it is *that* view the contexts
/// want. Only the luma-style planes are ever 8x8-transformed — 4:2:0 and
/// 4:2:2 chroma has no 8x8 transform at all — so the chroma counts are
/// spread only in 4:4:4, where they are luma-style planes themselves.
fn cur_residual(
    intra: bool,
    transform_8x8: bool,
    chroma_format_idc: u32,
    cbp_luma: u8,
    nz_luma: &[u8; 16],
    nz_chroma: &[[u8; 16]; 2],
) -> CurMbResidual {
    let c444 = chroma_format_idc == 3;
    CurMbResidual {
        intra,
        transform_8x8,
        cbp_luma,
        nz_luma: spread8(transform_8x8, nz_luma),
        nz_chroma: [
            spread8(transform_8x8 && c444, &nz_chroma[0]),
            spread8(transform_8x8 && c444, &nz_chroma[1]),
        ],
    }
}

/// An 8x8 block's whole nonzero count, out of counts already spread over
/// its four 4x4s — which is what the one CABAC residual block for that
/// 8x8 must come out with.
fn quad_total(nz: &[u8; 16], blk8: usize) -> usize {
    let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
    nz[by8 * 4 + bx8] as usize
}

/// One luma-like plane's residual bins — the DC block for `Intra_16x16`,
/// then the coded 8x8s' blocks, which are four 4x4s or one 8x8 by the
/// macroblock's `transform_size_8x8_flag` — with plane `p`'s own
/// categories ([`PLANE_CATS`]) and coded_block_flag contexts. Luma is
/// plane 0; in 4:4:4 Cb and Cr are planes 1 and 2 coded the same way,
/// gated by the *same* luma coded-block-pattern bits, exactly as
/// `parse_residual_luma_like_cabac` reads them.
#[allow(clippy::too_many_arguments)]
fn write_plane_residual_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    cur: &CurMbResidual,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    rows: usize,
    c444: bool,
    p: usize,
    dc: Option<&[i16; 16]>,
    levels: &[[i16; 16]; 16],
) {
    let [cat_dc, cat_ac, cat_4x4, cat_8x8] = PLANE_CATS[p];
    let scan4: &[u8; 16] = if field { &FIELD_SCAN4X4 } else { &ZIGZAG4X4 };
    let scan8: &[u8; 64] = if field { &FIELD_SCAN8X8 } else { &ZIGZAG8X8 };
    let mut buf = [0i32; 16];
    if let Some(dc) = dc {
        debug_assert!(!cur.transform_8x8, "Intra_16x16 carries no transform_size_8x8_flag");
        for (o, &v) in buf.iter_mut().zip(dc) {
            *o = v as i32;
        }
        let inc = enc_cbf_ctx_inc(cur, left, above, rows, cat_dc, 0, 0, 0, 0);
        write_residual_block_cabac(e, st, field, cat_dc, Some(inc), &buf, scan4, 0, 16);
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if cur.cbp_luma & (1 << blk8) == 0 {
            // The two layouts put an 8x8's coefficients in different
            // places — quad `blk8` is the flat range `blk8 * 64` under
            // the 8x8 transform, and its four *raster* 4x4s otherwise,
            // which are not the same slots — so the emptiness check has
            // to follow the layout the levels are actually in.
            debug_assert!(
                if cur.transform_8x8 {
                    levels.as_flattened()[blk8 * 64..blk8 * 64 + 64].iter().all(|&v| v == 0)
                } else {
                    (0..4).all(|sub| {
                        let raster = (by8 + (sub >> 1)) * 4 + bx8 + (sub & 1);
                        levels[raster].iter().all(|&v| v == 0)
                    })
                },
                "plane {p} 8x8 {blk8} has coefficients but no cbp bit — they would be lost"
            );
            continue;
        }
        if cur.transform_8x8 {
            // One block of sixty-four, out of the storage `levels` shares
            // between the two transform layouts (quad `blk8` at flat
            // offset `blk8 * 64`), taken in the 8x8 scan.
            //
            // The coded_block_flag exists only in 4:4:4; everywhere else
            // it is inferred 1, which is what makes an 8x8 with a cbp bit
            // and no coefficient unspellable — the decision has to clear
            // the bit instead, and `write_residual_block_cabac` asserts
            // so rather than emitting a block the reader cannot parse.
            let mut buf8 = [0i32; 64];
            for (o, &v) in
                buf8.iter_mut().zip(&levels.as_flattened()[blk8 * 64..blk8 * 64 + 64])
            {
                *o = v as i32;
            }
            let cbf = c444
                .then(|| enc_cbf_ctx_inc(cur, left, above, rows, cat_8x8, bx8, by8, 0, 0));
            let n = write_residual_block_cabac(e, st, field, cat_8x8, cbf, &buf8, scan8, 0, 64);
            debug_assert_eq!(
                n,
                quad_total(cur.nz_plane(p), blk8),
                "plane {p} 8x8 {blk8}: the decision's counts disagree with the writer's"
            );
            continue;
        }
        for sub in 0..4 {
            let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
            let raster = by * 4 + bx;
            for (o, &v) in buf.iter_mut().zip(&levels[raster]) {
                *o = v as i32;
            }
            // Intra_16x16 codes the 15 AC coefficients (position 0 lives
            // in the DC block); everything else codes all 16.
            let (cat, start, max_coeff) = if dc.is_some() { (cat_ac, 1, 15) } else { (cat_4x4, 0, 16) };
            debug_assert!(dc.is_none() || buf[0] == 0, "I_16x16 AC keeps position 0 free");
            let inc = enc_cbf_ctx_inc(cur, left, above, rows, cat, bx, by, 0, 0);
            let n = write_residual_block_cabac(
                e, st, field, cat, Some(inc), &buf, scan4, start, max_coeff,
            );
            debug_assert_eq!(
                n,
                cur.nz_plane(p)[raster] as usize,
                "plane {p} nz[{raster}] disagrees with the levels; neighbour contexts would desync"
            );
        }
    }
}

/// The residual walk behind [`write_intra_residual_cabac`] and
/// [`write_inter_residual_cabac`] — one body, because the reader it
/// inverts ([`parse_residual_cabac`]) likewise serves both and two copies
/// of the walk could drift apart invisibly. `luma_dc` is `Some` exactly
/// for Intra_16x16: its DC block is written first, unconditionally (the
/// coded_block_flag may be zero), and the 4x4 blocks become
/// 15-coefficient AC; every other shape's blocks carry all 16
/// coefficients with the DC in place. Then, when the chroma cbp is
/// nonzero, both components' chroma DC; at chroma cbp 2, every chroma AC
/// block.
#[allow(clippy::too_many_arguments)]
fn write_residual_walk_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    chroma_format_idc: u32,
    cur: &CurMbResidual,
    luma_dc: Option<&[i16; 16]>,
    luma: &[[i16; 16]; 16],
    cbp_chroma: u8,
    chroma_dc: &[[i16; 16]; 2],
    chroma_ac: &[[[i16; 16]; 16]; 2],
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
) {
    let c422 = chroma_format_idc == 2;
    let rows = if c422 { 4 } else { 2 };

    // Luma, then (4:4:4) Cb and Cr coded the same way — the mirror of
    // `parse_residual_cabac`'s plane order, each plane's contexts its own
    // ([`PLANE_CATS`]).
    let c444 = chroma_format_idc == 3;
    write_plane_residual_cabac(e, st, field, cur, left, above, rows, c444, 0, luma_dc, luma);
    if c444 {
        for p in 1..3usize {
            write_plane_residual_cabac(
                e, st, field, cur, left, above, rows, c444, p,
                luma_dc.is_some().then_some(&chroma_dc[p - 1]),
                &chroma_ac[p - 1],
            );
        }
        debug_assert_eq!(cbp_chroma, 0, "ChromaArrayType 3 has no chroma cbp");
        return;
    }

    if chroma_format_idc == 0 || cbp_chroma == 0 {
        debug_assert!(chroma_format_idc != 0 || cbp_chroma == 0, "monochrome has no chroma cbp");
        return;
    }
    let scan4: &[u8; 16] = if field { &FIELD_SCAN4X4 } else { &ZIGZAG4X4 };
    let mut buf = [0i32; 16];
    let n_dc = if c422 { 8 } else { 4 };
    let dc_scan: &[u8] = if c422 { &SCAN_CHROMA_DC_422[..] } else { &IDENTITY_OFF[..4] };
    for comp in 0..2 {
        debug_assert!(
            c422 || chroma_dc[comp][4..].iter().all(|&v| v == 0),
            "4:2:0 chroma DC has four coefficients"
        );
        buf = [0; 16];
        for (o, &v) in buf.iter_mut().zip(&chroma_dc[comp][..n_dc]) {
            *o = v as i32;
        }
        let inc = enc_cbf_ctx_inc(cur, left, above, rows, CAT_CHROMA_DC, 0, 0, comp, 0);
        write_residual_block_cabac(e, st, field, CAT_CHROMA_DC, Some(inc), &buf, dc_scan, 0, n_dc);
    }
    if cbp_chroma == 2 {
        for comp in 0..2 {
            for blk in 0..2 * rows {
                for (o, &v) in buf.iter_mut().zip(&chroma_ac[comp][blk]) {
                    *o = v as i32;
                }
                debug_assert!(buf[0] == 0, "chroma AC keeps position 0 free");
                let inc = enc_cbf_ctx_inc(cur, left, above, rows, CAT_CHROMA_AC, 0, 0, comp, blk);
                let n = write_residual_block_cabac(
                    e, st, field, CAT_CHROMA_AC, Some(inc), &buf, scan4, 1, 15,
                );
                debug_assert_eq!(
                    n, cur.nz_chroma[comp][blk] as usize,
                    "nz_chroma[{comp}][{blk}] disagrees with the levels"
                );
            }
        }
    } else {
        debug_assert!(
            chroma_ac.iter().all(|c| c.iter().all(|b| b.iter().all(|&v| v == 0))),
            "chroma AC coefficients below chroma cbp 2 would be lost"
        );
    }
}

/// Parse one CABAC macroblock (after `mb_skip_flag` was 0), including the
/// residual. `frame_motion` is the current picture's motion so far (for the
/// ref_idx contexts of neighbours in other macroblocks).
pub fn parse_mb_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    frame_motion: &[Vec<super::frame::BlockMotion>; 2],
    layer: &mut MbLayer,
    dq: &super::transform::Dequant,
    qps: &mut super::recon::QpState,
) -> Result<()> {
    if super::mb::syntax_trace() {
        eprintln!("mbstart addr={} pos={}", nb.addr, c.position());
    }
    layer.reset(MbKind::I4x4, true);
    let t = decode_mb_type(c, st, ctx, info, nb)?;
    match ctx.slice_type {
        SliceType::I | SliceType::Si => super::cavlc::intra_mb_type(t, layer)?,
        SliceType::P | SliceType::Sp => {
            if t < 5 {
                super::cavlc::p_mb_type(t, layer)?;
            } else {
                super::cavlc::intra_mb_type(t - 5, layer)?;
            }
        }
        SliceType::B => {
            if t < 23 {
                super::cavlc::b_mb_type(t, layer)?;
            } else {
                super::cavlc::intra_mb_type(t - 23, layer)?;
            }
        }
    }

    if layer.kind == MbKind::IPcm {
        // The arithmetic decoder stopped after the terminate bin; raw
        // samples follow at the next byte boundary, then it re-initialises.
        let r = c.reader();
        r.align();
        let n = 256
            + match ctx.chroma_format_idc {
                0 => 0,
                1 => 128,
                2 => 256,
                _ => 512,
            };
        layer.pcm = (0..n).map(|_| r.bits(ctx.bit_depth) as u16).collect();
        if r.overrun() {
            return Err(Error::bitstream("I_PCM samples truncated"));
        }
        c.reinit();
        st.prev_qp_delta_nonzero = false;
        return Ok(());
    }

    let mut no_sub_mb_part_less_than_8x8 = true;
    if layer.kind == MbKind::Inter8x8 {
        for part in 0..4 {
            let t = if ctx.slice_type.is_b() {
                decode_sub_mb_type_b(c, st)
            } else {
                decode_sub_mb_type_p(c, st)
            };
            let (shape, dir) = if ctx.slice_type.is_b() {
                b_sub_mb_type(t)?
            } else {
                p_sub_mb_type(t)?
            };
            layer.sub_shape[part] = shape;
            layer.pred_dir[part] = dir;
            if shape == SubMbShape::Direct {
                if !ctx.direct_8x8_inference {
                    no_sub_mb_part_less_than_8x8 = false;
                }
            } else if shape.count() > 1 {
                no_sub_mb_part_less_than_8x8 = false;
            }
        }
        for list in 0..2 {
            for part in 0..4 {
                if layer.sub_shape[part] == SubMbShape::Direct
                    || layer.pred_dir[part] & (1 << list) == 0
                {
                    continue;
                }
                let n = ctx.num_ref_idx[list] * if layer.field && !ctx.field_pic { 2 } else { 1 };
                let (bx, by) = (((part & 1) * 2) as i32, ((part >> 1) * 2) as i32);
                let ri = if n <= 1 {
                    0
                } else {
                    decode_ref_idx(c, st, info, &layer, nb, frame_motion, list, bx, by)?
                };
                if ri as u32 >= n.max(1) {
                    return Err(Error::bitstream("ref_idx out of range"));
                }
                layer.ref_idx[list][part] = ri;
            }
        }
        for list in 0..2 {
            for part in 0..4 {
                let shape = layer.sub_shape[part];
                if shape == SubMbShape::Direct || layer.pred_dir[part] & (1 << list) == 0 {
                    continue;
                }
                for sub in 0..shape.count() {
                    let (x, y, w, h) = sub_partition_rect(part, shape, sub);
                    let (bx, by) = ((x / 4) as i32, (y / 4) as i32);
                    let (mx, my) = decode_mvd(c, st, info, &layer, nb, list, bx, by)?;
                    // The mvd applies to every 4x4 of the sub-partition (for
                    // later neighbours' contexts).
                    for yy in y / 4..(y + h) / 4 {
                        for xx in x / 4..(x + w) / 4 {
                            layer.mvd[yy * 4 + xx].mvd[list] = Mv::new(mx, my);
                        }
                    }
                }
            }
        }
    } else {
        if ctx.transform_8x8_mode && layer.kind == MbKind::I4x4 {
            layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
            if layer.transform_8x8 {
                layer.kind = MbKind::I8x8;
            }
        }
        match layer.kind {
            MbKind::I4x4 => {
                for blk in 0..16 {
                    let raster = super::mb::raster_of_blk(blk);
                    let (bx, by) = (raster % 4, raster / 4);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, false);
                    layer.intra_modes[raster] = decode_intra_pred_mode(c, st, pred);
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I8x8 => {
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, true);
                    let mode = decode_intra_pred_mode(c, st, pred);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            layer.intra_modes[(by + dy) * 4 + bx + dx] = mode;
                        }
                    }
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I16x16 => {
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::BDirect16x16 => {}
            _ => {
                let parts = mb_partitions(layer.kind);
                for list in 0..2 {
                    for &(x, y, w, h) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let n = ctx.num_ref_idx[list]
                            * if layer.field && !ctx.field_pic { 2 } else { 1 };
                        let ri = if n <= 1 {
                            0
                        } else {
                            decode_ref_idx(
                                c,
                                st,
                                info,
                                &layer,
                                nb,
                                frame_motion,
                                list,
                                (x / 4) as i32,
                                (y / 4) as i32,
                            )?
                        };
                        if ri as u32 >= n.max(1) {
                            return Err(Error::bitstream("ref_idx out of range"));
                        }
                        for by in y / 8..(y + h) / 8 {
                            for bx in x / 8..(x + w) / 8 {
                                layer.ref_idx[list][by * 2 + bx] = ri;
                            }
                        }
                    }
                }
                for list in 0..2 {
                    for &(x, y, w, h) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let (bx, by) = ((x / 4) as i32, (y / 4) as i32);
                        let (mx, my) = decode_mvd(c, st, info, &layer, nb, list, bx, by)?;
                        for yy in y / 4..(y + h) / 4 {
                            for xx in x / 4..(x + w) / 4 {
                                layer.mvd[yy * 4 + xx].mvd[list] = Mv::new(mx, my);
                            }
                        }
                    }
                }
            }
        }
    }

    if layer.kind != MbKind::I16x16 {
        layer.cbp = decode_cbp(
            c,
            st,
            info,
            nb,
            ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2,
        );
        if layer.cbp & 15 != 0
            && ctx.transform_8x8_mode
            && !layer.kind.is_intra()
            && no_sub_mb_part_less_than_8x8
            && (layer.kind != MbKind::BDirect16x16 || ctx.direct_8x8_inference)
        {
            layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
        }
    }

    if layer.has_residual() {
        layer.qp_delta = decode_qp_delta(c, st)?;
        if !(-26..=25).contains(&layer.qp_delta) {
            return Err(Error::bitstream("mb_qp_delta out of range"));
        }
        st.prev_qp_delta_nonzero = layer.qp_delta != 0;
        layer.qp = super::mb::next_qp(qps.prev_qp, layer.qp_delta, ctx.bit_depth);
        qps.prev_qp = layer.qp;
        let mbdq = MbDequant::for_mb(dq, ctx, qps.chroma_offset, layer.kind, layer.qp);
        parse_residual_cabac(c, st, ctx, info, nb, layer, mbdq.as_ref())?;
    } else {
        st.prev_qp_delta_nonzero = false;
        layer.qp = qps.prev_qp;
    }
    if c.overrun() {
        return Err(Error::bitstream("slice data truncated in macroblock"));
    }
    Ok(())
}

/// The value the CAVLC and CABAC parsers agree `PRED_*` on (re-exported so
/// callers have one import).
pub const _PRED_CHECK: (u8, u8, u8) = (PRED_L0, PRED_L1, PRED_BI);


#[cfg(test)]
mod residual_round_trip {
    use super::*;
    use crate::bitwriter::BitWriter;

    /// Encode a block, decode it with the production reader, and require the
    /// coefficients, the count and the context states all to come back.
    ///
    /// The context comparison is the half that catches a desync: two
    /// binarisations can spell the same coefficients while leaving the
    /// probability model in different places, and nothing goes wrong until a
    /// later block reads a bin against a state the writer never had.
    fn round_trip(cat: usize, max_coeff: usize, cbf_inc: Option<usize>, field: bool, levels: &[i32]) {
        let scan: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let mut enc_st = CabacState::new(SliceType::I, 0, 26);
        let mut w = BitWriter::new();
        let n_enc = {
            let mut e = CabacEncoder::new(&mut w);
            let n = write_residual_block_cabac(
                &mut e, &mut enc_st, field, cat, cbf_inc, levels, &scan, 0, max_coeff,
            );
            e.encode_terminate(1);
            n
        };
        w.align_zero();
        let data = w.into_rbsp();

        let mut dec_st = CabacState::new(SliceType::I, 0, 26);
        let mut c = Cabac::new(&data);
        let mut out = vec![0i32; max_coeff];
        let n_dec = residual_block_cabac(
            &mut c, &mut dec_st, field, cat, cbf_inc, &mut out, &scan, 0, max_coeff, None,
        )
        .expect("the reader rejected what the writer produced");

        let want = &levels[..max_coeff];
        assert_eq!(&out[..], want, "cat={cat} field={field} coefficients differ");
        assert_eq!(n_dec, n_enc, "cat={cat} field={field} significant count differs");
        assert_eq!(
            enc_st.ctx, dec_st.ctx,
            "cat={cat} field={field} context states diverged"
        );
    }

    /// Every category, both field settings, over the level patterns that each
    /// exercise a different rule of the binarisation.
    #[test]
    fn round_trips_every_category() {
        // (cat, max_coeff, cbf_inc): 8x8 luma outside 4:4:4 has no
        // coded_block_flag, so it is the one that cannot spell an empty block.
        let shapes: [(usize, usize, Option<usize>); 5] = [
            (0, 16, Some(0)),  // Intra16x16 luma DC
            (2, 16, Some(1)),  // luma 4x4
            (3, 4, Some(2)),   // chroma DC, 4:2:0
            (4, 15, Some(3)),  // chroma AC
            (5, 64, None),     // luma 8x8, flag inferred
        ];
        for (cat, max_coeff, cbf_inc) in shapes {
            for field in [false, true] {
                let mut cases: Vec<Vec<i32>> = Vec::new();
                let z = vec![0i32; 64];
                if cbf_inc.is_some() {
                    cases.push(z.clone()); // the coded_block_flag = 0 path
                }
                // One coefficient, at the front and at the very back: the
                // back is where the reader infers significance instead of
                // coding it.
                for pos in [0usize, max_coeff - 1] {
                    let mut v = z.clone();
                    v[pos] = 1;
                    cases.push(v.clone());
                    v[pos] = -1;
                    cases.push(v.clone());
                    // Magnitudes either side of the unary cap, and past it.
                    for a in [2, 13, 14, 15, 16, 20, 100, 1000, 32767] {
                        let mut v = z.clone();
                        v[pos] = a;
                        cases.push(v.clone());
                        v[pos] = -a;
                        cases.push(v);
                    }
                }
                // Dense: every position significant, alternating sign, which
                // drives the greater-than-one counters to their caps.
                let mut dense = z.clone();
                for (i, v) in dense.iter_mut().enumerate().take(max_coeff) {
                    *v = if i % 2 == 0 { (i as i32) + 1 } else { -(i as i32) - 1 };
                }
                cases.push(dense);
                // Runs of magnitude one, which is what quantised high
                // frequencies actually look like, and the only thing that
                // drives `num_eq1` to its cap of four. Without a run of five
                // the cap is unreachable and removing it changes nothing —
                // a mutation that survived until these cases existed.
                let mut ones = z.clone();
                for v in ones.iter_mut().take(max_coeff) {
                    *v = 1;
                }
                cases.push(ones.clone());
                for (i, v) in ones.iter_mut().enumerate().take(max_coeff) {
                    *v = if i % 2 == 0 { 1 } else { -1 };
                }
                cases.push(ones);
                // A run of ones above one larger value: in reverse scan order
                // the ones come first, so the counter climbs before anything
                // greater than one resets it.
                for run in [4usize, 5, 6, 9] {
                    if run + 1 > max_coeff {
                        continue;
                    }
                    let mut v = z.clone();
                    v[0] = 7;
                    for slot in v.iter_mut().take(run + 1).skip(1) {
                        *slot = 1;
                    }
                    cases.push(v.clone());
                    // And the same with the run at the very top of the scan,
                    // so the inferred-last position is one of the ones.
                    let mut v = z.clone();
                    v[max_coeff - 1 - run] = 7;
                    for slot in v.iter_mut().take(max_coeff).skip(max_coeff - run) {
                        *slot = -1;
                    }
                    cases.push(v);
                }
                // Pseudo-random, mostly zero, which is what real residual is.
                let mut seed = 0x243f6a88u32 ^ (cat as u32) << 8;
                let mut lcg = || {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    seed >> 16
                };
                for _ in 0..60 {
                    let mut v = z.clone();
                    let mut any = false;
                    for slot in v.iter_mut().take(max_coeff) {
                        if lcg() % 4 == 0 {
                            let m = 1 + (lcg() % 40) as i32;
                            *slot = if lcg() % 2 == 0 { m } else { -m };
                            any = true;
                        }
                    }
                    if any || cbf_inc.is_some() {
                        cases.push(v);
                    }
                }
                for levels in &cases {
                    round_trip(cat, max_coeff, cbf_inc, field, levels);
                }
            }
        }
    }

    /// 4:2:2 chroma DC is the one category whose significance increments come
    /// from a table rather than the scan position, and it caps at two.
    #[test]
    fn round_trips_422_chroma_dc() {
        let mut z = vec![0i32; 64];
        for pos in 0..8 {
            let mut v = z.clone();
            v[pos] = if pos % 2 == 0 { 3 } else { -17 };
            round_trip(CAT_CHROMA_DC, 8, Some(1), false, &v);
        }
        for (i, v) in z.iter_mut().enumerate().take(8) {
            *v = (i as i32) - 4;
        }
        z[4] = 9; // no zero in the middle, so every position is significant
        round_trip(CAT_CHROMA_DC, 8, Some(0), false, &z);
    }
}

#[cfg(test)]
mod mb_round_trip {
    use super::*;
    use crate::bitwriter::BitWriter;
    use crate::encode::h264_intra::{PredMode, quad_rasters};
    use crate::h264::cavlc::{intra_mb_type, mb_partitions, p_mb_type, sub_block_counts_8x8};
    use crate::h264::frame::BlockMotion;
    use crate::h264::mb::raster_of_blk;

    /// Slice facts for an all-intra test slice.
    fn slice_ctx(
        slice_type: SliceType,
        num_ref: u32,
        chroma_format_idc: u32,
        field_pic: bool,
        transform_8x8_mode: bool,
    ) -> SliceCtx {
        SliceCtx {
            slice_type,
            slice_num: 0,
            num_ref_idx: [num_ref, 0],
            direct_spatial: false,
            transform_8x8_mode,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc,
            cabac: true,
            bit_depth: 8,
            transform_bypass: false,
            scaling_plane: 0,
            x264_old_444: false,
            field_pic,
            mbaff: false,
        }
    }

    fn lcg(seed: u32) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s >> 16
        }
    }

    /// Fill raster positions `lo..hi` of a block with one of the level
    /// shapes that each exercise a different rule of the binarisation, and
    /// return the nonzero count — the `nz` contract of `MbDecision`.
    fn fill_block(rng: &mut impl FnMut() -> u32, out: &mut [i16; 16], lo: usize, hi: usize) -> u8 {
        let n = match rng() % 5 {
            // All zero: a written block whose coded_block_flag is 0.
            0 => 0,
            // A run of ±1: what quantised residual mostly is, and the
            // binarisation rule a mutation once survived until runs of five
            // existed (the eq-one counter's cap).
            1 => {
                let len = 1 + (rng() as usize) % (hi - lo);
                for i in lo..lo + len {
                    out[i] = if rng() % 2 == 0 { 1 } else { -1 };
                }
                len
            }
            // Sparse, magnitudes either side of the unary cap of fourteen.
            2 => {
                let mut n = 0;
                for i in lo..hi {
                    if rng() % 4 == 0 {
                        let mag = [1, 1, 2, 3, 13, 14, 15, 40][(rng() % 8) as usize];
                        out[i] = if rng() % 2 == 0 { mag } else { -mag };
                        n += 1;
                    }
                }
                n
            }
            // The final scan position significant: the reader infers it
            // rather than decoding it (the scans all end on the highest
            // raster index of the span).
            3 => {
                let mag = [1, 14, 17][(rng() % 3) as usize];
                out[hi - 1] = if rng() % 2 == 0 { mag } else { -mag };
                1
            }
            // Dense ±1 with a sprinkling of 2s: drives both level counters.
            _ => {
                let mut n = 0;
                for i in lo..hi {
                    if rng() % 8 != 0 {
                        let v = if rng() % 4 == 0 { 2 } else { 1 };
                        out[i] = if rng() % 2 == 0 { v } else { -v };
                        n += 1;
                    }
                }
                n
            }
        };
        n as u8
    }

    /// A pseudo-random intra decision that is internally consistent the way
    /// the real mode decision's output is: cbp bits match the levels, `nz`
    /// counts match them too, AC blocks keep position 0 free.
    fn synth_intra(rng: &mut impl FnMut() -> u32, cfi: u32, force: Option<IntraKind>) -> MbDecision {
        let mut d = MbDecision::default();
        d.kind = force
            .unwrap_or(if rng() % 2 == 0 { IntraKind::I4x4 } else { IntraKind::I16x16 });
        d.transform_8x8 = d.kind == IntraKind::I8x8;
        match d.kind {
            // `I_NxN` with the 8x8 transform: four modes, and four blocks
            // of sixty-four in the storage `luma` shares between layouts.
            // `nz_luma` carries the four CAVLC sub-scan counts, which is
            // what the decision side produces and what `spread8` turns
            // into the decoder's view.
            IntraKind::I8x8 => {
                for &raster in &[0usize, 2, 8, 10] {
                    let m = if rng() % 2 == 0 {
                        PredMode { use_predicted: true, rem: 0 }
                    } else {
                        PredMode { use_predicted: false, rem: (rng() % 8) as u8 }
                    };
                    for r in quad_rasters(raster_quad(raster)) {
                        d.luma_pred[r] = m;
                    }
                }
                d.cbp_luma = (rng() % 16) as u8;
                for blk8 in 0..4usize {
                    if d.cbp_luma & (1 << blk8) == 0 {
                        continue;
                    }
                    let mut b = [0i16; 64];
                    fill_block8(rng, &mut b);
                    d.luma.as_flattened_mut()[blk8 * 64..blk8 * 64 + 64].copy_from_slice(&b);
                    let counts = sub_block_counts_8x8(&b);
                    for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
                        d.nz_luma[r] = counts[sub];
                    }
                }
                // An 8x8 whose coded_block_flag is inferred (everything
                // but 4:4:4) cannot be empty with its cbp bit set, so the
                // bit comes off — exactly what the decision does.
                if cfi != 3 {
                    for blk8 in 0..4usize {
                        if quad_rasters(blk8).iter().all(|&r| d.nz_luma[r] == 0) {
                            d.cbp_luma &= !(1 << blk8);
                        }
                    }
                }
            }
            IntraKind::I4x4 => {
                for r in 0..16 {
                    d.luma_pred[r] = if rng() % 2 == 0 {
                        PredMode { use_predicted: true, rem: 0 }
                    } else {
                        PredMode { use_predicted: false, rem: (rng() % 8) as u8 }
                    };
                }
                d.cbp_luma = (rng() % 16) as u8;
                for blk8 in 0..4usize {
                    if d.cbp_luma & (1 << blk8) == 0 {
                        continue;
                    }
                    let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    for sub in 0..4 {
                        let raster = (by8 + (sub >> 1)) * 4 + bx8 + (sub & 1);
                        let mut b = [0i16; 16];
                        d.nz_luma[raster] = fill_block(rng, &mut b, 0, 16);
                        d.luma[raster] = b;
                    }
                }
            }
            IntraKind::I16x16 => {
                d.intra16_mode = (rng() % 4) as u8;
                if rng() % 3 != 0 {
                    let mut b = [0i16; 16];
                    let _ = fill_block(rng, &mut b, 0, 16);
                    d.luma_dc = b;
                }
                d.cbp_luma = if rng() % 2 == 0 { 15 } else { 0 };
                if d.cbp_luma != 0 {
                    for raster in 0..16 {
                        let mut b = [0i16; 16];
                        d.nz_luma[raster] = fill_block(rng, &mut b, 1, 16);
                        d.luma[raster] = b;
                    }
                }
            }
        }
        d.chroma_mode = if cfi == 0 || cfi == 3 { 0 } else { (rng() % 4) as u8 };
        if cfi == 3 {
            // ChromaArrayType 3: Cb and Cr are luma-style planes, coded
            // with the *same* coded block pattern, transform size and
            // (for I_16x16) DC split as luma — and there is no chroma cbp
            // at all. `chroma_ac` holds each plane's blocks and
            // `chroma_dc` its Intra_16x16 DC.
            synth_444_planes(rng, &mut d);
            return d;
        }
        let (cbp_c, cdc, cac, cnz) = synth_chroma(rng, cfi);
        d.cbp_chroma = cbp_c;
        d.chroma_dc = cdc;
        d.chroma_ac = cac;
        d.nz_chroma = cnz;
        d
    }

    /// Which 8x8 quad a 4x4 raster index belongs to.
    fn raster_quad(raster: usize) -> usize {
        (raster / 8) * 2 + (raster % 4) / 2
    }

    /// A block of sixty-four levels with the same shape [`fill_block`]
    /// gives a 4x4: some zero, some at the escape thresholds.
    fn fill_block8(rng: &mut impl FnMut() -> u32, out: &mut [i16; 64]) {
        for o in out.iter_mut() {
            *o = match rng() % 6 {
                0 | 1 | 2 => 0,
                3 => 1,
                4 => -1,
                _ => {
                    let mag = match rng() % 4 {
                        0 => 2 + (rng() % 6) as i16,
                        1 => 9 + (rng() % 8) as i16,
                        2 => 15 + (rng() % 200) as i16,
                        _ => 1 + (rng() % 3000) as i16,
                    };
                    if rng() % 2 == 0 { mag } else { -mag }
                }
            };
        }
        // The reader infers the final significant coefficient rather than
        // coding it, so a block whose only coefficient is at the last scan
        // position is still spellable — but an all-zero block with a cbp
        // bit set is not, outside 4:4:4. Guarantee one.
        if out.iter().all(|&v| v == 0) {
            out[0] = 1;
        }
    }

    /// 4:4:4's Cb and Cr planes for an intra decision whose luma is
    /// already synthesised: the same kind, transform size and coded block
    /// pattern, with each plane's own levels.
    fn synth_444_planes(rng: &mut impl FnMut() -> u32, d: &mut MbDecision) {
        d.cbp_chroma = 0;
        for comp in 0..2 {
            match d.kind {
                IntraKind::I16x16 => {
                    let mut b = [0i16; 16];
                    let _ = fill_block(rng, &mut b, 0, 16);
                    d.chroma_dc[comp] = b;
                    if d.cbp_luma != 0 {
                        for raster in 0..16 {
                            let mut b = [0i16; 16];
                            d.nz_chroma[comp][raster] = fill_block(rng, &mut b, 1, 16);
                            d.chroma_ac[comp][raster] = b;
                        }
                    }
                }
                IntraKind::I4x4 => {
                    for blk8 in 0..4usize {
                        if d.cbp_luma & (1 << blk8) == 0 {
                            continue;
                        }
                        for &raster in &quad_rasters(blk8) {
                            let mut b = [0i16; 16];
                            d.nz_chroma[comp][raster] = fill_block(rng, &mut b, 0, 16);
                            d.chroma_ac[comp][raster] = b;
                        }
                    }
                }
                IntraKind::I8x8 => {
                    for blk8 in 0..4usize {
                        if d.cbp_luma & (1 << blk8) == 0 {
                            continue;
                        }
                        let mut b = [0i16; 64];
                        fill_block8(rng, &mut b);
                        // A plane's 8x8 may legitimately be empty in
                        // 4:4:4: its coded_block_flag is coded, not
                        // inferred. Let a third of them be.
                        if rng() % 3 == 0 {
                            b = [0; 64];
                        }
                        d.chroma_ac[comp].as_flattened_mut()[blk8 * 64..blk8 * 64 + 64]
                            .copy_from_slice(&b);
                        let counts = sub_block_counts_8x8(&b);
                        for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
                            d.nz_chroma[comp][r] = counts[sub];
                        }
                    }
                }
            }
        }
    }

    /// Chroma residual shared by the intra and inter synthesisers: a cbp
    /// value with DC / AC contents to match.
    #[allow(clippy::type_complexity)]
    fn synth_chroma(
        rng: &mut impl FnMut() -> u32,
        cfi: u32,
    ) -> (u8, [[i16; 16]; 2], [[[i16; 16]; 16]; 2], [[u8; 16]; 2]) {
        let mut chroma_dc = [[0i16; 16]; 2];
        let mut chroma_ac = [[[0i16; 16]; 16]; 2];
        let mut nz_chroma = [[0u8; 16]; 2];
        if cfi != 1 && cfi != 2 {
            return (0, chroma_dc, chroma_ac, nz_chroma);
        }
        let n_dc = if cfi == 2 { 8 } else { 4 };
        let rows = if cfi == 2 { 4 } else { 2 };
        let cbp_chroma = (rng() % 3) as u8;
        if cbp_chroma >= 1 {
            for comp in 0..2 {
                let mut b = [0i16; 16];
                let _ = fill_block(rng, &mut b, 0, n_dc);
                for (i, o) in chroma_dc[comp][..n_dc].iter_mut().enumerate() {
                    *o = b[i];
                }
            }
        }
        if cbp_chroma == 2 {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    let mut b = [0i16; 16];
                    nz_chroma[comp][blk] = fill_block(rng, &mut b, 1, 16);
                    chroma_ac[comp][blk] = b;
                }
            }
        }
        (cbp_chroma, chroma_dc, chroma_ac, nz_chroma)
    }

    /// One mvd component whose magnitude lands in each context regime:
    /// zero, small, around the sum thresholds of 3 and 32, the prefix cap
    /// of nine, and far into the UEG3 escape.
    fn mvd_comp(rng: &mut impl FnMut() -> u32) -> i16 {
        let mag = match rng() % 5 {
            0 => 0,
            1 => (rng() % 4) as i16,
            2 => 3 + (rng() % 30) as i16,
            3 => 9 + (rng() % 8) as i16,
            _ => 200 + (rng() % 2000) as i16,
        };
        if rng() % 2 == 0 { mag } else { -mag }
    }

    /// A pseudo-random P macroblock, internally consistent the way
    /// [`synth_intra`]'s output is: a skip carries nothing, cbp bits match
    /// the levels, and the reference index respects the list length.
    fn synth_inter(rng: &mut impl FnMut() -> u32, cfi: u32, num_ref: u32) -> InterDecision {
        let mut d = InterDecision::default();
        if rng() % 4 == 0 {
            d.kind = InterMbKind::PSkip;
            return d;
        }
        // A shape, then one mvd per partition of it. The shapes must be
        // mixed within a slice: a 16x8's lower half reads its context
        // from the upper half, and a slice of nothing but 16x16 would
        // never exercise an in-macroblock mvd neighbour at all.
        // `ref_idx` is written by the 16x16-only
        // `write_ref_idx_16x16_cabac`, which hardwires its neighbour
        // blocks; a multi-reference slice therefore stays 16x16 until a
        // partition-aware ref_idx writer exists. The production encoder
        // never reaches this — it declares one reference, so the element
        // is absent — and `write_p16x16_mb` asserts the limit rather than
        // spelling something a reader would not take.
        d.kind = if num_ref > 1 {
            InterMbKind::P16x16
        } else {
            match rng() % 4 {
                0 => InterMbKind::P16x16,
                1 => InterMbKind::P16x8,
                2 => InterMbKind::P8x16,
                _ => InterMbKind::P8x8,
            }
        };
        if d.kind == InterMbKind::P8x8 {
            for part in 0..4 {
                d.sub_shape[part] = match rng() % 4 {
                    0 => SubMbShape::S8x8,
                    1 => SubMbShape::S8x4,
                    2 => SubMbShape::S4x8,
                    _ => SubMbShape::S4x4,
                };
            }
        }
        d.ref_idx = if num_ref > 1 { (rng() % num_ref) as i8 } else { 0 };
        // One mvd per prediction rectangle, replicated over it — the
        // layout the decoder's CABAC parser stores and the contexts read.
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = d.rects(&mut rects);
        for &(x, y, w, h) in rects.iter().take(n) {
            let m = Mv::new(mvd_comp(rng), mvd_comp(rng));
            for by in y / 4..(y + h) / 4 {
                for bx in x / 4..(x + w) / 4 {
                    d.mvd[by * 4 + bx] = m;
                }
            }
        }
        d.cbp_luma = (rng() % 16) as u8;
        for blk8 in 0..4usize {
            if d.cbp_luma & (1 << blk8) == 0 {
                continue;
            }
            let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
            for sub in 0..4 {
                let raster = (by8 + (sub >> 1)) * 4 + bx8 + (sub & 1);
                let mut b = [0i16; 16];
                d.nz_luma[raster] = fill_block(rng, &mut b, 0, 16);
                d.luma[raster] = b;
            }
        }
        let (cbp_c, cdc, cac, cnz) = synth_chroma(rng, cfi);
        d.cbp_chroma = cbp_c;
        d.chroma_dc = cdc;
        d.chroma_ac = cac;
        d.nz_chroma = cnz;
        d
    }

    /// What the writer-side picture loop keeps per macroblock beyond
    /// [`WrittenMb`]: the two context facts the mb_type and chroma-mode
    /// writers take as arguments.
    struct Coded {
        nb: WrittenMb,
        /// `mb_type != I_NxN` (mb_type's first-bin ctxIdxInc counts these).
        not_nxn: bool,
        /// Intra, not I_PCM, `intra_chroma_pred_mode != 0`.
        chroma_nonzero: bool,
    }

    /// One macroblock of a synthetic slice.
    enum TestMb {
        Intra(MbDecision),
        Pcm(Vec<u16>),
        Inter(InterDecision),
    }

    /// Write one intra macroblock in the syntax order [`parse_mb_cabac`]
    /// reads it: mb_type, prediction modes, cbp (I_NxN only), mb_qp_delta
    /// and residual when there is residual — the reference for the picture
    /// loop this module's writers will be wired into.
    #[allow(clippy::too_many_arguments)]
    fn write_mb(
        e: &mut CabacEncoder,
        st: &mut CabacState,
        d: &MbDecision,
        left: Option<&Coded>,
        above: Option<&Coded>,
        cfi: u32,
        field: bool,
        t8x8: bool,
    ) {
        let inc = left.map_or(0, |m| m.not_nxn as usize) + above.map_or(0, |m| m.not_nxn as usize);
        write_mb_type_i_cabac(e, st, inc, intra_mb_type_code(d));
        write_intra_mb_body(e, st, d, left, above, cfi, field, t8x8);
    }

    /// Everything after `mb_type` for an intra macroblock — shared by the
    /// I-slice composite and intra-in-P, exactly as the readers share it.
    #[allow(clippy::too_many_arguments)]
    fn write_intra_mb_body(
        e: &mut CabacEncoder,
        st: &mut CabacState,
        d: &MbDecision,
        left: Option<&Coded>,
        above: Option<&Coded>,
        cfi: u32,
        field: bool,
        t8x8: bool,
    ) {
        let chroma_nb = (cfi == 1 || cfi == 2).then(|| {
            [
                left.is_some_and(|m| m.chroma_nonzero),
                above.is_some_and(|m| m.chroma_nonzero),
            ]
        });
        let lnb = left.map(|m| &m.nb);
        let anb = above.map(|m| &m.nb);
        // Before `mb_pred()`, and only for I_NxN.
        if t8x8 && d.kind.is_nxn() {
            write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
        }
        write_intra_pred_modes_cabac(e, st, d, chroma_nb);
        if d.kind.is_nxn() {
            write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), cfi == 1 || cfi == 2);
        }
        let has_residual = d.kind == IntraKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
        if has_residual {
            write_mb_qp_delta_cabac(e, st, d.qp_delta as i32);
            st.prev_qp_delta_nonzero = d.qp_delta != 0;
            write_intra_residual_cabac(e, st, field, cfi, d, lnb, anb);
        } else {
            st.prev_qp_delta_nonzero = false;
        }
    }

    /// One coded (non-skipped) `P_L0_16x16` macroblock in
    /// [`parse_mb_cabac`]'s order: mb_type, ref_idx when the list is
    /// longer than one, mvd, cbp, then qp_delta and residual when cbp is
    /// nonzero — the reference for the picture loop this will be wired
    /// into.
    #[allow(clippy::too_many_arguments)]
    fn write_p16x16_mb(
        e: &mut CabacEncoder,
        st: &mut CabacState,
        d: &InterDecision,
        left: Option<&Coded>,
        above: Option<&Coded>,
        cfi: u32,
        field: bool,
        num_ref: u32,
        t8x8: bool,
    ) {
        write_mb_type_p_cabac(e, st, d.kind.p_mb_type());
        let lnb = left.map(|m| &m.nb);
        let anb = above.map(|m| &m.nb);
        if num_ref > 1 {
            debug_assert_eq!(
                d.kind,
                InterMbKind::P16x16,
                "a multi-reference test writes one ref_idx, so one partition"
            );
            write_ref_idx_16x16_cabac(e, st, lnb, anb, d.ref_idx);
        }
        if d.kind == InterMbKind::P8x8 {
            for part in 0..4 {
                write_sub_mb_type_p_cabac(
                    e,
                    st,
                    crate::encode::h264_cavlc_mb::sub_mb_type_p(d.sub_shape[part]),
                );
            }
        }
        let mut cur = CurMbMvd::default();
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = d.rects(&mut rects);
        for &(x, y, w, h) in rects.iter().take(n) {
            let m = d.mvd[(y / 4) * 4 + x / 4];
            write_mvd_cabac(e, st, &cur, lnb, anb, 0, x / 4, y / 4, m);
            cur.set(0, x, y, w, h, m);
        }
        write_cbp_cabac(e, st, lnb, anb, d.cbp_luma | (d.cbp_chroma << 4), cfi == 1 || cfi == 2);
        // After the coded block pattern, only when some luma block is
        // coded, and only when every sub-macroblock partition is at least
        // 8x8 (7.3.5) — a split `P_8x8` suppresses the flag, and writing
        // it where the reader does not take it desyncs the very next
        // element.
        if t8x8 && d.cbp_luma != 0 && d.no_sub_mb_part_less_than_8x8() {
            write_transform_8x8_cabac(e, st, lnb, anb, d.transform_8x8);
        }
        if d.cbp_luma != 0 || d.cbp_chroma != 0 {
            write_mb_qp_delta_cabac(e, st, d.qp_delta as i32);
            st.prev_qp_delta_nonzero = d.qp_delta != 0;
            write_inter_residual_cabac(e, st, field, cfi, d, lnb, anb);
        } else {
            st.prev_qp_delta_nonzero = false;
        }
    }

    /// [`parse_mb_cabac`] mirrored — the intra path and the P_L0_16x16
    /// walk — with `dq = None` (raw levels), returning the (prev, rem)
    /// syntax recovered per 4x4 block: with the predicted mode in hand the
    /// map mode -> (prev, rem) is bijective, so comparing recovered syntax
    /// against what was written is exact.
    fn parse_mb(
        c: &mut Cabac,
        st: &mut CabacState,
        ctx: &SliceCtx,
        info: &PicInfo,
        nb: &MbNeighbours,
        frame_motion: &[Vec<BlockMotion>; 2],
        layer: &mut MbLayer,
    ) -> [(bool, u8); 16] {
        let mut syntax = [(true, 0u8); 16];
        layer.reset(MbKind::I4x4, true);
        let t = decode_mb_type(c, st, ctx, info, nb).expect("mb_type rejected");
        let intra_t = if ctx.slice_type == SliceType::P {
            if t < 5 {
                // `parse_mb_cabac`'s inter partition walk, over the
                // shapes these tests write: 16x16, 16x8 and 8x16. Both
                // passes go in the reader's order — every ref_idx, then
                // every mvd — because that ordering is part of what the
                // writers have to invert.
                p_mb_type(t, layer).expect("P mb_type rejected");
                let mut return_early = false;
                if layer.kind == MbKind::Inter8x8 {
                    // `sub_mb_pred()`: all four `sub_mb_type`s, then all
                    // `ref_idx` (nothing, at one reference), then all
                    // mvds — the reader's three separate passes.
                    for part in 0..4 {
                        let t = decode_sub_mb_type_p(c, st);
                        let (shape, dir) =
                            crate::h264::cavlc::p_sub_mb_type(t).expect("P sub_mb_type rejected");
                        layer.sub_shape[part] = shape;
                        layer.pred_dir[part] = dir;
                        layer.ref_idx[0][part] = 0;
                    }
                    for part in 0..4 {
                        let shape = layer.sub_shape[part];
                        for sub in 0..shape.count() {
                            let (x, y, w, h) = sub_partition_rect(part, shape, sub);
                            let (mx, my) = decode_mvd(
                                c, st, info, layer, nb, 0, (x / 4) as i32, (y / 4) as i32,
                            )
                            .expect("mvd rejected");
                            for by in y / 4..(y + h) / 4 {
                                for bx in x / 4..(x + w) / 4 {
                                    layer.mvd[by * 4 + bx].mvd[0] = Mv::new(mx, my);
                                }
                            }
                        }
                    }
                    return_early = true;
                }
                let parts = if return_early { &[][..] } else { mb_partitions(layer.kind) };
                for &(x, y, w, h) in parts {
                    let n = ctx.num_ref_idx[0];
                    let ri = if n <= 1 {
                        0
                    } else {
                        decode_ref_idx(
                            c, st, info, layer, nb, frame_motion, 0, (x / 4) as i32, (y / 4) as i32,
                        )
                        .expect("ref_idx rejected")
                    };
                    assert!((ri as u32) < n.max(1), "ref_idx out of range");
                    for by in y / 8..(y + h) / 8 {
                        for bx in x / 8..(x + w) / 8 {
                            layer.ref_idx[0][by * 2 + bx] = ri;
                        }
                    }
                }
                for &(x, y, w, h) in parts {
                    let (mx, my) =
                        decode_mvd(c, st, info, layer, nb, 0, (x / 4) as i32, (y / 4) as i32)
                            .expect("mvd rejected");
                    for by in y / 4..(y + h) / 4 {
                        for bx in x / 4..(x + w) / 4 {
                            layer.mvd[by * 4 + bx].mvd[0] = Mv::new(mx, my);
                        }
                    }
                }
                None
            } else {
                Some(t - 5)
            }
        } else {
            Some(t)
        };
        if let Some(it) = intra_t {
            intra_mb_type(it, layer).expect("mb_type out of range");
            // `transform_size_8x8_flag` for I_NxN, before `mb_pred()`.
            if ctx.transform_8x8_mode && layer.kind == MbKind::I4x4 {
                layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
                if layer.transform_8x8 {
                    layer.kind = MbKind::I8x8;
                }
            }
        }
        match layer.kind {
            MbKind::Inter16x16
            | MbKind::Inter16x8
            | MbKind::Inter8x16
            | MbKind::Inter8x8 => {}
            MbKind::IPcm => {
                let n = 256
                    + match ctx.chroma_format_idc {
                        0 => 0,
                        1 => 128,
                        2 => 256,
                        _ => 512,
                    };
                let r = c.reader();
                r.align();
                layer.pcm = (0..n).map(|_| r.bits(ctx.bit_depth) as u16).collect();
                assert!(!r.overrun(), "I_PCM samples truncated");
                c.reinit();
                st.prev_qp_delta_nonzero = false;
                return syntax;
            }
            MbKind::I4x4 => {
                for blk in 0..16 {
                    let raster = raster_of_blk(blk);
                    let (bx, by) = (raster % 4, raster / 4);
                    let pred = predicted_intra_mode(info, layer, nb, ctx, bx, by, false);
                    let mode = decode_intra_pred_mode(c, st, pred);
                    layer.intra_modes[raster] = mode;
                    syntax[raster] = if mode == pred {
                        (true, 0)
                    } else {
                        (false, if mode < pred { mode } else { mode - 1 })
                    };
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I8x8 => {
                // Four modes in quad order; the reader replicates each
                // over its quad, and the recovered syntax is reported at
                // indices 0..4 so the check can compare it directly.
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let pred = predicted_intra_mode(info, layer, nb, ctx, bx, by, true);
                    let mode = decode_intra_pred_mode(c, st, pred);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            layer.intra_modes[(by + dy) * 4 + bx + dx] = mode;
                        }
                    }
                    syntax[blk8] = if mode == pred {
                        (true, 0)
                    } else {
                        (false, if mode < pred { mode } else { mode - 1 })
                    };
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I16x16 => {
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            k => panic!("unexpected macroblock kind {k:?}"),
        }
        if layer.kind != MbKind::I16x16 {
            layer.cbp = decode_cbp(
                c,
                st,
                info,
                nb,
                ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2,
            );
            // An inter macroblock's flag, after the coded block pattern
            // — and only when every sub-macroblock partition is at least
            // 8x8 (7.3.5's `noSubMbPartSizeLessThan8x8Flag`).
            let no_sub_lt_8x8 = layer.kind != MbKind::Inter8x8
                || layer.sub_shape.iter().all(|s| s.count() == 1);
            if layer.cbp & 15 != 0
                && ctx.transform_8x8_mode
                && !layer.kind.is_intra()
                && no_sub_lt_8x8
            {
                layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
            }
        }
        if layer.has_residual() {
            layer.qp_delta = decode_qp_delta(c, st).expect("mb_qp_delta rejected");
            st.prev_qp_delta_nonzero = layer.qp_delta != 0;
            parse_residual_cabac(c, st, ctx, info, nb, layer, None).expect("residual rejected");
        } else {
            st.prev_qp_delta_nonzero = false;
        }
        syntax
    }

    /// The decoder's per-macroblock bookkeeping (`recon.rs`), for the
    /// fields the intra CABAC contexts read back from neighbours.
    fn commit(info: &mut PicInfo, addr: usize, layer: &MbLayer, c444: bool) {
        let m = &mut info.mbs[addr];
        m.kind = layer.kind;
        m.slice = 0;
        m.decoded = true;
        m.cbp = layer.cbp;
        m.transform_8x8 = layer.transform_8x8;
        m.chroma_mode = layer.chroma_mode;
        m.qp_delta_nonzero = layer.has_residual() && layer.qp_delta != 0;
        m.dc_cbf = layer.dc_cbf;
        let base = addr * 16;
        if layer.kind == MbKind::IPcm {
            info.luma_nz[base..base + 16].fill(16);
            info.chroma_nz[addr * 32..addr * 32 + 32].fill(16);
        } else {
            info.luma_nz[base..base + 16].copy_from_slice(&layer.nz[0]);
            if c444 {
                // 4:4:4 stores the two planes' luma-style counts where
                // the 4:2:x chroma AC counts would go — `derive()`'s own
                // layout (src/h264/recon.rs).
                info.chroma_nz[addr * 32..addr * 32 + 16].copy_from_slice(&layer.nz[1]);
                info.chroma_nz[addr * 32 + 16..addr * 32 + 32].copy_from_slice(&layer.nz[2]);
            } else {
                for comp in 0..2 {
                    info.chroma_nz[addr * 32 + comp * 16..addr * 32 + comp * 16 + 8]
                        .copy_from_slice(&layer.chroma_nz[comp]);
                }
            }
        }
        if matches!(layer.kind, MbKind::I4x4 | MbKind::I8x8) {
            info.intra_modes[base..base + 16].copy_from_slice(&layer.intra_modes);
        } else {
            info.intra_modes[base..base + 16].fill(2);
        }
        for l in 0..2 {
            for (dst, ent) in info.mvd[l][base..base + 16].iter_mut().zip(&layer.mvd) {
                *dst = ent.mvd[l];
            }
        }
    }

    /// Every field of one parsed macroblock against the decision that was
    /// written — and the writer's own [`WrittenMb`] against what the
    /// decoder stores, which is the proof of `from_decision`.
    fn check_mb(addr: usize, d: &MbDecision, layer: &MbLayer, syntax: &[(bool, u8); 16], ctx: &SliceCtx) {
        match d.kind {
            IntraKind::I8x8 => {
                assert_eq!(layer.kind, MbKind::I8x8, "mb {addr} kind");
                // Four modes, at each quad's top-left 4x4, and the reader
                // replicates each over its quad.
                for (i, raster) in [0usize, 2, 8, 10].into_iter().enumerate() {
                    let p = d.luma_pred[raster];
                    let want = (p.use_predicted, if p.use_predicted { 0 } else { p.rem });
                    assert_eq!(syntax[i], want, "mb {addr} quad {i} pred-mode syntax");
                }
            }
            IntraKind::I4x4 => {
                assert_eq!(layer.kind, MbKind::I4x4, "mb {addr} kind");
                for r in 0..16 {
                    let p = d.luma_pred[r];
                    let want = (p.use_predicted, if p.use_predicted { 0 } else { p.rem });
                    assert_eq!(syntax[r], want, "mb {addr} block {r} pred-mode syntax");
                }
            }
            IntraKind::I16x16 => {
                assert_eq!(layer.kind, MbKind::I16x16, "mb {addr} kind");
                assert_eq!(layer.intra16_mode, d.intra16_mode, "mb {addr} intra16 mode");
            }
        }
        assert_eq!(layer.transform_8x8, d.transform_8x8, "mb {addr} transform_size_8x8_flag");
        assert_eq!(layer.cbp, (d.cbp_luma & 15) | (d.cbp_chroma << 4), "mb {addr} cbp");
        let chroma = ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2;
        let c444 = ctx.chroma_format_idc == 3;
        if chroma {
            assert_eq!(layer.chroma_mode, d.chroma_mode, "mb {addr} chroma mode");
        }
        let has_residual = d.kind == IntraKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
        assert_eq!(
            layer.qp_delta,
            if has_residual { d.qp_delta as i32 } else { 0 },
            "mb {addr} qp_delta"
        );
        if d.kind == IntraKind::I16x16 {
            for i in 0..16 {
                assert_eq!(layer.dc[0][i], d.luma_dc[i] as i32, "mb {addr} luma DC coeff {i}");
            }
        }
        // The coefficient storage means two different things by transform
        // size, and comparing it in the wrong layout would agree with a
        // wrong writer — quad `blk8` is the flat range `blk8 * 64` under
        // the 8x8 transform, and its four raster 4x4s otherwise.
        let want_plane = |plane: usize, levels: &[[i16; 16]; 16], nz: &[u8; 16], label: &str| {
            for i in 0..256 {
                assert_eq!(
                    layer.coef[plane][i],
                    levels.as_flattened()[i] as i32,
                    "mb {addr} {label} coeff {i}"
                );
            }
            if d.transform_8x8 {
                // The decoder stores one count per 8x8 on all four of its
                // 4x4s; the decision counts sub-scans. `spread8` is the
                // bridge, and this is what proves it.
                assert_eq!(
                    layer.nz[plane],
                    gate_nz_luma(d.cbp_luma, &spread8(true, nz)),
                    "mb {addr} {label} nz"
                );
            } else {
                assert_eq!(&layer.nz[plane], nz, "mb {addr} {label} nz");
            }
        };
        want_plane(0, &d.luma, &d.nz_luma, "luma");
        if c444 {
            // The two luma-style planes, coded exactly like luma.
            for comp in 0..2 {
                want_plane(1 + comp, &d.chroma_ac[comp], &d.nz_chroma[comp], "plane");
            }
            if d.kind == IntraKind::I16x16 {
                for comp in 0..2 {
                    for i in 0..16 {
                        assert_eq!(
                            layer.dc[1 + comp][i],
                            d.chroma_dc[comp][i] as i32,
                            "mb {addr} plane {comp} DC coeff {i}"
                        );
                    }
                }
            }
            assert_eq!(d.cbp_chroma, 0, "ChromaArrayType 3 has no chroma cbp");
        }
        if chroma {
            let n_dc = if ctx.chroma_format_idc == 2 { 8 } else { 4 };
            let rows = if ctx.chroma_format_idc == 2 { 4 } else { 2 };
            for comp in 0..2 {
                for i in 0..n_dc {
                    assert_eq!(
                        layer.chroma_dc[comp][i],
                        d.chroma_dc[comp][i] as i32,
                        "mb {addr} chroma {comp} DC coeff {i}"
                    );
                }
                for blk in 0..2 * rows {
                    for k in 0..16 {
                        assert_eq!(
                            layer.chroma_ac[comp][blk][k],
                            d.chroma_ac[comp][blk][k] as i32,
                            "mb {addr} chroma {comp} AC block {blk} coeff {k}"
                        );
                    }
                }
                assert_eq!(layer.chroma_nz[comp][..], d.nz_chroma[comp][..8], "mb {addr} chroma {comp} nz");
            }
        }
        // The writer's own neighbour record against what the decoder
        // stores, which is the proof of `from_decision` — and, under the
        // 8x8 transform, of `spread8`.
        let wm = WrittenMb::from_decision(d, c444);
        assert_eq!(wm.cbp, layer.cbp, "mb {addr} WrittenMb cbp");
        assert_eq!(wm.dc_cbf, layer.dc_cbf, "mb {addr} WrittenMb dc_cbf");
        assert_eq!(wm.transform_8x8, layer.transform_8x8, "mb {addr} WrittenMb transform_8x8");
        assert_eq!(wm.nz_luma, layer.nz[0], "mb {addr} WrittenMb nz_luma");
        if c444 {
            assert_eq!(wm.nz_chroma[0], layer.nz[1], "mb {addr} WrittenMb plane Cb nz");
            assert_eq!(wm.nz_chroma[1], layer.nz[2], "mb {addr} WrittenMb plane Cr nz");
        } else {
            assert_eq!(wm.nz_chroma[0][..8], layer.chroma_nz[0][..], "mb {addr} WrittenMb nz_chroma Cb");
            assert_eq!(wm.nz_chroma[1][..8], layer.chroma_nz[1][..], "mb {addr} WrittenMb nz_chroma Cr");
        }
    }

    /// Every field of one parsed P macroblock against the decision that
    /// was written, and [`WrittenMb::from_inter_decision`] against what
    /// the decoder's bookkeeping stores.
    fn check_inter_mb(addr: usize, d: &InterDecision, layer: &MbLayer, ctx: &SliceCtx, num_ref: u32) {
        let c444 = ctx.chroma_format_idc == 3;
        assert_eq!(layer.kind, d.kind.dec_kind(), "mb {addr} kind");
        let want_ri = if num_ref > 1 { d.ref_idx } else { 0 };
        assert_eq!(layer.ref_idx[0], [want_ri; 4], "mb {addr} ref_idx");
        // Each rectangle's mvd over the blocks it covers — the reader
        // replicates it there, and the *next* rectangle's context reads
        // it back out, so comparing every block and not just the corner
        // is what makes a wrong replication visible.
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = d.rects(&mut rects);
        for (i, &(x, y, w, h)) in rects.iter().take(n).enumerate() {
            for by in y / 4..(y + h) / 4 {
                for bx in x / 4..(x + w) / 4 {
                    assert_eq!(
                        layer.mvd[by * 4 + bx].mvd[0],
                        d.mvd[(y / 4) * 4 + x / 4],
                        "mb {addr} rectangle {i} mvd block ({bx},{by})"
                    );
                }
            }
        }
        if d.kind == InterMbKind::P8x8 {
            assert_eq!(&layer.sub_shape, &d.sub_shape, "mb {addr} sub_mb_type");
        }
        assert_eq!(layer.transform_8x8, d.transform_8x8, "mb {addr} transform_size_8x8_flag");
        assert_eq!(layer.cbp, (d.cbp_luma & 15) | (d.cbp_chroma << 4), "mb {addr} cbp");
        let has_residual = d.cbp_luma != 0 || d.cbp_chroma != 0;
        assert_eq!(
            layer.qp_delta,
            if has_residual { d.qp_delta as i32 } else { 0 },
            "mb {addr} qp_delta"
        );
        // As in `check_mb`: the storage means two different things by
        // transform size, and the counts the decoder keeps are the
        // spread ones.
        let want_plane = |plane: usize, levels: &[[i16; 16]; 16], nz: &[u8; 16], label: &str| {
            for i in 0..256 {
                assert_eq!(
                    layer.coef[plane][i],
                    levels.as_flattened()[i] as i32,
                    "mb {addr} inter {label} coeff {i}"
                );
            }
            if d.transform_8x8 {
                assert_eq!(
                    layer.nz[plane],
                    gate_nz_luma(d.cbp_luma, &spread8(true, nz)),
                    "mb {addr} inter {label} nz"
                );
            } else {
                assert_eq!(&layer.nz[plane], nz, "mb {addr} inter {label} nz");
            }
        };
        want_plane(0, &d.luma, &d.nz_luma, "luma");
        if c444 {
            for comp in 0..2 {
                want_plane(1 + comp, &d.chroma_ac[comp], &d.nz_chroma[comp], "plane");
            }
        }
        if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
            let n_dc = if ctx.chroma_format_idc == 2 { 8 } else { 4 };
            let rows = if ctx.chroma_format_idc == 2 { 4 } else { 2 };
            for comp in 0..2 {
                for i in 0..n_dc {
                    assert_eq!(
                        layer.chroma_dc[comp][i],
                        d.chroma_dc[comp][i] as i32,
                        "mb {addr} chroma {comp} DC coeff {i}"
                    );
                }
                for blk in 0..2 * rows {
                    for k in 0..16 {
                        assert_eq!(
                            layer.chroma_ac[comp][blk][k],
                            d.chroma_ac[comp][blk][k] as i32,
                            "mb {addr} chroma {comp} AC block {blk} coeff {k}"
                        );
                    }
                }
                assert_eq!(layer.chroma_nz[comp][..], d.nz_chroma[comp][..8], "mb {addr} chroma {comp} nz");
            }
        }
        let wm = WrittenMb::from_inter_decision(d, c444);
        assert_eq!(wm.cbp, layer.cbp, "mb {addr} WrittenMb cbp");
        assert_eq!(wm.dc_cbf, layer.dc_cbf, "mb {addr} WrittenMb dc_cbf");
        assert_eq!(wm.transform_8x8, layer.transform_8x8, "mb {addr} WrittenMb transform_8x8");
        assert_eq!(wm.nz_luma, layer.nz[0], "mb {addr} WrittenMb nz_luma");
        if c444 {
            assert_eq!(wm.nz_chroma[0], layer.nz[1], "mb {addr} WrittenMb plane Cb nz");
            assert_eq!(wm.nz_chroma[1], layer.nz[2], "mb {addr} WrittenMb plane Cr nz");
        } else {
            assert_eq!(wm.nz_chroma[0][..8], layer.chroma_nz[0][..], "mb {addr} WrittenMb nz_chroma Cb");
            assert_eq!(wm.nz_chroma[1][..8], layer.chroma_nz[1][..], "mb {addr} WrittenMb nz_chroma Cr");
        }
        for blk in 0..16 {
            assert_eq!(wm.mvd[0][blk], layer.mvd[blk].mvd[0], "mb {addr} WrittenMb mvd {blk}");
            assert_eq!(wm.ref_idx[blk], want_ri, "mb {addr} WrittenMb ref_idx {blk}");
        }
    }

    /// Write a whole synthetic I slice with the writer functions, decode it
    /// with the production readers over the production neighbour machinery,
    /// and require every field — and the entire context array — to come
    /// back. The context comparison is the half that catches a desync: two
    /// spellings can agree on this slice's bins while leaving the
    /// probability model in different places, and nothing goes wrong until
    /// a later bin reads against a state the writer never had.
    #[allow(clippy::too_many_arguments)]
    fn round_trip_slice(
        mbs: &[TestMb],
        mb_width: usize,
        cfi: u32,
        field: bool,
        qp: i32,
        p_slice: bool,
        num_ref: u32,
        t8x8: bool,
    ) {
        let c444 = cfi == 3;
        let total = mbs.len();
        assert_eq!(total % mb_width, 0);
        let slice_type = if p_slice { SliceType::P } else { SliceType::I };

        // ---- write ----
        let mut w = BitWriter::new();
        w.align_one(); // cabac_alignment_one_bit (a fresh writer is aligned)
        let mut enc_st = CabacState::new(slice_type, 0, qp);
        let mut coded: Vec<Coded> = Vec::with_capacity(total);
        let mut i = 0usize;
        // A codeword flushed by I_PCM leaves that macroblock's
        // end_of_slice_flag to the next engine.
        let mut open_terminate = false;
        while i < total {
            let mut e = CabacEncoder::new(&mut w);
            if open_terminate {
                e.encode_terminate(0);
                open_terminate = false;
            }
            let mut pcm: Option<&[u16]> = None;
            while i < total {
                let left = (i % mb_width > 0).then(|| &coded[i - 1]);
                let above = (i >= mb_width).then(|| &coded[i - mb_width]);
                if p_slice {
                    // Every macroblock of a P slice codes mb_skip_flag,
                    // whatever it turns out to be.
                    let is_skip =
                        matches!(&mbs[i], TestMb::Inter(d) if d.kind == InterMbKind::PSkip);
                    let lnb = left.map(|m| &m.nb);
                    let anb = above.map(|m| &m.nb);
                    write_mb_skip_cabac(&mut e, &mut enc_st, lnb, anb, false, is_skip);
                }
                match &mbs[i] {
                    TestMb::Intra(d) => {
                        if p_slice {
                            write_mb_type_p_cabac(&mut e, &mut enc_st, 5 + intra_mb_type_code(d));
                            write_intra_mb_body(
                                &mut e, &mut enc_st, d, left, above, cfi, field, t8x8,
                            );
                        } else {
                            write_mb(&mut e, &mut enc_st, d, left, above, cfi, field, t8x8);
                        }
                        coded.push(Coded {
                            nb: WrittenMb::from_decision(d, c444),
                            not_nxn: !d.kind.is_nxn(),
                            chroma_nonzero: d.chroma_mode != 0,
                        });
                        i += 1;
                        e.encode_terminate((i == total) as u32);
                        if i == total {
                            break;
                        }
                    }
                    TestMb::Inter(d) => {
                        match d.kind {
                            // A skip wrote its flag and writes nothing else;
                            // the qp-delta carry clears, as the decoder's
                            // slice loop clears it.
                            InterMbKind::PSkip => enc_st.prev_qp_delta_nonzero = false,
                            InterMbKind::P16x16
                            | InterMbKind::P16x8
                            | InterMbKind::P8x16
                            | InterMbKind::P8x8 => {
                                write_p16x16_mb(
                                    &mut e, &mut enc_st, d, left, above, cfi, field, num_ref,
                                    t8x8,
                                )
                            }
                            InterMbKind::UseIntra => {
                                unreachable!("tests spell intra via TestMb::Intra")
                            }
                        }
                        coded.push(Coded {
                            nb: WrittenMb::from_inter_decision(d, c444),
                            not_nxn: true,
                            chroma_nonzero: false,
                        });
                        i += 1;
                        e.encode_terminate((i == total) as u32);
                        if i == total {
                            break;
                        }
                    }
                    TestMb::Pcm(samples) => {
                        if p_slice {
                            write_mb_type_p_cabac(&mut e, &mut enc_st, 5 + MB_TYPE_I_PCM);
                        } else {
                            let inc = left.map_or(0, |m| m.not_nxn as usize)
                                + above.map_or(0, |m| m.not_nxn as usize);
                            write_mb_type_i_cabac(&mut e, &mut enc_st, inc, MB_TYPE_I_PCM);
                        }
                        enc_st.prev_qp_delta_nonzero = false;
                        coded.push(Coded {
                            nb: WrittenMb::pcm(),
                            not_nxn: true,
                            chroma_nonzero: false,
                        });
                        i += 1;
                        pcm = Some(samples);
                        break; // the terminate of 1 flushed this codeword
                    }
                }
            }
            drop(e);
            if let Some(samples) = pcm {
                w.align_zero(); // pcm_alignment_zero_bit
                for &s in samples {
                    w.bits(8, s as u32);
                }
                open_terminate = true;
                if i == total {
                    let mut e = CabacEncoder::new(&mut w);
                    e.encode_terminate(1);
                    open_terminate = false;
                }
            }
        }
        w.align_zero();
        let data = w.into_rbsp();

        // ---- read back ----
        let ctx = slice_ctx(slice_type, num_ref, cfi, field, t8x8);
        let chroma_rows = match cfi {
            1 => 2,
            2 => 4,
            _ => 0,
        };
        let planes = if c444 { 3 } else { 1 };
        let mut dec_st = CabacState::new(slice_type, 0, qp);
        let mut c = Cabac::new(&data);
        let mut info = PicInfo::new(mb_width, total / mb_width);
        let mut layer = MbLayer::new(MbKind::I4x4);
        let mut nb = MbNeighbours::default();
        // The current picture's motion so far, which the ref_idx contexts
        // read (the decoder's `cur.motion`); only `ref_idx` is consulted.
        let mut frame_motion: [Vec<BlockMotion>; 2] = [
            vec![BlockMotion::default(); total * 16],
            vec![BlockMotion::default(); total * 16],
        ];
        for (addr, mb) in mbs.iter().enumerate() {
            nb.derive_into(&info, addr, 0);
            nb.gather_nz(&info, planes, chroma_rows);
            let want_skip = matches!(mb, TestMb::Inter(d) if d.kind == InterMbKind::PSkip);
            let mut skipped = false;
            if p_slice {
                let skip = decode_mb_skip(&mut c, &mut dec_st, &info, &nb, false);
                assert_eq!(skip, want_skip, "mb {addr} skip flag");
                if skip {
                    // The decoder's slice loop: a blank P_Skip layer and a
                    // cleared qp-delta carry.
                    layer.reset(MbKind::PSkip, true);
                    dec_st.prev_qp_delta_nonzero = false;
                    skipped = true;
                }
            } else {
                assert!(!want_skip, "an I-slice test cannot hold a skip");
            }
            if !skipped {
                let syntax =
                    parse_mb(&mut c, &mut dec_st, &ctx, &info, &nb, &frame_motion, &mut layer);
                match mb {
                    TestMb::Pcm(samples) => {
                        assert_eq!(layer.kind, MbKind::IPcm, "mb {addr} kind");
                        assert_eq!(&layer.pcm[..], &samples[..], "mb {addr} PCM samples");
                    }
                    TestMb::Intra(d) => check_mb(addr, d, &layer, &syntax, &ctx),
                    TestMb::Inter(d) => check_inter_mb(addr, d, &layer, &ctx, num_ref),
                }
            }
            commit(&mut info, addr, &layer, c444);
            let bm = match layer.kind {
                MbKind::Inter16x16 => {
                    BlockMotion { ref_idx: layer.ref_idx[0][0], ..BlockMotion::default() }
                }
                MbKind::PSkip => BlockMotion { ref_idx: 0, ..BlockMotion::default() },
                _ => BlockMotion::default(),
            };
            frame_motion[0][addr * 16..addr * 16 + 16].fill(bm);
            let eos = decode_end_of_slice(&mut c);
            assert_eq!(eos, addr + 1 == total, "end_of_slice after mb {addr}");
        }
        assert!(!c.overrun(), "the reader ran past what the writer produced");
        assert_eq!(enc_st.ctx, dec_st.ctx, "context states diverged");
        assert_eq!(
            enc_st.prev_qp_delta_nonzero, dec_st.prev_qp_delta_nonzero,
            "the qp-delta carry diverged"
        );
    }

    /// Every B `sub_mb_type` of Table 7-18, written and decoded by
    /// [`decode_sub_mb_type_b`] with the contexts compared afterwards —
    /// the binarisation has four branches and a value on the wrong one
    /// would still decode to *something*.
    #[test]
    fn b_sub_mb_type_round_trips() {
        for t in 0..=12u32 {
            let mut w = BitWriter::new();
            let mut enc_st = CabacState::new(SliceType::B, 0, 30);
            {
                let mut e = CabacEncoder::new(&mut w);
                write_sub_mb_type_b_cabac(&mut e, &mut enc_st, t);
                e.encode_terminate(1);
            }
            w.align_zero();
            let data = w.into_rbsp();
            let mut dec_st = CabacState::new(SliceType::B, 0, 30);
            let mut c = Cabac::new(&data);
            assert_eq!(decode_sub_mb_type_b(&mut c, &mut dec_st), t);
            assert_eq!(c.terminate(), 1, "t {t}");
            assert!(!c.overrun());
            assert_eq!(enc_st.ctx, dec_st.ctx, "t {t}: contexts diverged");
        }
    }

    /// Every B `mb_type` — Table 7-14's whole numbering, 0..=22, plus the
    /// intra tree behind the B prefix at 23+ — against every first-bin
    /// increment the neighbour rule can produce, decoded by
    /// [`decode_mb_type`] over real neighbour configurations. (I_PCM in a
    /// B slice has no producer — the intra fallback never chooses PCM —
    /// and stays untested here.)
    #[test]
    fn b_mb_type_round_trips() {
        for (inc, kinds) in [
            (0usize, Some((MbKind::BSkip, MbKind::BDirect16x16))),
            (1, Some((MbKind::Inter16x16, MbKind::BSkip))),
            (2, Some((MbKind::Inter16x16, MbKind::I16x16))),
        ] {
            for t in (0u32..=22).chain([23, 24, 30, 47]) {
                let mut w = BitWriter::new();
                let mut enc_st = CabacState::new(SliceType::B, 0, 30);
                {
                    let mut e = CabacEncoder::new(&mut w);
                    write_mb_type_b_cabac(&mut e, &mut enc_st, inc, t);
                    e.encode_terminate(1);
                }
                w.align_zero();
                let data = w.into_rbsp();

                let mut info = PicInfo::new(2, 2);
                let addr = 3;
                if let Some((left, above)) = kinds {
                    for (a, k) in [(2usize, left), (1usize, above)] {
                        info.mbs[a].kind = k;
                        info.mbs[a].decoded = true;
                    }
                }
                let mut nb = MbNeighbours::default();
                nb.derive_into(&info, addr, 0);
                let ctx = slice_ctx(SliceType::B, 1, 1, false, false);
                let mut dec_st = CabacState::new(SliceType::B, 0, 30);
                let mut c = Cabac::new(&data);
                let got = decode_mb_type(&mut c, &mut dec_st, &ctx, &info, &nb).unwrap();
                assert_eq!(got, t, "inc {inc}");
                assert_eq!(c.terminate(), 1, "inc {inc} t {t}");
                assert!(!c.overrun());
                assert_eq!(enc_st.ctx, dec_st.ctx, "inc {inc} t {t}: contexts diverged");
            }
        }
    }

    /// The mvd writer's context sums read the *list's own* neighbour
    /// mvds: a large list-0 mvd next door must not push the list-1
    /// component into a different context. Written per list against
    /// neighbours whose two lists carry very different magnitudes, and
    /// decoded by [`decode_mvd`] over the decoder's own per-list arrays.
    #[test]
    fn mvd_contexts_read_the_lists_own_neighbours() {
        for list in 0..2usize {
            for mvd in [Mv::new(3, -7), Mv::new(-40, 1), Mv::ZERO] {
                // Left and above neighbours: big mvds in list 0, tiny in
                // list 1 — so the two lists select different contexts.
                let mut nbmb = WrittenMb::from_inter_decision(
                    &InterDecision { kind: InterMbKind::P16x16, ..InterDecision::default() },
                    false,
                );
                nbmb.mvd = [[Mv::new(30, 30); 16], [Mv::new(1, 0); 16]];
                let mut w = BitWriter::new();
                let mut enc_st = CabacState::new(SliceType::B, 0, 28);
                {
                    let mut e = CabacEncoder::new(&mut w);
                    write_mvd_16x16_cabac(&mut e, &mut enc_st, Some(&nbmb), Some(&nbmb), list, mvd);
                    e.encode_terminate(1);
                }
                w.align_zero();
                let data = w.into_rbsp();

                let mut info = PicInfo::new(2, 2);
                let addr = 3;
                for a in [1usize, 2] {
                    info.mbs[a].kind = MbKind::Inter16x16;
                    info.mbs[a].decoded = true;
                    for blk in 0..16 {
                        info.mvd[0][a * 16 + blk] = Mv::new(30, 30);
                        info.mvd[1][a * 16 + blk] = Mv::new(1, 0);
                    }
                }
                let mut nb = MbNeighbours::default();
                nb.derive_into(&info, addr, 0);
                let mut layer = MbLayer::new(MbKind::Inter16x16);
                layer.reset(MbKind::Inter16x16, true);
                let mut dec_st = CabacState::new(SliceType::B, 0, 28);
                let mut c = Cabac::new(&data);
                let (x, y) = decode_mvd(&mut c, &mut dec_st, &info, &layer, &nb, list, 0, 0).unwrap();
                assert_eq!((x, y), (mvd.x, mvd.y), "list {list}");
                assert_eq!(c.terminate(), 1);
                assert!(!c.overrun());
                assert_eq!(enc_st.ctx, dec_st.ctx, "list {list} mvd {mvd:?}: contexts diverged");
            }
        }
    }

    /// Every intra `mb_type` value against every first-bin increment,
    /// decoded by [`decode_mb_type`] over real neighbour configurations
    /// that produce those increments.
    #[test]
    fn mb_type_numbering_round_trips() {
        for (inc, kinds) in [
            (0usize, None),
            (1, Some((MbKind::I4x4, MbKind::I16x16))),
            (2, Some((MbKind::I16x16, MbKind::IPcm))),
        ] {
            for t in 0..=MB_TYPE_I_PCM {
                let mut w = BitWriter::new();
                let mut enc_st = CabacState::new(SliceType::I, 0, 30);
                {
                    let mut e = CabacEncoder::new(&mut w);
                    write_mb_type_i_cabac(&mut e, &mut enc_st, inc, t);
                    if t != MB_TYPE_I_PCM {
                        e.encode_terminate(1);
                    }
                }
                w.align_zero();
                let data = w.into_rbsp();

                let mut info = PicInfo::new(2, 2);
                let addr = if kinds.is_some() { 3 } else { 0 };
                if let Some((left, above)) = kinds {
                    for (a, k) in [(2usize, left), (1usize, above)] {
                        info.mbs[a].kind = k;
                        info.mbs[a].decoded = true;
                    }
                }
                let mut nb = MbNeighbours::default();
                nb.derive_into(&info, addr, 0);
                let ctx = slice_ctx(SliceType::I, 0, 1, false, false);
                let mut dec_st = CabacState::new(SliceType::I, 0, 30);
                let mut c = Cabac::new(&data);
                let got = decode_mb_type(&mut c, &mut dec_st, &ctx, &info, &nb).unwrap();
                assert_eq!(got, t, "inc {inc}");
                if t != MB_TYPE_I_PCM {
                    assert_eq!(c.terminate(), 1, "inc {inc} t {t}");
                }
                assert!(!c.overrun());
                assert_eq!(enc_st.ctx, dec_st.ctx, "inc {inc} t {t}: contexts diverged");
            }
        }
    }

    /// I_16x16 over every (prediction mode, chroma cbp, luma cbp) — the
    /// whole `mb_type` space — with qp_delta of each sign, luma DC blocks
    /// that are sometimes empty (a coded_block_flag of 0), and AC blocks
    /// holding runs of ±1.
    #[test]
    fn i16x16_sweep_round_trips() {
        let qpds = [-26i8, -3, -1, 0, 1, 2, 25];
        let mut qpd = qpds.iter().cycle();
        for mode in 0..4u8 {
            for cbp_chroma in 0..3u8 {
                for cbp_luma in [0u8, 15] {
                    let mut d = MbDecision::default();
                    d.kind = IntraKind::I16x16;
                    d.intra16_mode = mode;
                    d.cbp_luma = cbp_luma;
                    d.cbp_chroma = cbp_chroma;
                    d.chroma_mode = (mode + 1) % 4;
                    d.qp_delta = *qpd.next().unwrap();
                    if mode != 1 {
                        for i in 0..6 {
                            d.luma_dc[i] = if i % 2 == 0 { 1 } else { -1 };
                        }
                        d.luma_dc[15] = 3;
                    }
                    if cbp_luma == 15 {
                        for r in 0..16usize {
                            let mut b = [0i16; 16];
                            if r % 3 != 0 {
                                let n = 1 + r % 6;
                                for (k, slot) in b[1..1 + n].iter_mut().enumerate() {
                                    *slot = if k % 2 == 0 { -1 } else { 1 };
                                }
                                d.nz_luma[r] = n as u8;
                            }
                            d.luma[r] = b;
                        }
                    }
                    if cbp_chroma >= 1 {
                        d.chroma_dc[0][0] = 5;
                        d.chroma_dc[1][2] = -1;
                    }
                    if cbp_chroma == 2 {
                        for comp in 0..2 {
                            for blk in 0..4 {
                                let mut b = [0i16; 16];
                                if (comp + blk) % 2 == 0 {
                                    b[1] = 1;
                                    b[2] = -1;
                                    b[3] = 1;
                                    d.nz_chroma[comp][blk] = 3;
                                }
                                d.chroma_ac[comp][blk] = b;
                            }
                        }
                    }
                    round_trip_slice(&[TestMb::Intra(d)], 1, 1, false, 26, false, 0, false);
                }
            }
        }
    }

    /// `mb_qp_delta` over a chain of macroblocks: consecutive nonzero
    /// deltas exercise the prev-nonzero context, zeros reset it, and the
    /// values cover both signs and both extremes.
    #[test]
    fn qp_delta_chain_round_trips() {
        let deltas = [0i8, 5, -3, 0, 1, -1, 25, -26, 0, 2];
        let mbs: Vec<TestMb> = deltas
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let mut rng = lcg(0x9000 + i as u32);
                let mut d = synth_intra(&mut rng, 1, Some(IntraKind::I16x16));
                d.qp_delta = q;
                TestMb::Intra(d)
            })
            .collect();
        let n = mbs.len();
        round_trip_slice(&mbs, n, 1, false, 26, false, 0, false);
    }

    /// Whole synthetic slices over a macroblock grid: mixed I_4x4 /
    /// I_16x16 / I_PCM, every chroma format the writer covers, frame and
    /// field coding — so every neighbour-dependent context (mb_type,
    /// chroma mode, cbp, every coded_block_flag category) is derived from
    /// real written neighbours on both sides.
    #[test]
    fn grids_round_trip() {
        for (cfi, field, seed) in [
            (1u32, false, 1u32),
            (1, false, 2),
            (1, false, 3),
            (2, false, 4),
            (2, false, 5),
            (0, false, 6),
            (1, true, 7),
        ] {
            let (w_mb, h_mb) = (4usize, 3usize);
            let mut rng = lcg(0xC0FFEE ^ seed);
            let mut mbs = Vec::new();
            for i in 0..w_mb * h_mb {
                // A couple of I_PCM macroblocks, never the last (this
                // harness keeps the final end_of_slice out of the PCM
                // path; write_pcm_slice_data_cabac covers that ending).
                if i % 7 == 3 && i + 1 != w_mb * h_mb {
                    let n = 256
                        + match cfi {
                            1 => 128,
                            2 => 256,
                            _ => 0,
                        };
                    let samples: Vec<u16> = (0..n).map(|_| (rng() % 256) as u16).collect();
                    mbs.push(TestMb::Pcm(samples));
                    continue;
                }
                let mut d = synth_intra(&mut rng, cfi, None);
                let has_residual =
                    d.kind == IntraKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
                if has_residual {
                    d.qp_delta = [0i8, 0, 3, -2, 1, -26, 25][(rng() % 7) as usize];
                }
                mbs.push(TestMb::Intra(d));
            }
            round_trip_slice(&mbs, w_mb, cfi, field, 28, false, 0, false);
        }
    }

    /// The same grids with the 8x8 transform on offer, over every chroma
    /// format including 4:4:4 — where the two chroma planes are coded
    /// luma-style and the 8x8 blocks carry a coded_block_flag of their
    /// own (categories 5 / 9 / 13), which is the only place those
    /// contexts are ever exercised.
    ///
    /// The mixture matters more than the count: a slice of macroblocks
    /// that all chose the same transform size would never exercise the
    /// `transform_size_8x8_flag` context increment, which counts
    /// *neighbours whose flag was set*, nor the 8x8 coded_block_flag's
    /// rule that a neighbour transformed 4x4 contributes zero however
    /// many coefficients it has. So the synthesiser mixes all three
    /// intra kinds and the harness compares the full context array at the
    /// end, where a wrong increment shows up even when every coefficient
    /// came back right.
    #[test]
    fn grids_with_the_8x8_transform_round_trip() {
        for (cfi, field, seed) in [
            (1u32, false, 11u32),
            (1, false, 12),
            (2, false, 13),
            (0, false, 14),
            (3, false, 15),
            (3, false, 16),
            (1, true, 17),
            (3, true, 18),
        ] {
            let (w_mb, h_mb) = (4usize, 3usize);
            let mut rng = lcg(0x8888 ^ seed);
            let mut mbs = Vec::new();
            for _ in 0..w_mb * h_mb {
                let force = match rng() % 3 {
                    0 => IntraKind::I4x4,
                    1 => IntraKind::I8x8,
                    _ => IntraKind::I16x16,
                };
                let mut d = synth_intra(&mut rng, cfi, Some(force));
                let has_residual =
                    d.kind == IntraKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
                if has_residual {
                    d.qp_delta = [0i8, 0, 3, -2, 1, -26, 25][(rng() % 7) as usize];
                }
                mbs.push(TestMb::Intra(d));
            }
            round_trip_slice(&mbs, w_mb, cfi, field, 28, false, 0, true);
        }
    }

    /// A P slice whose coded macroblocks carry the 8x8 transform: the
    /// flag's *other* placement, after `coded_block_pattern` and only
    /// when some luma block is coded — with skips and intra macroblocks
    /// mixed in, since a skipped neighbour contributes zero to the flag's
    /// increment and an intra one contributes its own flag.
    #[test]
    fn a_p_slice_with_the_8x8_transform_round_trips() {
        for (cfi, seed, num_ref) in [(1u32, 21u32, 1u32), (1, 22, 3), (2, 23, 1), (3, 24, 1), (0, 25, 2)] {
            let (w_mb, h_mb) = (4usize, 3usize);
            let mut rng = lcg(0x5151 ^ seed);
            let mut mbs = Vec::new();
            for _ in 0..w_mb * h_mb {
                if rng() % 4 == 0 {
                    let force = if rng() % 2 == 0 { IntraKind::I8x8 } else { IntraKind::I4x4 };
                    let mut d = synth_intra(&mut rng, cfi, Some(force));
                    if d.cbp_luma != 0 || d.cbp_chroma != 0 {
                        d.qp_delta = [0i8, 2, -3][(rng() % 3) as usize];
                    }
                    mbs.push(TestMb::Intra(d));
                    continue;
                }
                let mut d = synth_inter(&mut rng, cfi, num_ref);
                if d.kind == InterMbKind::P16x16 && d.cbp_luma != 0 && rng() % 2 == 0 {
                    make_inter_8x8(&mut rng, &mut d, cfi);
                }
                if d.cbp_luma != 0 || d.cbp_chroma != 0 {
                    d.qp_delta = [0i8, 1, -2, 25][(rng() % 4) as usize];
                }
                mbs.push(TestMb::Inter(d));
            }
            round_trip_slice(&mbs, w_mb, cfi, false, 28, true, num_ref, true);
        }
    }

    /// Turn an already-synthesised P macroblock into an 8x8-transformed
    /// one: the luma-style planes recoded as four blocks of sixty-four,
    /// with the cbp bits its own contents justify. The chroma of a
    /// 4:2:0 / 4:2:2 macroblock is untouched — there is no 8x8 chroma
    /// transform outside 4:4:4.
    fn make_inter_8x8(rng: &mut impl FnMut() -> u32, d: &mut InterDecision, cfi: u32) {
        d.transform_8x8 = true;
        d.luma = [[0; 16]; 16];
        d.nz_luma = [0; 16];
        for blk8 in 0..4usize {
            if d.cbp_luma & (1 << blk8) == 0 {
                continue;
            }
            let mut b = [0i16; 64];
            fill_block8(rng, &mut b);
            d.luma.as_flattened_mut()[blk8 * 64..blk8 * 64 + 64].copy_from_slice(&b);
            let counts = sub_block_counts_8x8(&b);
            for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
                d.nz_luma[r] = counts[sub];
            }
        }
        if cfi == 3 {
            d.chroma_ac = [[[0; 16]; 16]; 2];
            d.nz_chroma = [[0; 16]; 2];
            for comp in 0..2 {
                for blk8 in 0..4usize {
                    if d.cbp_luma & (1 << blk8) == 0 {
                        continue;
                    }
                    let mut b = [0i16; 64];
                    fill_block8(rng, &mut b);
                    if rng() % 3 == 0 {
                        b = [0; 64];
                    }
                    d.chroma_ac[comp].as_flattened_mut()[blk8 * 64..blk8 * 64 + 64]
                        .copy_from_slice(&b);
                    let counts = sub_block_counts_8x8(&b);
                    for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
                        d.nz_chroma[comp][r] = counts[sub];
                    }
                }
            }
        }
        // A macroblock with no coded luma block carries no flag.
        if d.cbp_luma == 0 {
            d.transform_8x8 = false;
        }
    }

    /// Every P-slice `mb_type` the writer spells, decoded by
    /// [`decode_mb_type`] over a P-slice context: the inter shapes and the
    /// intra tree behind the P prefix, I_PCM (30, which flushes) included.
    #[test]
    fn p_mb_type_round_trips() {
        for t in [0u32, 1, 2, 5, 6, 10, 17, 22, 29, 30] {
            let mut w = BitWriter::new();
            let mut enc_st = CabacState::new(SliceType::P, 0, 30);
            {
                let mut e = CabacEncoder::new(&mut w);
                write_mb_type_p_cabac(&mut e, &mut enc_st, t);
                if t != 30 {
                    e.encode_terminate(1);
                }
            }
            w.align_zero();
            let data = w.into_rbsp();

            let info = PicInfo::new(1, 1);
            let mut nb = MbNeighbours::default();
            nb.derive_into(&info, 0, 0);
            let ctx = slice_ctx(SliceType::P, 1, 1, false, false);
            let mut dec_st = CabacState::new(SliceType::P, 0, 30);
            let mut c = Cabac::new(&data);
            let got = decode_mb_type(&mut c, &mut dec_st, &ctx, &info, &nb).unwrap();
            assert_eq!(got, t);
            if t != 30 {
                assert_eq!(c.terminate(), 1, "t {t}");
            }
            assert!(!c.overrun());
            assert_eq!(enc_st.ctx, dec_st.ctx, "t {t}: contexts diverged");
        }
    }

    /// `ref_idx` unary against the production reader, over every first-bin
    /// increment (each neighbour's refIdxZeroFlag on or off) and values
    /// through both context switches of the unary tail.
    #[test]
    fn ref_idx_round_trips() {
        for (l_on, a_on) in [(false, false), (true, false), (false, true), (true, true)] {
            for v in 0..6i8 {
                let mk = |ref_idx: i8| {
                    WrittenMb::from_inter_decision(
                        &InterDecision {
                            kind: InterMbKind::P16x16,
                            ref_idx,
                            ..InterDecision::default()
                        },
                        false,
                    )
                };
                let (lw, aw) = (mk(l_on as i8), mk(a_on as i8));
                let mut w = BitWriter::new();
                let mut enc_st = CabacState::new(SliceType::P, 0, 28);
                {
                    let mut e = CabacEncoder::new(&mut w);
                    write_ref_idx_16x16_cabac(&mut e, &mut enc_st, Some(&lw), Some(&aw), v);
                    e.encode_terminate(1);
                }
                w.align_zero();
                let data = w.into_rbsp();

                let mut info = PicInfo::new(2, 2);
                let mut fm: [Vec<BlockMotion>; 2] = [
                    vec![BlockMotion::default(); 4 * 16],
                    vec![BlockMotion::default(); 4 * 16],
                ];
                for (a, on) in [(2usize, l_on), (1usize, a_on)] {
                    info.mbs[a].kind = MbKind::Inter16x16;
                    info.mbs[a].decoded = true;
                    let bm = BlockMotion { ref_idx: on as i8, ..BlockMotion::default() };
                    fm[0][a * 16..a * 16 + 16].fill(bm);
                }
                let mut nb = MbNeighbours::default();
                nb.derive_into(&info, 3, 0);
                let layer = MbLayer::new(MbKind::Inter16x16);
                let mut dec_st = CabacState::new(SliceType::P, 0, 28);
                let mut c = Cabac::new(&data);
                let got =
                    decode_ref_idx(&mut c, &mut dec_st, &info, &layer, &nb, &fm, 0, 0, 0).unwrap();
                assert_eq!(got, v, "l={l_on} a={a_on}");
                assert_eq!(c.terminate(), 1);
                assert!(!c.overrun());
                assert_eq!(enc_st.ctx, dec_st.ctx, "l={l_on} a={a_on} v={v}: contexts diverged");
            }
        }
    }

    /// One mvd component against the production reader across the context
    /// thresholds (sums 2/3 and 32/33), the prefix cap of nine, and the
    /// UEG3 escape — chained in one codeword so the context state carries
    /// between values.
    #[test]
    fn mvd_component_round_trips() {
        let sums = [0i32, 2, 3, 32, 33, 500];
        let vals = [
            0i32, 1, -1, 2, -3, 8, -8, 9, -9, 10, 16, -17, 24, -25, 40, -100, 1000, -32767, 32767,
        ];
        let mut w = BitWriter::new();
        let mut enc_st = CabacState::new(SliceType::P, 0, 26);
        {
            let mut e = CabacEncoder::new(&mut w);
            for (i, &s) in sums.iter().enumerate() {
                for &v in &vals {
                    write_mvd_component_cabac(&mut e, &mut enc_st, s, i % 2, v);
                }
            }
            e.encode_terminate(1);
        }
        w.align_zero();
        let data = w.into_rbsp();

        let mut dec_st = CabacState::new(SliceType::P, 0, 26);
        let mut c = Cabac::new(&data);
        for (i, &s) in sums.iter().enumerate() {
            for &v in &vals {
                let got = decode_mvd_component(&mut c, &mut dec_st, s, 0, i % 2).unwrap();
                assert_eq!(got as i32, v, "sum={s}");
            }
        }
        assert_eq!(c.terminate(), 1);
        assert!(!c.overrun());
        assert_eq!(enc_st.ctx, dec_st.ctx, "contexts diverged");
    }

    /// Whole synthetic P slices: skip runs, `P_L0_16x16` with residual and
    /// mvds spanning the escape, intra-in-P, and one I_PCM through the
    /// P-slice spelling — with and without a coded ref_idx, 4:2:0 and
    /// 4:2:2 — so every neighbour-dependent context (skip, cbp,
    /// coded_block_flag, mvd sums, refIdxZeroFlag) is derived from real
    /// written neighbours on both sides.
    #[test]
    fn p_grids_round_trip() {
        for (cfi, num_ref, seed) in [(1u32, 1u32, 11u32), (1, 4, 12), (1, 4, 13), (2, 2, 14)] {
            let (w_mb, h_mb) = (4usize, 3usize);
            let mut rng = lcg(0xBEEF ^ seed);
            let mut mbs: Vec<TestMb> = Vec::new();
            for i in 0..w_mb * h_mb {
                match i {
                    // A macroblock with a nonzero qp_delta directly before
                    // the forced skips: the carry must clear across them.
                    4 => {
                        let mut d = InterDecision::default();
                        // One 16x16 partition: the same mvd on all sixteen blocks.
                        d.mvd = [Mv::new(7, -3); 16];
                        d.cbp_luma = 1;
                        d.luma[0][0] = 4;
                        d.luma[0][5] = -1;
                        d.nz_luma[0] = 2;
                        d.qp_delta = 3;
                        mbs.push(TestMb::Inter(d));
                    }
                    // A run of skips (the second sees a skipped left
                    // neighbour, ctxIdxInc 0 on that side).
                    5 | 6 => {
                        let d = InterDecision {
                            kind: InterMbKind::PSkip,
                            ..InterDecision::default()
                        };
                        mbs.push(TestMb::Inter(d));
                    }
                    // One I_PCM, through the P-slice mb_type spelling.
                    8 => {
                        let n = 256 + if cfi == 2 { 256 } else { 128 };
                        let samples: Vec<u16> =
                            (0..n).map(|_| (rng() % 256) as u16).collect();
                        mbs.push(TestMb::Pcm(samples));
                    }
                    // Intra-in-P.
                    _ if i % 5 == 3 => {
                        let mut d = synth_intra(&mut rng, cfi, None);
                        let has_residual = d.kind == IntraKind::I16x16
                            || d.cbp_luma != 0
                            || d.cbp_chroma != 0;
                        if has_residual {
                            d.qp_delta = [0i8, 2, -2][(rng() % 3) as usize];
                        }
                        mbs.push(TestMb::Intra(d));
                    }
                    _ => {
                        let mut d = synth_inter(&mut rng, cfi, num_ref);
                        if d.kind == InterMbKind::P16x16
                            && (d.cbp_luma != 0 || d.cbp_chroma != 0)
                        {
                            d.qp_delta = [0i8, 0, 3, -2, 25, -26][(rng() % 6) as usize];
                        }
                        mbs.push(TestMb::Inter(d));
                    }
                }
            }
            round_trip_slice(&mbs, w_mb, cfi, false, 28, true, num_ref, false);
        }
    }
}
