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
    debug_assert!(t <= MB_TYPE_I_PCM, "mb_type {t} out of the intra range");
    debug_assert!(inc <= 2, "the first bin has three contexts");
    let base = CTX_MB_TYPE_I;
    if t == 0 {
        e.encode_decision(&mut st.ctx[base + inc], 0); // I_NxN
        return;
    }
    e.encode_decision(&mut st.ctx[base + inc], 1);
    e.encode_terminate((t == MB_TYPE_I_PCM) as u32);
    if t == MB_TYPE_I_PCM {
        return; // flushed; the PCM samples follow outside the engine
    }
    // I_16x16. The reader sums t = 1 + 12·luma + 4·chroma + pred, so the
    // three fields come back out by the inverse arithmetic.
    let t = t - 1;
    e.encode_decision(&mut st.ctx[base + 3], (t >= 12) as u32);
    let chroma = (t / 4) % 3;
    e.encode_decision(&mut st.ctx[base + 4], (chroma != 0) as u32);
    if chroma != 0 {
        e.encode_decision(&mut st.ctx[base + 5], (chroma == 2) as u32);
    }
    let pred = t % 4;
    e.encode_decision(&mut st.ctx[base + 6], pred >> 1);
    e.encode_decision(&mut st.ctx[base + 7], pred & 1);
}

/// The I-slice `mb_type` of an intra `MbDecision`, in the numbering
/// [`write_mb_type_i_cabac`] takes (CAVLC spells the same value as
/// `ue(v)`): the inverse of [`super::cavlc::intra_mb_type`]'s I_16x16
/// arithmetic.
#[allow(dead_code)] // the picture loop being built is the caller
pub(crate) fn intra_mb_type_code(d: &MbDecision) -> u32 {
    match d.kind {
        IntraKind::I4x4 => 0,
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
    if d.kind == IntraKind::I4x4 {
        for blk in 0..16 {
            let m = d.luma_pred[super::mb::raster_of_blk(blk)];
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
/// already-written *intra* macroblock — the writer-side mirror of what the
/// decoder stores per macroblock (`MbInfo` plus the picture's nonzero-count
/// arrays), kept by the caller that walks the picture and handed to
/// [`write_cbp_cabac`] and [`write_intra_residual_cabac`] as the left and
/// above neighbours (`None` = not available: outside the picture or the
/// slice).
///
/// Build one with [`WrittenMb::from_decision`] after writing a macroblock,
/// or [`WrittenMb::pcm`] for an I_PCM one. There is no representation for
/// skipped or inter macroblocks: this seam covers intra slices, which is
/// all the encoder writes today, and grows a field when inter does.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WrittenMb {
    /// The macroblock is I_PCM.
    pub pcm: bool,
    /// The macroblock is I_16x16 (the luma-DC coded_block_flag context
    /// reads it).
    pub i16x16: bool,
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
    pub nz_chroma: [[u8; 8]; 2],
}

impl WrittenMb {
    /// The state of a macroblock written from `d` — derived here, next to
    /// the contexts that read it, so a caller cannot get the gating subtly
    /// wrong: blocks of an uncoded 8x8 count zero whatever the decision
    /// arrays hold, and a DC flag is set only when that DC block was
    /// actually written.
    #[allow(dead_code)] // the picture loop being built is the caller
    pub(crate) fn from_decision(d: &MbDecision) -> Self {
        let i16x16 = d.kind == IntraKind::I16x16;
        let mut nz_luma = [0u8; 16];
        for (r, nz) in nz_luma.iter_mut().enumerate() {
            let b8 = (r / 8) * 2 + (r % 4) / 2;
            if d.cbp_luma & (1 << b8) != 0 {
                *nz = d.nz_luma[r];
            }
        }
        let nz_chroma = if d.cbp_chroma == 2 { d.nz_chroma } else { [[0; 8]; 2] };
        let mut dc_cbf = 0u8;
        if i16x16 && d.luma_dc.iter().any(|&v| v != 0) {
            dc_cbf |= 1;
        }
        if d.cbp_chroma != 0 {
            for comp in 0..2 {
                if d.chroma_dc[comp].iter().any(|&v| v != 0) {
                    dc_cbf |= 2 << comp;
                }
            }
        }
        WrittenMb {
            pcm: false,
            i16x16,
            cbp: (d.cbp_luma & 15) | (d.cbp_chroma << 4),
            dc_cbf,
            nz_luma,
            nz_chroma,
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
            cbp: 0,
            dc_cbf: 0,
            nz_luma: [16; 16],
            nz_chroma: [[16; 8]; 2],
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

/// The `coded_block_flag` ctxIdxInc for a residual block of the intra
/// macroblock being *written*: [`cbf_ctx_inc`]'s mirror over the state an
/// encoder's picture loop keeps ([`WrittenMb`] neighbours plus the current
/// macroblock's own `MbDecision`) instead of the decoder's picture arrays.
/// It covers the categories an intra 4:2:0 / 4:2:2 macroblock codes (luma
/// DC / AC / 4x4, chroma DC / AC); the current macroblock is intra, so an
/// unavailable neighbour's condTermFlag is 1 throughout (9.3.3.1.1.9).
///
/// `rows` is the chroma block-row count (2 for 4:2:0, 4 for 4:2:2): it
/// picks which of the above neighbour's chroma blocks sit on the shared
/// edge. Blocks *inside* the current macroblock read `d`'s counts and cbp
/// directly — every left / above neighbour inside a macroblock comes
/// earlier in block order, so its count is what the decoder will have
/// stored by the time it derives this same increment.
#[allow(clippy::too_many_arguments)]
fn enc_cbf_ctx_inc(
    d: &MbDecision,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
    rows: usize,
    cat: usize,
    bx: usize,
    by: usize,
    comp: usize,
    blk: usize,
) -> usize {
    let cond_luma = |dx: i32, dy: i32| -> usize {
        let (nx, ny) = (bx as i32 + dx, by as i32 + dy);
        if nx >= 0 && ny >= 0 {
            let nblk = (ny * 4 + nx) as usize;
            let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
            if d.cbp_luma & (1 << b8) == 0 {
                return 0;
            }
            return (d.nz_luma[nblk] != 0) as usize;
        }
        let (m, edge) = if nx < 0 {
            (left, ny as usize * 4 + 3) // the left MB's right column
        } else {
            (above, 12 + nx as usize) // the above MB's bottom row
        };
        match m {
            None => 1,
            Some(m) => (m.nz_luma[edge] != 0) as usize,
        }
    };
    let cond_dc = |m: Option<&WrittenMb>, bit: u8, needs_i16: bool| -> usize {
        match m {
            None => 1,
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
            return (d.nz_chroma[comp][(cy * 2 + cx) as usize] != 0) as usize;
        }
        let (m, edge) = if cx < 0 {
            (left, cy as usize * 2 + 1)
        } else {
            (above, (rows - 1) * 2 + cx as usize)
        };
        match m {
            None => 1,
            Some(m) => (m.nz_chroma[comp][edge] != 0) as usize,
        }
    };
    match cat {
        0 => cond_dc(left, 0, true) + 2 * cond_dc(above, 0, true),
        1 | 2 => cond_luma(-1, 0) + 2 * cond_luma(0, -1),
        CAT_CHROMA_DC => {
            cond_dc(left, 1 + comp as u8, false) + 2 * cond_dc(above, 1 + comp as u8, false)
        }
        CAT_CHROMA_AC => cond_chroma_ac(-1, 0) + 2 * cond_chroma_ac(0, -1),
        _ => unreachable!("no category {cat} in an intra 4:2:x macroblock"),
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
/// transform, 4:2:0 / 4:2:2 / monochrome (4:4:4's luma-style chroma planes
/// and the 8x8 categories have no writer yet). The walk is the reader's:
/// I_16x16's luma DC first (always — its coded_block_flag may be zero),
/// then the luma 4x4 (or I_16x16 AC) blocks of each 8x8 whose cbp bit is
/// set, then, when the chroma cbp is nonzero, both components' chroma DC,
/// then, at chroma cbp 2, every chroma AC block.
///
/// The caller writes this *after* `mb_qp_delta`, and only when the
/// macroblock has residual at all (`cbp != 0` or I_16x16), matching
/// [`parse_mb_cabac`]. `left` / `above` are the neighbouring macroblocks'
/// [`WrittenMb`] state for the coded_block_flag contexts. `field` selects
/// the field scans and context tables — the current encoder is frame-only
/// and passes false.
#[allow(clippy::too_many_arguments, dead_code)] // the picture loop being built is the caller
pub(crate) fn write_intra_residual_cabac(
    e: &mut CabacEncoder,
    st: &mut CabacState,
    field: bool,
    chroma_format_idc: u32,
    d: &MbDecision,
    left: Option<&WrittenMb>,
    above: Option<&WrittenMb>,
) {
    debug_assert!(
        chroma_format_idc <= 2,
        "4:4:4 writes chroma as luma-style planes, which this writer does not spell"
    );
    let scan4: &[u8; 16] = if field { &FIELD_SCAN4X4 } else { &ZIGZAG4X4 };
    let c422 = chroma_format_idc == 2;
    let rows = if c422 { 4 } else { 2 };
    let i16 = d.kind == IntraKind::I16x16;
    let mut buf = [0i32; 16];

    if i16 {
        for (o, &v) in buf.iter_mut().zip(&d.luma_dc) {
            *o = v as i32;
        }
        let inc = enc_cbf_ctx_inc(d, left, above, rows, 0, 0, 0, 0, 0);
        write_residual_block_cabac(e, st, field, 0, Some(inc), &buf, scan4, 0, 16);
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if d.cbp_luma & (1 << blk8) == 0 {
            debug_assert!(
                (0..4).all(|sub| {
                    let raster = (by8 + (sub >> 1)) * 4 + bx8 + (sub & 1);
                    d.luma[raster].iter().all(|&v| v == 0)
                }),
                "8x8 {blk8} has coefficients but no cbp bit — they would be lost"
            );
            continue;
        }
        for sub in 0..4 {
            let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
            let raster = by * 4 + bx;
            for (o, &v) in buf.iter_mut().zip(&d.luma[raster]) {
                *o = v as i32;
            }
            // I_16x16 codes the 15 AC coefficients (position 0 lives in the
            // DC block); I_NxN codes all 16.
            let (cat, start, max_coeff) = if i16 { (1, 1, 15) } else { (2, 0, 16) };
            debug_assert!(!i16 || buf[0] == 0, "I_16x16 AC keeps position 0 free");
            let inc = enc_cbf_ctx_inc(d, left, above, rows, cat, bx, by, 0, 0);
            let n = write_residual_block_cabac(
                e, st, field, cat, Some(inc), &buf, scan4, start, max_coeff,
            );
            debug_assert_eq!(
                n, d.nz_luma[raster] as usize,
                "nz_luma[{raster}] disagrees with the levels; neighbour contexts would desync"
            );
        }
    }

    if chroma_format_idc == 0 || d.cbp_chroma == 0 {
        debug_assert!(chroma_format_idc != 0 || d.cbp_chroma == 0, "monochrome has no chroma cbp");
        return;
    }
    let n_dc = if c422 { 8 } else { 4 };
    let dc_scan: &[u8] = if c422 { &SCAN_CHROMA_DC_422[..] } else { &IDENTITY_OFF[..4] };
    for comp in 0..2 {
        debug_assert!(
            c422 || d.chroma_dc[comp][4..].iter().all(|&v| v == 0),
            "4:2:0 chroma DC has four coefficients"
        );
        buf = [0; 16];
        for (o, &v) in buf.iter_mut().zip(&d.chroma_dc[comp][..n_dc]) {
            *o = v as i32;
        }
        let inc = enc_cbf_ctx_inc(d, left, above, rows, CAT_CHROMA_DC, 0, 0, comp, 0);
        write_residual_block_cabac(e, st, field, CAT_CHROMA_DC, Some(inc), &buf, dc_scan, 0, n_dc);
    }
    if d.cbp_chroma == 2 {
        for comp in 0..2 {
            for blk in 0..2 * rows {
                for (o, &v) in buf.iter_mut().zip(&d.chroma_ac[comp][blk]) {
                    *o = v as i32;
                }
                debug_assert!(buf[0] == 0, "chroma AC keeps position 0 free");
                let inc = enc_cbf_ctx_inc(d, left, above, rows, CAT_CHROMA_AC, 0, 0, comp, blk);
                let n = write_residual_block_cabac(
                    e, st, field, CAT_CHROMA_AC, Some(inc), &buf, scan4, 1, 15,
                );
                debug_assert_eq!(
                    n, d.nz_chroma[comp][blk] as usize,
                    "nz_chroma[{comp}][{blk}] disagrees with the levels"
                );
            }
        }
    } else {
        debug_assert!(
            d.chroma_ac.iter().all(|c| c.iter().all(|b| b.iter().all(|&v| v == 0))),
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
mod intra_mb_round_trip {
    use super::*;
    use crate::bitwriter::BitWriter;
    use crate::encode::h264_intra::PredMode;
    use crate::h264::cavlc::intra_mb_type;
    use crate::h264::mb::raster_of_blk;

    /// Slice facts for an all-intra test slice.
    fn slice_ctx(chroma_format_idc: u32, field_pic: bool) -> SliceCtx {
        SliceCtx {
            slice_type: SliceType::I,
            slice_num: 0,
            num_ref_idx: [0, 0],
            direct_spatial: false,
            transform_8x8_mode: false,
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
        match d.kind {
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
        d.chroma_mode = if cfi == 0 { 0 } else { (rng() % 4) as u8 };
        if cfi == 1 || cfi == 2 {
            let n_dc = if cfi == 2 { 8 } else { 4 };
            let rows = if cfi == 2 { 4 } else { 2 };
            d.cbp_chroma = (rng() % 3) as u8;
            if d.cbp_chroma >= 1 {
                for comp in 0..2 {
                    let mut b = [0i16; 16];
                    let _ = fill_block(rng, &mut b, 0, n_dc);
                    for (i, o) in d.chroma_dc[comp][..n_dc].iter_mut().enumerate() {
                        *o = b[i];
                    }
                }
            }
            if d.cbp_chroma == 2 {
                for comp in 0..2 {
                    for blk in 0..2 * rows {
                        let mut b = [0i16; 16];
                        d.nz_chroma[comp][blk] = fill_block(rng, &mut b, 1, 16);
                        d.chroma_ac[comp][blk] = b;
                    }
                }
            }
        }
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
    }

    /// Write one intra macroblock in the syntax order [`parse_mb_cabac`]
    /// reads it: mb_type, prediction modes, cbp (I_NxN only), mb_qp_delta
    /// and residual when there is residual — the reference for the picture
    /// loop this module's writers will be wired into.
    fn write_mb(
        e: &mut CabacEncoder,
        st: &mut CabacState,
        d: &MbDecision,
        left: Option<&Coded>,
        above: Option<&Coded>,
        cfi: u32,
        field: bool,
    ) {
        let inc = left.map_or(0, |m| m.not_nxn as usize) + above.map_or(0, |m| m.not_nxn as usize);
        write_mb_type_i_cabac(e, st, inc, intra_mb_type_code(d));
        let chroma_nb = (cfi == 1 || cfi == 2).then(|| {
            [
                left.is_some_and(|m| m.chroma_nonzero),
                above.is_some_and(|m| m.chroma_nonzero),
            ]
        });
        write_intra_pred_modes_cabac(e, st, d, chroma_nb);
        let lnb = left.map(|m| &m.nb);
        let anb = above.map(|m| &m.nb);
        if d.kind == IntraKind::I4x4 {
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

    /// [`parse_mb_cabac`]'s intra path, with `dq = None` so the raw levels
    /// come back, returning the recovered `(prev, rem)` syntax per 4x4
    /// block: given the predicted mode, mode ↔ (prev, rem) is a bijection
    /// (prev spells exactly the predicted mode; a remainder never does), so
    /// comparing the recovery against what was written is exact.
    fn parse_intra_mb(
        c: &mut Cabac,
        st: &mut CabacState,
        ctx: &SliceCtx,
        info: &PicInfo,
        nb: &MbNeighbours,
        layer: &mut MbLayer,
    ) -> [(bool, u8); 16] {
        let mut syntax = [(true, 0u8); 16];
        layer.reset(MbKind::I4x4, true);
        let t = decode_mb_type(c, st, ctx, info, nb).expect("mb_type rejected");
        intra_mb_type(t, layer).expect("mb_type out of range");
        match layer.kind {
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
    fn commit(info: &mut PicInfo, addr: usize, layer: &MbLayer) {
        let m = &mut info.mbs[addr];
        m.kind = layer.kind;
        m.slice = 0;
        m.decoded = true;
        m.cbp = layer.cbp;
        m.transform_8x8 = false;
        m.chroma_mode = layer.chroma_mode;
        m.qp_delta_nonzero = layer.has_residual() && layer.qp_delta != 0;
        m.dc_cbf = layer.dc_cbf;
        let base = addr * 16;
        if layer.kind == MbKind::IPcm {
            info.luma_nz[base..base + 16].fill(16);
            info.chroma_nz[addr * 32..addr * 32 + 32].fill(16);
        } else {
            info.luma_nz[base..base + 16].copy_from_slice(&layer.nz[0]);
            for comp in 0..2 {
                info.chroma_nz[addr * 32 + comp * 16..addr * 32 + comp * 16 + 8]
                    .copy_from_slice(&layer.chroma_nz[comp]);
            }
        }
        if layer.kind == MbKind::I4x4 {
            info.intra_modes[base..base + 16].copy_from_slice(&layer.intra_modes);
        } else {
            info.intra_modes[base..base + 16].fill(2);
        }
    }

    /// Every field of one parsed macroblock against the decision that was
    /// written — and the writer's own [`WrittenMb`] against what the
    /// decoder stores, which is the proof of `from_decision`.
    fn check_mb(addr: usize, d: &MbDecision, layer: &MbLayer, syntax: &[(bool, u8); 16], ctx: &SliceCtx) {
        match d.kind {
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
        assert_eq!(layer.cbp, (d.cbp_luma & 15) | (d.cbp_chroma << 4), "mb {addr} cbp");
        let chroma = ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2;
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
        for r in 0..16 {
            for k in 0..16 {
                assert_eq!(
                    layer.coef[0][r * 16 + k],
                    d.luma[r][k] as i32,
                    "mb {addr} luma block {r} coeff {k}"
                );
            }
        }
        assert_eq!(layer.nz[0], d.nz_luma, "mb {addr} luma nz");
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
                assert_eq!(layer.chroma_nz[comp], d.nz_chroma[comp], "mb {addr} chroma {comp} nz");
            }
        }
        let wm = WrittenMb::from_decision(d);
        assert_eq!(wm.cbp, layer.cbp, "mb {addr} WrittenMb cbp");
        assert_eq!(wm.dc_cbf, layer.dc_cbf, "mb {addr} WrittenMb dc_cbf");
        assert_eq!(wm.nz_luma, layer.nz[0], "mb {addr} WrittenMb nz_luma");
        assert_eq!(wm.nz_chroma[0], layer.chroma_nz[0], "mb {addr} WrittenMb nz_chroma Cb");
        assert_eq!(wm.nz_chroma[1], layer.chroma_nz[1], "mb {addr} WrittenMb nz_chroma Cr");
    }

    /// Write a whole synthetic I slice with the writer functions, decode it
    /// with the production readers over the production neighbour machinery,
    /// and require every field — and the entire context array — to come
    /// back. The context comparison is the half that catches a desync: two
    /// spellings can agree on this slice's bins while leaving the
    /// probability model in different places, and nothing goes wrong until
    /// a later bin reads against a state the writer never had.
    fn round_trip_slice(mbs: &[TestMb], mb_width: usize, cfi: u32, field: bool, qp: i32) {
        let total = mbs.len();
        assert_eq!(total % mb_width, 0);

        // ---- write ----
        let mut w = BitWriter::new();
        w.align_one(); // cabac_alignment_one_bit (a fresh writer is aligned)
        let mut enc_st = CabacState::new(SliceType::I, 0, qp);
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
                match &mbs[i] {
                    TestMb::Intra(d) => {
                        write_mb(&mut e, &mut enc_st, d, left, above, cfi, field);
                        coded.push(Coded {
                            nb: WrittenMb::from_decision(d),
                            not_nxn: d.kind != IntraKind::I4x4,
                            chroma_nonzero: d.chroma_mode != 0,
                        });
                        i += 1;
                        e.encode_terminate((i == total) as u32);
                        if i == total {
                            break;
                        }
                    }
                    TestMb::Pcm(samples) => {
                        let inc = left.map_or(0, |m| m.not_nxn as usize)
                            + above.map_or(0, |m| m.not_nxn as usize);
                        write_mb_type_i_cabac(&mut e, &mut enc_st, inc, MB_TYPE_I_PCM);
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
        let ctx = slice_ctx(cfi, field);
        let chroma_rows = match cfi {
            1 => 2,
            2 => 4,
            _ => 0,
        };
        let mut dec_st = CabacState::new(SliceType::I, 0, qp);
        let mut c = Cabac::new(&data);
        let mut info = PicInfo::new(mb_width, total / mb_width);
        let mut layer = MbLayer::new(MbKind::I4x4);
        let mut nb = MbNeighbours::default();
        for (addr, mb) in mbs.iter().enumerate() {
            nb.derive_into(&info, addr, 0);
            nb.gather_nz(&info, 1, chroma_rows);
            let syntax = parse_intra_mb(&mut c, &mut dec_st, &ctx, &info, &nb, &mut layer);
            match mb {
                TestMb::Pcm(samples) => {
                    assert_eq!(layer.kind, MbKind::IPcm, "mb {addr} kind");
                    assert_eq!(&layer.pcm[..], &samples[..], "mb {addr} PCM samples");
                }
                TestMb::Intra(d) => check_mb(addr, d, &layer, &syntax, &ctx),
            }
            commit(&mut info, addr, &layer);
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
                let ctx = slice_ctx(1, false);
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
                    round_trip_slice(&[TestMb::Intra(d)], 1, 1, false, 26);
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
        round_trip_slice(&mbs, n, 1, false, 26);
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
            round_trip_slice(&mbs, w_mb, cfi, field, 28);
        }
    }
}
