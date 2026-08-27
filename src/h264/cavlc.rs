//! CAVLC (H.264 clause 9.2) and the CAVLC-coded macroblock layer (7.3.5).

use std::sync::OnceLock;

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::{Error, Result};

use super::frame::Mv;
use super::mb::{MbDequant, 
    MbKind, MbLayer, MbNeighbours, PRED_BI, PRED_L0, PRED_L1, PicInfo, SliceCtx, SubMbShape,
};
use super::slice::SliceType;
use super::tables::*;

/// A decoded VLC entry: `(code length, value)`; length 0 = invalid code.
#[derive(Clone, Copy, Default)]
struct VlcEntry {
    len: u8,
    a: u8,
    b: u8,
}

/// Lookup tables built once: `coeff_token` for the five nC classes as two
/// levels of 8 bits (a first-level entry with `len == ESCAPE` names a
/// second-level table for the next 8 bits — the tables stay a few KiB and in
/// L1, where one flat 16-bit table per class was a megabyte of cache misses),
/// `total_zeros` (peek 9 bits) and `run_before` (peek 3 bits, or arithmetic).
struct VlcTables {
    coeff_token: Vec<[VlcEntry; 256]>,
    coeff_token2: Vec<[VlcEntry; 256]>,
    total_zeros: [[VlcEntry; 512]; 15],
    chroma_dc_total_zeros: [[VlcEntry; 512]; 3],
    /// 4:2:2 chroma DC (eight coefficients): tzVlcIndex 1..=7.
    chroma422_dc_total_zeros: [[VlcEntry; 512]; 7],
    /// `run_before` for zerosLeft 1..=6 (codes are at most 3 bits); zerosLeft
    /// > 6 is decoded arithmetically (see `read_run_before`).
    run_before: [[VlcEntry; 8]; 6],
}

/// `len` of a first-level `coeff_token` entry that continues in
/// `coeff_token2[a]`.
const ESCAPE: u8 = 0xff;

/// Build the two-level table for one class from `(len, code) -> (tc, t1)`.
fn build_coeff_token(
    first: &mut [VlcEntry; 256],
    second: &mut Vec<[VlcEntry; 256]>,
    codes: &[(u8, u32, u8, u8)],
) {
    for &(len, code, tc, t1) in codes {
        if len == 0 {
            continue;
        }
        if len <= 8 {
            build_table(first, 8, len, code, tc, t1);
        } else {
            let prefix = (code >> (len - 8)) as usize;
            let sub_id = if first[prefix].len == ESCAPE {
                first[prefix].a as usize
            } else {
                second.push([VlcEntry::default(); 256]);
                first[prefix] = VlcEntry {
                    len: ESCAPE,
                    a: (second.len() - 1) as u8,
                    b: 0,
                };
                second.len() - 1
            };
            let rest_len = len - 8;
            let rest_code = code & ((1u32 << rest_len) - 1);
            build_table(&mut second[sub_id], 8, rest_len, rest_code, tc, t1);
        }
    }
}

fn build_table(entries: &mut [VlcEntry], bits: u32, len: u8, code: u32, a: u8, b: u8) {
    if len == 0 {
        return;
    }
    let shift = bits - len as u32;
    let base = (code as usize) << shift;
    for i in 0..(1usize << shift) {
        entries[base + i] = VlcEntry { len, a, b };
    }
}

fn tables() -> &'static VlcTables {
    static T: OnceLock<Box<VlcTables>> = OnceLock::new();
    T.get_or_init(|| {
        let mut coeff_token: Vec<[VlcEntry; 256]> = Vec::with_capacity(6);
        let mut coeff_token2: Vec<[VlcEntry; 256]> = Vec::new();
        for cls in 0..4 {
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..17 {
                for t1 in 0..4 {
                    codes.push((
                        COEFF_TOKEN_LEN[cls][tc][t1],
                        COEFF_TOKEN_BITS[cls][tc][t1] as u32,
                        tc as u8,
                        t1 as u8,
                    ));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        {
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..5 {
                for t1 in 0..4 {
                    codes.push((
                        CHROMA_DC_COEFF_TOKEN_LEN[tc][t1],
                        CHROMA_DC_COEFF_TOKEN_BITS[tc][t1] as u32,
                        tc as u8,
                        t1 as u8,
                    ));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        {
            // Class 5: 4:2:2 chroma DC (nC == -2).
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..9 {
                for t1 in 0..4 {
                    codes.push((
                        CHROMA422_DC_COEFF_TOKEN_LEN[tc][t1],
                        CHROMA422_DC_COEFF_TOKEN_BITS[tc][t1] as u32,
                        tc as u8,
                        t1 as u8,
                    ));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        let mut total_zeros = [[VlcEntry::default(); 512]; 15];
        for (tc1, tab) in total_zeros.iter_mut().enumerate() {
            for tz in 0..16 {
                build_table(
                    tab,
                    9,
                    TOTAL_ZEROS_LEN[tc1][tz],
                    TOTAL_ZEROS_BITS[tc1][tz] as u32,
                    tz as u8,
                    0,
                );
            }
        }
        let mut chroma_dc_total_zeros = [[VlcEntry::default(); 512]; 3];
        for (tc1, tab) in chroma_dc_total_zeros.iter_mut().enumerate() {
            for tz in 0..4 {
                build_table(
                    tab,
                    9,
                    CHROMA_DC_TOTAL_ZEROS_LEN[tc1][tz],
                    CHROMA_DC_TOTAL_ZEROS_BITS[tc1][tz] as u32,
                    tz as u8,
                    0,
                );
            }
        }
        let mut chroma422_dc_total_zeros = [[VlcEntry::default(); 512]; 7];
        for (tc1, tab) in chroma422_dc_total_zeros.iter_mut().enumerate() {
            for tz in 0..8 {
                build_table(
                    tab,
                    9,
                    CHROMA422_DC_TOTAL_ZEROS_LEN[tc1][tz],
                    CHROMA422_DC_TOTAL_ZEROS_BITS[tc1][tz] as u32,
                    tz as u8,
                    0,
                );
            }
        }
        let mut run_before = [[VlcEntry::default(); 8]; 6];
        for (zl1, tab) in run_before.iter_mut().enumerate() {
            for rb in 0..16 {
                build_table(
                    tab,
                    3,
                    RUN_BEFORE_LEN[zl1][rb],
                    RUN_BEFORE_BITS[zl1][rb] as u32,
                    rb as u8,
                    0,
                );
            }
        }
        Box::new(VlcTables {
            coeff_token,
            coeff_token2,
            total_zeros,
            chroma_dc_total_zeros,
            chroma422_dc_total_zeros,
            run_before,
        })
    })
}

/// Decode `coeff_token` for the given `nC` (−1 for 4:2:0 chroma DC, −2 for
/// 4:2:2 chroma DC). Returns `(TotalCoeff, TrailingOnes)`.
#[inline(always)]
fn read_coeff_token(r: &mut BitReader, t: &VlcTables, nc: i32) -> Result<(usize, usize)> {
    let cls = match nc {
        -2 => 5,
        -1 => 4,
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        _ => 3,
    };
    let bits = r.peek(16) as usize;
    let mut e = t.coeff_token[cls][bits >> 8];
    let mut len = e.len as u32;
    if e.len == ESCAPE {
        e = t.coeff_token2[e.a as usize][bits & 0xff];
        len = 8 + e.len as u32;
    }
    if e.len == 0 {
        return Err(Error::bitstream("CAVLC: invalid coeff_token"));
    }
    r.skip(len);
    Ok((e.a as usize, e.b as usize))
}

/// `residual_block_cavlc` (7.3.5.3.2 / 9.2). Writes the nonzero levels into
/// `out[scan[i]]` for scan positions `i` in `start_idx..=end_idx` — `out` is
/// the block in raster order and must be zero where nothing is written (the
/// macroblock layer's reset guarantees it). Returns `TotalCoeff`.
/// `dq` is the `(table, shift)` scaling applied to each level as it is
/// written (see [`super::mb::MbDequant`]), or `None` for the DC blocks and
/// lossless macroblocks (levels as parsed).
#[allow(clippy::too_many_arguments)]
fn residual_block(
    r: &mut BitReader,
    t: &VlcTables,
    nc: i32,
    out: &mut [i32],
    scan: &[u8],
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
    dq: Option<(&[i32], u32)>,
) -> Result<usize> {
    // Work on a local copy of the reader so its fields live in registers
    // through the many small reads below, then hand the position back.
    let mut rd = r.clone();
    let res = match dq {
        Some((table, shift)) => residual_block_inner::<true>(&mut rd, t, nc, out, scan, start_idx, end_idx, max_num_coeff, table, shift),
        None => residual_block_inner::<false>(&mut rd, t, nc, out, scan, start_idx, end_idx, max_num_coeff, &[], 0),
    };
    *r = rd;
    res
}

/// The magnitude at which suffixLength grows, per current suffixLength
/// (Table 9-9's thresholds 3, 6, 12, 24, 48). Shared by the reader and the
/// writer: the two have to grow it in lockstep, or every level after the
/// first disagreement is read against the wrong suffix width.
const SUFFIX_LIMIT: [i32; 7] = [0, 3, 6, 12, 24, 48, i32::MAX];

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn residual_block_inner<const DQ: bool>(
    r: &mut BitReader,
    t: &VlcTables,
    nc: i32,
    out: &mut [i32],
    scan: &[u8],
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
    dq_table: &[i32],
    dq_shift: u32,
) -> Result<usize> {
    // Write one level to its raster position, scaled when the block is.
    let put = |out: &mut [i32], idx: usize, level: i32| {
        out[idx] = if DQ { super::mb::dequant_level(level, dq_table[idx], dq_shift) } else { level };
    };
    let (total_coeff, trailing_ones) = read_coeff_token(r, t, nc)?;
    if total_coeff > end_idx - start_idx + 1 {
        return Err(Error::bitstream("CAVLC: TotalCoeff larger than the block"));
    }
    if total_coeff == 0 {
        return Ok(0);
    }
    // Levels, highest frequency first (9.2.2): the trailing ones are sign
    // bits; the first coded level has suffixLength 0 (or 1 for a block
    // with many coefficients and few trailing ones), and its magnitude is
    // raised by one when fewer than three trailing ones preceded it;
    // every later level has suffixLength >= 1, growing with the magnitudes.
    let mut level_val = [0i32; 16];
    let mut i = 0;
    if trailing_ones > 0 {
        let signs = r.bits(trailing_ones as u32);
        for k in 0..trailing_ones {
            level_val[k] = 1 - 2 * ((signs >> (trailing_ones - 1 - k)) & 1) as i32;
        }
        i = trailing_ones;
    }
    // level_prefix: leading zeros before a one.
    #[inline(always)]
    fn level_prefix(r: &mut BitReader) -> Result<u32> {
        let p = r.peek(32).leading_zeros();
        if p >= 32 {
            return Err(Error::bitstream("CAVLC: level_prefix runaway"));
        }
        r.skip(p + 1);
        Ok(p)
    }
    // Sign from the parity of level_code: even is positive.
    #[inline(always)]
    fn signed(level_code: i32) -> i32 {
        let mask = -(level_code & 1);
        (((2 + level_code) >> 1) ^ mask) - mask
    }
    let mut suffix_length: usize;
    if i < total_coeff {
        // The first level.
        let prefix = level_prefix(r)?;
        let sl0 = (total_coeff > 10 && trailing_ones < 3) as u32;
        let mut level_code: i32 = if prefix < 14 {
            ((prefix << sl0) + if sl0 > 0 { r.bits(sl0) } else { 0 }) as i32
        } else if prefix == 14 {
            if sl0 == 0 { 14 + r.bits(4) as i32 } else { (14 << sl0) as i32 + r.bits(sl0) as i32 }
        } else {
            // prefix >= 15: a (prefix - 3)-bit suffix, escape from 15 (and
            // 15 more when suffixLength is 0): 30 either way.
            let mut lc = 30i32;
            if prefix >= 16 {
                lc += (1i32 << (prefix - 3)) - 4096;
            }
            lc + r.bits(prefix - 3) as i32
        };
        if trailing_ones < 3 {
            level_code += 2;
        }
        let v = signed(level_code);
        level_val[i] = v;
        i += 1;
        suffix_length = if sl0 == 0 {
            // suffixLength 0 -> 1, then the usual growth (magnitude > 3 -> 2).
            if v.unsigned_abs() > 3 { 2 } else { 1 }
        } else {
            1 + (v.unsigned_abs() > 3) as usize
        };
        // The remaining levels: suffixLength >= 1.
        while i < total_coeff {
            let prefix = level_prefix(r)?;
            let level_code: i32 = if prefix < 15 {
                ((prefix << suffix_length) + r.bits(suffix_length as u32)) as i32
            } else {
                let mut lc = (15 << suffix_length) as i32;
                if prefix >= 16 {
                    lc += (1i32 << (prefix - 3)) - 4096;
                }
                lc + r.bits(prefix - 3) as i32
            };
            let v = signed(level_code);
            level_val[i] = v;
            i += 1;
            if v.unsigned_abs() as i32 > SUFFIX_LIMIT[suffix_length] {
                suffix_length += 1;
            }
        }
    }

    // total_zeros, then the runs before each coefficient (highest frequency
    // first), writing each coefficient as its position becomes known.
    let mut zeros_left: usize = 0;
    if total_coeff < end_idx - start_idx + 1 {
        let e = if max_num_coeff == 4 {
            t.chroma_dc_total_zeros[total_coeff - 1][r.peek(9) as usize]
        } else if max_num_coeff == 8 {
            t.chroma422_dc_total_zeros[total_coeff - 1][r.peek(9) as usize]
        } else {
            t.total_zeros[total_coeff - 1][r.peek(9) as usize]
        };
        if e.len == 0 {
            return Err(Error::bitstream("CAVLC: invalid total_zeros"));
        }
        r.skip(e.len as u32);
        zeros_left = e.a as usize;
    }
    // Scan position (from start_idx) of the highest-frequency coefficient.
    let mut coeff_num = zeros_left + total_coeff - 1;
    if start_idx + coeff_num > end_idx {
        return Err(Error::bitstream("CAVLC: coefficient position past the block"));
    }
    put(out, scan[start_idx + coeff_num] as usize, level_val[0]);
    let mut i = 1;
    while i < total_coeff && zeros_left > 0 {
        let run = if zeros_left > 6 {
            // Table 9-10, zerosLeft > 6: three-bit codes 111..001 for
            // runs 0..6, then a run of `k` zeros and a one for run 4 + k.
            let bits = r.peek(11);
            let top3 = bits >> 8;
            if top3 != 0 {
                r.skip(3);
                (7 - top3) as usize
            } else {
                let lz = (bits << 21).leading_zeros(); // within the 11 peeked bits
                if lz >= 11 {
                    return Err(Error::bitstream("CAVLC: invalid run_before"));
                }
                r.skip(lz + 1);
                lz as usize + 4
            }
        } else {
            let e = t.run_before[zeros_left - 1][r.peek(3) as usize];
            if e.len == 0 {
                return Err(Error::bitstream("CAVLC: invalid run_before"));
            }
            r.skip(e.len as u32);
            e.a as usize
        };
        if run > zeros_left {
            return Err(Error::bitstream("CAVLC: run_before exceeds zerosLeft"));
        }
        zeros_left -= run;
        coeff_num -= 1 + run;
        put(out, scan[start_idx + coeff_num] as usize, level_val[i]);
        i += 1;
    }
    // No zeros left: the rest are consecutive.
    while i < total_coeff {
        coeff_num -= 1;
        put(out, scan[start_idx + coeff_num] as usize, level_val[i]);
        i += 1;
    }
    if r.overrun() {
        return Err(Error::bitstream("CAVLC: slice data truncated"));
    }
    Ok(total_coeff)
}

// ---------------------------------------------------------------------------
// nC derivation (9.2.1)
// ---------------------------------------------------------------------------

/// nC for the 4x4 block at raster `(bx, by)` of colour plane `p` (luma, or
/// Cb / Cr in 4:4:4, whose neighbours are the same plane's blocks): the
/// rounded mean of the left and above blocks' TotalCoeff (9.2.1), from
/// this macroblock's blocks decoded so far or the neighbours gathered in
/// [`MbNeighbours::gather_nz`] (skip 0, I_PCM 16, unavailable left out).
#[inline]
pub fn plane_nc(layer: &MbLayer, nb: &MbNeighbours, p: usize, bx: usize, by: usize) -> i32 {
    let a = if bx > 0 {
        Some(layer.nz[p][by * 4 + bx - 1])
    } else if nb.nz_avail[0] {
        Some(nb.nz_left[p][by])
    } else {
        None
    };
    let b = if by > 0 {
        Some(layer.nz[p][(by - 1) * 4 + bx])
    } else if nb.nz_avail[1] {
        Some(nb.nz_top[p][bx])
    } else {
        None
    };
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

/// nC for chroma AC block `(bx, by)` (a two-column, `rows`-row grid: 4:2:0
/// has two rows, 4:2:2 four) of component `comp` — as [`plane_nc`], on the
/// chroma blocks (6.4.11.5).
#[inline]
pub fn chroma_nc(layer: &MbLayer, nb: &MbNeighbours, comp: usize, bx: usize, by: usize) -> i32 {
    let a = if bx > 0 {
        Some(layer.chroma_nz[comp][by * 2 + bx - 1])
    } else if nb.nz_avail[0] {
        Some(nb.nzc_left[comp][by])
    } else {
        None
    };
    let b = if by > 0 {
        Some(layer.chroma_nz[comp][(by - 1) * 2 + bx])
    } else if nb.nz_avail[1] {
        Some(nb.nzc_top[comp][bx])
    } else {
        None
    };
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

// ---------------------------------------------------------------------------
// Intra prediction mode derivation (8.3.1.1 / 8.3.2.1), shared with CABAC
// ---------------------------------------------------------------------------

/// The predicted Intra4x4/8x8 mode for the 4x4 block at raster `(bx, by)`
/// (for 8x8 blocks, its top-left 4x4), from neighbours A and B.
pub fn predicted_intra_mode(
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    ctx: &SliceCtx,
    bx: usize,
    by: usize,
    is_8x8: bool,
) -> u8 {
    let mode_of = |addr: usize, blk: usize, is_a: bool| -> Option<u8> {
        if addr == nb.addr {
            return Some(layer.intra_modes[blk]);
        }
        let m = &info.mbs[addr];
        if !m.kind.is_intra() && ctx.constrained_intra_pred {
            // dcPredModePredictedFlag: inter neighbour under constrained
            // intra prediction -> DC.
            return None;
        }
        if m.kind == MbKind::Si && ctx.constrained_intra_pred && layer.kind != MbKind::Si {
            // ... and an SI neighbour of a macroblock that is not SI.
            return None;
        }
        match m.kind {
            MbKind::I4x4 | MbKind::Si => {
                if is_8x8 {
                    // 8.3.2.1: an I4x4 neighbour of an 8x8 block contributes
                    // the mode of the sub-block adjacent to the current
                    // block: n = 1 (A) or 2 (B) within the neighbouring 8x8
                    // — except in an MBAFF frame for the left neighbour of
                    // 8x8 block 2 of a frame macroblock when that neighbour
                    // is a field macroblock: n = 3.
                    let bx8 = (blk % 4) / 2 * 2;
                    let by8 = (blk / 4) / 2 * 2;
                    let cur_blk8 = (by / 2) * 2 + bx / 2;
                    let n3 = is_a && nb.mbaff && !nb.cur_field && m.field && cur_blk8 == 2;
                    let (sx, sy) = if n3 {
                        (bx8 + 1, by8 + 1)
                    } else if is_a {
                        (bx8 + 1, by8)
                    } else {
                        (bx8, by8 + 1)
                    };
                    Some(info.intra_modes[addr * 16 + sy * 4 + sx])
                } else {
                    Some(info.intra_modes[addr * 16 + blk])
                }
            }
            MbKind::I8x8 => Some(info.intra_modes[addr * 16 + blk]),
            _ => Some(2),
        }
    };
    let a = nb.block(bx as i32 - 1, by as i32);
    let b = nb.block(bx as i32, by as i32 - 1);
    let (Some((aa, ab)), Some((ba, bb))) = (a, b) else {
        return 2;
    };
    let ma = mode_of(aa, ab, true);
    let mb = mode_of(ba, bb, false);
    match (ma, mb) {
        (Some(x), Some(y)) => x.min(y),
        _ => 2,
    }
}

// ---------------------------------------------------------------------------
// The CAVLC macroblock layer
// ---------------------------------------------------------------------------

/// Map `mb_type` (ue) of an I slice / the intra part of P and B slices.
pub fn intra_mb_type(t: u32, layer: &mut MbLayer) -> Result<()> {
    match t {
        0 => layer.kind = MbKind::I4x4,
        1..=24 => {
            layer.kind = MbKind::I16x16;
            let t = t - 1;
            layer.intra16_mode = (t % 4) as u8;
            let chroma = ((t / 4) % 3) as u8;
            let luma = if t >= 12 { 15 } else { 0 };
            layer.cbp = luma | (chroma << 4);
        }
        25 => layer.kind = MbKind::IPcm,
        _ => {
            return Err(Error::bitstream(format!(
                "mb_type {t} out of range for an intra macroblock"
            )));
        }
    }
    Ok(())
}

/// Map a P-slice `mb_type` (0..=4) to kind and directions.
pub fn p_mb_type(t: u32, layer: &mut MbLayer) -> Result<bool> {
    // Returns whether the mb is P_8x8ref0.
    match t {
        0 => {
            layer.kind = MbKind::Inter16x16;
            layer.pred_dir = [PRED_L0; 4];
        }
        1 => {
            layer.kind = MbKind::Inter16x8;
            layer.pred_dir = [PRED_L0; 4];
        }
        2 => {
            layer.kind = MbKind::Inter8x16;
            layer.pred_dir = [PRED_L0; 4];
        }
        3 | 4 => {
            layer.kind = MbKind::Inter8x8;
            layer.pred_dir = [PRED_L0; 4];
            return Ok(t == 4);
        }
        _ => return Err(Error::bitstream("P mb_type out of range")),
    }
    Ok(false)
}

/// B-slice `mb_type` 0..=22 → kind and per-partition directions (Table 7-14).
pub fn b_mb_type(t: u32, layer: &mut MbLayer) -> Result<()> {
    const B16X16: [u8; 3] = [PRED_L0, PRED_L1, PRED_BI];
    // Table rows 4..=21: (part0 dir, part1 dir, is16x8) for t-4.
    const B2: [(u8, u8); 9] = [
        (PRED_L0, PRED_L0),
        (PRED_L1, PRED_L1),
        (PRED_L0, PRED_L1),
        (PRED_L1, PRED_L0),
        (PRED_L0, PRED_BI),
        (PRED_L1, PRED_BI),
        (PRED_BI, PRED_L0),
        (PRED_BI, PRED_L1),
        (PRED_BI, PRED_BI),
    ];
    match t {
        0 => layer.kind = MbKind::BDirect16x16,
        1..=3 => {
            layer.kind = MbKind::Inter16x16;
            layer.pred_dir = [B16X16[(t - 1) as usize]; 4];
        }
        4..=21 => {
            let i = ((t - 4) / 2) as usize;
            let (d0, d1) = B2[i];
            if (t - 4) % 2 == 0 {
                layer.kind = MbKind::Inter16x8;
                layer.pred_dir = [d0, d0, d1, d1];
            } else {
                layer.kind = MbKind::Inter8x16;
                layer.pred_dir = [d0, d1, d0, d1];
            }
        }
        22 => layer.kind = MbKind::Inter8x8,
        _ => return Err(Error::bitstream("B mb_type out of range")),
    }
    Ok(())
}

/// P `sub_mb_type` → shape (all L0).
pub fn p_sub_mb_type(t: u32) -> Result<(SubMbShape, u8)> {
    Ok(match t {
        0 => (SubMbShape::S8x8, PRED_L0),
        1 => (SubMbShape::S8x4, PRED_L0),
        2 => (SubMbShape::S4x8, PRED_L0),
        3 => (SubMbShape::S4x4, PRED_L0),
        _ => return Err(Error::bitstream("P sub_mb_type out of range")),
    })
}

/// B `sub_mb_type` → shape and direction (Table 7-18).
pub fn b_sub_mb_type(t: u32) -> Result<(SubMbShape, u8)> {
    Ok(match t {
        0 => (SubMbShape::Direct, PRED_BI),
        1 => (SubMbShape::S8x8, PRED_L0),
        2 => (SubMbShape::S8x8, PRED_L1),
        3 => (SubMbShape::S8x8, PRED_BI),
        4 => (SubMbShape::S8x4, PRED_L0),
        5 => (SubMbShape::S4x8, PRED_L0),
        6 => (SubMbShape::S8x4, PRED_L1),
        7 => (SubMbShape::S4x8, PRED_L1),
        8 => (SubMbShape::S8x4, PRED_BI),
        9 => (SubMbShape::S4x8, PRED_BI),
        10 => (SubMbShape::S4x4, PRED_L0),
        11 => (SubMbShape::S4x4, PRED_L1),
        12 => (SubMbShape::S4x4, PRED_BI),
        _ => return Err(Error::bitstream("B sub_mb_type out of range")),
    })
}

/// The 4x4 blocks (raster) covered by 8x8 partition `part` /
/// sub-partition `sub` of `shape`, as `(x, y, w, h)` in samples.
pub fn sub_partition_rect(
    part: usize,
    shape: SubMbShape,
    sub: usize,
) -> (usize, usize, usize, usize) {
    let px = (part & 1) * 8;
    let py = (part >> 1) * 8;
    match shape {
        SubMbShape::S8x8 | SubMbShape::Direct => (px, py, 8, 8),
        SubMbShape::S8x4 => (px, py + sub * 4, 8, 4),
        SubMbShape::S4x8 => (px + sub * 4, py, 4, 8),
        SubMbShape::S4x4 => (px + (sub & 1) * 4, py + (sub >> 1) * 4, 4, 4),
    }
}

/// The partitions of a macroblock kind as `(x, y, w, h)` rectangles.
pub fn mb_partitions(kind: MbKind) -> &'static [(usize, usize, usize, usize)] {
    match kind {
        MbKind::Inter16x8 => &[(0, 0, 16, 8), (0, 8, 16, 8)],
        MbKind::Inter8x16 => &[(0, 0, 8, 16), (8, 0, 8, 16)],
        MbKind::Inter8x8 => &[(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
        _ => &[(0, 0, 16, 16)],
    }
}

/// Which 8x8 partition a partition rectangle belongs to (its top-left).
#[inline]
pub fn part_index_of(x: usize, y: usize) -> usize {
    (y / 8) * 2 + x / 8
}

/// Read `mb_type` and the prediction syntax; then `coded_block_pattern`,
/// `transform_size_8x8_flag`, `mb_qp_delta` and the residual. `mb_type_raw`
/// is the already-read `mb_type` (the caller reads it because in P/B slices
/// it follows the skip run).
pub fn parse_mb_cavlc(
    r: &mut BitReader,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    mb_type_raw: u32,
    layer: &mut MbLayer,
    dq: &super::transform::Dequant,
    qps: &mut super::recon::QpState,
) -> Result<()> {
    if super::mb::syntax_trace() {
        eprintln!(
            "cavlc mbstart addr={} pos={} type={}",
            nb.addr,
            r.position(),
            mb_type_raw
        );
    }
    layer.reset(MbKind::I4x4, false);
    let mut p8x8ref0 = false;
    match ctx.slice_type {
        SliceType::I => intra_mb_type(mb_type_raw, layer)?,
        SliceType::Si => {
            // Table 7-12: mb_type 0 is SI, the rest are the I types shifted.
            if mb_type_raw == 0 {
                layer.kind = MbKind::Si;
            } else {
                intra_mb_type(mb_type_raw - 1, layer)?;
            }
        }
        SliceType::P | SliceType::Sp => {
            if mb_type_raw < 5 {
                p8x8ref0 = p_mb_type(mb_type_raw, layer)?;
            } else {
                intra_mb_type(mb_type_raw - 5, layer)?;
            }
        }
        SliceType::B => {
            if mb_type_raw < 23 {
                b_mb_type(mb_type_raw, layer)?;
            } else {
                intra_mb_type(mb_type_raw - 23, layer)?;
            }
        }
    }

    if layer.kind == MbKind::IPcm {
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
        return Ok(());
    }

    let mut no_sub_mb_part_less_than_8x8 = true;
    if layer.kind == MbKind::Inter8x8 {
        // sub_mb_pred()
        for part in 0..4 {
            let t = r.ue();
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
                layer.ref_idx[list][part] = if p8x8ref0 || n <= 1 {
                    0
                } else {
                    read_ref_idx(r, n)?
                };
            }
        }
        for list in 0..2 {
            for part in 0..4 {
                let shape = layer.sub_shape[part];
                if shape == SubMbShape::Direct || layer.pred_dir[part] & (1 << list) == 0 {
                    continue;
                }
                for sub in 0..shape.count() {
                    let (x, y, _, _) = sub_partition_rect(part, shape, sub);
                    let mvd = Mv::new(r.se() as i16, r.se() as i16);
                    layer.mvd[(y / 4) * 4 + x / 4].mvd[list] = mvd;
                }
            }
        }
    } else {
        if ctx.transform_8x8_mode && layer.kind == MbKind::I4x4 {
            layer.transform_8x8 = r.flag();
            if layer.transform_8x8 {
                layer.kind = MbKind::I8x8;
            }
        }
        // mb_pred()
        match layer.kind {
            MbKind::I4x4 | MbKind::Si => {
                for blk in 0..16 {
                    let raster = super::mb::raster_of_blk(blk);
                    let (bx, by) = (raster % 4, raster / 4);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, false);
                    let mode = if r.flag() {
                        pred
                    } else {
                        let rem = r.bits(3) as u8;
                        if rem < pred { rem } else { rem + 1 }
                    };
                    if super::mb::syntax_trace() {
                        eprintln!(
                            "cavlc ipm mb={} blk={} raster={} pred={} mode={} @{}",
                            nb.addr,
                            blk,
                            raster,
                            pred,
                            mode,
                            r.position()
                        );
                    }
                    layer.intra_modes[raster] = mode;
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::I8x8 => {
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, true);
                    let mode = if r.flag() {
                        pred
                    } else {
                        let rem = r.bits(3) as u8;
                        if rem < pred { rem } else { rem + 1 }
                    };
                    for dy in 0..2 {
                        for dx in 0..2 {
                            layer.intra_modes[(by + dy) * 4 + bx + dx] = mode;
                        }
                    }
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::I16x16 => {
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::BDirect16x16 => {}
            _ => {
                let parts = mb_partitions(layer.kind);
                for list in 0..2 {
                    for &(x, y, _, _) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let n = ctx.num_ref_idx[list]
                            * if layer.field && !ctx.field_pic { 2 } else { 1 };
                        let ri = if n <= 1 { 0 } else { read_ref_idx(r, n)? };
                        // The reference index applies to the whole partition.
                        for &(px, py, pw, ph) in parts.iter().filter(|p| p.0 == x && p.1 == y) {
                            for by in py / 8..(py + ph) / 8 {
                                for bx in px / 8..(px + pw) / 8 {
                                    layer.ref_idx[list][by * 2 + bx] = ri;
                                }
                            }
                        }
                    }
                }
                for list in 0..2 {
                    for &(x, y, _, _) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let mvd = Mv::new(r.se() as i16, r.se() as i16);
                        layer.mvd[(y / 4) * 4 + x / 4].mvd[list] = mvd;
                    }
                }
            }
        }
    }

    if layer.kind != MbKind::I16x16 {
        let code = r.ue();
        if code > 47 {
            return Err(Error::bitstream("coded_block_pattern out of range"));
        }
        let intra = layer.kind.is_intra();
        layer.cbp = if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
            if intra {
                GOLOMB_TO_INTRA4X4_CBP[code as usize]
            } else {
                GOLOMB_TO_INTER_CBP[code as usize]
            }
        } else {
            if code > 15 {
                return Err(Error::bitstream("coded_block_pattern out of range"));
            }
            if intra {
                GOLOMB_TO_INTRA4X4_CBP_GRAY[code as usize]
            } else {
                GOLOMB_TO_INTER_CBP_GRAY[code as usize]
            }
        };
        if layer.cbp & 15 != 0
            && ctx.transform_8x8_mode
            && !layer.kind.is_intra()
            && no_sub_mb_part_less_than_8x8
            && (layer.kind != MbKind::BDirect16x16 || ctx.direct_8x8_inference)
        {
            layer.transform_8x8 = r.flag();
        }
    }

    if layer.has_residual() {
        layer.qp_delta = r.se();
        if !(-26..=25).contains(&layer.qp_delta) {
            return Err(Error::bitstream("mb_qp_delta out of range"));
        }
        layer.qp = super::mb::next_qp(qps.prev_qp, layer.qp_delta, ctx.bit_depth);
        qps.prev_qp = layer.qp;
        let mbdq = MbDequant::for_mb(dq, ctx, qps.chroma_offset, layer.kind, layer.qp);
        parse_residual_cavlc(r, ctx, nb, layer, mbdq.as_ref())?;
    } else {
        layer.qp = qps.prev_qp;
    }
    if r.overrun() {
        return Err(Error::bitstream("slice data truncated in macroblock"));
    }
    Ok(())
}

fn read_chroma_mode(r: &mut BitReader) -> Result<u8> {
    let m = r.ue();
    if m > 3 {
        return Err(Error::bitstream("intra_chroma_pred_mode out of range"));
    }
    Ok(m as u8)
}

fn read_ref_idx(r: &mut BitReader, num_ref_idx: u32) -> Result<i8> {
    let v = r.te(num_ref_idx - 1);
    if v >= num_ref_idx {
        return Err(Error::bitstream("ref_idx out of range"));
    }
    Ok(v as i8)
}

/// `residual()` for CAVLC (7.3.5.3), filling the layer's coefficient
/// arrays (raster order) and nonzero counts.
/// For CAVLC 8x8 blocks (four interleaved 4x4 scans): where scan position
/// `i` of sub-block `sub` lands in the 8x8 raster.
static SCAN8_SUB_FIELD: [[u8; 16]; 4] = {
    let mut t = [[0u8; 16]; 4];
    let mut sub = 0;
    while sub < 4 {
        let mut i = 0;
        while i < 16 {
            t[sub][i] = FIELD_SCAN8X8[4 * i + sub];
            i += 1;
        }
        sub += 1;
    }
    t
};

/// See [`SCAN8_SUB_FIELD`]: the frame scan. Crate-visible because the
/// encoder's 8x8 residual writer hands the reader's own sub-scans to
/// [`write_residual_block_cavlc`] — one definition, both directions.
pub(crate) static SCAN8_SUB: [[u8; 16]; 4] = {
    let mut t = [[0u8; 16]; 4];
    let mut sub = 0;
    while sub < 4 {
        let mut i = 0;
        while i < 16 {
            t[sub][i] = ZIGZAG8X8[4 * i + sub];
            i += 1;
        }
        sub += 1;
    }
    t
};

/// Chroma DC scan (identity over the 2x2). Crate-visible because the
/// encoder's macroblock writer hands the same scan to the residual writer
/// below — one definition, both directions.
pub(crate) static SCAN_CHROMA_DC: [u8; 4] = [0, 1, 2, 3];

/// How many nonzero levels each of an 8x8 block's four CAVLC sub-blocks
/// carries, given the block in 8x8 raster order.
///
/// These are the counts the reader stores in `layer.nz` as it decodes the
/// four interleaved blocks above, and therefore the counts the *next*
/// block's `nC` predictor reads (9.2.1). They live here rather than in
/// the encoder because they are a property of [`SCAN8_SUB`]: the four
/// sub-scans are not the 8x8's four spatial quadrants but its scan
/// positions taken every fourth, and counting them anywhere else would be
/// a second, silently divergent, spelling of that.
///
/// Frame scan only — a field macroblock interleaves [`SCAN8_SUB_FIELD`]
/// instead, and the encoder that calls this codes frames.
pub(crate) fn sub_block_counts_8x8(levels: &[i16]) -> [u8; 4] {
    debug_assert_eq!(levels.len(), 64, "an 8x8 block has sixty-four coefficients");
    let mut n = [0u8; 4];
    for (sub, count) in n.iter_mut().enumerate() {
        *count = SCAN8_SUB[sub].iter().filter(|&&pos| levels[pos as usize] != 0).count() as u8;
    }
    n
}

/// `residual_luma()` (7.3.5.3.1) for colour plane `p`: the luma plane, or
/// Cb / Cr in 4:4:4, which are coded exactly like it.
fn parse_residual_luma_like(
    r: &mut BitReader,
    ctx: &SliceCtx,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
    p: usize,
    dq: Option<&MbDequant>,
) -> Result<()> {
    let t = tables();
    let dq4: Option<(&[i32], u32)> = dq.map(|d| (&d.q4[p].0[..], d.q4[p].1));
    let dq8: Option<(&[i32], u32)> = dq.map(|d| (&d.q8[p].0[..], d.q8[p].1));
    let trace = super::mb::syntax_trace();
    // Field pictures (and field macroblocks) use the field scans.
    let (scan4, scan8sub): (&[u8; 16], &[[u8; 16]; 4]) = if ctx.field_pic || layer.field {
        (&FIELD_SCAN4X4, &SCAN8_SUB_FIELD)
    } else {
        (&ZIGZAG4X4, &SCAN8_SUB)
    };
    if layer.kind == MbKind::I16x16 {
        let nc = plane_nc(layer, nb, p, 0, 0);
        residual_block(r, t, nc, &mut layer.dc[p], scan4, 0, 15, 16, None)?;
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if layer.cbp & (1 << blk8) == 0 {
            continue;
        }
        if !layer.transform_8x8 {
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                let nc = plane_nc(layer, nb, p, bx, by);
                let base = raster * 16;
                let pos0 = r.position();
                let n = if layer.kind == MbKind::I16x16 {
                    residual_block(r, t, nc, &mut layer.coef[p][base..base + 16], scan4, 1, 15, 15, dq4)?
                } else {
                    residual_block(r, t, nc, &mut layer.coef[p][base..base + 16], scan4, 0, 15, 16, dq4)?
                };
                if trace {
                    eprintln!(
                        "cavlc blk raster={raster} nc={nc} @{pos0} -> n={n} end={}",
                        r.position()
                    );
                }
                layer.nz[p][raster] = n as u8;
            }
        } else {
            // 8x8 transform with CAVLC: four interleaved 4x4 blocks.
            let base = blk8 * 64;
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                let nc = plane_nc(layer, nb, p, bx, by);
                let n = residual_block(
                    r,
                    t,
                    nc,
                    &mut layer.coef[p][base..base + 64],
                    &scan8sub[sub],
                    0,
                    15,
                    16,
                    dq8,
                )?;
                layer.nz[p][raster] = n as u8;
            }
        }
    }
    Ok(())
}

fn parse_residual_cavlc(
    r: &mut BitReader,
    ctx: &SliceCtx,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
    dq: Option<&MbDequant>,
) -> Result<()> {
    let t = tables();
    // Luma, then (4:4:4) Cb and Cr coded the same way.
    parse_residual_luma_like(r, ctx, nb, layer, 0, dq)?;
    if ctx.chroma_format_idc == 3 {
        parse_residual_luma_like(r, ctx, nb, layer, 1, dq)?;
        parse_residual_luma_like(r, ctx, nb, layer, 2, dq)?;
    }
    let scan4: &[u8; 16] = if ctx.field_pic || layer.field {
        &FIELD_SCAN4X4
    } else {
        &ZIGZAG4X4
    };
    // Chroma (4:2:0: 2x2 blocks per component and 4 DC coefficients; 4:2:2:
    // 2x4 blocks and 8 DC coefficients).
    if (ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2) && layer.cbp & 0x30 != 0 {
        let c422 = ctx.chroma_format_idc == 2;
        let (n_dc, rows) = if c422 { (8usize, 4usize) } else { (4, 2) };
        for comp in 0..2 {
            if c422 {
                residual_block(
                    r,
                    t,
                    -2,
                    &mut layer.chroma_dc[comp],
                    &SCAN_CHROMA_DC_422,
                    0,
                    7,
                    8,
                    None,
                )?;
            } else {
                residual_block(
                    r,
                    t,
                    -1,
                    &mut layer.chroma_dc[comp][..4],
                    &SCAN_CHROMA_DC,
                    0,
                    3,
                    4,
                    None,
                )?;
            }
        }
        let _ = n_dc;
        if layer.cbp & 0x20 != 0 {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    let (bx, by) = (blk & 1, blk >> 1);
                    let nc = chroma_nc(layer, nb, comp, bx, by);
                    let n =
                        residual_block(r, t, nc, &mut layer.chroma_ac[comp][blk], scan4, 1, 15, 15, dq.map(|d| (&d.q4[1 + comp].0[..], d.q4[1 + comp].1)))?;
                    layer.chroma_nz[comp][blk] = n as u8;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writing (9.2, the inverse of `residual_block`)
// ---------------------------------------------------------------------------

/// Write one residual block in CAVLC: the inverse of [`residual_block`].
///
/// `levels` is the block in raster order — the layout the reader decodes
/// *into* — and `scan` maps scan position to it the same way. `nc` is the
/// same predictor the reader is given. Returns `TotalCoeff`, which the caller
/// needs for the next block's `nc`.
///
/// It sits beside the reader for the reason the CABAC one does: the two are
/// inverses over a pile of small rules, and a change made to one and not the
/// other is a desync that stays invisible until a later block decodes as
/// rubbish.
pub(crate) fn write_residual_block_cavlc(
    w: &mut BitWriter,
    nc: i32,
    levels: &[i32],
    scan: &[u8],
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
) -> usize {
    let span = end_idx - start_idx + 1;
    // The nonzero coefficients in scan order, lowest frequency first.
    let mut pos = [0usize; 16];
    let mut val = [0i32; 16];
    let mut total_coeff = 0usize;
    for k in 0..span {
        let v = levels[scan[start_idx + k] as usize];
        if v != 0 {
            pos[total_coeff] = k;
            val[total_coeff] = v;
            total_coeff += 1;
        }
    }
    debug_assert!(total_coeff <= 16, "more coefficients than a block can hold");

    // TrailingOnes: magnitude-one coefficients at the high-frequency end,
    // at most three. They are coded as bare sign bits.
    let mut t1 = 0usize;
    while t1 < 3 && t1 < total_coeff && val[total_coeff - 1 - t1].abs() == 1 {
        t1 += 1;
    }

    let cls = coeff_token_class(nc);
    let (len, code) = match cls {
        4 => (
            CHROMA_DC_COEFF_TOKEN_LEN[total_coeff][t1],
            CHROMA_DC_COEFF_TOKEN_BITS[total_coeff][t1] as u32,
        ),
        5 => (
            CHROMA422_DC_COEFF_TOKEN_LEN[total_coeff][t1],
            CHROMA422_DC_COEFF_TOKEN_BITS[total_coeff][t1] as u32,
        ),
        _ => (
            COEFF_TOKEN_LEN[cls][total_coeff][t1],
            COEFF_TOKEN_BITS[cls][total_coeff][t1] as u32,
        ),
    };
    debug_assert!(len > 0, "no coeff_token for class {cls} total {total_coeff} t1 {t1}");
    w.bits(len as u32, code);
    if total_coeff == 0 {
        return 0;
    }

    // The trailing ones, highest frequency first: one bit each, set for
    // negative.
    for k in 0..t1 {
        w.bit((val[total_coeff - 1 - k] < 0) as u32);
    }

    // The remaining levels, still highest frequency first. suffixLength
    // adapts as the magnitudes grow, exactly as the reader grows it.
    let mut suffix_length = 0usize;
    for i in t1..total_coeff {
        let v = val[total_coeff - 1 - i];
        // The reader's `signed()` inverted: positives take the even code
        // numbers, negatives the odd.
        let mut level_code = (if v > 0 { 2 * v - 2 } else { -2 * v - 1 }) as u32;
        if i == t1 {
            // The first coded level. Fewer than three trailing ones means a
            // magnitude of one is impossible here — it would have been a
            // trailing one — so the code number is shifted down by two.
            let sl0 = (total_coeff > 10 && t1 < 3) as u32;
            if t1 < 3 {
                debug_assert!(level_code >= 2, "first level of magnitude one with room for a trailing one");
                level_code -= 2;
            }
            if (level_code >> sl0) < 14 {
                write_level_prefix(w, level_code >> sl0);
                if sl0 > 0 {
                    w.bits(sl0, level_code & ((1 << sl0) - 1));
                }
            } else {
                // `level_prefix` 14 is its own escape, with a four-bit
                // suffix when suffixLength is zero and `sl0` bits otherwise.
                let (base, width) = if sl0 == 0 { (14u32, 4u32) } else { (14 << sl0, sl0) };
                if level_code < base + (1 << width) {
                    write_level_prefix(w, 14);
                    w.bits(width, level_code - base);
                } else {
                    write_level_escape(w, level_code, 30);
                }
            }
            suffix_length = if sl0 == 0 {
                if v.unsigned_abs() > 3 { 2 } else { 1 }
            } else {
                1 + (v.unsigned_abs() > 3) as usize
            };
        } else {
            if (level_code >> suffix_length) < 15 {
                write_level_prefix(w, level_code >> suffix_length);
                w.bits(suffix_length as u32, level_code & ((1 << suffix_length) - 1));
            } else {
                write_level_escape(w, level_code, 15 << suffix_length);
            }
            if v.unsigned_abs() as i32 > SUFFIX_LIMIT[suffix_length] {
                suffix_length += 1;
            }
        }
    }

    // total_zeros: how many zeros precede the highest-frequency coefficient.
    // Absent when the block is full, because then there are none.
    let total_zeros = pos[total_coeff - 1] + 1 - total_coeff;
    if total_coeff < span {
        let (len, code) = if max_num_coeff == 4 {
            (
                CHROMA_DC_TOTAL_ZEROS_LEN[total_coeff - 1][total_zeros],
                CHROMA_DC_TOTAL_ZEROS_BITS[total_coeff - 1][total_zeros] as u32,
            )
        } else if max_num_coeff == 8 {
            (
                CHROMA422_DC_TOTAL_ZEROS_LEN[total_coeff - 1][total_zeros],
                CHROMA422_DC_TOTAL_ZEROS_BITS[total_coeff - 1][total_zeros] as u32,
            )
        } else {
            (
                TOTAL_ZEROS_LEN[total_coeff - 1][total_zeros],
                TOTAL_ZEROS_BITS[total_coeff - 1][total_zeros] as u32,
            )
        };
        debug_assert!(len > 0, "no total_zeros code for {total_coeff} coefficients, {total_zeros} zeros");
        w.bits(len as u32, code);
    } else {
        debug_assert_eq!(total_zeros, 0, "a full block cannot have zeros before its last coefficient");
    }

    // run_before, again highest frequency first, and only while zeros remain
    // to distribute: once they are gone the rest are consecutive and the
    // reader stops asking.
    let mut zeros_left = total_zeros;
    let mut i = 1usize;
    while i < total_coeff && zeros_left > 0 {
        let run = pos[total_coeff - i] - pos[total_coeff - 1 - i] - 1;
        write_run_before(w, run, zeros_left);
        zeros_left -= run;
        i += 1;
    }
    total_coeff
}

/// The coeff_token VLC class for a `nC` predictor (9.2.1): two of the six are
/// the chroma DC tables, chosen by the negative sentinels.
#[inline]
fn coeff_token_class(nc: i32) -> usize {
    match nc {
        -2 => 5,
        -1 => 4,
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        _ => 3,
    }
}

/// `level_prefix`: that many zero bits, then a one.
#[inline]
fn write_level_prefix(w: &mut BitWriter, prefix: u32) {
    w.zeros(prefix);
    w.bit(1);
}

/// The `level_prefix` 15-and-above escape. The suffix is twelve bits at
/// fifteen and one bit wider at each higher prefix, and the ranges tile
/// exactly from `base15`, so the shortest prefix that reaches the value is
/// the one the reader will read back.
fn write_level_escape(w: &mut BitWriter, level_code: u32, base15: u32) {
    let mut prefix = 15u32;
    loop {
        let width = prefix - 3;
        let base = if prefix == 15 { base15 } else { base15 + (1 << width) - 4096 };
        if level_code < base + (1 << width) {
            write_level_prefix(w, prefix);
            w.bits(width, level_code - base);
            return;
        }
        prefix += 1;
        assert!(prefix <= 31, "level too large for any level_prefix escape");
    }
}

/// `run_before` (Table 9-10).
///
/// The table's last row is the "zerosLeft > 6" column, so the row index
/// saturates rather than growing — above six zeros the code no longer depends
/// on how many there are. The reader derives that column arithmetically
/// instead, because peeking and counting leading zeros is faster than a
/// lookup when decoding; both spellings produce the same bits, which is what
/// the round trip checks.
fn write_run_before(w: &mut BitWriter, run: usize, zeros_left: usize) {
    debug_assert!(run <= zeros_left, "run_before longer than the zeros left");
    let row = zeros_left.min(7) - 1;
    debug_assert!(
        RUN_BEFORE_LEN[row][run] > 0,
        "no run_before code for run {run} with {zeros_left} left"
    );
    w.bits(RUN_BEFORE_LEN[row][run] as u32, RUN_BEFORE_BITS[row][run] as u32);
}


#[cfg(test)]
mod cavlc_round_trip {
    use super::*;

    /// Write a block, read it back with the production reader, and require
    /// the coefficients and TotalCoeff to match. CAVLC carries no adaptive
    /// state between blocks, so those two are the whole of the state and the
    /// check is complete rather than merely strong.
    fn round_trip(nc: i32, start_idx: usize, end_idx: usize, max_num_coeff: usize, levels: &[i32]) {
        let scan: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let mut w = BitWriter::new();
        let n_w = write_residual_block_cavlc(&mut w, nc, levels, &scan, start_idx, end_idx, max_num_coeff);
        w.rbsp_trailing_bits();
        let data = w.into_rbsp();

        let mut r = BitReader::new(&data);
        let mut out = vec![0i32; 16];
        let n_r = residual_block(
            &mut r, tables(), nc, &mut out, &scan, start_idx, end_idx, max_num_coeff, None,
        )
        .unwrap_or_else(|e| panic!("reader rejected nc={nc} levels={levels:?}: {e}"));

        assert_eq!(n_r, n_w, "nc={nc} TotalCoeff differs for {levels:?}");
        for k in start_idx..=end_idx {
            let idx = scan[k] as usize;
            assert_eq!(out[idx], levels[idx], "nc={nc} coefficient {idx} differs for {levels:?}");
        }
        assert!(!r.overrun(), "nc={nc} reader overran for {levels:?}");
    }

    /// Every nC class against every block shape, over the level patterns that
    /// reach a different rule of the coding.
    #[test]
    fn round_trips_every_class_and_shape() {
        // (nc, start, end, max_num_coeff)
        let shapes: [(i32, usize, usize, usize); 8] = [
            (0, 0, 15, 16),   // class 0
            (2, 0, 15, 16),   // class 1
            (5, 0, 15, 16),   // class 2
            (9, 0, 15, 16),   // class 3
            (0, 1, 15, 15),   // an AC block, which starts at one
            (8, 1, 15, 15),
            (-1, 0, 3, 4),    // chroma DC, 4:2:0
            (-2, 0, 7, 8),    // chroma DC, 4:2:2
        ];
        for (nc, start, end, maxc) in shapes {
            let span = end - start + 1;
            let mut cases: Vec<Vec<i32>> = Vec::new();
            let z = vec![0i32; 16];
            cases.push(z.clone()); // TotalCoeff 0
            // One coefficient, at each end, at every magnitude that changes
            // the shape of the level code — including the escapes.
            for k in [0usize, 1, span / 2, span - 1] {
                if k >= span {
                    continue;
                }
                for a in [1i32, 2, 3, 4, 7, 8, 15, 16, 17, 30, 31, 60, 100, 2000, 4000, 4200, 20000] {
                    for sign in [1i32, -1] {
                        let mut v = z.clone();
                        v[scan_idx(start + k)] = a * sign;
                        cases.push(v);
                    }
                }
            }
            // Trailing ones: zero through four magnitude-one coefficients at
            // the high-frequency end, since only three of them count and the
            // fourth becomes an ordinary level.
            for ones in 0..=4usize {
                if ones + 1 > span {
                    continue;
                }
                let mut v = z.clone();
                v[scan_idx(start)] = 6;
                for j in 0..ones {
                    v[scan_idx(start + span - 1 - j)] = if j % 2 == 0 { 1 } else { -1 };
                }
                cases.push(v);
            }
            // A full block: no total_zeros is written at all.
            let mut full = z.clone();
            for k in 0..span {
                full[scan_idx(start + k)] = if k % 2 == 0 { 1 } else { -2 };
            }
            cases.push(full);
            // More than ten coefficients with fewer than three trailing ones
            // is the one case that starts at suffixLength one rather than
            // zero. It needs a nearly full 16-coefficient block, so it only
            // arises for the luma shapes.
            if span >= 12 {
                let mut v = z.clone();
                for k in 0..12 {
                    v[scan_idx(start + k)] = if k == 11 { 9 } else { (k as i32 % 3) - 1 };
                }
                // Make sure nothing at the top is a one, so t1 stays below
                // three and the high-suffixLength start is really taken.
                v[scan_idx(start + 11)] = 9;
                v[scan_idx(start + 10)] = -4;
                let n = v.iter().filter(|x| **x != 0).count();
                if n > 10 {
                    cases.push(v);
                }
            }
            // Long zero runs, which is what drives run_before past six.
            for gap in [7usize, 8, 10, 13] {
                if gap + 1 >= span {
                    continue;
                }
                let mut v = z.clone();
                v[scan_idx(start)] = 3;
                v[scan_idx(start + gap)] = -5;
                v[scan_idx(start + gap + 1)] = 1;
                cases.push(v);
            }
            // Pseudo-random, mostly zero.
            let mut seed = 0xc0ffee_u32 ^ ((nc as u32) << 13) ^ ((maxc as u32) << 3);
            let mut lcg = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 16
            };
            for _ in 0..400 {
                let mut v = z.clone();
                for k in 0..span {
                    if lcg() % 3 == 0 {
                        let m = 1 + (lcg() % 60) as i32;
                        v[scan_idx(start + k)] = if lcg() % 2 == 0 { m } else { -m };
                    }
                }
                cases.push(v);
            }
            for levels in &cases {
                round_trip(nc, start, end, maxc, levels);
            }
        }
    }

    /// The scan used by the tests is the identity, so a scan position is its
    /// own raster index; naming it keeps the cases readable.
    fn scan_idx(k: usize) -> usize {
        k
    }
}
