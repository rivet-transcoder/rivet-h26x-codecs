//! The CAVLC intra macroblock layer: [`MbDecision`]s into bits.
//!
//! This is the writing half of a coded intra macroblock, and its single
//! rule is that it mirrors `h264::cavlc::parse_mb_cavlc` element for
//! element: the same syntax order, the same `nC` derivation, the same
//! coded-block-pattern mapping. Wherever the reader owns a table, this
//! side derives its inverse from that table rather than writing the
//! numbers down a second time — two copies of Table 9-4 would be two
//! places for one of them to be wrong, and the round-trip test below
//! asserts the derivation rather than trusting it.
//!
//! The state the reader keeps per picture — the nonzero-coefficient
//! counts its `nC` predictor reads from the left and upper neighbours —
//! is kept here too, in `NzState`, and it is fed from what
//! `write_residual_block_cavlc` *returns* rather than from the mode
//! decision's own counts. The two agree (a `debug_assert` says so where
//! the spans line up), but the returned value is what the reader will
//! store, so it is the one that cannot drift.
//!
//! One asymmetry with the reader is worth naming: the reader derives a
//! neighbouring block's intra prediction mode as DC (2) when the
//! neighbouring macroblock is available but not `I_NxN` (8.3.1.1). The
//! caller's `left_modes` / `top_modes` bookkeeping in
//! [`write_intra_picture`] therefore records `Some(2)` for an available
//! `I_16x16` neighbour — `None` is only ever "no macroblock there".
//! Recording `None` instead would predict DC where the reader predicts
//! `min(2, other)`, and the desync would surface as a wrong decoded mode
//! two macroblocks later.

use crate::bitwriter::BitWriter;
use crate::encode::h264_intra::{MbDecision, MbKind};
use crate::encode::h264_me::{InterDecision, InterMbKind};
use crate::encode::h264_pic::{IntraTools, PMb, code_intra_picture, code_p_picture};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::cavlc::{SCAN_CHROMA_DC, write_residual_block_cavlc};
use crate::h264::mb::raster_of_blk;
use crate::h264::tables::{
    GOLOMB_TO_INTER_CBP, GOLOMB_TO_INTER_CBP_GRAY, GOLOMB_TO_INTRA4X4_CBP,
    GOLOMB_TO_INTRA4X4_CBP_GRAY, SCAN_CHROMA_DC_422, ZIGZAG4X4,
};

/// `coded_block_pattern` me(v) for intra: cbp -> codeNum, the inverse of
/// the reader's codeNum -> cbp table, derived from it at compile time.
static INTRA_CBP_TO_GOLOMB: [u8; 48] = {
    let mut inv = [0u8; 48];
    let mut code = 0;
    while code < 48 {
        inv[GOLOMB_TO_INTRA4X4_CBP[code] as usize] = code as u8;
        code += 1;
    }
    inv
};

/// The same for monochrome (and 4:4:4, which this module refuses anyway),
/// whose cbp stops at 15.
static INTRA_CBP_TO_GOLOMB_GRAY: [u8; 16] = {
    let mut inv = [0u8; 16];
    let mut code = 0;
    while code < 16 {
        inv[GOLOMB_TO_INTRA4X4_CBP_GRAY[code] as usize] = code as u8;
        code += 1;
    }
    inv
};

/// `coded_block_pattern` me(v) for inter macroblocks — Table 9-4's other
/// column, inverted from the reader's table like the intra ones above.
static INTER_CBP_TO_GOLOMB: [u8; 48] = {
    let mut inv = [0u8; 48];
    let mut code = 0;
    while code < 48 {
        inv[GOLOMB_TO_INTER_CBP[code] as usize] = code as u8;
        code += 1;
    }
    inv
};

/// See [`INTER_CBP_TO_GOLOMB`]; monochrome.
static INTER_CBP_TO_GOLOMB_GRAY: [u8; 16] = {
    let mut inv = [0u8; 16];
    let mut code = 0;
    while code < 16 {
        inv[GOLOMB_TO_INTER_CBP_GRAY[code] as usize] = code as u8;
        code += 1;
    }
    inv
};

/// The per-picture nonzero-coefficient state `nC` (9.2.1) predicts from:
/// what the reader gathers per macroblock in `MbNeighbours::gather_nz`,
/// kept as one row of counts along the top edge and one column along the
/// left. Updated from the writer's returned `TotalCoeff`s, block by
/// block, exactly when the reader stores them; blocks a cleared cbp bit
/// skips stay zero, which is what the reader's per-macroblock reset
/// leaves behind.
struct NzState {
    /// Per 4x4 luma column of the picture: the count of the bottom block
    /// of the macroblock above (`mbs_wide * 4`).
    top_luma: Vec<u8>,
    /// Per 4x4 luma row: the count of the rightmost block of the
    /// macroblock to the left.
    left_luma: [u8; 4],
    /// The same for the chroma blocks, per component (`mbs_wide * 2`).
    top_chroma: [Vec<u8>; 2],
    /// Chroma left column; `rows` entries are meaningful.
    left_chroma: [[u8; 4]; 2],
    /// Chroma AC block rows: 0 (monochrome), 2 (4:2:0) or 4 (4:2:2).
    rows: usize,
}

impl NzState {
    fn new(mbs_wide: usize, rows: usize) -> Self {
        NzState {
            top_luma: vec![0; mbs_wide * 4],
            left_luma: [0; 4],
            top_chroma: [vec![0; mbs_wide * 2], vec![0; mbs_wide * 2]],
            left_chroma: [[0; 4]; 2],
            rows,
        }
    }
}

/// The rounded mean of 9.2.1: both neighbours average, one is taken as
/// is, none is zero.
fn nc_of(a: Option<u8>, b: Option<u8>) -> i32 {
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

/// A block's levels widened to what the residual writer takes. The reader
/// decodes into `i32`; the decision side stores `i16` because the levels
/// fit and a slice's worth of blocks is measured in kilobytes.
fn widen(levels: &[i16; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (o, &v) in out.iter_mut().zip(levels) {
        *o = v as i32;
    }
    out
}

/// The chroma DC block widened: four meaningful entries in 4:2:0, eight
/// in 4:2:2. Always eight wide — the writer only reads the scan's span.
fn widen_dc(levels: &[i16; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    for (o, &v) in out.iter_mut().zip(levels) {
        *o = v as i32;
    }
    out
}

/// Write one intra macroblock — `mb_type` through the residual — updating
/// `st` with the counts the next macroblocks' `nC` will read. `left` and
/// `top` say whether those neighbouring macroblocks exist.
///
/// `mb_type_offset` is what the slice type adds to an intra `mb_type`
/// before it is coded: 0 in an I slice, 5 in a P slice, 23 in a B slice
/// (Table 7-11's note — the reader subtracts the same constant).
fn write_macroblock(
    w: &mut BitWriter,
    dec: &MbDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
    mb_type_offset: u32,
) {
    let chroma = st.rows != 0;
    let i16x16 = dec.kind == MbKind::I16x16;
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;

    // mb_type (Table 7-11): I_NxN is 0; the I_16x16 types encode the
    // prediction mode and both halves of the coded block pattern.
    match dec.kind {
        MbKind::I4x4 => w.ue(mb_type_offset),
        MbKind::I16x16 => w.ue(
            mb_type_offset
                + 1
                + dec.intra16_mode as u32
                + 4 * dec.cbp_chroma as u32
                + 12 * (dec.cbp_luma == 15) as u32,
        ),
    }

    if dec.kind == MbKind::I4x4 {
        // The sixteen prediction modes, in luma4x4BlkIdx order — the
        // standard's 4x4 scan, not raster, which is why the raster-indexed
        // decision is walked through `raster_of_blk`.
        for blk in 0..16 {
            let p = dec.luma_pred[raster_of_blk(blk)];
            w.flag(p.use_predicted);
            if !p.use_predicted {
                w.bits(3, p.rem as u32);
            }
        }
    }
    if chroma {
        w.ue(dec.chroma_mode as u32);
    }
    if dec.kind == MbKind::I4x4 {
        // coded_block_pattern as me(v), from the derived inverse mapping.
        let code = if chroma {
            INTRA_CBP_TO_GOLOMB[cbp]
        } else {
            INTRA_CBP_TO_GOLOMB_GRAY[cbp]
        };
        w.ue(code as u32);
    }
    // mb_qp_delta is present exactly when the reader's `has_residual` says
    // so: any coded block, or I_16x16, whose DC block is always coded.
    if cbp != 0 || i16x16 {
        w.se(dec.qp_delta as i32);
    }

    write_mb_residual(
        w,
        st,
        mb_x,
        left,
        top,
        i16x16.then_some(&dec.luma_dc),
        cbp,
        &dec.luma,
        &dec.chroma_dc,
        &dec.chroma_ac,
        &dec.nz_luma,
        &dec.nz_chroma,
    );
}

/// Write one P_L0_16x16 macroblock — `mb_type` through the residual. The
/// skip run before it belongs to the caller, which is counting.
///
/// No `ref_idx_l0` is written: with exactly one active reference the
/// element is absent from the stream — the reader's `read_ref_idx` is
/// only reached when `num_ref_idx_active > 1` (7.3.5.1) — and this
/// encoder's slice headers always declare one. The `debug_assert` is the
/// tripwire for the day that stops being true.
fn write_p16_macroblock(
    w: &mut BitWriter,
    dec: &InterDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
) {
    debug_assert_eq!(dec.kind, InterMbKind::P16x16);
    debug_assert_eq!(dec.ref_idx, 0, "more than one reference needs te(v) ref_idx writing");
    w.ue(0); // mb_type: P_L0_16x16 (Table 7-13)
    // mvd_l0 for the single partition, x then y (7.3.5.1).
    w.se(dec.mvd.x as i32);
    w.se(dec.mvd.y as i32);
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;
    let code = if st.rows != 0 {
        INTER_CBP_TO_GOLOMB[cbp]
    } else {
        INTER_CBP_TO_GOLOMB_GRAY[cbp]
    };
    w.ue(code as u32);
    // mb_qp_delta is present exactly when the reader's `has_residual` says
    // so, which for an inter macroblock is any coded block at all.
    if cbp != 0 {
        w.se(dec.qp_delta as i32);
    }
    write_mb_residual(
        w,
        st,
        mb_x,
        left,
        top,
        None,
        cbp,
        &dec.luma,
        &dec.chroma_dc,
        &dec.chroma_ac,
        &dec.nz_luma,
        &dec.nz_chroma,
    );
}

/// A skipped macroblock's mark on the `nC` state: every count zero, which
/// is what the reader's per-macroblock reset leaves for its neighbours to
/// read. Forgetting this — leaving the previous coded macroblock's counts
/// in the left column — desyncs the very next residual block's tables.
fn skip_nz(st: &mut NzState, mb_x: usize) {
    st.left_luma = [0; 4];
    st.top_luma[mb_x * 4..mb_x * 4 + 4].fill(0);
    for comp in 0..2 {
        st.left_chroma[comp] = [0; 4];
        if st.rows != 0 {
            st.top_chroma[comp][mb_x * 2..mb_x * 2 + 2].fill(0);
        }
    }
}

/// The residual and its `nC` bookkeeping, shared by the intra and inter
/// macroblock writers: the syntax from `coded_block_pattern` onwards is
/// the same for both — only whether an Intra_16x16 DC block leads (and
/// shortens the AC spans) differs, and `luma_dc` carries exactly that.
///
/// Mirrors `parse_residual_luma_like` for plane 0 and then the chroma of
/// `parse_residual_cavlc`. `cur` / `curc` are the within-macroblock
/// counts decoded so far, which is what the reader's `nC` reads for a
/// block whose left or top neighbour is in this same macroblock.
#[allow(clippy::too_many_arguments)]
fn write_mb_residual(
    w: &mut BitWriter,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
    luma_dc: Option<&[i16; 16]>,
    cbp: usize,
    luma: &[[i16; 16]; 16],
    chroma_dc: &[[i16; 8]; 2],
    chroma_ac: &[[[i16; 16]; 8]; 2],
    nz_luma: &[u8; 16],
    nz_chroma: &[[u8; 8]; 2],
) {
    let chroma = st.rows != 0;
    let i16x16 = luma_dc.is_some();
    let mut cur = [0u8; 16];
    let mut curc = [[0u8; 8]; 2];
    let luma_nc = |cur: &[u8; 16], st: &NzState, bx: usize, by: usize| -> i32 {
        let a = if bx > 0 {
            Some(cur[by * 4 + bx - 1])
        } else if left {
            Some(st.left_luma[by])
        } else {
            None
        };
        let b = if by > 0 {
            Some(cur[(by - 1) * 4 + bx])
        } else if top {
            Some(st.top_luma[mb_x * 4 + bx])
        } else {
            None
        };
        nc_of(a, b)
    };

    if let Some(dc) = luma_dc {
        // The DC block first. Its own count is not stored anywhere — the
        // reader discards it too — so the return is deliberately dropped.
        let nc = luma_nc(&cur, st, 0, 0);
        let _ = write_residual_block_cavlc(w, nc, &widen(dc), &ZIGZAG4X4, 0, 15, 16);
    }
    for blk8 in 0..4 {
        if cbp & (1 << blk8) == 0 {
            continue;
        }
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        for sub in 0..4 {
            let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
            let raster = by * 4 + bx;
            let nc = luma_nc(&cur, st, bx, by);
            let lv = widen(&luma[raster]);
            // I_16x16 AC blocks start at scan position one — the DC went
            // in the block above — and so carry at most fifteen.
            let n = if i16x16 {
                write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 1, 15, 15)
            } else {
                write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 0, 15, 16)
            };
            debug_assert_eq!(
                n, nz_luma[raster] as usize,
                "luma block ({bx},{by}): the decision's count disagrees with the writer's"
            );
            cur[raster] = n as u8;
        }
    }
    if chroma && cbp & 0x30 != 0 {
        for comp in 0..2 {
            let dc = widen_dc(&chroma_dc[comp]);
            if st.rows == 4 {
                let _ = write_residual_block_cavlc(w, -2, &dc, &SCAN_CHROMA_DC_422, 0, 7, 8);
            } else {
                let _ = write_residual_block_cavlc(w, -1, &dc, &SCAN_CHROMA_DC, 0, 3, 4);
            }
        }
        if cbp & 0x20 != 0 {
            for comp in 0..2 {
                for blk in 0..2 * st.rows {
                    let (bx, by) = (blk & 1, blk >> 1);
                    let a = if bx > 0 {
                        Some(curc[comp][by * 2 + bx - 1])
                    } else if left {
                        Some(st.left_chroma[comp][by])
                    } else {
                        None
                    };
                    let b = if by > 0 {
                        Some(curc[comp][(by - 1) * 2 + bx])
                    } else if top {
                        Some(st.top_chroma[comp][mb_x * 2 + bx])
                    } else {
                        None
                    };
                    let lv = widen(&chroma_ac[comp][blk]);
                    let n = write_residual_block_cavlc(w, nc_of(a, b), &lv, &ZIGZAG4X4, 1, 15, 15);
                    debug_assert_eq!(
                        n, nz_chroma[comp][blk] as usize,
                        "chroma block {comp}/{blk}: the decision's count disagrees with the writer's"
                    );
                    curc[comp][blk] = n as u8;
                }
            }
        }
    }

    // What the neighbours will read: the right column and the bottom row,
    // including the zeros of blocks nothing coded.
    st.left_luma = [cur[3], cur[7], cur[11], cur[15]];
    st.top_luma[mb_x * 4..mb_x * 4 + 4].copy_from_slice(&cur[12..16]);
    for comp in 0..2 {
        for r in 0..st.rows {
            st.left_chroma[comp][r] = curc[comp][r * 2 + 1];
        }
        if st.rows != 0 {
            let base = (st.rows - 1) * 2;
            st.top_chroma[comp][mb_x * 2] = curc[comp][base];
            st.top_chroma[comp][mb_x * 2 + 1] = curc[comp][base + 1];
        }
    }
}

/// Write every macroblock of an all-intra CAVLC picture: the shared walk
/// (`h264_pic::code_intra_picture`) makes the decisions, reconstructs and runs the
/// loop filter; this side only spells bits and keeps the `nC` state. The
/// slice header is already written; the caller closes the RBSP.
///
/// `qp` is the slice QP, which every macroblock is coded at —
/// [`MbDecision::qp_delta`] passes through as `mb_qp_delta` and is zero
/// until something adapts the quantiser. Refuses nothing at run time: the
/// caller keeps 4:4:4 and lossless on the PCM path, and a `debug_assert`
/// holds the door.
pub fn write_intra_picture(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
) {
    let mbs_wide = g.mbs_wide as usize;
    let rows = g.chroma_mb().1 as usize / 4;
    let mut st = NzState::new(mbs_wide, rows);
    code_intra_picture(g, tools, qp, planes, rec, |mb_x, mb_y, dec| {
        write_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0, 0);
    });
}

/// Write every macroblock of a P CAVLC picture: the shared walk
/// (`h264_pic::code_p_picture`) owns the motion search, the skip decision, the
/// intra fallback and every neighbour state a decoder derives; this side
/// spells the bits — the `mb_skip_run` bookkeeping and the macroblock
/// layers — and keeps the `nC` state. The slice header is already
/// written; the caller closes the RBSP.
///
/// `refp` is the reference picture's reconstruction, borders already
/// replicated ([`crate::encode::h264_me::prepare_reference`]); exactly one
/// reference is active, which is why no `ref_idx` is ever written.
pub fn write_p_picture(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refp: &[Recon],
) {
    let mbs_wide = g.mbs_wide as usize;
    let rows = g.chroma_mb().1 as usize / 4;
    let mut st = NzState::new(mbs_wide, rows);
    // `mb_skip_run`: counted here, written before each coded macroblock,
    // and flushed after the last one — the reader expects a run before
    // *every* coded macroblock (zero included) and a bare trailing run
    // when the slice ends in skips (7.3.4).
    let mut skip_run: u32 = 0;
    code_p_picture(g, tools, qp, planes, rec, refp, |mb_x, mb_y, mb| match mb {
        PMb::Skip(_) => {
            skip_run += 1;
            skip_nz(&mut st, mb_x);
        }
        PMb::Coded(dec) => {
            w.ue(skip_run);
            skip_run = 0;
            write_p16_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0);
        }
        PMb::Intra(idec) => {
            w.ue(skip_run);
            skip_run = 0;
            // Intra in a P slice: the same macroblock, `mb_type` shifted
            // by 5 (Table 7-11's note).
            write_macroblock(w, idec, &mut st, mb_x, mb_x > 0, mb_y > 0, 5);
        }
    });
    if skip_run > 0 {
        w.ue(skip_run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::encode::h264_intra::{IntraCtx, MbAvail, PredMode, code_macroblock};
    use crate::encode::h264_syntax::recon_plane;
    use crate::h264::SliceType;
    use crate::h264::mb::chroma_qp;
    use crate::h264::sps::ScalingLists;
    use crate::h264::transform::Dequant;
    use crate::h264::cavlc::{intra_mb_type, parse_mb_cavlc};
    use crate::h264::frame::{CHROMA_PAD, LUMA_PAD};
    use crate::h264::mb::{MbKind as DecKind, MbLayer, MbNeighbours, PicInfo, SliceCtx};
    use crate::h264::recon::QpState;

    fn flat() -> ScalingLists {
        ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] }
    }

    /// The derived coded_block_pattern mappings really invert the reader's:
    /// codeNum -> cbp -> codeNum is the identity over every entry.
    #[test]
    fn cbp_mapping_round_trips() {
        for code in 0..48usize {
            assert_eq!(
                INTRA_CBP_TO_GOLOMB[GOLOMB_TO_INTRA4X4_CBP[code] as usize] as usize,
                code,
                "intra cbp codeNum {code}"
            );
        }
        for code in 0..16usize {
            assert_eq!(
                INTRA_CBP_TO_GOLOMB_GRAY[GOLOMB_TO_INTRA4X4_CBP_GRAY[code] as usize] as usize,
                code,
                "monochrome intra cbp codeNum {code}"
            );
        }
        for code in 0..48usize {
            assert_eq!(
                INTER_CBP_TO_GOLOMB[GOLOMB_TO_INTER_CBP[code] as usize] as usize,
                code,
                "inter cbp codeNum {code}"
            );
        }
        for code in 0..16usize {
            assert_eq!(
                INTER_CBP_TO_GOLOMB_GRAY[GOLOMB_TO_INTER_CBP_GRAY[code] as usize] as usize,
                code,
                "monochrome inter cbp codeNum {code}"
            );
        }
    }

    /// A `SliceCtx` for parsing one macroblock of a single-reference P
    /// slice, which is the only kind this encoder writes.
    fn p_ctx() -> SliceCtx {
        SliceCtx {
            slice_type: SliceType::P,
            slice_num: 0,
            num_ref_idx: [1, 0],
            direct_spatial: false,
            transform_8x8_mode: false,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc: 1,
            cabac: false,
            bit_depth: 8,
            transform_bypass: false,
            scaling_plane: 0,
            x264_old_444: false,
            field_pic: false,
            mbaff: false,
        }
    }

    /// Write one P_L0_16x16 macroblock and hand the bits to the production
    /// reader: kind, the absent ref_idx, the mvd, cbp, the nonzero counts
    /// (the `nC` state a next macroblock would read) and the QP
    /// bookkeeping must all come back as written. Covers an empty
    /// macroblock (cbp 0, so no `mb_qp_delta` on the wire), a dense one,
    /// and negative mvd components — the sign bit of se(v) is exactly the
    /// kind of thing only a round trip through the real reader catches.
    #[test]
    fn a_p16_macroblock_round_trips_through_the_reader() {
        for (mvdx, mvdy, coded) in [(0i16, 0i16, false), (7, -3, true), (-13, 21, true), (1, 0, false)] {
            let mut dec = InterDecision { mvd: crate::h264::frame::Mv::new(mvdx, mvdy), ..InterDecision::default() };
            if coded {
                dec.luma[0][0] = 5;
                dec.luma[0][7] = -2;
                dec.nz_luma[0] = 2;
                dec.luma[8][3] = 1;
                dec.nz_luma[8] = 1;
                dec.cbp_luma = 0b0101; // 8x8 blocks 0 (raster 0) and 2 (raster 8)
                dec.chroma_dc[1][0] = -3;
                dec.chroma_ac[0][1][2] = 2;
                dec.nz_chroma[0][1] = 1;
                dec.cbp_chroma = 2;
            }

            let mut st = NzState::new(1, 2);
            let mut w = BitWriter::new();
            write_p16_macroblock(&mut w, &dec, &mut st, 0, false, false);
            w.rbsp_trailing_bits();
            let rbsp = w.into_rbsp();

            let ctx = p_ctx();
            let info = PicInfo::new(1, 1);
            let nb = MbNeighbours { mb_width: 1, ..MbNeighbours::default() };
            let dq = Dequant::new(&flat());
            let mut qps = QpState { prev_qp: 28, chroma_offset: [0, 0] };
            let mut r = BitReader::new(&rbsp);
            let t = r.ue();
            let mut layer = MbLayer::new(DecKind::I4x4);
            parse_mb_cavlc(&mut r, &ctx, &info, &nb, t, &mut layer, &dq, &mut qps)
                .expect("the reader rejected what the writer produced");
            assert!(!r.overrun());

            assert_eq!(layer.kind, DecKind::Inter16x16, "mvd ({mvdx},{mvdy})");
            assert_eq!(layer.ref_idx[0][0], 0, "one active reference infers ref_idx 0");
            assert_eq!(layer.mvd[0].mvd[0], crate::h264::frame::Mv::new(mvdx, mvdy));
            assert_eq!(layer.cbp, dec.cbp_luma | (dec.cbp_chroma << 4));
            assert_eq!(layer.qp_delta, if coded { dec.qp_delta as i32 } else { 0 });
            assert_eq!(layer.qp, 28, "constant QP whether or not a delta was carried");
            for blk in 0..16 {
                assert_eq!(layer.nz[0][blk], dec.nz_luma[blk], "luma nz {blk}");
            }
            for comp in 0..2 {
                for blk in 0..4 {
                    assert_eq!(
                        layer.chroma_nz[comp][blk], dec.nz_chroma[comp][blk],
                        "chroma nz {comp}/{blk}"
                    );
                }
            }
        }
    }

    /// An intra macroblock in a P slice is the same macroblock with
    /// `mb_type` shifted by 5, and the production reader must unmap it to
    /// the same kind, mode and coded block pattern.
    #[test]
    fn an_intra_macroblock_in_a_p_slice_round_trips_with_its_offset() {
        let dec = MbDecision {
            intra16_mode: 1,
            chroma_mode: 2,
            ..MbDecision::default()
        };
        let mut st = NzState::new(1, 2);
        let mut w = BitWriter::new();
        w.ue(0); // the mb_skip_run a coded macroblock follows
        write_macroblock(&mut w, &dec, &mut st, 0, false, false, 5);
        w.rbsp_trailing_bits();
        let rbsp = w.into_rbsp();

        let ctx = p_ctx();
        let info = PicInfo::new(1, 1);
        let nb = MbNeighbours { mb_width: 1, ..MbNeighbours::default() };
        let dq = Dequant::new(&flat());
        let mut qps = QpState { prev_qp: 26, chroma_offset: [0, 0] };
        let mut r = BitReader::new(&rbsp);
        assert_eq!(r.ue(), 0, "the skip run before the coded macroblock");
        let t = r.ue();
        assert!(t >= 5, "an intra mb_type in a P slice starts at 5");
        let mut layer = MbLayer::new(DecKind::I4x4);
        parse_mb_cavlc(&mut r, &ctx, &info, &nb, t, &mut layer, &dq, &mut qps)
            .expect("the reader rejected the offset intra macroblock");
        assert_eq!(layer.kind, DecKind::I16x16);
        assert_eq!(layer.intra16_mode, 1);
        assert_eq!(layer.chroma_mode, 2);
        assert_eq!(layer.cbp, 0);
    }

    /// The I_16x16 `mb_type` arithmetic against the reader's unmapping,
    /// over every combination it can carry.
    #[test]
    fn mb_type_matches_the_readers_unmapping() {
        for mode in 0..4u8 {
            for chroma in 0..3u8 {
                for luma in [0u8, 15] {
                    let t = 1 + mode as u32 + 4 * chroma as u32 + 12 * (luma == 15) as u32;
                    let mut layer = MbLayer::new(DecKind::I4x4);
                    intra_mb_type(t, &mut layer).unwrap();
                    assert_eq!(layer.kind, DecKind::I16x16);
                    assert_eq!(layer.intra16_mode, mode, "t={t}");
                    assert_eq!(layer.cbp, luma | (chroma << 4), "t={t}");
                }
            }
        }
        let mut layer = MbLayer::new(DecKind::I16x16);
        intra_mb_type(0, &mut layer).unwrap();
        assert_eq!(layer.kind, DecKind::I4x4);
    }

    /// Write one macroblock and hand the bits to the production reader.
    /// The comparison covers everything the reader stores unscaled: the
    /// kind, modes, coded block pattern, QP bookkeeping, every nonzero
    /// count (which is the `nC` state the next macroblock would read),
    /// and the raw DC levels. The AC levels come back dequantised, and
    /// their coverage is the residual writer's own round-trip test.
    fn round_trip(dec: &MbDecision, chosen_modes: Option<&[u8; 16]>, qp: u8) {
        let mut st = NzState::new(1, 2);
        let mut w = BitWriter::new();
        write_macroblock(&mut w, dec, &mut st, 0, false, false, 0);
        w.rbsp_trailing_bits();
        let rbsp = w.into_rbsp();

        let ctx = SliceCtx {
            slice_type: SliceType::I,
            slice_num: 0,
            num_ref_idx: [0; 2],
            direct_spatial: false,
            transform_8x8_mode: false,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc: 1,
            cabac: false,
            bit_depth: 8,
            transform_bypass: false,
            scaling_plane: 0,
            x264_old_444: false,
            field_pic: false,
            mbaff: false,
        };
        let info = PicInfo::new(1, 1);
        let nb = MbNeighbours { mb_width: 1, ..MbNeighbours::default() };
        let dq = Dequant::new(&flat());
        let mut qps = QpState { prev_qp: qp as i32, chroma_offset: [0, 0] };
        let mut r = BitReader::new(&rbsp);
        let t = r.ue();
        let mut layer = MbLayer::new(DecKind::I4x4);
        parse_mb_cavlc(&mut r, &ctx, &info, &nb, t, &mut layer, &dq, &mut qps)
            .expect("the reader rejected what the writer produced");
        assert!(!r.overrun());

        match dec.kind {
            MbKind::I4x4 => {
                assert_eq!(layer.kind, DecKind::I4x4);
                if let Some(modes) = chosen_modes {
                    assert_eq!(&layer.intra_modes, modes, "decoded 4x4 modes");
                }
            }
            MbKind::I16x16 => {
                assert_eq!(layer.kind, DecKind::I16x16);
                assert_eq!(layer.intra16_mode, dec.intra16_mode);
                let dc: Vec<i32> = dec.luma_dc.iter().map(|&v| v as i32).collect();
                assert_eq!(&layer.dc[0][..], &dc[..], "luma DC levels");
            }
        }
        assert_eq!(layer.cbp, dec.cbp_luma | (dec.cbp_chroma << 4), "cbp");
        assert_eq!(layer.chroma_mode, dec.chroma_mode);
        assert_eq!(layer.qp_delta, dec.qp_delta as i32);
        assert_eq!(layer.qp, qp as i32);
        for blk in 0..16 {
            assert_eq!(layer.nz[0][blk], dec.nz_luma[blk], "luma nz {blk}");
        }
        for comp in 0..2 {
            for blk in 0..4 {
                assert_eq!(
                    layer.chroma_nz[comp][blk], dec.nz_chroma[comp][blk],
                    "chroma nz {comp}/{blk}"
                );
            }
            if dec.cbp_chroma != 0 {
                let dc: Vec<i32> = dec.chroma_dc[comp][..4].iter().map(|&v| v as i32).collect();
                assert_eq!(&layer.chroma_dc[comp][..4], &dc[..], "chroma DC {comp}");
            }
        }
    }

    /// Decide a real macroblock from samples and round-trip it, over
    /// sources that reach the interesting shapes: flat (nothing coded),
    /// a gradient (I_16x16 plane prediction country) and noise (dense
    /// residual), across the QP range.
    #[test]
    fn coded_macroblocks_round_trip_through_the_reader() {
        let tools = IntraTools::new();
        let fill = |f: &mut dyn FnMut(usize, usize) -> u8| {
            let mut y = vec![0u8; 16 * 16];
            for r in 0..16 {
                for c in 0..16 {
                    y[r * 16 + c] = f(c, r);
                }
            }
            let mut cb = vec![0u8; 8 * 8];
            let mut cr = vec![0u8; 8 * 8];
            for r in 0..8 {
                for c in 0..8 {
                    cb[r * 8 + c] = f(c * 2, r * 2).wrapping_add(30);
                    cr[r * 8 + c] = f(c * 2 + 1, r * 2).wrapping_sub(30);
                }
            }
            (y, cb, cr)
        };
        let mut seed = 0x2718_2818u32;
        let mut lcg = move |x: usize, y: usize| -> u8 {
            seed = seed
                .wrapping_mul(1664525)
                .wrapping_add(1013904223 ^ ((x * 31 + y) as u32));
            (seed >> 16) as u8
        };
        let sources: [(&str, (Vec<u8>, Vec<u8>, Vec<u8>)); 3] = [
            ("flat", fill(&mut |_, _| 128)),
            ("gradient", fill(&mut |x, y| (60 + 6 * x + 3 * y) as u8)),
            ("noise", fill(&mut |x, y| lcg(x, y))),
        ];
        for (_name, (y, cb, cr)) in &sources {
            for qp in [10u8, 26, 40] {
                let qpc = chroma_qp(qp as i32, 0, 0);
                let ctx = IntraCtx {
                    dsp: &tools.dsp,
                    enc: &tools.enc,
                    dist: &tools.dist,
                    quant: &tools.quant,
                    dequant: &tools.dequant,
                    qp: qp as i32,
                    qpc: [qpc; 2],
                    chroma_h: 8,
                };
                let mut rec = vec![
                    recon_plane(16, 16, LUMA_PAD),
                    recon_plane(8, 8, CHROMA_PAD),
                    recon_plane(8, 8, CHROMA_PAD),
                ];
                let mb = MbAvail { left: false, top: false, top_left: false, top_right: false };
                let (dec, modes) = code_macroblock(
                    &ctx, &mut rec, 0, 0, y, 16, [cb, cr], 8, mb, &[None; 4], &[None; 4],
                );
                let chosen = (dec.kind == MbKind::I4x4).then_some(&modes);
                round_trip(&dec, chosen, qp);
            }
        }
    }

    /// A hand-built I_4x4 decision, so the writer's I_NxN syntax — the
    /// sixteen mode elements, the me(v) coded block pattern and the
    /// full-span residual blocks — is exercised whatever the mode
    /// decision above happens to choose. All modes DC keeps the
    /// prediction bookkeeping trivially consistent with an isolated
    /// macroblock, whose every predicted mode is DC.
    #[test]
    fn a_synthetic_i4x4_macroblock_round_trips() {
        let mut dec = MbDecision {
            kind: MbKind::I4x4,
            luma_pred: [PredMode { use_predicted: true, rem: 0 }; 16],
            chroma_mode: 1,
            ..MbDecision::default()
        };
        // Blocks 0 and 5 coded (both in luma 8x8 block 0), block 15 too,
        // with counts the levels really have.
        dec.luma[0][0] = 7;
        dec.luma[0][3] = -2;
        dec.luma[0][10] = 1;
        dec.nz_luma[0] = 3;
        dec.luma[5][1] = -1;
        dec.nz_luma[5] = 1;
        dec.luma[15][0] = 4;
        dec.luma[15][15] = 1;
        dec.nz_luma[15] = 2;
        dec.cbp_luma = 0b1001;
        // Chroma: DC on both components, AC on Cb block 2.
        dec.chroma_dc[0][0] = 3;
        dec.chroma_dc[0][2] = -1;
        dec.chroma_dc[1][1] = 2;
        dec.chroma_ac[0][2][5] = -4;
        dec.chroma_ac[0][2][1] = 1;
        dec.nz_chroma[0][2] = 2;
        dec.cbp_chroma = 2;
        let modes = [2u8; 16];
        round_trip(&dec, Some(&modes), 28);
    }
}
