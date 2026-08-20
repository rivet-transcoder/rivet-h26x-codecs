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
use crate::encode::h264_me::{BDecision, InterDecision, InterMbKind};
use crate::encode::h264_pic::{
    BMb, IntraTools, PMb, PicMotion, code_b_picture, code_intra_picture, code_p_picture,
};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::cavlc::{SCAN8_SUB, SCAN_CHROMA_DC, write_residual_block_cavlc};
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
    /// Per luma-*like* plane (luma; in 4:4:4 also Cb and Cr, which the
    /// reader's `nC` treats as three luma planes — `nb.nz_top` /
    /// `nb.nz_left` in `MbNeighbours::gather_nz`), per 4x4 column of the
    /// picture: the count of the bottom block of the macroblock above
    /// (`mbs_wide * 4`).
    top_luma: [Vec<u8>; 3],
    /// Per luma-like plane and 4x4 row: the count of the rightmost block
    /// of the macroblock to the left.
    left_luma: [[u8; 4]; 3],
    /// The same for the 4:2:x chroma blocks, per component
    /// (`mbs_wide * 2`). Unused in 4:4:4, whose chroma is luma-like.
    top_chroma: [Vec<u8>; 2],
    /// Chroma left column; `rows` entries are meaningful.
    left_chroma: [[u8; 4]; 2],
    /// Chroma AC block rows: 2 (4:2:0), 4 (4:2:2), 0 (monochrome and
    /// 4:4:4 — the latter flagged separately below).
    rows: usize,
    /// ChromaArrayType 3: planes 1 and 2 of the luma-like state are live.
    c444: bool,
}

impl NzState {
    fn new(mbs_wide: usize, rows: usize, c444: bool) -> Self {
        debug_assert!(!c444 || rows == 0, "4:4:4 has no 4:2:x chroma rows");
        NzState {
            top_luma: [vec![0; mbs_wide * 4], vec![0; mbs_wide * 4], vec![0; mbs_wide * 4]],
            left_luma: [[0; 4]; 3],
            top_chroma: [vec![0; mbs_wide * 2], vec![0; mbs_wide * 2]],
            left_chroma: [[0; 4]; 2],
            rows,
            c444,
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
fn widen_dc(levels: &[i16; 16]) -> [i32; 8] {
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
#[allow(clippy::too_many_arguments)]
fn write_macroblock(
    w: &mut BitWriter,
    dec: &MbDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
    mb_type_offset: u32,
    t8x8_mode: bool,
) {
    let chroma = st.rows != 0;
    let i16x16 = dec.kind == MbKind::I16x16;
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;

    // mb_type (Table 7-11): I_NxN is 0 for *both* transform sizes — which
    // is what `transform_size_8x8_flag` below is for; the I_16x16 types
    // encode the prediction mode and both halves of the coded block
    // pattern.
    match dec.kind {
        MbKind::I4x4 | MbKind::I8x8 => w.ue(mb_type_offset),
        MbKind::I16x16 => w.ue(
            mb_type_offset
                + 1
                + dec.intra16_mode as u32
                + 4 * dec.cbp_chroma as u32
                + 12 * (dec.cbp_luma == 15) as u32,
        ),
    }

    // `transform_size_8x8_flag`, before `mb_pred()` and only for I_NxN:
    // the reader takes it under `ctx.transform_8x8_mode && layer.kind ==
    // MbKind::I4x4` (`parse_mb_cavlc` in src/h264/cavlc.rs), where
    // `I4x4` is still standing for I_NxN because the flag has not yet
    // renamed it.
    if t8x8_mode && dec.kind.is_nxn() {
        w.flag(dec.transform_8x8);
    }
    debug_assert!(t8x8_mode || !dec.transform_8x8, "no PPS flag, no 8x8 transform");

    match dec.kind {
        // The sixteen prediction modes, in luma4x4BlkIdx order — the
        // standard's 4x4 scan, not raster, which is why the raster-indexed
        // decision is walked through `raster_of_blk`.
        MbKind::I4x4 => {
            for blk in 0..16 {
                let p = dec.luma_pred[raster_of_blk(blk)];
                w.flag(p.use_predicted);
                if !p.use_predicted {
                    w.bits(3, p.rem as u32);
                }
            }
        }
        // Four modes instead of sixteen, one per 8x8 quad in raster
        // order, with the same two syntax elements; the decision stored
        // each on all four of its quad's 4x4s, so the quad's top-left is
        // where it is read.
        MbKind::I8x8 => {
            for &raster in &[0usize, 2, 8, 10] {
                let p = dec.luma_pred[raster];
                w.flag(p.use_predicted);
                if !p.use_predicted {
                    w.bits(3, p.rem as u32);
                }
            }
        }
        MbKind::I16x16 => {}
    }
    if chroma {
        w.ue(dec.chroma_mode as u32);
    }
    if dec.kind.is_nxn() {
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
        dec.transform_8x8,
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
/// One macroblock of any coded P shape — 16x16, 16x8 or 8x16 — since
/// after `mb_type` they differ only in how many mvds follow.
///
/// No `ref_idx_l0` is written: with exactly one active reference the
/// element is absent from the stream — the reader's `read_ref_idx` is
/// only reached when `num_ref_idx_active > 1` (7.3.5.1) — and this
/// encoder's slice headers always declare one. The `debug_assert` is the
/// tripwire for the day that stops being true.
#[allow(clippy::too_many_arguments)]
fn write_p16_macroblock(
    w: &mut BitWriter,
    dec: &InterDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
    t8x8_mode: bool,
) {
    debug_assert!(
        matches!(
            dec.kind,
            InterMbKind::P16x16 | InterMbKind::P16x8 | InterMbKind::P8x16
        ),
        "only a coded P macroblock carries this syntax"
    );
    debug_assert_eq!(dec.ref_idx, 0, "more than one reference needs te(v) ref_idx writing");
    w.ue(dec.kind.p_mb_type()); // Table 7-13
    // `ref_idx_l0` is absent: one active reference, so the reader infers
    // 0 for every partition (7.3.5.1). Then one mvd per partition, x then
    // y, in the order `mb_partitions` lists them — the reader's own two
    // passes, of which the first has nothing to read.
    for i in 0..dec.kind.parts().len() {
        w.se(dec.mvd[i].x as i32);
        w.se(dec.mvd[i].y as i32);
    }
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;
    let code = if st.rows != 0 {
        INTER_CBP_TO_GOLOMB[cbp]
    } else {
        INTER_CBP_TO_GOLOMB_GRAY[cbp]
    };
    w.ue(code as u32);
    // An inter macroblock's `transform_size_8x8_flag` comes after the
    // coded block pattern and only when some luma block is coded;
    // `no_sub_mb_part_less_than_8x8` holds trivially for one 16x16
    // partition.
    if t8x8_mode && dec.cbp_luma != 0 {
        w.flag(dec.transform_8x8);
    }
    debug_assert!(!dec.transform_8x8 || (t8x8_mode && dec.cbp_luma != 0));
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
        dec.transform_8x8,
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
    for p in 0..3 {
        st.left_luma[p] = [0; 4];
        st.top_luma[p][mb_x * 4..mb_x * 4 + 4].fill(0);
    }
    for comp in 0..2 {
        st.left_chroma[comp] = [0; 4];
        if st.rows != 0 {
            st.top_chroma[comp][mb_x * 2..mb_x * 2 + 2].fill(0);
        }
    }
}

/// One luma-like plane's residual — the DC block for `Intra_16x16`, then
/// the coded 8x8s' 4x4 blocks — with plane `p`'s own `nC` bookkeeping,
/// updated for the next macroblocks. The mirror of
/// `parse_residual_luma_like` (src/h264/cavlc.rs) for one plane: luma is
/// plane 0; in 4:4:4 Cb and Cr are planes 1 and 2 coded the same way,
/// gated by the *same* luma coded-block-pattern bits.
#[allow(clippy::too_many_arguments)]
fn write_plane_residual(
    w: &mut BitWriter,
    st: &mut NzState,
    p: usize,
    mb_x: usize,
    left: bool,
    top: bool,
    dc: Option<&[i16; 16]>,
    transform_8x8: bool,
    cbp: usize,
    levels: &[[i16; 16]; 16],
    nz: &[u8; 16],
) {
    let mut cur = [0u8; 16];
    let nc_at = |cur: &[u8; 16], st: &NzState, bx: usize, by: usize| -> i32 {
        let a = if bx > 0 {
            Some(cur[by * 4 + bx - 1])
        } else if left {
            Some(st.left_luma[p][by])
        } else {
            None
        };
        let b = if by > 0 {
            Some(cur[(by - 1) * 4 + bx])
        } else if top {
            Some(st.top_luma[p][mb_x * 4 + bx])
        } else {
            None
        };
        nc_of(a, b)
    };
    if let Some(dc) = dc {
        debug_assert!(!transform_8x8, "Intra_16x16 carries no transform_size_8x8_flag");
        // The DC block first. Its own count is not stored anywhere — the
        // reader discards it too — so the return is deliberately dropped.
        let nc = nc_at(&cur, st, 0, 0);
        let _ = write_residual_block_cavlc(w, nc, &widen(dc), &ZIGZAG4X4, 0, 15, 16);
    }
    for blk8 in 0..4 {
        if cbp & (1 << blk8) == 0 {
            continue;
        }
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        // Under the 8x8 transform CAVLC still codes four blocks per 8x8,
        // but they are not its four 4x4s: they are its sixty-four scan
        // positions taken every fourth (`SCAN8_SUB` in src/h264/cavlc.rs,
        // built from `ZIGZAG8X8`), each written as a sixteen-coefficient
        // block into the *8x8*'s storage. Everything else is unchanged —
        // the same four `nC` predictions in the same order, the same
        // counts stored on the same 4x4s — which is exactly why this is a
        // choice of scan and span rather than a second walk.
        let scan8 = transform_8x8.then(|| &levels.as_flattened()[blk8 * 64..blk8 * 64 + 64]);
        for sub in 0..4 {
            let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
            let raster = by * 4 + bx;
            let nc = nc_at(&cur, st, bx, by);
            let n = if let Some(block8) = scan8 {
                let mut lv = [0i32; 64];
                for (o, &v) in lv.iter_mut().zip(block8) {
                    *o = v as i32;
                }
                write_residual_block_cavlc(w, nc, &lv, &SCAN8_SUB[sub], 0, 15, 16)
            } else {
                let lv = widen(&levels[raster]);
                // I_16x16 AC blocks start at scan position one — the DC
                // went in the block above — and so carry at most fifteen.
                if dc.is_some() {
                    write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 1, 15, 15)
                } else {
                    write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 0, 15, 16)
                }
            };
            debug_assert_eq!(
                n, nz[raster] as usize,
                "plane {p} block ({bx},{by}): the decision's count disagrees with the writer's"
            );
            cur[raster] = n as u8;
        }
    }
    // What the neighbours will read: the right column and the bottom row,
    // including the zeros of blocks nothing coded.
    st.left_luma[p] = [cur[3], cur[7], cur[11], cur[15]];
    st.top_luma[p][mb_x * 4..mb_x * 4 + 4].copy_from_slice(&cur[12..16]);
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
    transform_8x8: bool,
    cbp: usize,
    luma: &[[i16; 16]; 16],
    chroma_dc: &[[i16; 16]; 2],
    chroma_ac: &[[[i16; 16]; 16]; 2],
    nz_luma: &[u8; 16],
    nz_chroma: &[[u8; 16]; 2],
) {
    let chroma = st.rows != 0;
    let mut curc = [[0u8; 8]; 2];

    // Luma, then (4:4:4) Cb and Cr coded the same way — the mirror of
    // `parse_residual_cavlc`'s plane order, each plane's `nC` from its own
    // neighbour counts (`plane_nc` in src/h264/cavlc.rs).
    write_plane_residual(w, st, 0, mb_x, left, top, luma_dc, transform_8x8, cbp, luma, nz_luma);
    if st.c444 {
        // 4:4:4's chroma planes are luma-style, transform size included.
        write_plane_residual(
            w, st, 1, mb_x, left, top,
            luma_dc.is_some().then_some(&chroma_dc[0]),
            transform_8x8, cbp, &chroma_ac[0], &nz_chroma[0],
        );
        write_plane_residual(
            w, st, 2, mb_x, left, top,
            luma_dc.is_some().then_some(&chroma_dc[1]),
            transform_8x8, cbp, &chroma_ac[1], &nz_chroma[1],
        );
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

    // What the neighbours will read of the 4:2:x chroma state (the
    // luma-like planes updated theirs above).
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
) -> PicMotion {
    let mbs_wide = g.mbs_wide as usize;
    let rows = if g.chroma == crate::picture::ChromaFormat::Yuv444 { 0 } else { g.chroma_mb().1 as usize / 4 };
    let mut st = NzState::new(mbs_wide, rows, g.chroma == crate::picture::ChromaFormat::Yuv444);
    let t8x8 = tools.transform_8x8;
    code_intra_picture(g, tools, qp, planes, rec, |mb_x, mb_y, dec| {
        write_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0, 0, t8x8);
    })
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
) -> PicMotion {
    let mbs_wide = g.mbs_wide as usize;
    let rows = if g.chroma == crate::picture::ChromaFormat::Yuv444 { 0 } else { g.chroma_mb().1 as usize / 4 };
    let mut st = NzState::new(mbs_wide, rows, g.chroma == crate::picture::ChromaFormat::Yuv444);
    // `mb_skip_run`: counted here, written before each coded macroblock,
    // and flushed after the last one — the reader expects a run before
    // *every* coded macroblock (zero included) and a bare trailing run
    // when the slice ends in skips (7.3.4).
    let mut skip_run: u32 = 0;
    let t8x8 = tools.transform_8x8;
    let fmbs = code_p_picture(g, tools, qp, planes, rec, refp, |mb_x, mb_y, mb| match mb {
        PMb::Skip(_) => {
            skip_run += 1;
            skip_nz(&mut st, mb_x);
        }
        PMb::Coded(dec) => {
            w.ue(skip_run);
            skip_run = 0;
            write_p16_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0, t8x8);
        }
        PMb::Intra(idec) => {
            w.ue(skip_run);
            skip_run = 0;
            // Intra in a P slice: the same macroblock, `mb_type` shifted
            // by 5 (Table 7-11's note).
            write_macroblock(w, idec, &mut st, mb_x, mb_x > 0, mb_y > 0, 5, t8x8);
        }
    });
    if skip_run > 0 {
        w.ue(skip_run);
    }
    fmbs
}

/// Write one coded B macroblock — `mb_type` (Table 7-14's 16x16 rows)
/// through the residual. The skip run belongs to the caller; no `ref_idx`
/// is written because exactly one reference is active per list, and the
/// mvds come in list order for the lists the direction uses (7.3.5.1's
/// prediction loops). `B_Direct_16x16` carries no motion syntax at all —
/// `mb_type` 0, then straight to the coded block pattern.
#[allow(clippy::too_many_arguments)]
fn write_b16_macroblock(
    w: &mut BitWriter,
    dec: &BDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
    direct: bool,
    t8x8_mode: bool,
) {
    debug_assert!(dec.ref_idx.iter().all(|&r| r <= 0), "multi-reference lists need te(v) ref_idx");
    if direct {
        w.ue(0); // B_Direct_16x16
    } else {
        // B_L0_16x16 (1), B_L1_16x16 (2), B_Bi_16x16 (3).
        let t = match dec.used {
            [true, false] => 1,
            [false, true] => 2,
            [true, true] => 3,
            [false, false] => unreachable!("an explicit B macroblock uses a list"),
        };
        w.ue(t);
        for l in 0..2 {
            if dec.used[l] {
                w.se(dec.mvd[l].x as i32);
                w.se(dec.mvd[l].y as i32);
            }
        }
    }
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;
    let code = if st.rows != 0 {
        INTER_CBP_TO_GOLOMB[cbp]
    } else {
        INTER_CBP_TO_GOLOMB_GRAY[cbp]
    };
    w.ue(code as u32);
    // `B_Direct_16x16` carries the flag too, because the SPS this encoder
    // writes sets `direct_8x8_inference_flag`.
    if t8x8_mode && dec.cbp_luma != 0 {
        w.flag(dec.transform_8x8);
    }
    debug_assert!(!dec.transform_8x8 || (t8x8_mode && dec.cbp_luma != 0));
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
        dec.transform_8x8,
        cbp,
        &dec.luma,
        &dec.chroma_dc,
        &dec.chroma_ac,
        &dec.nz_luma,
        &dec.nz_chroma,
    );
}

/// Write every macroblock of a B CAVLC picture: the shared walk
/// (`h264_pic::code_b_picture`) owns the searches, the direct derivation and the
/// intra fallback; this side spells the bits — the same `mb_skip_run`
/// bookkeeping as P — and keeps the `nC` state. Returns the picture's
/// motion record for the caller's reference bookkeeping.
#[allow(clippy::too_many_arguments)]
pub fn write_b_picture(
    w: &mut BitWriter,
    g: &Geometry,
    tools: &IntraTools,
    qp: u8,
    planes: &[Plane<'_>],
    rec: &mut [Recon],
    refs: [&[Recon]; 2],
    col: &PicMotion,
) -> PicMotion {
    let mbs_wide = g.mbs_wide as usize;
    let rows = if g.chroma == crate::picture::ChromaFormat::Yuv444 { 0 } else { g.chroma_mb().1 as usize / 4 };
    let mut st = NzState::new(mbs_wide, rows, g.chroma == crate::picture::ChromaFormat::Yuv444);
    let mut skip_run: u32 = 0;
    let t8x8 = tools.transform_8x8;
    let fmbs = code_b_picture(g, tools, qp, planes, rec, refs, col, |mb_x, mb_y, mb| match mb {
        BMb::Skip(_) => {
            skip_run += 1;
            skip_nz(&mut st, mb_x);
        }
        BMb::Direct(dec) => {
            w.ue(skip_run);
            skip_run = 0;
            write_b16_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0, true, t8x8);
        }
        BMb::Explicit(dec) => {
            w.ue(skip_run);
            skip_run = 0;
            write_b16_macroblock(w, dec, &mut st, mb_x, mb_x > 0, mb_y > 0, false, t8x8);
        }
        BMb::Intra(idec) => {
            w.ue(skip_run);
            skip_run = 0;
            // Intra in a B slice: the same macroblock, `mb_type` shifted
            // by 23 (Table 7-14's note).
            write_macroblock(w, idec, &mut st, mb_x, mb_x > 0, mb_y > 0, 23, t8x8);
        }
    });
    if skip_run > 0 {
        w.ue(skip_run);
    }
    fmbs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::encode::h264_intra::{IntraCtx, MbAvail, PredMode, code_macroblock, quad_rasters};
    use crate::h264::tables::ZIGZAG8X8;
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
            let mut dec = InterDecision {
                mvd: [crate::h264::frame::Mv::new(mvdx, mvdy), crate::h264::frame::Mv::ZERO,
                      crate::h264::frame::Mv::ZERO, crate::h264::frame::Mv::ZERO],
                ..InterDecision::default()
            };
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

            let mut st = NzState::new(1, 2, false);
            let mut w = BitWriter::new();
            write_p16_macroblock(&mut w, &dec, &mut st, 0, false, false, false);
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

    /// Write one B macroblock of each 16x16 shape and hand the bits to
    /// the production reader: the direction, the per-list mvds in list
    /// order, the absent ref_idx, the INTER coded-block-pattern column
    /// and the nonzero counts must all come back as written — and
    /// `B_Direct_16x16` must come back as exactly `mb_type` 0 with no
    /// motion syntax consumed.
    #[test]
    fn a_b16_macroblock_round_trips_through_the_reader() {
        use crate::h264::mb::PRED_L0;
        let cases: [(bool, [bool; 2], [crate::h264::frame::Mv; 2]); 5] = [
            (true, [true, true], [crate::h264::frame::Mv::ZERO; 2]),
            (false, [true, false], [crate::h264::frame::Mv::new(7, -3), crate::h264::frame::Mv::ZERO]),
            (false, [false, true], [crate::h264::frame::Mv::ZERO, crate::h264::frame::Mv::new(-13, 21)]),
            (false, [true, true], [crate::h264::frame::Mv::new(2, 2), crate::h264::frame::Mv::new(-1, 5)]),
            (false, [true, true], [crate::h264::frame::Mv::ZERO, crate::h264::frame::Mv::ZERO]),
        ];
        for (direct, used, mvd) in cases {
            let mut dec = BDecision { used, mvd, ..BDecision::default() };
            dec.ref_idx = [if used[0] { 0 } else { -1 }, if used[1] { 0 } else { -1 }];
            dec.luma[0][0] = 4;
            dec.luma[0][5] = -1;
            dec.nz_luma[0] = 2;
            dec.cbp_luma = 0b0001;
            dec.chroma_dc[0][1] = 2;
            dec.cbp_chroma = 1;

            let mut st = NzState::new(1, 2, false);
            let mut w = BitWriter::new();
            write_b16_macroblock(&mut w, &dec, &mut st, 0, false, false, direct, false);
            w.rbsp_trailing_bits();
            let rbsp = w.into_rbsp();

            let ctx = SliceCtx {
                slice_type: SliceType::B,
                num_ref_idx: [1, 1],
                direct_spatial: true,
                ..p_ctx()
            };
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

            if direct {
                assert_eq!(t, 0);
                assert_eq!(layer.kind, DecKind::BDirect16x16);
            } else {
                assert_eq!(layer.kind, DecKind::Inter16x16, "used {used:?}");
                let want_dir = (used[0] as u8) * PRED_L0 + (used[1] as u8) * 2;
                assert_eq!(layer.pred_dir[0], want_dir, "used {used:?}");
                for l in 0..2 {
                    if used[l] {
                        assert_eq!(layer.mvd[0].mvd[l], mvd[l], "list {l}");
                        assert_eq!(layer.ref_idx[l][0], 0, "list {l}");
                    }
                }
            }
            assert_eq!(layer.cbp, dec.cbp_luma | (dec.cbp_chroma << 4));
            for blk in 0..16 {
                assert_eq!(layer.nz[0][blk], dec.nz_luma[blk], "luma nz {blk}");
            }
        }
    }

    /// A 4:4:4 intra macroblock — decided by the real mode decision,
    /// its chroma planes replaying the luma modes luma-style — written
    /// and handed to the production reader with `chroma_format_idc` 3:
    /// the kind, the shared coded block pattern, and every plane's
    /// nonzero counts (the `nC` state a next macroblock's three planes
    /// would read) must come back as coded. No `intra_chroma_pred_mode`
    /// exists on the wire, which the reader enforces by construction.
    #[test]
    fn a_444_intra_macroblock_round_trips_through_the_reader() {
        use crate::h264::frame::LUMA_PAD;
        let tools = IntraTools::new(false, false);
        let mut seed = 77u32;
        let mut lcg = move || -> u8 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        };
        for qp in [14u8, 26, 40] {
            let qpc = chroma_qp(qp as i32, 0, 0);
            let ctx = IntraCtx {
                dsp: &tools.dsp,
                enc: &tools.enc,
                dist: &tools.dist,
                quant: &tools.quant,
                dequant: &tools.dequant,
                qp: qp as i32,
                qpc: [qpc; 2],
                chroma_h: 16,
                c444: true,
                t8x8: false,
                subparts: false,
            };
            let mut rec = vec![
                recon_plane(16, 16, LUMA_PAD),
                recon_plane(16, 16, LUMA_PAD),
                recon_plane(16, 16, LUMA_PAD),
            ];
            let mut y = vec![0u8; 16 * 16];
            let mut cb = vec![0u8; 16 * 16];
            let mut cr = vec![0u8; 16 * 16];
            for i in 0..256 {
                y[i] = lcg();
                cb[i] = lcg();
                cr[i] = lcg();
            }
            let mb = MbAvail { left: false, top: false, top_left: false, top_right: false };
            let (dec, _modes) = code_macroblock(
                &ctx, &mut rec, 0, 0, &y, 16, [&cb, &cr], 16, mb, &[None; 4], &[None; 4],
            );
            assert_eq!(dec.cbp_chroma, 0, "ChromaArrayType 3 has no chroma cbp");

            let mut st = NzState::new(1, 0, true);
            let mut w = BitWriter::new();
            write_macroblock(&mut w, &dec, &mut st, 0, false, false, 0, false);
            w.rbsp_trailing_bits();
            let rbsp = w.into_rbsp();

            let sctx = SliceCtx { chroma_format_idc: 3, ..p_ctx() };
            let sctx = SliceCtx { slice_type: SliceType::I, num_ref_idx: [0, 0], ..sctx };
            let info = PicInfo::new(1, 1);
            let nb = MbNeighbours { mb_width: 1, ..MbNeighbours::default() };
            let dq = Dequant::new(&flat());
            let mut qps = QpState { prev_qp: qp as i32, chroma_offset: [0, 0] };
            let mut r = BitReader::new(&rbsp);
            let t = r.ue();
            let mut layer = MbLayer::new(DecKind::I4x4);
            parse_mb_cavlc(&mut r, &sctx, &info, &nb, t, &mut layer, &dq, &mut qps)
                .expect("the reader rejected the 4:4:4 macroblock");
            assert!(!r.overrun());

            match dec.kind {
                MbKind::I4x4 => assert_eq!(layer.kind, DecKind::I4x4, "qp={qp}"),
                MbKind::I8x8 => assert_eq!(layer.kind, DecKind::I8x8, "qp={qp}"),
                MbKind::I16x16 => {
                    assert_eq!(layer.kind, DecKind::I16x16, "qp={qp}");
                    assert_eq!(layer.intra16_mode, dec.intra16_mode);
                    let want: Vec<i32> = dec.luma_dc.iter().map(|&v| v as i32).collect();
                    assert_eq!(&layer.dc[0][..], &want[..], "luma DC");
                    for comp in 0..2 {
                        let want: Vec<i32> =
                            dec.chroma_dc[comp].iter().map(|&v| v as i32).collect();
                        assert_eq!(&layer.dc[1 + comp][..], &want[..], "plane {comp} DC");
                    }
                }
            }
            assert_eq!(layer.cbp, dec.cbp_luma, "qp={qp} shared cbp");
            for blk in 0..16 {
                assert_eq!(layer.nz[0][blk], dec.nz_luma[blk], "qp={qp} luma nz {blk}");
                for comp in 0..2 {
                    assert_eq!(
                        layer.nz[1 + comp][blk], dec.nz_chroma[comp][blk],
                        "qp={qp} plane {comp} nz {blk}"
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
        let mut st = NzState::new(1, 2, false);
        let mut w = BitWriter::new();
        w.ue(0); // the mb_skip_run a coded macroblock follows
        write_macroblock(&mut w, &dec, &mut st, 0, false, false, 5, false);
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
        round_trip_fmt(dec, chosen_modes, qp, 1, false)
    }

    /// As [`round_trip`], with the chroma format and whether the PPS
    /// offers the 8x8 transform spelled out.
    fn round_trip_fmt(
        dec: &MbDecision,
        chosen_modes: Option<&[u8; 16]>,
        qp: u8,
        cfi: u32,
        t8x8: bool,
    ) {
        let c444 = cfi == 3;
        let rows = match cfi {
            1 => 2,
            2 => 4,
            _ => 0,
        };
        let mut st = NzState::new(1, rows, c444);
        let mut w = BitWriter::new();
        write_macroblock(&mut w, dec, &mut st, 0, false, false, 0, t8x8);
        w.rbsp_trailing_bits();
        let rbsp = w.into_rbsp();

        let ctx = SliceCtx {
            slice_type: SliceType::I,
            slice_num: 0,
            num_ref_idx: [0; 2],
            direct_spatial: false,
            transform_8x8_mode: t8x8,
            constrained_intra_pred: false,
            direct_8x8_inference: true,
            chroma_format_idc: cfi,
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
            MbKind::I8x8 => {
                assert_eq!(layer.kind, DecKind::I8x8);
                // The reader replicates each quad's mode over its four
                // 4x4s, which is the form the decision keeps too.
                if let Some(modes) = chosen_modes {
                    assert_eq!(&layer.intra_modes, modes, "decoded 8x8 modes");
                }
            }
            MbKind::I16x16 => {
                assert_eq!(layer.kind, DecKind::I16x16);
                assert_eq!(layer.intra16_mode, dec.intra16_mode);
                let dc: Vec<i32> = dec.luma_dc.iter().map(|&v| v as i32).collect();
                assert_eq!(&layer.dc[0][..], &dc[..], "luma DC levels");
            }
        }
        assert_eq!(layer.transform_8x8, dec.transform_8x8, "transform_size_8x8_flag");
        assert_eq!(layer.cbp, dec.cbp_luma | (dec.cbp_chroma << 4), "cbp");
        assert_eq!(layer.chroma_mode, dec.chroma_mode);
        assert_eq!(layer.qp_delta, dec.qp_delta as i32);
        assert_eq!(layer.qp, qp as i32);
        // CAVLC stores exactly the counts the decision carries — its four
        // sub-scan counts under the 8x8 transform, one per 4x4 otherwise —
        // because those *are* the four blocks it codes.
        assert_eq!(layer.nz[0], dec.nz_luma, "luma nz");
        if c444 {
            for comp in 0..2 {
                assert_eq!(layer.nz[1 + comp], dec.nz_chroma[comp], "plane {comp} nz");
            }
        } else {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    assert_eq!(
                        layer.chroma_nz[comp][blk], dec.nz_chroma[comp][blk],
                        "chroma nz {comp}/{blk}"
                    );
                }
                if dec.cbp_chroma != 0 {
                    let n_dc = if rows == 4 { 8 } else { 4 };
                    let dc: Vec<i32> =
                        dec.chroma_dc[comp][..n_dc].iter().map(|&v| v as i32).collect();
                    assert_eq!(&layer.chroma_dc[comp][..n_dc], &dc[..], "chroma DC {comp}");
                }
            }
        }
    }

    /// Decide a real macroblock from samples and round-trip it, over
    /// sources that reach the interesting shapes: flat (nothing coded),
    /// a gradient (I_16x16 plane prediction country) and noise (dense
    /// residual), across the QP range.
    #[test]
    fn coded_macroblocks_round_trip_through_the_reader() {
        let tools = IntraTools::new(false, false);
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
                    c444: false,
                    t8x8: false,
                    subparts: false,
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

    /// Real `I_8x8` macroblocks, decided by the real mode decision with
    /// the 8x8 transform on offer, written and handed to the production
    /// CAVLC reader — over the chroma formats and the QP range, and over
    /// sources that reach the shapes the decision actually picks between.
    ///
    /// The two things this covers that no 4x4 test can: the four
    /// prediction-mode elements land where the reader takes them (before
    /// the coded block pattern, after a `transform_size_8x8_flag` that
    /// is itself read before `mb_pred()`), and the residual is four
    /// *interleaved* sub-scans of one 8x8 rather than four 4x4 blocks —
    /// so the `nC` bookkeeping is per sub-scan, and the levels come back
    /// in the 8x8's raster and not in four separate ones.
    #[test]
    fn coded_8x8_macroblocks_round_trip_through_the_reader() {
        let tools = IntraTools::new(true, false);
        let mut seed = 0x8080_8080u32;
        let mut lcg = move |x: usize, y: usize| -> u8 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223 ^ ((x * 31 + y) as u32));
            (seed >> 16) as u8
        };
        // Smooth enough that the 8x8 candidate wins somewhere, detailed
        // enough that it does not always.
        let mut y = vec![0u8; 16 * 16];
        let mut cb = vec![0u8; 16 * 16];
        let mut cr = vec![0u8; 16 * 16];
        for r in 0..16 {
            for c in 0..16 {
                let smooth = (40 + 5 * c + 3 * r) as u8;
                y[r * 16 + c] = smooth.wrapping_add(lcg(c, r) / 8);
                cb[r * 16 + c] = smooth.wrapping_add(20);
                cr[r * 16 + c] = smooth.wrapping_sub(20);
            }
        }
        let mut saw_8x8 = false;
        for &(cfi, chroma_h, c444) in &[(1u32, 8usize, false), (2, 16, false), (3, 16, true)] {
            for qp in [10u8, 26, 33, 40] {
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
                    c444,
                    t8x8: true,
                    subparts: false,
                };
                let cpad = if c444 { LUMA_PAD } else { CHROMA_PAD };
                let cw = if c444 { 16 } else { 8 };
                let mut rec = vec![
                    recon_plane(16, 16, LUMA_PAD),
                    recon_plane(cw, chroma_h as u32, cpad),
                    recon_plane(cw, chroma_h as u32, cpad),
                ];
                let mb = MbAvail { left: false, top: false, top_left: false, top_right: false };
                let cstride = cw as usize;
                let (dec, modes) = code_macroblock(
                    &ctx, &mut rec, 0, 0, &y, 16, [&cb, &cr], cstride, mb, &[None; 4], &[None; 4],
                );
                saw_8x8 |= dec.kind == MbKind::I8x8;
                let chosen = dec.kind.is_nxn().then_some(&modes);
                round_trip_fmt(&dec, chosen, qp, cfi, true);
            }
        }
        assert!(saw_8x8, "no configuration chose the 8x8 transform; the test proved nothing");
    }

    /// A hand-built `I_8x8` decision, so the writer's 8x8 syntax is
    /// exercised whatever the mode decision above happens to choose —
    /// including an 8x8 whose coefficients land in only some of its four
    /// sub-scans, which is the case where a sub-scan count of zero has to
    /// travel intact through `nC` (and where a writer that summed them,
    /// as CABAC's neighbour record must, would desync the very next
    /// block's tables).
    #[test]
    fn a_synthetic_i8x8_macroblock_round_trips() {
        use crate::h264::cavlc::sub_block_counts_8x8;
        let mut dec = MbDecision {
            kind: MbKind::I8x8,
            transform_8x8: true,
            luma_pred: [PredMode { use_predicted: true, rem: 0 }; 16],
            chroma_mode: 1,
            ..MbDecision::default()
        };
        // 8x8 blocks 0 and 3 coded. Block 0's coefficients sit only at
        // scan positions congruent to 0 and 2 mod 4, so sub-scans 1 and 3
        // count zero; block 3 is dense.
        let mut b0 = [0i16; 64];
        for i in 0..16 {
            b0[ZIGZAG8X8[4 * i] as usize] = if i % 3 == 0 { 3 } else { 0 };
            b0[ZIGZAG8X8[4 * i + 2] as usize] = if i % 5 == 0 { -2 } else { 0 };
        }
        let mut b3 = [0i16; 64];
        for (i, v) in b3.iter_mut().enumerate() {
            *v = ((i as i16 % 7) - 3) * if i % 2 == 0 { 1 } else { -1 };
        }
        dec.luma.as_flattened_mut()[0..64].copy_from_slice(&b0);
        dec.luma.as_flattened_mut()[192..256].copy_from_slice(&b3);
        for (blk8, b) in [(0usize, &b0), (3, &b3)] {
            let counts = sub_block_counts_8x8(b);
            for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
                dec.nz_luma[r] = counts[sub];
            }
        }
        assert!(
            quad_rasters(0).iter().any(|&r| dec.nz_luma[r] == 0),
            "the interesting case is an 8x8 with an empty sub-scan"
        );
        dec.cbp_luma = 0b1001;
        dec.chroma_dc[0][0] = 3;
        dec.chroma_dc[1][1] = 2;
        dec.chroma_ac[0][2][5] = -4;
        dec.nz_chroma[0][2] = 1;
        dec.cbp_chroma = 2;
        round_trip_fmt(&dec, Some(&[2u8; 16]), 28, 1, true);
    }

    /// The inter placement of the flag: after `coded_block_pattern`, and
    /// only when some luma block is coded. Both states of the flag, and a
    /// macroblock with no luma residual at all — where the element is
    /// absent from the wire and a decoder infers zero.
    #[test]
    fn a_p16_macroblock_with_the_8x8_transform_round_trips() {
        use crate::h264::cavlc::sub_block_counts_8x8;
        for (t8x8, coded) in [(true, true), (false, true), (false, false)] {
            let mut dec = InterDecision {
                mvd: [crate::h264::frame::Mv::new(5, -9), crate::h264::frame::Mv::ZERO,
                      crate::h264::frame::Mv::ZERO, crate::h264::frame::Mv::ZERO],
                transform_8x8: t8x8 && coded,
                ..InterDecision::default()
            };
            if coded {
                if t8x8 {
                    let mut b = [0i16; 64];
                    for (i, v) in b.iter_mut().enumerate() {
                        *v = ((i as i16 % 5) - 2) * if i % 3 == 0 { 2 } else { -1 };
                    }
                    dec.luma.as_flattened_mut()[64..128].copy_from_slice(&b);
                    let counts = sub_block_counts_8x8(&b);
                    for (sub, &r) in quad_rasters(1).iter().enumerate() {
                        dec.nz_luma[r] = counts[sub];
                    }
                    dec.cbp_luma = 0b0010;
                } else {
                    dec.luma[2][0] = 5;
                    dec.luma[2][7] = -2;
                    dec.nz_luma[2] = 2;
                    dec.cbp_luma = 0b0010;
                }
                dec.chroma_dc[1][0] = -3;
                dec.cbp_chroma = 1;
            }

            let mut st = NzState::new(1, 2, false);
            let mut w = BitWriter::new();
            write_p16_macroblock(&mut w, &dec, &mut st, 0, false, false, true);
            w.rbsp_trailing_bits();
            let rbsp = w.into_rbsp();

            let ctx = SliceCtx { transform_8x8_mode: true, ..p_ctx() };
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
            assert_eq!(layer.kind, DecKind::Inter16x16);
            assert_eq!(layer.transform_8x8, dec.transform_8x8, "t8x8={t8x8} coded={coded}");
            assert_eq!(layer.cbp, dec.cbp_luma | (dec.cbp_chroma << 4));
            assert_eq!(layer.nz[0], dec.nz_luma, "t8x8={t8x8} coded={coded} luma nz");
            // The reader scales as it parses, so the levels come back
            // dequantised — and asking *it* for the table and shift is
            // what pins the 8x8 inter scaling list (index 1, since the
            // 8x8 lists run `2 * plane + inter`) and the `qP / 6` shift.
            let mbdq = crate::h264::mb::MbDequant::for_mb(
                &dq, &ctx, [0, 0], DecKind::Inter16x16, layer.qp,
            )
            .expect("not lossless");
            let (table, shift) = mbdq.q8[0];
            for i in 0..256 {
                let want = if dec.transform_8x8 {
                    crate::h264::mb::dequant_level(
                        dec.luma.as_flattened()[i] as i32,
                        table[i % 64],
                        shift,
                    )
                } else {
                    let (blk, k) = (i / 16, i % 16);
                    crate::h264::mb::dequant_level(
                        dec.luma[blk][k] as i32,
                        mbdq.q4[0].0[k],
                        mbdq.q4[0].1,
                    )
                };
                assert_eq!(layer.coef[0][i], want, "t8x8={t8x8} coded={coded} coeff {i}");
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
