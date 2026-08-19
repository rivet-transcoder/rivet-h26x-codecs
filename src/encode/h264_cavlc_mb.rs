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
use crate::dsp::Cpu;
use crate::dsp::distortion::DistortionDsp;
use crate::dsp::h264::H264Dsp;
use crate::dsp::h264_enc::{H264EncDsp, Quant};
use crate::encode::h264_intra::{IntraCtx, MbAvail, MbDecision, MbKind, code_macroblock};
use crate::encode::h264_syntax::{Geometry, Plane, Recon};
use crate::h264::cavlc::{SCAN_CHROMA_DC, write_residual_block_cavlc};
use crate::h264::mb::{chroma_qp, raster_of_blk};
use crate::h264::sps::ScalingLists;
use crate::h264::tables::{
    GOLOMB_TO_INTRA4X4_CBP, GOLOMB_TO_INTRA4X4_CBP_GRAY, SCAN_CHROMA_DC_422, ZIGZAG4X4,
};
use crate::h264::transform::Dequant;
use crate::picture::ChromaFormat;

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

/// The kernels and derived tables the intra transform path runs on, built
/// once per encoder. Nothing in here is CAVLC-specific — the CABAC intra
/// writer will want the same set — but this module is its first user.
pub struct IntraTools {
    dsp: H264Dsp<u8>,
    enc: H264EncDsp,
    dist: DistortionDsp<u8>,
    quant: Quant,
    dequant: Dequant,
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
fn write_macroblock(
    w: &mut BitWriter,
    dec: &MbDecision,
    st: &mut NzState,
    mb_x: usize,
    left: bool,
    top: bool,
) {
    let chroma = st.rows != 0;
    let i16x16 = dec.kind == MbKind::I16x16;
    let cbp = (dec.cbp_luma | (dec.cbp_chroma << 4)) as usize;

    // mb_type (Table 7-11): I_NxN is 0; the I_16x16 types encode the
    // prediction mode and both halves of the coded block pattern.
    match dec.kind {
        MbKind::I4x4 => w.ue(0),
        MbKind::I16x16 => w.ue(
            1 + dec.intra16_mode as u32
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

    // The residual, mirroring `parse_residual_luma_like` for plane 0 and
    // then the chroma of `parse_residual_cavlc`. `cur` / `curc` are the
    // within-macroblock counts decoded so far, which is what the reader's
    // `nC` reads for a block whose left or top neighbour is in this same
    // macroblock.
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

    if i16x16 {
        // The DC block first. Its own count is not stored anywhere — the
        // reader discards it too — so the return is deliberately dropped.
        let nc = luma_nc(&cur, st, 0, 0);
        let _ = write_residual_block_cavlc(w, nc, &widen(&dec.luma_dc), &ZIGZAG4X4, 0, 15, 16);
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
            let lv = widen(&dec.luma[raster]);
            // I_16x16 AC blocks start at scan position one — the DC went
            // in the block above — and so carry at most fifteen.
            let n = if i16x16 {
                write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 1, 15, 15)
            } else {
                write_residual_block_cavlc(w, nc, &lv, &ZIGZAG4X4, 0, 15, 16)
            };
            debug_assert_eq!(
                n, dec.nz_luma[raster] as usize,
                "luma block ({bx},{by}): the decision's count disagrees with the writer's"
            );
            cur[raster] = n as u8;
        }
    }
    if chroma && cbp & 0x30 != 0 {
        for comp in 0..2 {
            let dc = widen_dc(&dec.chroma_dc[comp]);
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
                    let lv = widen(&dec.chroma_ac[comp][blk]);
                    let n = write_residual_block_cavlc(w, nc_of(a, b), &lv, &ZIGZAG4X4, 1, 15, 15);
                    debug_assert_eq!(
                        n, dec.nz_chroma[comp][blk] as usize,
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

/// Code and write every macroblock of an all-intra CAVLC picture, leaving
/// the reconstruction in `rec`. The slice header is already written; the
/// caller closes the RBSP.
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
    debug_assert!(g.chroma != ChromaFormat::Yuv444, "ChromaArrayType 3 has no path here");
    let (cw, ch) = g.chroma_mb();
    let chroma_h = ch as usize;
    let rows = chroma_h / 4;
    // 8-bit only (the encoder refuses deeper at construction), and the PPS
    // writes both chroma QP offsets as zero.
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

    let mut st = NzState::new(mbs_wide, rows);
    // The neighbouring macroblocks' 4x4 modes along the shared edges, for
    // the prediction of 8.3.1.1: `Some(2)` for an available macroblock
    // that was not I_NxN (see the module documentation), `None` only where
    // there is no macroblock — though an absent side is never read, since
    // the availability flags gate it first.
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
                &ctx,
                rec,
                mb_x,
                mb_y,
                &src_y,
                luma_stride,
                [&src_cb, &src_cr],
                chroma_stride,
                mb,
                &left_modes,
                &top_modes[mb_x],
            );
            write_macroblock(w, &dec, &mut st, mb_x, mb.left, mb.top);
            (left_modes, top_modes[mb_x]) = match dec.kind {
                MbKind::I4x4 => (
                    [Some(modes[3]), Some(modes[7]), Some(modes[11]), Some(modes[15])],
                    [Some(modes[12]), Some(modes[13]), Some(modes[14]), Some(modes[15])],
                ),
                MbKind::I16x16 => ([Some(2); 4], [Some(2); 4]),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::encode::h264_intra::PredMode;
    use crate::encode::h264_syntax::recon_plane;
    use crate::h264::SliceType;
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
        write_macroblock(&mut w, dec, &mut st, 0, false, false);
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
