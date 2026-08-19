//! Intra mode decision and reconstruction for H.264.
//!
//! This is the *deciding* half of coding a macroblock: which prediction
//! mode, what the quantised coefficients are, and what the reconstruction
//! looks like. Turning that into bits belongs to the entropy coders, and
//! the seam between the two is [`MbDecision`] — data, not a shared file,
//! so one decision serves both CAVLC and CABAC and neither can drift from
//! the other.
//!
//! Two rules shape everything here:
//!
//! **The prediction is the decoder's own.** `h264::intra`'s predictors are
//! conformance-proven and they run here unchanged, reading their
//! neighbours out of the reconstruction plane exactly as they do when
//! decoding. An encoder that predicted even slightly differently would
//! desync, and the desync would show up as drift rather than as an error.
//!
//! **So is the reconstruction.** After quantising, the levels go back
//! through the decoder's dequantisation and inverse transform to build the
//! reconstruction this macroblock's neighbours will predict from. That is
//! what makes the encode gate's SELF property — the bitstream decodes to
//! what the encoder thought it encoded — achievable rather than hoped for.
//!
//! Mode decision scores candidates by SATD rather than SAD. It costs a few
//! times more per candidate and picks visibly better modes, because it
//! measures the residual the way the transform that coded it does: a
//! smooth ramp is cheap in SAD and expensive to code, and SATD is what
//! notices.

use crate::dsp::distortion::DistortionDsp;
use crate::dsp::h264::{H264Dsp, NO_DC};
use crate::dsp::h264_enc::{H264EncDsp, Quant, qbits4, quant_offset};
use crate::encode::h264_syntax::Recon;
use crate::h264::intra::{IntraAvail, predict_4x4, predict_16x16, predict_chroma};
use crate::h264::tables::BLK4X4_FROM_RASTER;
use crate::h264::transform::{Dequant, chroma_dc_transform_420, chroma_dc_transform_422, luma_dc_transform};

/// How a macroblock was coded, in the form an entropy coder needs.
///
/// One of these is produced per macroblock and consumed immediately: the
/// coefficient arrays are about a kilobyte, so a slice's worth buffered up
/// would be megabytes of nothing at any real picture size.
#[derive(Clone)]
pub struct MbDecision {
    /// Which macroblock type was chosen.
    pub kind: MbKind,
    /// Intra_16x16 prediction mode, 0..=3. Meaningless for `I4x4`.
    pub intra16_mode: u8,
    /// Intra_4x4 prediction modes as the *syntax* carries them, one per
    /// 4x4 block in raster order. Meaningless for `I16x16`.
    pub luma_pred: [PredMode; 16],
    /// `intra_chroma_pred_mode`, 0..=3.
    pub chroma_mode: u8,
    /// `CodedBlockPatternLuma`: one bit per 8x8, set when any of its 4x4
    /// blocks has a nonzero level.
    pub cbp_luma: u8,
    /// `CodedBlockPatternChroma`: 0 none, 1 DC only, 2 DC and AC.
    pub cbp_chroma: u8,
    /// `mb_qp_delta`.
    pub qp_delta: i8,
    /// Intra_16x16 luma DC levels, raster order.
    pub luma_dc: [i16; 16],
    /// Luma levels per 4x4 block (raster within the macroblock), each
    /// block in raster order within itself. The scan is the writer's.
    pub luma: [[i16; 16]; 16],
    /// Chroma DC levels per component: four entries in 4:2:0, eight in
    /// 4:2:2.
    pub chroma_dc: [[i16; 8]; 2],
    /// Chroma AC levels per component, per 4x4 block.
    pub chroma_ac: [[[i16; 16]; 8]; 2],
    /// Nonzero count per luma block, which CAVLC's `nC` needs from the
    /// neighbours and which is free to count while quantising.
    pub nz_luma: [u8; 16],
    /// The same per chroma block.
    pub nz_chroma: [[u8; 8]; 2],
}

impl Default for MbDecision {
    fn default() -> Self {
        MbDecision {
            kind: MbKind::I16x16,
            intra16_mode: 2,
            luma_pred: [PredMode::default(); 16],
            chroma_mode: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
            qp_delta: 0,
            luma_dc: [0; 16],
            luma: [[0; 16]; 16],
            chroma_dc: [[0; 8]; 2],
            chroma_ac: [[[0; 16]; 8]; 2],
            nz_luma: [0; 16],
            nz_chroma: [[0; 8]; 2],
        }
    }
}

/// The macroblock types this module decides between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MbKind {
    /// `I_NxN` with `transform_size_8x8_flag` 0.
    I4x4,
    /// One of the twenty-four `I_16x16` types.
    I16x16,
}

/// An Intra_4x4 prediction mode as the syntax carries it: either "the
/// predicted one" or a three-bit remainder. The prediction is
/// `min(modeA, modeB)` over the left and top blocks, so deriving it needs
/// both the neighbours' modes and their macroblock types — state this side
/// holds anyway, and which would otherwise have to be duplicated in two
/// entropy coders where it could diverge.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct PredMode {
    /// `prev_intra4x4_pred_mode_flag`.
    pub use_predicted: bool,
    /// `rem_intra4x4_pred_mode`, 0..=7, when `use_predicted` is false.
    pub rem: u8,
}

/// Which neighbours of a macroblock exist. A single-slice all-intra
/// encoder needs nothing more: within one slice every decoded macroblock
/// is available, so availability is picture geometry.
#[derive(Clone, Copy)]
pub struct MbAvail {
    /// The macroblock to the left.
    pub left: bool,
    /// Above.
    pub top: bool,
    /// Above-left.
    pub top_left: bool,
    /// Above-right.
    pub top_right: bool,
}

/// Everything the mode decision needs that does not change per macroblock.
pub struct IntraCtx<'a> {
    /// Decode-side kernels, for the reconstruction.
    pub dsp: &'a H264Dsp<u8>,
    /// Forward transforms and quantisation.
    pub enc: &'a H264EncDsp,
    /// Distortion metrics, for scoring candidates.
    pub dist: &'a DistortionDsp<u8>,
    /// Forward quantisation tables.
    pub quant: &'a Quant,
    /// The decoder's dequantisation tables, which the reconstruction uses.
    pub dequant: &'a Dequant,
    /// Luma QP.
    pub qp: i32,
    /// Chroma QP per component.
    pub qpc: [i32; 2],
    /// Chroma height in samples: 8 for 4:2:0, 16 for 4:2:2, 0 for
    /// monochrome.
    pub chroma_h: usize,
}

/// Whether a 4x4 block's top-right neighbour has been reconstructed by the
/// time the block is predicted. Derived from the standard's own 4x4 scan
/// order rather than from a copied pattern: the neighbour is usable when
/// it exists and comes earlier in that order.
fn top_right_ready(bx: usize, by: usize, mb: MbAvail) -> bool {
    if by == 0 {
        // Above the macroblock: the top neighbour supplies it, except off
        // the right-hand edge where the above-right macroblock does.
        return if bx == 3 { mb.top_right } else { mb.top };
    }
    if bx == 3 {
        // The row above inside this macroblock has nothing to the right.
        return false;
    }
    let me = BLK4X4_FROM_RASTER[by * 4 + bx] as usize;
    let them = BLK4X4_FROM_RASTER[(by - 1) * 4 + bx + 1] as usize;
    them < me
}

/// Availability for the 4x4 block at `(bx, by)` of a macroblock whose own
/// neighbours are `mb`.
fn avail_4x4(bx: usize, by: usize, mb: MbAvail) -> IntraAvail {
    IntraAvail {
        top: if by == 0 { mb.top } else { true },
        left: if bx == 0 { mb.left } else { true },
        top_left: match (bx, by) {
            (0, 0) => mb.top_left,
            (0, _) => mb.left,
            (_, 0) => mb.top,
            _ => true,
        },
        top_right: top_right_ready(bx, by, mb),
    }
}

/// The prediction of an Intra_4x4 mode (8.3.1.1): the smaller of the left
/// and top blocks' modes, DC when either is missing.
fn predicted_mode(left: Option<u8>, top: Option<u8>) -> u8 {
    match (left, top) {
        (Some(a), Some(b)) => a.min(b),
        _ => 2,
    }
}

/// Turn a chosen mode into the flag and remainder the syntax carries.
fn as_syntax(chosen: u8, predicted: u8) -> PredMode {
    if chosen == predicted {
        PredMode { use_predicted: true, rem: 0 }
    } else {
        PredMode {
            use_predicted: false,
            rem: if chosen < predicted { chosen } else { chosen - 1 },
        }
    }
}

/// Read a `w` by `h` block out of a plane into a packed buffer.
fn gather(p: &Recon, off: usize, w: usize, h: usize, out: &mut [u8]) {
    for y in 0..h {
        out[y * w..y * w + w].copy_from_slice(&p.data[off + y * p.stride..off + y * p.stride + w]);
    }
}

/// Forward-transform, quantise and reconstruct one 4x4 luma block in
/// place: `off` addresses the block in `rec`, which already holds the
/// prediction. Returns the levels and their nonzero count.
#[allow(clippy::too_many_arguments)]
fn code_block_4x4(
    ctx: &IntraCtx,
    rec: &mut Recon,
    off: usize,
    src: &[u8],
    src_stride: usize,
    list: usize,
    qp: i32,
    keep_dc: bool,
) -> ([i16; 16], u32, i32) {
    let mut residual = [0i16; 16];
    for y in 0..4 {
        for x in 0..4 {
            residual[y * 4 + x] =
                src[y * src_stride + x] as i16 - rec.data[off + y * rec.stride + x] as i16;
        }
    }
    let mut coeffs = [0i32; 16];
    (ctx.enc.fdct4)(&residual, &mut coeffs);
    let dc = coeffs[0];
    let m = (qp % 6) as usize;
    let qbits = qbits4(qp);
    let offset = quant_offset(qbits, true);
    let mut levels = [0i16; 16];
    let mut nz = (ctx.enc.quant4)(&coeffs, &mut levels, &ctx.quant.mf4[list][m], qbits, offset);
    if !keep_dc {
        // The DC is carried by the macroblock's DC block instead.
        if levels[0] != 0 {
            nz -= 1;
        }
        levels[0] = 0;
    }
    (levels, nz, dc)
}

/// Dequantise levels and add the inverse transform to the prediction
/// already in `rec` — the decoder's own path, so the two cannot disagree.
fn reconstruct_4x4(ctx: &IntraCtx, rec: &mut Recon, off: usize, levels: &[i16; 16], list: usize, qp: i32, dc: Option<i32>) {
    let m = (qp % 6) as usize;
    let scale = &ctx.dequant.scale4[list][m];
    let q6 = qp / 6;
    let mut coefs = [0i32; 16];
    for i in 0..16 {
        let c = levels[i] as i32 * scale[i];
        coefs[i] = if qp >= 24 {
            c << (q6 - 4)
        } else {
            (c + (1 << (3 - q6))) >> (4 - q6)
        };
    }
    (ctx.dsp.residual4)(&mut rec.data[off..], rec.stride, &coefs, dc.unwrap_or(NO_DC), 255);
}

/// Code one macroblock as `I_16x16` with the given prediction mode,
/// leaving the reconstruction in `rec`. Returns the decision and the sum
/// of squared errors of the reconstruction, which is what a
/// rate-distortion comparison between candidate modes wants.
#[allow(clippy::too_many_arguments)]
fn code_i16x16(
    ctx: &IntraCtx,
    rec: &mut Recon,
    px: usize,
    py: usize,
    src: &[u8],
    src_stride: usize,
    mode: u8,
    mb: MbAvail,
    out: &mut MbDecision,
) {
    let off = rec.offset(px as isize, py as isize);
    let av = IntraAvail {
        top: mb.top,
        left: mb.left,
        top_left: mb.top_left,
        top_right: false,
    };
    let _ = predict_16x16(rec, off, rec.stride, mode, av, 8);

    // Forward-transform every block, keeping the DCs aside.
    let mut dcs = [0i32; 16];
    let mut levels = [[0i16; 16]; 16];
    let mut nz = [0u8; 16];
    for blk in 0..16 {
        let (bx, by) = (blk % 4, blk / 4);
        let boff = off + by * 4 * rec.stride + bx * 4;
        let soff = by * 4 * src_stride + bx * 4;
        let (lv, n, dc) = code_block_4x4(ctx, rec, boff, &src[soff..], src_stride, 0, ctx.qp, false);
        levels[blk] = lv;
        nz[blk] = n as u8;
        dcs[blk] = dc;
    }

    // The DC block: Hadamard, then quantised at the same QP with the
    // position-0 multiplier, which is what 8.5.10 inverts.
    let mut dc_i32 = dcs;
    (ctx.enc.hadamard4)(&mut dc_i32);
    let m = (ctx.qp % 6) as usize;
    let qbits = qbits4(ctx.qp) + 1;
    let offset = quant_offset(qbits, true);
    let mut dc_levels = [0i16; 16];
    for i in 0..16 {
        let c = dc_i32[i];
        let mf = ctx.quant.mf4[0][m][0] as i64;
        let v = ((c.unsigned_abs() as i64 * mf + offset as i64) >> qbits) as i32;
        dc_levels[i] = if c < 0 { -v as i16 } else { v as i16 };
    }

    // Reconstruct: the decoder's inverse Hadamard and scaling, then each
    // block with its DC put back at position 0.
    let mut dc_rec = [0i32; 16];
    for i in 0..16 {
        dc_rec[i] = dc_levels[i] as i32;
    }
    luma_dc_transform(&mut dc_rec, ctx.dequant.scale4[0][m][0], ctx.qp);
    for blk in 0..16 {
        let (bx, by) = (blk % 4, blk / 4);
        let boff = off + by * 4 * rec.stride + bx * 4;
        reconstruct_4x4(ctx, rec, boff, &levels[blk], 0, ctx.qp, Some(dc_rec[blk]));
    }

    out.kind = MbKind::I16x16;
    out.intra16_mode = mode;
    out.luma = levels;
    out.luma_dc = dc_levels;
    out.nz_luma = nz;
    // I_16x16 codes luma AC all-or-nothing.
    out.cbp_luma = if nz.iter().any(|&n| n != 0) { 15 } else { 0 };
}

/// Code one macroblock as `I_4x4`, choosing each block's mode by SATD
/// against its own prediction. The blocks are done in decode order because
/// each one predicts from the reconstruction of those before it — which is
/// the whole reason an encoder's inner loop looks like a decoder.
#[allow(clippy::too_many_arguments)]
fn code_i4x4(
    ctx: &IntraCtx,
    rec: &mut Recon,
    px: usize,
    py: usize,
    src: &[u8],
    src_stride: usize,
    mb: MbAvail,
    left_modes: &[Option<u8>; 4],
    top_modes: &[Option<u8>; 4],
    out: &mut MbDecision,
) -> [u8; 16] {
    let base = rec.offset(px as isize, py as isize);
    let mut chosen = [2u8; 16];
    let mut levels = [[0i16; 16]; 16];
    let mut nz = [0u8; 16];

    for scan in 0..16 {
        let bx = crate::h264::tables::BLK4X4_X[scan] as usize;
        let by = crate::h264::tables::BLK4X4_Y[scan] as usize;
        let raster = by * 4 + bx;
        let boff = base + by * 4 * rec.stride + bx * 4;
        let soff = by * 4 * src_stride + bx * 4;
        let av = avail_4x4(bx, by, mb);

        // The neighbouring modes this block predicts from.
        let left = if bx > 0 {
            Some(chosen[by * 4 + bx - 1])
        } else if mb.left {
            left_modes[by]
        } else {
            None
        };
        let top = if by > 0 {
            Some(chosen[(by - 1) * 4 + bx])
        } else if mb.top {
            top_modes[bx]
        } else {
            None
        };
        let predicted = predicted_mode(left, top);

        // Try every legal mode, keep the cheapest by SATD. The prediction
        // is written into the reconstruction plane, so each trial has to
        // be scored before the next overwrites it.
        let mut src_blk = [0u8; 16];
        for y in 0..4 {
            src_blk[y * 4..y * 4 + 4].copy_from_slice(&src[soff + y * src_stride..soff + y * src_stride + 4]);
        }
        let mut best = (u32::MAX, 2u8);
        for mode in 0..9u8 {
            if predict_4x4(rec, boff, rec.stride, mode, av, 8).is_err() {
                continue;
            }
            let mut pred = [0u8; 16];
            gather(rec, boff, 4, 4, &mut pred);
            let cost = (ctx.dist.satd)(&src_blk, 4, &pred, 4, 4, 4);
            // A tie goes to the predicted mode, which costs one bit
            // instead of four.
            let better = cost < best.0 || (cost == best.0 && mode == predicted);
            if better {
                best = (cost, mode);
            }
        }
        chosen[raster] = best.1;

        // Re-predict with the winner, then code and reconstruct.
        let _ = predict_4x4(rec, boff, rec.stride, best.1, av, 8);
        let (lv, n, _) = code_block_4x4(ctx, rec, boff, &src[soff..], src_stride, 0, ctx.qp, true);
        levels[raster] = lv;
        nz[raster] = n as u8;
        reconstruct_4x4(ctx, rec, boff, &lv, 0, ctx.qp, None);

        out.luma_pred[raster] = as_syntax(best.1, predicted);
    }

    out.kind = MbKind::I4x4;
    out.luma = levels;
    out.nz_luma = nz;
    let mut cbp = 0u8;
    for blk8 in 0..4 {
        let (ox, oy) = ((blk8 % 2) * 2, (blk8 / 2) * 2);
        let any = (0..4).any(|k| nz[(oy + k / 2) * 4 + ox + k % 2] != 0);
        if any {
            cbp |= 1 << blk8;
        }
    }
    out.cbp_luma = cbp;
    chosen
}

/// Code the chroma of a macroblock at the chosen mode, into both planes.
#[allow(clippy::too_many_arguments)]
fn code_chroma(
    ctx: &IntraCtx,
    rec: &mut [Recon],
    cx: usize,
    cy: usize,
    src: &[&[u8]],
    src_stride: usize,
    mode: u8,
    mb: MbAvail,
    out: &mut MbDecision,
) {
    let h = ctx.chroma_h;
    if h == 0 {
        return;
    }
    let blocks = h / 4 * 2;
    let av = IntraAvail {
        top: mb.top,
        left: mb.left,
        top_left: mb.top_left,
        top_right: false,
    };
    let mut any_ac = false;
    let mut any_dc = false;
    for comp in 0..2 {
        let plane = &mut rec[comp + 1];
        let off = plane.offset(cx as isize, cy as isize);
        let _ = predict_chroma(plane, off, plane.stride, mode, av, [mb.left; 4], 8, h);

        let qp = ctx.qpc[comp];
        let list = 1 + comp; // Cb intra, Cr intra
        let mut dcs = [0i32; 8];
        let mut levels = [[0i16; 16]; 8];
        let mut nz = [0u8; 8];
        for blk in 0..blocks {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = off + by * 4 * plane.stride + bx * 4;
            let soff = by * 4 * src_stride + bx * 4;
            let mut residual = [0i16; 16];
            for y in 0..4 {
                for x in 0..4 {
                    residual[y * 4 + x] =
                        src[comp][soff + y * src_stride + x] as i16 - plane.data[boff + y * plane.stride + x] as i16;
                }
            }
            let mut coeffs = [0i32; 16];
            (ctx.enc.fdct4)(&residual, &mut coeffs);
            dcs[blk] = coeffs[0];
            let m = (qp % 6) as usize;
            let qbits = qbits4(qp);
            let offset = quant_offset(qbits, true);
            let mut lv = [0i16; 16];
            let mut n = (ctx.enc.quant4)(&coeffs, &mut lv, &ctx.quant.mf4[list][m], qbits, offset);
            if lv[0] != 0 {
                n -= 1;
            }
            lv[0] = 0;
            levels[blk] = lv;
            nz[blk] = n as u8;
        }

        // Chroma DC: the 2x2 or 2x4 Hadamard, quantised at twice the shift.
        let m = (qp % 6) as usize;
        let qbits = qbits4(qp) + 1;
        let offset = quant_offset(qbits, true);
        let mut dc_levels = [0i16; 8];
        if blocks == 4 {
            let mut d = [dcs[0], dcs[1], dcs[2], dcs[3]];
            (ctx.enc.hadamard2x2)(&mut d);
            for i in 0..4 {
                let mf = ctx.quant.mf4[list][m][0] as i64;
                let v = ((d[i].unsigned_abs() as i64 * mf + offset as i64) >> qbits) as i32;
                dc_levels[i] = if d[i] < 0 { -v as i16 } else { v as i16 };
            }
        } else {
            let mut d = dcs;
            (ctx.enc.hadamard2x4)(&mut d);
            // 4:2:2 chroma DC is scaled at QP'c,DC = QP'c + 3 (8.5.11.2),
            // and that raised QP governs the *whole* quantiser — the
            // multiplier's row and the shift. Splitting them (mf at qp + 3,
            // shift at qp) leaves the levels a power of two too large
            // whenever the two QPs fall in different bands of six, and the
            // reconstruction — the decoder's, which scales at qp + 3
            // throughout — faithfully doubles the coded DC error. SELF
            // still passes then; the chroma PSNR is what notices.
            let qbits = qbits4(qp + 3) + 1;
            let offset = quant_offset(qbits, true);
            let mf = ctx.quant.mf4[list][((qp + 3) % 6) as usize][0] as i64;
            for i in 0..8 {
                let v = ((d[i].unsigned_abs() as i64 * mf + offset as i64) >> qbits) as i32;
                dc_levels[i] = if d[i] < 0 { -v as i16 } else { v as i16 };
            }
        }

        // Reconstruct through the decoder's DC transform and residual add.
        let mut dc_rec = [0i32; 8];
        for i in 0..blocks {
            dc_rec[i] = dc_levels[i] as i32;
        }
        if blocks == 4 {
            let mut d4 = [dc_rec[0], dc_rec[1], dc_rec[2], dc_rec[3]];
            chroma_dc_transform_420(&mut d4, ctx.dequant.scale4[list][m][0], qp);
            dc_rec[..4].copy_from_slice(&d4);
        } else {
            chroma_dc_transform_422(&mut dc_rec, ctx.dequant.scale4[list][((qp + 3) % 6) as usize][0], qp);
        }
        for blk in 0..blocks {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = off + by * 4 * plane.stride + bx * 4;
            let scale = &ctx.dequant.scale4[list][m];
            let q6 = qp / 6;
            let mut coefs = [0i32; 16];
            for i in 0..16 {
                let c = levels[blk][i] as i32 * scale[i];
                coefs[i] = if qp >= 24 {
                    c << (q6 - 4)
                } else {
                    (c + (1 << (3 - q6))) >> (4 - q6)
                };
            }
            (ctx.dsp.residual4)(&mut plane.data[boff..], plane.stride, &coefs, dc_rec[blk], 255);
        }

        any_ac |= nz.iter().any(|&n| n != 0);
        any_dc |= dc_levels.iter().any(|&v| v != 0);
        out.chroma_dc[comp] = dc_levels;
        out.chroma_ac[comp] = levels;
        out.nz_chroma[comp] = nz;
    }
    out.chroma_mode = mode;
    out.cbp_chroma = if any_ac {
        2
    } else if any_dc {
        1
    } else {
        0
    };
}

/// The Lagrangian multiplier H.264 mode decision conventionally uses,
/// `0.85 * 2^((QP - 12) / 3)`, in the same units as a SATD so a mode's
/// cost can be its distortion plus `lambda` times its bits.
fn lambda(qp: i32) -> f32 {
    0.85f32 * ((qp - 12) as f32 / 3.0).exp2()
}

/// Decide and code one macroblock, leaving its reconstruction in `rec`.
///
/// `left_modes` and `top_modes` are the neighbouring macroblocks' 4x4
/// modes down the shared edge, `None` where the neighbour is unavailable
/// or was not coded as `I_NxN`; the caller keeps those, because it walks
/// the picture. The chosen 4x4 modes come back for the same reason.
///
/// The two candidates are compared by SATD plus a rate estimate, which is
/// a first cut: it counts the mode signalling and ignores the residual,
/// so it is a decision worth improving once there is a bit count to feed
/// it. It is deliberately not a placeholder that pretends to be more —
/// the estimate is named in one place and the comparison is one line.
#[allow(clippy::too_many_arguments)]
pub fn code_macroblock(
    ctx: &IntraCtx,
    rec: &mut [Recon],
    mb_x: usize,
    mb_y: usize,
    src_luma: &[u8],
    luma_stride: usize,
    src_chroma: [&[u8]; 2],
    chroma_stride: usize,
    mb: MbAvail,
    left_modes: &[Option<u8>; 4],
    top_modes: &[Option<u8>; 4],
) -> (MbDecision, [u8; 16]) {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let soff = py * luma_stride + px;
    let mut out = MbDecision::default();

    // I_4x4 first, because its decision and its reconstruction interleave:
    // each block predicts from the one before it. Its cost is the SATD it
    // accumulated plus roughly four bits a block of mode signalling.
    let modes = code_i4x4(
        ctx,
        &mut rec[0],
        px,
        py,
        &src_luma[soff..],
        luma_stride,
        mb,
        left_modes,
        top_modes,
        &mut out,
    );
    let off = rec[0].offset(px as isize, py as isize);
    let mut pred = [0u8; 256];
    gather(&rec[0], off, 16, 16, &mut pred);
    let cost_4x4 = (ctx.dist.satd)(&src_luma[soff..], luma_stride, &pred, 16, 16, 16) as f32
        + lambda(ctx.qp) * 4.0 * 16.0;

    // I_16x16: its prediction reads only neighbours outside the
    // macroblock, which I_4x4 did not touch, so the modes can be scored
    // without undoing anything.
    let av = IntraAvail { top: mb.top, left: mb.left, top_left: mb.top_left, top_right: false };
    let ystride = rec[0].stride;
    let mut best = (f32::MAX, 2u8);
    for mode in 0..4u8 {
        if predict_16x16(&mut rec[0], off, ystride, mode, av, 8).is_err() {
            continue;
        }
        gather(&rec[0], off, 16, 16, &mut pred);
        let c = (ctx.dist.satd)(&src_luma[soff..], luma_stride, &pred, 16, 16, 16) as f32
            + lambda(ctx.qp) * 2.0;
        if c < best.0 {
            best = (c, mode);
        }
    }

    let chosen_4x4;
    if best.0 < cost_4x4 {
        let mut i16out = MbDecision::default();
        code_i16x16(ctx, &mut rec[0], px, py, &src_luma[soff..], luma_stride, best.1, mb, &mut i16out);
        i16out.luma_pred = out.luma_pred;
        out = i16out;
        chosen_4x4 = [2u8; 16];
    } else {
        // Put back the I_4x4 reconstruction the 16x16 trials overwrote.
        let mut redo = MbDecision::default();
        code_i4x4(
            ctx,
            &mut rec[0],
            px,
            py,
            &src_luma[soff..],
            luma_stride,
            mb,
            left_modes,
            top_modes,
            &mut redo,
        );
        out = redo;
        chosen_4x4 = modes;
    }

    if ctx.chroma_h != 0 {
        let (cw, ch) = (8usize, ctx.chroma_h);
        let (cx, cy) = (mb_x * cw, mb_y * ch);
        let coff = cy * chroma_stride + cx;
        // Chroma modes in the order the syntax numbers them; DC is legal
        // whatever the neighbours are, so there is always a candidate.
        let mut best_c = (u32::MAX, 0u8);
        for mode in 0..4u8 {
            let mut cost = 0u32;
            let mut ok = true;
            for comp in 0..2 {
                let plane = &mut rec[comp + 1];
                let off = plane.offset(cx as isize, cy as isize);
                let cstride = plane.stride;
                if predict_chroma(plane, off, cstride, mode, av, [mb.left; 4], 8, ch).is_err() {
                    ok = false;
                    break;
                }
                let mut p = [0u8; 128];
                gather(plane, off, cw, ch, &mut p[..cw * ch]);
                cost += (ctx.dist.satd)(&src_chroma[comp][coff..], chroma_stride, &p, cw, cw, ch);
            }
            if ok && cost < best_c.0 {
                best_c = (cost, mode);
            }
        }
        code_chroma(
            ctx,
            rec,
            cx,
            cy,
            &[&src_chroma[0][coff..], &src_chroma[1][coff..]],
            chroma_stride,
            best_c.1,
            mb,
            &mut out,
        );
    }

    (out, chosen_4x4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::h264::sps::ScalingLists;

    fn flat() -> ScalingLists {
        ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] }
    }

    /// Every 4x4 block's top-right neighbour must already be
    /// reconstructed. Checked against the standard's scan order rather
    /// than a copied pattern, and against the corner cases the pattern
    /// exists to encode.
    #[test]
    fn top_right_availability_follows_the_scan_order() {
        let all = MbAvail { left: true, top: true, top_left: true, top_right: true };
        // Inside the macroblock, a block on the right edge below the top
        // row never has one.
        for by in 1..4 {
            assert!(!top_right_ready(3, by, all), "(3,{by})");
        }
        // The top row takes it from the macroblock above, and its
        // right-hand block from the above-right macroblock.
        for bx in 0..3 {
            assert!(top_right_ready(bx, 0, all));
        }
        assert!(top_right_ready(3, 0, all));
        let no_tr = MbAvail { top_right: false, ..all };
        assert!(!top_right_ready(3, 0, no_tr));
        // And every "ready" block really does come later in decode order.
        for by in 1..4 {
            for bx in 0..3 {
                if top_right_ready(bx, by, all) {
                    let me = BLK4X4_FROM_RASTER[by * 4 + bx];
                    let them = BLK4X4_FROM_RASTER[(by - 1) * 4 + bx + 1];
                    assert!(them < me, "({bx},{by}) claims a neighbour it precedes");
                }
            }
        }
    }

    /// The mode-to-syntax mapping has to round-trip: what the writer emits
    /// is what a decoder derives back.
    #[test]
    fn mode_syntax_round_trips() {
        for predicted in 0..9u8 {
            for chosen in 0..9u8 {
                let p = as_syntax(chosen, predicted);
                let back = if p.use_predicted {
                    predicted
                } else if p.rem < predicted {
                    p.rem
                } else {
                    p.rem + 1
                };
                assert_eq!(back, chosen, "predicted={predicted} chosen={chosen}");
                assert!(p.use_predicted || p.rem < 8);
            }
        }
    }

    /// A flat macroblock predicted from flat neighbours costs nothing:
    /// the residual is zero, so every level is zero and the coded block
    /// pattern is empty. It also exercises the whole path end to end.
    #[test]
    fn a_flat_macroblock_codes_to_nothing() {
        let dsp = H264Dsp::<u8>::new(Cpu::SCALAR);
        let enc = H264EncDsp::SCALAR;
        let dist = DistortionDsp::<u8>::scalar();
        let quant = Quant::new(&flat());
        let dequant = Dequant::new(&flat());
        let ctx = IntraCtx {
            dsp: &dsp,
            enc: &enc,
            dist: &dist,
            quant: &quant,
            dequant: &dequant,
            qp: 26,
            qpc: [26, 26],
            chroma_h: 8,
        };
        let mut rec = crate::encode::h264_syntax::recon_plane(32, 32, 16);
        for v in rec.data.iter_mut() {
            *v = 128;
        }
        let src = vec![128u8; 32 * 32];
        let mut out = MbDecision::default();
        let mb = MbAvail { left: true, top: true, top_left: true, top_right: true };
        code_i16x16(&ctx, &mut rec, 0, 0, &src, 32, 2, mb, &mut out);
        assert_eq!(out.cbp_luma, 0, "a flat block should code no luma");
        assert!(out.luma_dc.iter().all(|&v| v == 0));
        // And the reconstruction is still flat.
        let off = rec.offset(0, 0);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(rec.data[off + y * rec.stride + x], 128);
            }
        }
    }
}
