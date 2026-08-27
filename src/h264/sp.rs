//! SP and SI slice reconstruction (H.264 clause 8.6): the prediction is
//! transformed, the residual levels are added to it in the transform domain
//! and the sum is quantised and dequantised with `QSY` (`QSC` for chroma)
//! before the inverse transform — so that the reconstruction depends only
//! on the transmitted levels and the quantised prediction, which is what
//! lets a switching picture (SP with `sp_for_switch_flag`, or SI) land on
//! the same samples from a different prediction.
//!
//! Two paths: 8.6.1 for the P macroblocks of a non-switching SP slice
//! (the levels are scaled with `QPY` into the prediction's transform
//! coefficients, then everything is requantised with `QSY`), and 8.6.2 for
//! the switching pictures (the prediction is quantised with `QSY` and the
//! levels are added to the quantised values). Written to the JVT reference
//! decoder of the conformance streams' era (JM 8.6, whose SP output is what
//! the suite's reconstructed YUV records) — the current JM pairs the
//! chroma DC levels with the transposed prediction coefficients, and its
//! output is not the reference.
//!
//! Only the flat scaling of the Extended profile exists here: scaling
//! matrices and the 8x8 transform are High-profile tools, and SP / SI
//! slices are Extended-profile ones.

use crate::sample::Sample;

/// `LevelScale(m, i, j)` of the 2003 text (normAdjust4x4): by `m = qP % 6`
/// and the position class — both indices even, one odd, both odd.
const DEQUANT: [[i32; 3]; 6] = [
    [10, 13, 16],
    [11, 14, 18],
    [13, 16, 20],
    [14, 18, 23],
    [16, 20, 25],
    [18, 23, 29],
];

/// The forward quantisation scale of 8.6.1 (`LevelScale2` / JM's
/// `quant_coef`), by `qP % 6` and position class.
const QUANT: [[i32; 3]; 6] = [
    [13107, 8066, 5243],
    [11916, 7490, 4660],
    [10082, 6554, 4194],
    [9362, 5825, 3647],
    [8192, 5243, 3355],
    [7282, 4559, 2893],
];

/// `A_ij` (8.6.1): 16 for both indices even, 25 for both odd, 20 otherwise.
const A: [i32; 3] = [16, 20, 25];

/// The position class of `(i, j)`: 0 both even, 1 mixed, 2 both odd.
#[inline(always)]
fn class(i: usize, j: usize) -> usize {
    (i & 1) + (j & 1)
}

/// The forward 4x4 core transform `Cf X Cf^T` of a raster block.
fn forward4x4(p: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let (p0, p1, p2, p3) = (p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
        let (t0, t1, t2, t3) = (p0 + p3, p1 + p2, p1 - p2, p0 - p3);
        t[i * 4] = t0 + t1;
        t[i * 4 + 1] = (t3 << 1) + t2;
        t[i * 4 + 2] = t0 - t1;
        t[i * 4 + 3] = t3 - (t2 << 1);
    }
    let mut out = [0i32; 16];
    for j in 0..4 {
        let (p0, p1, p2, p3) = (t[j], t[4 + j], t[8 + j], t[12 + j]);
        let (t0, t1, t2, t3) = (p0 + p3, p1 + p2, p1 - p2, p0 - p3);
        out[j] = t0 + t1;
        out[4 + j] = (t3 << 1) + t2;
        out[8 + j] = t0 - t1;
        out[12 + j] = t3 - (t2 << 1);
    }
    out
}

/// The inverse 4x4 transform (8.5.12.2) with its `(x + 32) >> 6`, in place.
fn inverse4x4(d: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (d0, d1, d2, d3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        tmp[i * 4] = e0 + e3;
        tmp[i * 4 + 1] = e1 + e2;
        tmp[i * 4 + 2] = e1 - e2;
        tmp[i * 4 + 3] = e0 - e3;
    }
    for j in 0..4 {
        let (f0, f1, f2, f3) = (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
        let g0 = f0 + f2;
        let g1 = f0 - f2;
        let g2 = (f1 >> 1) - f3;
        let g3 = f1 + (f3 >> 1);
        d[j] = (g0 + g3 + 32) >> 6;
        d[4 + j] = (g1 + g2 + 32) >> 6;
        d[8 + j] = (g1 - g2 + 32) >> 6;
        d[12 + j] = (g0 - g3 + 32) >> 6;
    }
}

/// `Sign(v) * ((Abs(v) * scale + (1 << (bits - 1))) >> bits)`.
#[inline(always)]
fn quant(v: i32, scale: i32, bits: u32) -> i32 {
    let q = (v.abs() * scale + (1 << (bits - 1))) >> bits;
    if v < 0 { -q } else { q }
}

/// The transform-domain step shared by luma and chroma AC: the prediction
/// coefficient `p` and the level `lev` at class `c` become the dequantised
/// coefficient for the inverse transform.
#[inline(always)]
fn coefficient(p: i32, lev: i32, c: usize, qp: i32, qs: i32, switching: bool) -> i32 {
    let (qs_per, qs_rem) = ((qs / 6) as u32, (qs % 6) as usize);
    let q = if switching {
        // 8.6.2: quantise the prediction with QS, add the level.
        quant(p, QUANT[qs_rem][c], 15 + qs_per) + lev
    } else {
        // 8.6.1: scale the level with QP into the prediction, requantise with QS.
        let (qp_per, qp_rem) = ((qp / 6) as u32, (qp % 6) as usize);
        let s = p + ((lev * DEQUANT[qp_rem][c] * A[c]) << qp_per >> 6);
        quant(s, QUANT[qs_rem][c], 15 + qs_per)
    };
    (q * DEQUANT[qs_rem][c]) << qs_per
}

/// One 4x4 luma block (or, in the chroma path, the AC of one chroma block
/// with the DC already placed): `dst` holds the prediction and receives the
/// reconstruction; `lev` are the block's levels in raster order.
#[allow(clippy::too_many_arguments)]
pub fn luma_block<S: Sample>(
    dst: &mut [S],
    stride: usize,
    lev: &[i32],
    qp: i32,
    qs: i32,
    switching: bool,
    max: i32,
) {
    let mut p = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            p[i * 4 + j] = dst[i * stride + j].to_i32();
        }
    }
    let mut c = forward4x4(&p);
    for i in 0..4 {
        for j in 0..4 {
            let k = i * 4 + j;
            c[k] = coefficient(c[k], lev[k], class(i, j), qp, qs, switching);
        }
    }
    inverse4x4(&mut c);
    for i in 0..4 {
        for j in 0..4 {
            dst[i * stride + j] = S::from_i32(c[i * 4 + j].clamp(0, max));
        }
    }
}

/// One 4:2:0 chroma component of a macroblock: `dst` holds the 8x8
/// prediction and receives the reconstruction; `dc` are the four DC levels
/// (2x2 raster) and `ac[blk]` the AC levels of block `blk` (raster within
/// the block, position 0 ignored). `qpc` / `qsc` are `QPC` and `QSC`.
#[allow(clippy::too_many_arguments)]
pub fn chroma_420<S: Sample>(
    dst: &mut [S],
    stride: usize,
    dc: &[i32],
    ac: &[[i32; 16]],
    qpc: i32,
    qsc: i32,
    switching: bool,
    max: i32,
) {
    // The four blocks' transformed predictions.
    let mut c = [[0i32; 16]; 4];
    for (blk, cb) in c.iter_mut().enumerate() {
        let (bx, by) = (blk % 2, blk / 2);
        let mut p = [0i32; 16];
        for i in 0..4 {
            for j in 0..4 {
                p[i * 4 + j] = dst[(by * 4 + i) * stride + bx * 4 + j].to_i32();
            }
        }
        *cb = forward4x4(&p);
    }
    // DC: the 2x2 Hadamard of the blocks' DC coefficients (f = H P H, so
    // f01 is the horizontal frequency, pairing with the level c01), the
    // levels added as 8.6.1 / 8.6.2 say with the DC's extra bit, then the
    // inverse 2x2 and the >> 1 of the chroma DC scaling.
    let (tl, tr, bl, br) = (c[0][0], c[1][0], c[2][0], c[3][0]);
    let mut f = [
        tl + tr + bl + br,
        tl - tr + bl - br,
        tl + tr - bl - br,
        tl - tr - bl + br,
    ];
    let (qs_per, qs_rem) = ((qsc / 6) as u32, (qsc % 6) as usize);
    for (k, fk) in f.iter_mut().enumerate() {
        let q = if switching {
            quant(*fk, QUANT[qs_rem][0], 16 + qs_per) + dc[k]
        } else {
            let (qp_per, qp_rem) = ((qpc / 6) as u32, (qpc % 6) as usize);
            let s = *fk + ((dc[k] * DEQUANT[qp_rem][0] * A[0]) << qp_per >> 5);
            quant(s, QUANT[qs_rem][0], 16 + qs_per)
        };
        *fk = (q * DEQUANT[qs_rem][0]) << qs_per;
    }
    c[0][0] = (f[0] + f[1] + f[2] + f[3]) >> 1;
    c[1][0] = (f[0] - f[1] + f[2] - f[3]) >> 1;
    c[2][0] = (f[0] + f[1] - f[2] - f[3]) >> 1;
    c[3][0] = (f[0] - f[1] - f[2] + f[3]) >> 1;
    // AC, then the inverse transform of each block.
    for (blk, cb) in c.iter_mut().enumerate() {
        for i in 0..4 {
            for j in 0..4 {
                let k = i * 4 + j;
                if k == 0 {
                    continue;
                }
                cb[k] = coefficient(cb[k], ac[blk][k], class(i, j), qpc, qsc, switching);
            }
        }
        inverse4x4(cb);
        let (bx, by) = (blk % 2, blk / 2);
        for i in 0..4 {
            for j in 0..4 {
                dst[(by * 4 + i) * stride + bx * 4 + j] = S::from_i32(cb[i * 4 + j].clamp(0, max));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_then_inverse_is_the_scaled_identity() {
        // Cf X Cf^T followed by the inverse transform (which carries the
        // 1/64 scale) reproduces a block whose values are multiples of the
        // transform's own scaling — a DC-only block exactly.
        let p = [7i32; 16];
        let mut c = forward4x4(&p);
        assert_eq!(c[0], 7 * 16);
        assert!(c[1..].iter().all(|&v| v == 0));
        // The inverse expects the 8.5.12 scaling: DC * 16 * 4 = 7 * 64 * 16 / 16.
        for v in c.iter_mut() {
            *v *= 4;
        }
        inverse4x4(&mut c);
        assert!(c.iter().all(|&v| v == 7));
    }

    #[test]
    fn zero_levels_requantise_the_prediction_only() {
        // A flat prediction of 100 at QS 24 (qs_per 4, qs_rem 0): the DC
        // coefficient 1600 is quantised with 13107 >> 19 and dequantised
        // with 10 << 4, so the block comes back flat at the requantised
        // value — 8.6.1 with no residual is a requantisation, not a copy.
        let mut dst = [100u8; 16];
        luma_block(&mut dst, 4, &[0; 16], 28, 24, false, 255);
        let q = (1600 * 13107 + (1 << 18)) >> 19; // 40
        let dc = (q * 10) << 4; // 6400
        let expect = ((dc + 32) >> 6).clamp(0, 255); // 100
        assert_eq!(q, 40);
        assert!(dst.iter().all(|&v| v as i32 == expect), "{dst:?}");
        // The switching path at the same QS gives the same answer with no
        // levels, since both quantise the same coefficient with QS.
        let mut dst2 = [100u8; 16];
        luma_block(&mut dst2, 4, &[0; 16], 28, 24, true, 255);
        assert_eq!(dst, dst2);
    }

    #[test]
    fn a_dc_level_moves_the_block_by_its_scaled_step() {
        // 8.6.1 at QP 28 (per 4, rem 4): level 1 at DC adds
        // (1 * 16 * 16 << 4) >> 6 = 64 to the prediction coefficient.
        let mut flat = [100u8; 16];
        let mut lev = [0i32; 16];
        lev[0] = 1;
        luma_block(&mut flat, 4, &lev, 28, 28, false, 255);
        let s = 1600 + 64;
        let q = (s * 8192 + (1 << 18)) >> 19; // QS 28: quant 8192, bits 19
        let dc = (q * 16) << 4;
        let expect = (dc + 32) >> 6;
        assert!(
            flat.iter().all(|&v| v as i32 == expect),
            "{flat:?} vs {expect}"
        );
    }

    #[test]
    fn chroma_dc_pairs_levels_with_their_frequencies() {
        // A prediction that differs left / right only: its 2x2 DC
        // Hadamard has energy in f01 (horizontal). Adding a level at the
        // horizontal DC position (c01, raster index 1) in the switching
        // path changes the left-right contrast and nothing else.
        let mut pred = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                pred[i * 8 + j] = if j < 4 { 80 } else { 120 };
            }
        }
        let ac = [[0i32; 16]; 4];
        let mut a = pred;
        chroma_420(&mut a, 8, &[0, 0, 0, 0], &ac, 28, 28, true, 255);
        let mut b = pred;
        chroma_420(&mut b, 8, &[0, 1, 0, 0], &ac, 28, 28, true, 255);
        // Rows stay uniform (no vertical change); the two halves moved
        // apart by the same amount.
        for i in 0..8 {
            assert!((0..4).all(|j| b[i * 8 + j] == b[i * 8]));
            assert!((4..8).all(|j| b[i * 8 + j] == b[i * 8 + 4]));
        }
        let d_left = b[0] as i32 - a[0] as i32;
        let d_right = b[4] as i32 - a[4] as i32;
        assert!(d_left > 0 && d_right < 0, "{d_left} {d_right}");
        assert!((d_left + d_right).abs() <= 1);
        assert_eq!(b[0], b[4 * 8]);
        // The vertical position (c10, raster index 2) moves top against
        // bottom instead, leaving the columns of each half uniform.
        let mut c = pred;
        chroma_420(&mut c, 8, &[0, 0, 1, 0], &ac, 28, 28, true, 255);
        let d_top = c[0] as i32 - a[0] as i32;
        let d_bottom = c[4 * 8] as i32 - a[4 * 8] as i32;
        assert!(d_top > 0 && d_bottom < 0, "{d_top} {d_bottom}");
        assert!((d_top + d_bottom).abs() <= 1);
        assert_eq!(c[0], c[3]);
        assert_eq!(c[4], c[7]);
    }
}
