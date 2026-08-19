//! H.265 forward transforms and quantisation — the encode-side
//! counterpart of `dsp::hevc`'s inverse transforms and of
//! `hevc::residual`'s scaling.
//!
//! Built the same way as [`super::h264_enc`] and for the same reasons. The
//! forward direction is not normative, so what is checked is the pairing:
//! these kernels' output goes back through the decoder's own scaling and
//! inverse transform, which the JCT-VC suites hold in place, and has to
//! come back. And the tables are *derived from the decoder's* rather than
//! written down a second time — `quantScale` is `round(2^20 / levelScale)`,
//! entry for entry, so the two directions cannot drift apart.
//!
//! The same warning applies here as there: a round-trip test is blind to a
//! fault applied symmetrically to both sides of an inverse pair. What
//! protects these is that the inverse is the decoder's and not ours, so a
//! mistake on this side has nothing to cancel against. Writing a private
//! inverse for the tests would silently remove that.
//!
//! The shifts are where H.265 differs most from H.264, and they are worth
//! stating because getting one wrong produces a pure scale error rather
//! than anything that looks like a bug:
//!
//! - forward stage one shifts by `log2N + BitDepth - 9`, stage two by
//!   `log2N + 6`;
//! - quantisation shifts by `29 + QP/6 - BitDepth - log2N`, which is HM's
//!   `QUANT_SHIFT + qpPer + (MAX_TR_DYNAMIC_RANGE - BitDepth - log2N)`
//!   with the constants folded;
//! - the decoder then scales by `BitDepth + log2N - 5` and inverse-
//!   transforms by `20 - BitDepth`.
//!
//! Unlike H.264 the block size enters every one of them, so a shift that
//! is right at 4x4 can be wrong at 32x32 — which is why the round-trip
//! test runs all four sizes at every QP rather than sampling.

use super::Cpu;
use crate::hevc::tables::TRANSFORM32;

/// Forward DCT of an `n x n` block: residual samples in, coefficients out,
/// both raster. `log2` is the block's, `bit_depth` the picture's.
pub type FdctFn = fn(block: &mut [i16], log2: u32, bit_depth: u32);
/// Forward DST 4x4 (`trType == 1`), the intra luma 4x4 transform.
pub type Fdst4Fn = fn(block: &mut [i16], bit_depth: u32);
/// Forward quantisation: coefficients in, levels out, returning the count
/// of nonzero levels.
pub type HevcQuantFn = fn(coeffs: &[i16], levels: &mut [i16], n: usize, scale: i32, qbits: u32, offset: i32) -> u32;

/// The H.265 encode-side kernel table.
#[derive(Clone)]
pub struct HevcEncDsp {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Forward DCT by `log2 - 2`: 4, 8, 16 and 32.
    pub fdct: [FdctFn; 4],
    /// Forward DST 4x4.
    pub fdst4: Fdst4Fn,
    /// Forward quantisation.
    pub quant: HevcQuantFn,
    /// Forward transform-skip scaling.
    pub fskip: FdctFn,
}

impl HevcEncDsp {
    /// The scalar reference table.
    pub fn scalar() -> Self {
        HevcEncDsp {
            cpu: Cpu::SCALAR,
            fdct: [fdct_scalar::<4>, fdct_scalar::<8>, fdct_scalar::<16>, fdct_scalar::<32>],
            fdst4: fdst4_scalar,
            quant: quant_scalar,
            fskip: fskip_scalar,
        }
    }

    /// The best table for `cpu`. No rung replaces anything yet: which of
    /// these earns a hand-written kernel is a question for a profile of a
    /// real encode.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        d
    }
}

impl Default for HevcEncDsp {
    fn default() -> Self {
        Self::scalar()
    }
}

/// `levelScale` of 8.6.3, which the decoder scales by and this module
/// divides by.
pub const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// The forward multiplier for `qp % 6`, derived from [`LEVEL_SCALE`] so
/// there is one table and not two that must agree. Reproduces HM's
/// `quantScales` exactly, which a test asserts.
#[inline]
pub fn quant_scale(m: usize) -> i32 {
    let ls = LEVEL_SCALE[m] as i64;
    ((1i64 << 20) + ls / 2) as i32 / ls as i32
}

/// The quantiser's shift: `QUANT_SHIFT + QP/6 + (MAX_TR_DYNAMIC_RANGE -
/// BitDepth - log2N)`, with HM's constants folded.
#[inline]
pub fn qbits(qp: i32, log2: u32, bit_depth: u32) -> u32 {
    (29 + qp / 6 - bit_depth as i32 - log2 as i32) as u32
}

/// A dead-zone rounding offset: a third of a step for intra, a sixth for
/// inter, matching HM's 171/512 and 85/512.
#[inline]
pub fn quant_offset(qbits: u32, intra: bool) -> i32 {
    let denom = if intra { 3 } else { 6 };
    (1i32 << qbits) / denom
}

/// One 1-D forward transform of `n` points: `out[j] = sum_k M[j][k] x[k]`,
/// the same matrix the inverse reads down the other index.
#[inline]
fn fdct1(x: &[i32], n: usize, out: &mut [i32]) {
    let step = 32 / n;
    for (j, o) in out.iter_mut().enumerate().take(n) {
        let row = &TRANSFORM32[j * step];
        let mut s = 0i32;
        for k in 0..n {
            s += row[k] as i32 * x[k];
        }
        *o = s;
    }
}

fn fdct_scalar<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
    let s1 = log2 as i32 + bit_depth as i32 - 9;
    let s2 = log2 as i32 + 6;
    let mut tmp = [0i16; 32 * 32];
    let mut row = [0i32; 32];
    let mut out = [0i32; 32];
    // Rows, then columns — the mirror of the inverse, which does columns
    // then rows.
    for y in 0..N {
        for x in 0..N {
            row[x] = block[y * N + x] as i32;
        }
        fdct1(&row, N, &mut out);
        for x in 0..N {
            let v = if s1 > 0 { (out[x] + (1 << (s1 - 1))) >> s1 } else { out[x] };
            tmp[y * N + x] = v.clamp(-32768, 32767) as i16;
        }
    }
    let r2 = 1 << (s2 - 1);
    for x in 0..N {
        for y in 0..N {
            row[y] = tmp[y * N + x] as i32;
        }
        fdct1(&row, N, &mut out);
        for y in 0..N {
            block[y * N + x] = ((out[y] + r2) >> s2).clamp(-32768, 32767) as i16;
        }
    }
}

/// The DST matrix of 8.6.4.2, read the way a forward transform reads it.
const DST4: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];

fn fdst4_scalar(block: &mut [i16], bit_depth: u32) {
    let s1 = 2 + bit_depth as i32 - 9;
    let s2 = 2 + 6;
    let mut tmp = [0i16; 16];
    for y in 0..4 {
        for j in 0..4 {
            let mut s = 0i32;
            for k in 0..4 {
                s += DST4[j][k] * block[y * 4 + k] as i32;
            }
            let v = if s1 > 0 { (s + (1 << (s1 - 1))) >> s1 } else { s };
            tmp[y * 4 + j] = v.clamp(-32768, 32767) as i16;
        }
    }
    let r2 = 1 << (s2 - 1);
    for x in 0..4 {
        for j in 0..4 {
            let mut s = 0i32;
            for k in 0..4 {
                s += DST4[j][k] * tmp[k * 4 + x] as i32;
            }
            block[j * 4 + x] = ((s + r2) >> s2).clamp(-32768, 32767) as i16;
        }
    }
}

/// The forward of `transform_skip_residual`: the decoder shifts left by
/// `5 + log2` and right by `20 - BitDepth`, so this does the reverse with
/// rounding.
fn fskip_scalar(block: &mut [i16], log2: u32, bit_depth: u32) {
    let n = 1usize << log2;
    let ts_shift = 5 + log2 as i32;
    let bd_shift = 20 - bit_depth as i32;
    let sh = ts_shift - bd_shift;
    for v in block.iter_mut().take(n * n) {
        let x = *v as i32;
        *v = if sh >= 0 {
            (x + (1 << sh >> 1)) >> sh
        } else {
            x << -sh
        }
        .clamp(-32768, 32767) as i16;
    }
}

fn quant_scalar(coeffs: &[i16], levels: &mut [i16], n: usize, scale: i32, qbits: u32, offset: i32) -> u32 {
    let mut nz = 0;
    for i in 0..n * n {
        let c = coeffs[i] as i32;
        let m = (c.unsigned_abs() as i64 * scale as i64 + offset as i64) >> qbits;
        let v = (if c < 0 { -(m as i32) } else { m as i32 }).clamp(-32768, 32767);
        levels[i] = v as i16;
        nz += (v != 0) as u32;
    }
    nz
}

/// Forward RDPCM (7.3.8.11 read backwards): the decoder accumulates
/// residual along a row or column, so the encoder differences it.
pub fn rdpcm_forward(block: &mut [i16], log2: u32, vertical: bool) {
    let n = 1usize << log2;
    if vertical {
        for x in 0..n {
            for y in (1..n).rev() {
                block[y * n + x] = block[y * n + x].wrapping_sub(block[(y - 1) * n + x]);
            }
        }
    } else {
        for y in 0..n {
            for x in (1..n).rev() {
                block[y * n + x] = block[y * n + x].wrapping_sub(block[y * n + x - 1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::residual::{ScalingSource, rdpcm_residual, scale_coefficients, transform_skip_residual};

    fn lcg(s: &mut u64) -> i32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) & 0x1ff) as i32 - 255
    }

    /// The forward multipliers HM carries, which this module derives from
    /// the decoder's `levelScale` instead of repeating.
    #[test]
    fn quant_scales_match_the_canonical_ones() {
        const CANON: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
        for m in 0..6 {
            assert_eq!(quant_scale(m), CANON[m], "m={m}");
        }
    }

    /// The invariant: what this quantises, the decoder's own scaling and
    /// inverse transform bring back. Every size at every QP, because the
    /// block size enters all four shifts and a mistake can be invisible at
    /// one size and gross at another.
    #[test]
    fn forward_and_inverse_round_trip() {
        let dsp = crate::dsp::hevc::HevcDsp::<u16>::new(Cpu::SCALAR);
        let mut seed = 0xfeed_1234u64;
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            for qp in 0..52 {
                let bd = 8u32;
                let qb = qbits(qp, log2, bd);
                let off = quant_offset(qb, true);
                let scale = quant_scale((qp % 6) as usize);
                let step = 1i32 << (qp / 6);
                let mut worst = 0;
                for _ in 0..8 {
                    let mut res = vec![0i16; n * n];
                    for r in res.iter_mut() {
                        *r = lcg(&mut seed) as i16;
                    }
                    let mut block = res.clone();
                    (HevcEncDsp::scalar().fdct[(log2 - 2) as usize])(&mut block, log2, bd);
                    let mut levels = vec![0i16; n * n];
                    quant_scalar(&block, &mut levels, n, scale, qb, off);
                    scale_coefficients(&mut levels, log2, qp, bd, ScalingSource::Flat, false, n - 1, n - 1);
                    (dsp.idct[(log2 - 2) as usize])(&mut levels, 20 - bd as i32, n - 1, n - 1);
                    for i in 0..n * n {
                        worst = worst.max((levels[i] as i32 - res[i] as i32).abs());
                    }
                }
                assert!(worst <= 8 * step + 16, "log2={log2} qp={qp} worst={worst} step={step}");
            }
        }
    }

    /// The DST is the 4x4 intra luma transform and gets the same check.
    #[test]
    fn dst_round_trips() {
        let dsp = crate::dsp::hevc::HevcDsp::<u16>::new(Cpu::SCALAR);
        let mut seed = 0xd57u64;
        for qp in 0..52 {
            let bd = 8u32;
            let qb = qbits(qp, 2, bd);
            let off = quant_offset(qb, true);
            let scale = quant_scale((qp % 6) as usize);
            let step = 1i32 << (qp / 6);
            let mut worst = 0;
            for _ in 0..16 {
                let mut res = [0i16; 16];
                for r in res.iter_mut() {
                    *r = lcg(&mut seed) as i16;
                }
                let mut block = res;
                fdst4_scalar(&mut block, bd);
                let mut levels = [0i16; 16];
                quant_scalar(&block, &mut levels, 4, scale, qb, off);
                scale_coefficients(&mut levels, 2, qp, bd, ScalingSource::Flat, false, 3, 3);
                (dsp.idst4)(&mut levels, 20 - bd as i32, 3, 3);
                for i in 0..16 {
                    worst = worst.max((levels[i] as i32 - res[i] as i32).abs());
                }
            }
            assert!(worst <= 8 * step + 16, "qp={qp} worst={worst}");
        }
    }

    /// Transform skip is a pure scale, so its round trip is exact up to
    /// the rounding of one shift.
    #[test]
    fn transform_skip_round_trips() {
        let mut seed = 7u64;
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            for bd in [8u32, 10] {
                let mut res = vec![0i16; n * n];
                for r in res.iter_mut() {
                    *r = lcg(&mut seed) as i16;
                }
                let mut block = res.clone();
                fskip_scalar(&mut block, log2, bd);
                transform_skip_residual(&mut block, log2, bd);
                for i in 0..n * n {
                    assert!(
                        (block[i] as i32 - res[i] as i32).abs() <= 1,
                        "log2={log2} bd={bd} i={i} {} vs {}",
                        block[i],
                        res[i]
                    );
                }
            }
        }
    }

    /// RDPCM differences what the decoder accumulates, so the pair is
    /// exactly lossless — this one is an equality, not a bound.
    #[test]
    fn rdpcm_round_trips_exactly() {
        let mut seed = 31u64;
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            for vertical in [false, true] {
                let mut res = vec![0i16; n * n];
                for r in res.iter_mut() {
                    *r = lcg(&mut seed) as i16;
                }
                let mut block = res.clone();
                rdpcm_forward(&mut block, log2, vertical);
                rdpcm_residual(&mut block, log2, vertical);
                assert_eq!(block, res, "log2={log2} vertical={vertical}");
            }
        }
    }

    /// A flat block is DC-only whichever size runs, which catches a
    /// transposed stage before the round trip has to.
    #[test]
    fn a_flat_block_is_dc_only() {
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            let mut block = vec![64i16; n * n];
            (HevcEncDsp::scalar().fdct[(log2 - 2) as usize])(&mut block, log2, 8);
            assert!(block[0] != 0, "log2={log2} lost its DC");
            assert!(block[1..].iter().all(|&v| v == 0), "log2={log2} is not DC-only");
        }
    }
}
