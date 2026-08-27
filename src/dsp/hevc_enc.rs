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

    /// The best table for `cpu`: the scalar reference, then each rung of
    /// the ladder replacing the entries it has a kernel for. The profile
    /// that chose which (docs/encode_speed.md) put `fdct[16]` and `quant`
    /// at 4–7% and 3–4% of an inter encode.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        #[allow(unused_variables)]
        if !super::enc_simd_disabled("hevc_enc") {
            #[cfg(target_arch = "x86_64")]
            super::hevc_enc_x86::install(&mut d, cpu);
            #[cfg(target_arch = "aarch64")]
            super::hevc_enc_neon::install(&mut d, cpu);
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if cpu.simd128 {
                super::hevc_enc_wasm128::install(&mut d);
            }
        }
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

pub(crate) fn fdct_scalar<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
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
pub(crate) const DST4: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];

/// The matrices laid out for the SIMD tiers.
///
/// Built at compile time from the decoder's `TRANSFORM32` (and `DST4`) so
/// no tier can disagree with the reference about a coefficient — the tests
/// below read every layout back against the matrix. Shared here rather than
/// kept in one tier's file because the x86, NEON and wasm kernels read them
/// and none of them is compiled on the others' targets; which layouts a
/// target uses varies, hence the `dead_code` allowance on the module.
pub(crate) mod layouts {
    #![allow(dead_code)]

    use super::DST4;
    use crate::hevc::tables::TRANSFORM32;

    /// The pair table the `pmaddwd`-shaped kernels (x86, wasm `i32x4_dot_i16x8`)
    /// read: `FP[q * 2N + 2j + t] = M[j][2q + t]` for the `N`-point matrix
    /// `M` (`TRANSFORM32` every `32 / N`th row), flattened so a generic kernel
    /// can index it. `L` is `N * N`.
    pub(crate) const fn build_fp<const L: usize>(n: usize) -> [i16; L] {
        let mut t = [0i16; L];
        let step = 32 / n;
        let mut q = 0;
        while q < n / 2 {
            let mut j = 0;
            while j < n {
                t[q * 2 * n + 2 * j] = TRANSFORM32[j * step][2 * q] as i16;
                t[q * 2 * n + 2 * j + 1] = TRANSFORM32[j * step][2 * q + 1] as i16;
                j += 1;
            }
            q += 1;
        }
        t
    }

    /// The DST's pair table, from the same matrix the scalar kernel reads.
    pub(crate) const fn build_fdst() -> [i16; 16] {
        let mut t = [0i16; 16];
        let mut q = 0;
        while q < 2 {
            let mut j = 0;
            while j < 4 {
                t[q * 8 + 2 * j] = DST4[j][2 * q] as i16;
                t[q * 8 + 2 * j + 1] = DST4[j][2 * q + 1] as i16;
                j += 1;
            }
            q += 1;
        }
        t
    }

    /// The column table the multiply-accumulate-shaped kernels (NEON `smlal`
    /// by lane) read: `CT[k * N + j] = M[j][k]` — the matrix transposed, so
    /// that for one input sample `x[k]` the eight outputs `j..j+8` it feeds
    /// are one contiguous vector of weights.
    pub(crate) const fn build_ct<const L: usize>(n: usize) -> [i16; L] {
        let mut t = [0i16; L];
        let step = 32 / n;
        let mut k = 0;
        while k < n {
            let mut j = 0;
            while j < n {
                t[k * n + j] = TRANSFORM32[j * step][k] as i16;
                j += 1;
            }
            k += 1;
        }
        t
    }

    /// The DST's column table.
    pub(crate) const fn build_cdst() -> [i16; 16] {
        let mut t = [0i16; 16];
        let mut k = 0;
        while k < 4 {
            let mut j = 0;
            while j < 4 {
                t[k * 4 + j] = DST4[j][k] as i16;
                j += 1;
            }
            k += 1;
        }
        t
    }

    /// The matrix itself as i16, rows contiguous: `MT[j * N + k] = M[j][k]`.
    /// What the column table is the transpose of; the same kernels read this
    /// one in their second stage, where the weights of one output row are
    /// the vector and the data rows are what gets strided through.
    pub(crate) const fn build_mt<const L: usize>(n: usize) -> [i16; L] {
        let mut t = [0i16; L];
        let step = 32 / n;
        let mut j = 0;
        while j < n {
            let mut k = 0;
            while k < n {
                t[j * n + k] = TRANSFORM32[j * step][k] as i16;
                k += 1;
            }
            j += 1;
        }
        t
    }

    /// The DST matrix as i16, rows contiguous.
    pub(crate) const fn build_mdst() -> [i16; 16] {
        let mut t = [0i16; 16];
        let mut j = 0;
        while j < 4 {
            let mut k = 0;
            while k < 4 {
                t[j * 4 + k] = DST4[j][k] as i16;
                k += 1;
            }
            j += 1;
        }
        t
    }

    pub(crate) static FP4: [i16; 16] = build_fp::<16>(4);
    pub(crate) static FP8: [i16; 64] = build_fp::<64>(8);
    pub(crate) static FP16: [i16; 256] = build_fp::<256>(16);
    pub(crate) static FP32: [i16; 1024] = build_fp::<1024>(32);
    pub(crate) static FDST: [i16; 16] = build_fdst();

    pub(crate) static CT4: [i16; 16] = build_ct::<16>(4);
    pub(crate) static CT8: [i16; 64] = build_ct::<64>(8);
    pub(crate) static CT16: [i16; 256] = build_ct::<256>(16);
    pub(crate) static CT32: [i16; 1024] = build_ct::<1024>(32);
    pub(crate) static CDST: [i16; 16] = build_cdst();

    pub(crate) static MT4: [i16; 16] = build_mt::<16>(4);
    pub(crate) static MT8: [i16; 64] = build_mt::<64>(8);
    pub(crate) static MT16: [i16; 256] = build_mt::<256>(16);
    pub(crate) static MT32: [i16; 1024] = build_mt::<1024>(32);
    pub(crate) static MDST: [i16; 16] = build_mdst();

    /// The `N`-point matrix, rows contiguous.
    #[inline(always)]
    pub(crate) fn mt<const N: usize>() -> &'static [i16] {
        match N {
            32 => &MT32,
            16 => &MT16,
            8 => &MT8,
            _ => &MT4,
        }
    }

    /// The `N`-point pair table.
    #[inline(always)]
    pub(crate) fn fp<const N: usize>() -> &'static [i16] {
        match N {
            32 => &FP32,
            16 => &FP16,
            8 => &FP8,
            _ => &FP4,
        }
    }

    /// The `N`-point column table.
    #[inline(always)]
    pub(crate) fn ct<const N: usize>() -> &'static [i16] {
        match N {
            32 => &CT32,
            16 => &CT16,
            8 => &CT8,
            _ => &CT4,
        }
    }
}

pub(crate) fn fdst4_scalar(block: &mut [i16], bit_depth: u32) {
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

pub(crate) fn quant_scalar(coeffs: &[i16], levels: &mut [i16], n: usize, scale: i32, qbits: u32, offset: i32) -> u32 {
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
    use super::layouts::*;
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

    /// The pair table really is the matrix: every entry against
    /// `TRANSFORM32` directly, so a transposed build cannot pass by being
    /// consistently wrong in both stages of every tier that reads it.
    #[test]
    fn pair_table_is_the_matrix() {
        for &(n, t) in &[(4usize, &FP4[..]), (8, &FP8[..]), (16, &FP16[..]), (32, &FP32[..])] {
            let step = 32 / n;
            assert_eq!(t.len(), n * n, "n={n}");
            for q in 0..n / 2 {
                for j in 0..n {
                    assert_eq!(t[q * 2 * n + 2 * j], TRANSFORM32[j * step][2 * q] as i16, "n={n} q={q} j={j}");
                    assert_eq!(t[q * 2 * n + 2 * j + 1], TRANSFORM32[j * step][2 * q + 1] as i16, "n={n} q={q} j={j}");
                }
            }
        }
        for q in 0..2 {
            for j in 0..4 {
                assert_eq!(FDST[q * 8 + 2 * j], DST4[j][2 * q] as i16);
                assert_eq!(FDST[q * 8 + 2 * j + 1], DST4[j][2 * q + 1] as i16);
            }
        }
    }

    /// And the column table is its transpose, entry for entry.
    #[test]
    fn column_table_is_the_matrix_transposed() {
        for &(n, t) in &[(4usize, &CT4[..]), (8, &CT8[..]), (16, &CT16[..]), (32, &CT32[..])] {
            let step = 32 / n;
            assert_eq!(t.len(), n * n, "n={n}");
            for k in 0..n {
                for j in 0..n {
                    assert_eq!(t[k * n + j], TRANSFORM32[j * step][k] as i16, "n={n} k={k} j={j}");
                }
            }
        }
        for k in 0..4 {
            for j in 0..4 {
                assert_eq!(CDST[k * 4 + j], DST4[j][k] as i16);
                assert_eq!(MDST[j * 4 + k], DST4[j][k] as i16);
            }
        }
        // And the row-major copy is the matrix, entry for entry.
        for &(n, t) in &[(4usize, &MT4[..]), (8, &MT8[..]), (16, &MT16[..]), (32, &MT32[..])] {
            let step = 32 / n;
            for j in 0..n {
                for k in 0..n {
                    assert_eq!(t[j * n + k], TRANSFORM32[j * step][k] as i16, "n={n} j={j} k={k}");
                }
            }
        }
        assert_eq!(fp::<16>().len(), 256);
        assert_eq!(ct::<32>().len(), 1024);
        assert_eq!(mt::<8>().len(), 64);
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
