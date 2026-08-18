//! AVX-512 versions of the sample-size-independent H.265 kernels (x86-64),
//! installed over [`super::hevc_avx2`] for both the 8- and 16-bit tables.
//!
//! Only the 32-point inverse transform is here, and it is the one place in
//! either codec where a 512-bit vector is the natural width rather than a
//! way of stacking rows: a row of a 32x32 transform block is exactly
//! thirty-two 16-bit coefficients. The AVX2 kernel runs both of its stages
//! in two 16-lane steps per output row; this runs one. The smaller
//! transforms have nothing to widen — a 16-point row *is* a 256-bit vector —
//! and stay on AVX2.
//!
//! Bit-exact against the scalar reference in the tests below.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::hevc::HevcDsp;
use super::hevc_avx2 as w16;
use crate::hevc::tables::TRANSFORM32;

/// The 32-point inverse DCT. Shared by both sample widths — the transform
/// works on 16-bit coefficients whatever the samples are.
pub(super) const IDCT32: super::hevc::IdctFn = idct32_avx512;

/// Replace the AVX2 entries of `d` that AVX-512 improves on (16-bit
/// samples). Called after [`super::hevc_avx2::install`].
pub fn install(d: &mut HevcDsp<u16>) {
    d.idct[3] = IDCT32;
}

/// A 32x32 block's side.
const N: usize = 32;

fn idct32_avx512(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    // The DC shortcut and any short buffer stay with the AVX2 kernel.
    if coeffs.len() < N * N || (max_x == 0 && max_y == 0) {
        return w16::idct_avx2::<32>(coeffs, bd_shift, max_x, max_y);
    }
    unsafe { idct32_impl(coeffs, bd_shift, max_x, max_y) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn idct32_impl(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    unsafe {
        let mut tmp = [0i16; N * N];
        let nzy = max_y + 1;
        let npairs = nzy.div_ceil(2);
        // Stage 1 (columns): tmp[y][x] = clip((sum_j c[j][y] · coef[j][x] + 64) >> 7),
        // vectorised across x, pairs of input rows at a time. `madd` over
        // `unpacklo`/`unpackhi` leaves lane j of `lo` holding outputs
        // 8j..8j+3 and lane j of `hi` holding 8j+4..8j+7, so `packs` — also
        // per 128-bit lane — puts them back in order for free.
        if max_x >= 16 {
            for y in 0..N {
                let mut lo = _mm512_set1_epi32(64);
                let mut hi = lo;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = _mm512_loadu_si512(coeffs.as_ptr().add(j * N) as *const __m512i);
                    let b = if j + 1 < nzy { _mm512_loadu_si512(coeffs.as_ptr().add((j + 1) * N) as *const __m512i) } else { _mm512_setzero_si512() };
                    let c = _mm512_set1_epi32(w16::pair(TRANSFORM32[j][y], TRANSFORM32[j + 1][y]));
                    lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c));
                    hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c));
                }
                let r = _mm512_packs_epi32(_mm512_srai_epi32::<7>(lo), _mm512_srai_epi32::<7>(hi));
                _mm512_storeu_si512(tmp.as_mut_ptr().add(y * N) as *mut __m512i, r);
            }
        } else {
            // Nothing past column 15 to transform. The AVX2 kernel's 16-lane
            // step already skips that half, and transforming zeros twice as
            // wide would cost more than the wider vector saves.
            for y in 0..N {
                let mut lo = _mm256_set1_epi32(64);
                let mut hi = lo;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = w16::load_n(coeffs.as_ptr().add(j * N), N);
                    let b = if j + 1 < nzy { w16::load_n(coeffs.as_ptr().add((j + 1) * N), N) } else { _mm256_setzero_si256() };
                    let c = _mm256_set1_epi32(w16::pair(TRANSFORM32[j][y], TRANSFORM32[j + 1][y]));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c));
                }
                let r = _mm256_packs_epi32(_mm256_srai_epi32::<7>(lo), _mm256_srai_epi32::<7>(hi));
                w16::store_n(tmp.as_mut_ptr().add(y * N), r, 16);
            }
        }
        // Stage 2 (rows): out[y][x] = clip((sum_j c[j][x] · tmp[y][j] + round) >> shift),
        // vectorised across all thirty-two x at once against the matrix's
        // pre-interleaved pair rows. Here `lo` holds columns 0..15 and `hi`
        // columns 16..31, so the pack *does* interleave and one permute
        // undoes it.
        let nzx = max_x + 1;
        let npairs = nzx.div_ceil(2);
        let round2 = _mm512_set1_epi32(1 << (bd_shift - 1));
        let sh = _mm_cvtsi32_si128(bd_shift);
        let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        for y in 0..N {
            let row = tmp.as_ptr().add(y * N);
            let mut lo = round2;
            let mut hi = round2;
            for p in 0..npairs {
                let j = 2 * p;
                let t0 = *row.add(j) as i32;
                let t1 = if j + 1 < nzx { *row.add(j + 1) as i32 } else { 0 };
                let tv = _mm512_set1_epi32((t0 as u16 as i32) | ((t1 as u16 as i32) << 16));
                let pr = w16::pair_row(N, p);
                lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_loadu_si512(pr.as_ptr() as *const __m512i), tv));
                hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_loadu_si512(pr.as_ptr().add(32) as *const __m512i), tv));
            }
            let r = _mm512_packs_epi32(_mm512_sra_epi32(lo, sh), _mm512_sra_epi32(hi, sh));
            _mm512_storeu_si512(coeffs.as_mut_ptr().add(y * N) as *mut __m512i, _mm512_permutexvar_epi64(idx, r));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn idct32_matches_scalar() {
        // Same skip-is-not-coverage rule as the interpolation kernels.
        if super::super::hevc_avx512_u8::tests::avx512().is_none() {
            return;
        }
        let s = HevcDsp::<u16>::SCALAR;
        let mut seed = 0x1d_c7_u64;
        for trial in 0..300 {
            // The standard clips coefficients to 16 bits (8.6.2); the
            // nonzero bounds sweep both stage-1 branches and the tail where
            // the last coefficient pair is odd.
            let (max_x, max_y) = match trial % 6 {
                0 => (31, 31),
                1 => (15, 31),
                2 => (31, 15),
                3 => (0, 7),
                4 => (16, 1),
                _ => ((lcg(&mut seed) % 32) as usize, (lcg(&mut seed) % 32) as usize),
            };
            let range = if trial % 3 == 0 { 32767 } else { 900 };
            let mut a = [0i16; N * N];
            for y in 0..=max_y {
                for x in 0..=max_x {
                    a[y * N + x] = (lcg(&mut seed) % (2 * range + 1)) as i16 - range as i16;
                }
            }
            let mut b = a;
            let bd_shift = 20 - [8, 10, 12][trial % 3];
            (s.idct[3])(&mut a, bd_shift, max_x, max_y);
            idct32_avx512(&mut b, bd_shift, max_x, max_y);
            assert_eq!(a, b, "idct32 trial {trial} max {max_x},{max_y} shift {bd_shift}");
        }
    }
}
