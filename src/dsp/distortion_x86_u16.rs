//! x86-64 SIMD versions of the distortion metrics for 16-bit samples — what
//! the deep encoders' mode decisions and motion searches spend their time in.
//!
//! The lane layout is [`super::distortion_x86`]'s: eight samples to a
//! 128-bit vector (sixteen at AVX2), two SATD tiles to a row of eight. What
//! changes is that these kernels are told no bit depth. The encoders hand
//! them 9- to 14-bit samples and a `u16` holds 16, so each is exact for every
//! `u16` input, and the one metric whose exactness would cost every block
//! picks its arithmetic per tile pair instead:
//!
//! - **SAD** is `psubusw` both ways, or'd — the absolute difference, exact
//!   for any u16 — folded into i32 lanes by `pmaddwd` after flipping the
//!   sign bit (`d − 32768` is an i16 for every `d`), with the 32768 a lane
//!   put back once at the end.
//! - **SSD** splits each square into its halves, `pmullw` the low sixteen
//!   bits and `pmulhuw` the high, folds each half the way SAD folds, and
//!   recombines them as `hi · 65536 + lo` in 64 bits once a row.
//! - **SATD** transforms `a − b`. A difference of samples below 2048 keeps
//!   the whole Hadamard inside i16 — at most 4 · 2047 after the first stage,
//!   32752 after the second — and the four absolute coefficients of a column
//!   inside u16, since for a first-stage column `v`, ‖Hv‖₁ ≤ 2‖Hv‖₂ = 4‖v‖₂
//!   ≤ 65504. So a tile pair whose samples are all below 2048 takes the
//!   8-bit kernel's shape in 16-bit lanes (its per-lane sums folded like
//!   SAD's), and any other takes the same transform in 32-bit lanes, one
//!   tile to a vector, which is exact for any u16. The test that picks is
//!   one `por` a row, on vectors the kernel loaded anyway.
//!
//! Written once and compiled per rung by `kernels!`. SSE2 carries
//! everything; SSSE3 replaces the SATD's absolute values, SSE4.1 its
//! all-zero test and widening; AVX re-encodes SSE4.1 for VEX; AVX2 takes the
//! shapes sixteen or more samples wide. The tests sweep every rung against
//! the scalar reference at 9 to 16 bits (`super::u16_sweep`).

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::distortion::DistortionDsp;

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::distortion::{DistortionDsp, sad_scalar, satd_scalar, ssd_scalar};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        /// Every kernel.
        pub(crate) fn install_all(d: &mut DistortionDsp<u16>) {
            d.sad = sad;
            d.satd = satd;
            d.ssd = ssd;
        }

        /// The SATD alone: the rungs whose changes are its absolute value,
        /// widening and all-zero test.
        pub(crate) fn install_satd(d: &mut DistortionDsp<u16>) {
            d.satd = satd;
        }

        /// Eight samples at `p`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load8(p: *const u16) -> __m128i {
            unsafe { _mm_loadu_si128(p as *const __m128i) }
        }

        /// Four samples at `p` in the low lanes, the rest zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load4(p: *const u16) -> __m128i {
            unsafe { _mm_loadl_epi64(p as *const __m128i) }
        }

        /// `|a − b|` per u16 lane, exact for any samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn absdiff(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a)) }
        }

        /// Adjacent u16 lanes summed into four i32, each sum less 65536:
        /// the sign bit flipped makes every lane an i16 for `pmaddwd`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fold(v: __m128i) -> __m128i {
            unsafe { _mm_madd_epi16(_mm_xor_si128(v, _mm_set1_epi16(i16::MIN)), _mm_set1_epi16(1)) }
        }

        /// The four i32 lanes of `v`, summed in i64.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn lanes_i64(v: __m128i) -> i64 {
            unsafe {
                let mut t = [0i32; 4];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                t.iter().map(|&x| x as i64).sum()
            }
        }

        // ------------------------------------------------------------------
        // SAD
        // ------------------------------------------------------------------

        #[target_feature(enable = $feat)]
        unsafe fn sad_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
            unsafe {
                let mut acc = _mm_setzero_si128();
                // Lanes folded, each 32768 short.
                let mut lanes = 0u32;
                for y in 0..h {
                    let ra = a.add(y * sa);
                    let rb = b.add(y * sb);
                    let mut x = 0;
                    while x + 8 <= w {
                        acc = _mm_add_epi32(acc, fold(absdiff(load8(ra.add(x)), load8(rb.add(x)))));
                        x += 8;
                        lanes += 8;
                    }
                    if x + 4 <= w {
                        // The zero lanes above the four samples fold too.
                        acc = _mm_add_epi32(acc, fold(absdiff(load4(ra.add(x)), load4(rb.add(x)))));
                        lanes += 8;
                    }
                }
                // Wrapping, as the scalar sum's u32 does: each lane's sum is
                // right modulo 2^32, so their total is.
                (lanes_i64(acc) as u32).wrapping_add(lanes.wrapping_mul(32768))
            }
        }

        pub(crate) fn sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
            if w % 4 != 0 || h == 0 {
                return sad_scalar(a, a_stride, b, b_stride, w, h);
            }
            assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
            unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }

        // ------------------------------------------------------------------
        // SSD
        // ------------------------------------------------------------------

        #[target_feature(enable = $feat)]
        unsafe fn ssd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u64 {
            unsafe {
                let (mut total_lo, mut total_hi) = (0i64, 0i64);
                let mut lanes = 0i64;
                for y in 0..h {
                    let ra = a.add(y * sa);
                    let rb = b.add(y * sb);
                    // A row's folds stay far inside i32 (at most 65536 a
                    // vector per lane); they are taken to 64 bits once a row.
                    let (mut lo, mut hi) = (_mm_setzero_si128(), _mm_setzero_si128());
                    let mut sq = |d: __m128i| {
                        lo = _mm_add_epi32(lo, fold(_mm_mullo_epi16(d, d)));
                        hi = _mm_add_epi32(hi, fold(_mm_mulhi_epu16(d, d)));
                    };
                    let mut x = 0;
                    while x + 8 <= w {
                        sq(absdiff(load8(ra.add(x)), load8(rb.add(x))));
                        x += 8;
                        lanes += 8;
                    }
                    if x + 4 <= w {
                        sq(absdiff(load4(ra.add(x)), load4(rb.add(x))));
                        lanes += 8;
                    }
                    total_lo += lanes_i64(lo);
                    total_hi += lanes_i64(hi);
                }
                let fix = lanes * 32768;
                (((total_hi + fix) as u64) << 16).wrapping_add((total_lo + fix) as u64)
            }
        }

        pub(crate) fn ssd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u64 {
            if w % 4 != 0 || h == 0 {
                return ssd_scalar(a, a_stride, b, b_stride, w, h);
            }
            assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
            unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }

        // ------------------------------------------------------------------
        // SATD
        // ------------------------------------------------------------------

        /// The 4-point Hadamard butterfly, lane-wise across four i16 vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn butterfly16(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
            unsafe {
                let s0 = _mm_add_epi16(r0, r3);
                let s1 = _mm_add_epi16(r1, r2);
                let s2 = _mm_sub_epi16(r1, r2);
                let s3 = _mm_sub_epi16(r0, r3);
                [_mm_add_epi16(s0, s1), _mm_add_epi16(s3, s2), _mm_sub_epi16(s0, s1), _mm_sub_epi16(s3, s2)]
            }
        }

        /// The same across four i32 vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn butterfly32(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
            unsafe {
                let s0 = _mm_add_epi32(r0, r3);
                let s1 = _mm_add_epi32(r1, r2);
                let s2 = _mm_sub_epi32(r1, r2);
                let s3 = _mm_sub_epi32(r0, r3);
                [_mm_add_epi32(s0, s1), _mm_add_epi32(s3, s2), _mm_sub_epi32(s0, s1), _mm_sub_epi32(s3, s2)]
            }
        }

        /// SATD of the two tiles of differences in `r0..r3` (one row each,
        /// tile A in the low half, B in the high), in i16 — for differences
        /// of samples below 2048 — as `[A, A, B, B]` with the per-tile
        /// `(sum + 1) >> 1` applied.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn pair16(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> __m128i {
            unsafe {
                let [t0, t1, t2, t3] = butterfly16(r0, r1, r2, r3);
                // Transpose each tile: rows become columns, halves stay put.
                let u0 = _mm_unpacklo_epi16(t0, t1);
                let u1 = _mm_unpacklo_epi16(t2, t3);
                let u2 = _mm_unpackhi_epi16(t0, t1);
                let u3 = _mm_unpackhi_epi16(t2, t3);
                let v0 = _mm_unpacklo_epi32(u0, u1);
                let v1 = _mm_unpackhi_epi32(u0, u1);
                let v2 = _mm_unpacklo_epi32(u2, u3);
                let v3 = _mm_unpackhi_epi32(u2, u3);
                let c0 = _mm_unpacklo_epi64(v0, v2);
                let c1 = _mm_unpackhi_epi64(v0, v2);
                let c2 = _mm_unpacklo_epi64(v1, v3);
                let c3 = _mm_unpackhi_epi64(v1, v3);
                let [w0, w1, w2, w3] = butterfly16(c0, c1, c2, c3);
                // Each column's four absolute values: at most 65504, a u16.
                let s = _mm_add_epi16(_mm_add_epi16(abs16(w0), abs16(w1)), _mm_add_epi16(abs16(w2), abs16(w3)));
                // [A01, A23, B01, B23] each 65536 short -> [A, A, B, B]
                // 131072 short; the shortfall and the rounding go back at once.
                let p = fold(s);
                let q = _mm_add_epi32(p, _mm_shuffle_epi32(p, 0b10_11_00_01));
                _mm_srli_epi32(_mm_add_epi32(q, _mm_set1_epi32(131072 + 1)), 1)
            }
        }

        /// SATD of one tile of i32 differences (four rows of four), as four
        /// equal lanes with the rounding applied — exact for any u16 samples,
        /// whose coefficients reach 16 · 65535.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tile32(r: [__m128i; 4]) -> __m128i {
            unsafe {
                let [t0, t1, t2, t3] = butterfly32(r[0], r[1], r[2], r[3]);
                let u0 = _mm_unpacklo_epi32(t0, t1);
                let u1 = _mm_unpacklo_epi32(t2, t3);
                let u2 = _mm_unpackhi_epi32(t0, t1);
                let u3 = _mm_unpackhi_epi32(t2, t3);
                let [w0, w1, w2, w3] =
                    butterfly32(_mm_unpacklo_epi64(u0, u1), _mm_unpackhi_epi64(u0, u1), _mm_unpacklo_epi64(u2, u3), _mm_unpackhi_epi64(u2, u3));
                let s = _mm_add_epi32(_mm_add_epi32(abs32(w0), abs32(w1)), _mm_add_epi32(abs32(w2), abs32(w3)));
                let x = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let t = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b10_11_00_01));
                _mm_srli_epi32(_mm_add_epi32(t, _mm_set1_epi32(1)), 1)
            }
        }

        /// SATD of the two tiles whose sample rows are `ra` and `rb` (tile A
        /// in the low four lanes, B in the high four), as `[A, A, B, B]`:
        /// in i16 if every sample is below 2048, in i32 otherwise.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn pair(ra: [__m128i; 4], rb: [__m128i; 4]) -> __m128i {
            unsafe {
                let seen = _mm_or_si128(
                    _mm_or_si128(_mm_or_si128(ra[0], ra[1]), _mm_or_si128(ra[2], ra[3])),
                    _mm_or_si128(_mm_or_si128(rb[0], rb[1]), _mm_or_si128(rb[2], rb[3])),
                );
                if is_zero(_mm_and_si128(seen, _mm_set1_epi16(!2047))) {
                    let d = |k: usize| _mm_sub_epi16(ra[k], rb[k]);
                    pair16(d(0), d(1), d(2), d(3))
                } else {
                    let lo = |k: usize| _mm_sub_epi32(zx16(ra[k]), zx16(rb[k]));
                    let hi = |k: usize| _mm_sub_epi32(zx16h(ra[k]), zx16h(rb[k]));
                    _mm_unpacklo_epi64(tile32([lo(0), lo(1), lo(2), lo(3)]), tile32([hi(0), hi(1), hi(2), hi(3)]))
                }
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn satd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
            unsafe {
                let mut acc = _mm_setzero_si128();
                if w == 4 {
                    // Two tiles one above the other: rows y and y + 4 share a vector.
                    let mut y = 0;
                    while y + 8 <= h {
                        let row = |p: *const u16, s: usize, r: usize| _mm_unpacklo_epi64(load4(p.add((y + r) * s)), load4(p.add((y + r + 4) * s)));
                        let ra = [row(a, sa, 0), row(a, sa, 1), row(a, sa, 2), row(a, sa, 3)];
                        let rb = [row(b, sb, 0), row(b, sb, 1), row(b, sb, 2), row(b, sb, 3)];
                        acc = _mm_add_epi32(acc, pair(ra, rb));
                        y += 8;
                    }
                    if y < h {
                        // One tile, in the low half; the zero tile above it costs nothing.
                        let row = |p: *const u16, s: usize, r: usize| load4(p.add((y + r) * s));
                        acc = _mm_add_epi32(acc, pair([row(a, sa, 0), row(a, sa, 1), row(a, sa, 2), row(a, sa, 3)], [row(b, sb, 0), row(b, sb, 1), row(b, sb, 2), row(b, sb, 3)]));
                    }
                } else {
                    let mut y = 0;
                    while y < h {
                        let ra = a.add(y * sa);
                        let rb = b.add(y * sb);
                        let mut x = 0;
                        while x + 8 <= w {
                            let row = |p: *const u16, s: usize, r: usize| load8(p.add(r * s + x));
                            acc = _mm_add_epi32(acc, pair([row(ra, sa, 0), row(ra, sa, 1), row(ra, sa, 2), row(ra, sa, 3)], [row(rb, sb, 0), row(rb, sb, 1), row(rb, sb, 2), row(rb, sb, 3)]));
                            x += 8;
                        }
                        if x < w {
                            let row = |p: *const u16, s: usize, r: usize| load4(p.add(r * s + x));
                            acc = _mm_add_epi32(acc, pair([row(ra, sa, 0), row(ra, sa, 1), row(ra, sa, 2), row(ra, sa, 3)], [row(rb, sb, 0), row(rb, sb, 1), row(rb, sb, 2), row(rb, sb, 3)]));
                        }
                        y += 4;
                    }
                }
                // Lanes are [A, A, B, B] sums: one of each.
                (_mm_cvtsi128_si32(acc) as u32).wrapping_add(_mm_cvtsi128_si32(_mm_srli_si128(acc, 8)) as u32)
            }
        }

        pub(crate) fn satd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
            if w % 4 != 0 || h % 4 != 0 || h == 0 {
                return satd_scalar(a, a_stride, b, b_stride, w, h);
            }
            assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
            unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }
    };
}

/// SSE2: baseline on x86-64.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSSE3: `pabsw` / `pabsd` in the SATD.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// SSE4.1: `ptest` and `pmovzxwd` in the SATD.
pub(crate) mod sse41 {
    #![allow(dead_code)]
    kernels!("sse4.1", sse41);
}

/// AVX: the SSE4.1 primitive set, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// AVX2: sixteen samples a vector, for the block widths that have them.
/// Narrower blocks take the [`avx`] kernels.
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    use crate::dsp::distortion::DistortionDsp;

    pub(crate) fn install(d: &mut DistortionDsp<u16>) {
        d.sad = sad;
        d.satd = satd;
        d.ssd = ssd;
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load16(p: *const u16) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn absdiff(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a)) }
    }

    /// Adjacent u16 lanes summed into i32, each sum less 65536.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fold(v: __m256i) -> __m256i {
        unsafe { _mm256_madd_epi16(_mm256_xor_si256(v, _mm256_set1_epi16(i16::MIN)), _mm256_set1_epi16(1)) }
    }

    /// The eight i32 lanes of `v`, summed in i64.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn lanes_i64(v: __m256i) -> i64 {
        unsafe {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            t.iter().map(|&x| x as i64).sum()
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sad_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut lanes = 0u32;
            for y in 0..h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x < w {
                    acc = _mm256_add_epi32(acc, fold(absdiff(load16(ra.add(x)), load16(rb.add(x)))));
                    x += 16;
                    lanes += 16;
                }
            }
            (lanes_i64(acc) as u32).wrapping_add(lanes.wrapping_mul(32768))
        }
    }

    fn sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
        if w % 16 != 0 || h == 0 {
            return super::avx::sad(a, a_stride, b, b_stride, w, h);
        }
        assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
        unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn ssd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u64 {
        unsafe {
            let (mut total_lo, mut total_hi) = (0i64, 0i64);
            let mut lanes = 0i64;
            for y in 0..h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                let mut x = 0;
                while x < w {
                    let d = absdiff(load16(ra.add(x)), load16(rb.add(x)));
                    lo = _mm256_add_epi32(lo, fold(_mm256_mullo_epi16(d, d)));
                    hi = _mm256_add_epi32(hi, fold(_mm256_mulhi_epu16(d, d)));
                    x += 16;
                    lanes += 16;
                }
                total_lo += lanes_i64(lo);
                total_hi += lanes_i64(hi);
            }
            let fix = lanes * 32768;
            (((total_hi + fix) as u64) << 16).wrapping_add((total_lo + fix) as u64)
        }
    }

    fn ssd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u64 {
        if w % 16 != 0 || h == 0 {
            return super::avx::ssd(a, a_stride, b, b_stride, w, h);
        }
        assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
        unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn butterfly16(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> [__m256i; 4] {
        unsafe {
            let s0 = _mm256_add_epi16(r0, r3);
            let s1 = _mm256_add_epi16(r1, r2);
            let s2 = _mm256_sub_epi16(r1, r2);
            let s3 = _mm256_sub_epi16(r0, r3);
            [_mm256_add_epi16(s0, s1), _mm256_add_epi16(s3, s2), _mm256_sub_epi16(s0, s1), _mm256_sub_epi16(s3, s2)]
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn butterfly32(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> [__m256i; 4] {
        unsafe {
            let s0 = _mm256_add_epi32(r0, r3);
            let s1 = _mm256_add_epi32(r1, r2);
            let s2 = _mm256_sub_epi32(r1, r2);
            let s3 = _mm256_sub_epi32(r0, r3);
            [_mm256_add_epi32(s0, s1), _mm256_add_epi32(s3, s2), _mm256_sub_epi32(s0, s1), _mm256_sub_epi32(s3, s2)]
        }
    }

    /// Four tiles across in i16, two per 128-bit lane: the 128-bit `pair16`
    /// twice over, every unpack and shuffle staying in its lane. Returns
    /// `[A, A, B, B | C, C, D, D]`, rounded per tile.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quad16(d: [__m256i; 4]) -> __m256i {
        unsafe {
            let [t0, t1, t2, t3] = butterfly16(d[0], d[1], d[2], d[3]);
            let u0 = _mm256_unpacklo_epi16(t0, t1);
            let u1 = _mm256_unpacklo_epi16(t2, t3);
            let u2 = _mm256_unpackhi_epi16(t0, t1);
            let u3 = _mm256_unpackhi_epi16(t2, t3);
            let v0 = _mm256_unpacklo_epi32(u0, u1);
            let v1 = _mm256_unpackhi_epi32(u0, u1);
            let v2 = _mm256_unpacklo_epi32(u2, u3);
            let v3 = _mm256_unpackhi_epi32(u2, u3);
            let [w0, w1, w2, w3] =
                butterfly16(_mm256_unpacklo_epi64(v0, v2), _mm256_unpackhi_epi64(v0, v2), _mm256_unpacklo_epi64(v1, v3), _mm256_unpackhi_epi64(v1, v3));
            let s = _mm256_add_epi16(_mm256_add_epi16(_mm256_abs_epi16(w0), _mm256_abs_epi16(w1)), _mm256_add_epi16(_mm256_abs_epi16(w2), _mm256_abs_epi16(w3)));
            let p = fold(s);
            let q = _mm256_add_epi32(p, _mm256_shuffle_epi32(p, 0b10_11_00_01));
            _mm256_srli_epi32(_mm256_add_epi32(q, _mm256_set1_epi32(131072 + 1)), 1)
        }
    }

    /// Two tiles of i32 differences per 128-bit lane (four rows of four
    /// each), as four equal lanes per tile, rounded.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn tiles32(r: [__m256i; 4]) -> __m256i {
        unsafe {
            let [t0, t1, t2, t3] = butterfly32(r[0], r[1], r[2], r[3]);
            let u0 = _mm256_unpacklo_epi32(t0, t1);
            let u1 = _mm256_unpacklo_epi32(t2, t3);
            let u2 = _mm256_unpackhi_epi32(t0, t1);
            let u3 = _mm256_unpackhi_epi32(t2, t3);
            let [w0, w1, w2, w3] =
                butterfly32(_mm256_unpacklo_epi64(u0, u1), _mm256_unpackhi_epi64(u0, u1), _mm256_unpacklo_epi64(u2, u3), _mm256_unpackhi_epi64(u2, u3));
            let s = _mm256_add_epi32(_mm256_add_epi32(_mm256_abs_epi32(w0), _mm256_abs_epi32(w1)), _mm256_add_epi32(_mm256_abs_epi32(w2), _mm256_abs_epi32(w3)));
            let x = _mm256_add_epi32(s, _mm256_shuffle_epi32(s, 0b01_00_11_10));
            let t = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, 0b10_11_00_01));
            _mm256_srli_epi32(_mm256_add_epi32(t, _mm256_set1_epi32(1)), 1)
        }
    }

    /// The four tiles whose sample rows are `ra` and `rb`, sixteen wide:
    /// in i16 if every sample is below 2048, else in i32 — the tiles of
    /// columns 0..4 and 8..12 from the low unpack, 4..8 and 12..16 from the
    /// high one, laid out `[A, A, B, B | C, C, D, D]` like the narrow path's.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quad(ra: [__m256i; 4], rb: [__m256i; 4]) -> __m256i {
        unsafe {
            let seen = _mm256_or_si256(
                _mm256_or_si256(_mm256_or_si256(ra[0], ra[1]), _mm256_or_si256(ra[2], ra[3])),
                _mm256_or_si256(_mm256_or_si256(rb[0], rb[1]), _mm256_or_si256(rb[2], rb[3])),
            );
            let wide = _mm256_and_si256(seen, _mm256_set1_epi16(!2047));
            if _mm256_testz_si256(wide, wide) != 0 {
                quad16(std::array::from_fn(|k| _mm256_sub_epi16(ra[k], rb[k])))
            } else {
                let zero = _mm256_setzero_si256();
                let lo = std::array::from_fn(|k| _mm256_sub_epi32(_mm256_unpacklo_epi16(ra[k], zero), _mm256_unpacklo_epi16(rb[k], zero)));
                let hi = std::array::from_fn(|k| _mm256_sub_epi32(_mm256_unpackhi_epi16(ra[k], zero), _mm256_unpackhi_epi16(rb[k], zero)));
                _mm256_unpacklo_epi64(tiles32(lo), tiles32(hi))
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn satd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut y = 0;
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x < w {
                    let rows = |p: *const u16, s: usize| std::array::from_fn(|r| load16(p.add(r * s + x)));
                    acc = _mm256_add_epi32(acc, quad(rows(ra, sa), rows(rb, sb)));
                    x += 16;
                }
                y += 4;
            }
            // [A, A, B, B | C, C, D, D]: fold the lanes, then one of each.
            let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            (_mm_cvtsi128_si32(s) as u32).wrapping_add(_mm_cvtsi128_si32(_mm_srli_si128(s, 8)) as u32)
        }
    }

    fn satd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
        if w % 16 != 0 || h % 4 != 0 || h == 0 {
            return super::avx::satd(a, a_stride, b, b_stride, w, h);
        }
        assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
        unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }
}

/// Install the best 16-bit-sample distortion kernels `cpu` can run, one rung
/// at a time.
pub fn install(d: &mut DistortionDsp<u16>, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_satd(d);
    }
    if cpu.sse41 {
        sse41::install_satd(d);
    }
    if cpu.avx {
        avx::install_all(d);
    }
    if cpu.avx2 {
        avx2::install(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::u16_sweep;

    /// Every rung the host can run, installed cumulatively as in the field.
    fn rungs() -> Vec<(&'static str, DistortionDsp<u16>)> {
        let base = Cpu::SCALAR;
        [
            ("sse2", Cpu { sse2: true, ..base }, std::is_x86_feature_detected!("sse2")),
            ("ssse3", Cpu { sse2: true, ssse3: true, ..base }, std::is_x86_feature_detected!("ssse3")),
            ("sse4.1", Cpu { sse2: true, ssse3: true, sse41: true, ..base }, std::is_x86_feature_detected!("sse4.1")),
            ("avx", Cpu { sse2: true, ssse3: true, sse41: true, avx: true, ..base }, std::is_x86_feature_detected!("avx")),
            ("avx2", Cpu { sse2: true, ssse3: true, sse41: true, avx: true, avx2: true, ..base }, std::is_x86_feature_detected!("avx2")),
        ]
        .into_iter()
        .filter(|&(_, _, have)| have)
        .map(|(n, c, _)| {
            let mut d = DistortionDsp::<u16>::scalar();
            install(&mut d, c);
            (n, d)
        })
        .collect()
    }

    #[test]
    fn every_rung_matches_scalar_at_every_depth() {
        let r = rungs();
        assert!(!r.is_empty(), "no x86 rung to test");
        match u16_sweep::distortion(&r) {
            Ok(n) => assert!(n > 0, "the sweep compared nothing"),
            Err(e) => panic!("{e}"),
        }
    }

    /// `DistortionDsp::<u16>::new` reaches these through the sample-type
    /// dispatch.
    #[test]
    fn new_installs_the_u16_tiers() {
        let cpu = Cpu::detect();
        if !cpu.sse2 {
            return;
        }
        let d = DistortionDsp::<u16>::new(cpu);
        let s = DistortionDsp::<u16>::scalar();
        assert!(d.sad as usize != s.sad as usize, "u16 sad still scalar");
        assert!(d.satd as usize != s.satd as usize, "u16 satd still scalar");
        assert!(d.ssd as usize != s.ssd as usize, "u16 ssd still scalar");
    }

    /// Nanoseconds per call of each metric, scalar against each rung, over
    /// the shapes the encoders use, at 10 and at 12 bits (whose SATD takes
    /// the other path). Not a correctness test: `cargo test --release
    /// distortion_x86_u16 -- --ignored --nocapture` prints it; the two
    /// scalar rows are the same-table control.
    #[test]
    #[ignore]
    fn kernel_bench() {
        use std::time::Instant;
        let mut seed = 0xbe9c_u64;
        let mut lcg = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let s = DistortionDsp::<u16>::scalar();
        let mut tables = vec![("scalar", s.clone()), ("scalar-again", s)];
        tables.extend(rungs());
        for bits in [10u32, 12] {
            // Two frames of content a motion search compares: a source, and a
            // prediction a small residual away from it.
            let m = (1u32 << bits) - 1;
            let a: Vec<u16> = (0..96 * 96).map(|_| (lcg() % (m + 1)) as u16).collect();
            let b: Vec<u16> = a.iter().map(|&v| (v as i32 + (lcg() % 33) as i32 - 16).clamp(0, m as i32) as u16).collect();
            for &(w, h) in &[(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)] {
                let iters = (2_000_000 / (w * h) * 16) as u32;
                for (name, d) in &tables {
                    for (metric, f) in [("sad", 0), ("satd", 1), ("ssd", 2)] {
                        let mut sink = 0u64;
                        let t = Instant::now();
                        for i in 0..iters as usize {
                            let o = (i & 31) * 3;
                            sink = sink.wrapping_add(match f {
                                0 => (d.sad)(&a[o..], 96, &b[o..], 96, w, h) as u64,
                                1 => (d.satd)(&a[o..], 96, &b[o..], 96, w, h) as u64,
                                _ => (d.ssd)(&a[o..], 96, &b[o..], 96, w, h),
                            });
                        }
                        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
                        println!("{bits}-bit {w}x{h} {metric:4} {name:13} {ns:8.1} ns/call [{}]", sink & 1);
                    }
                }
            }
        }
    }
}
