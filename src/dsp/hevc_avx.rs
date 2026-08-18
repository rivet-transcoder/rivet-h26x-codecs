//! 128-bit SIMD versions of the H.265 kernels (x86-64), for CPUs without AVX2.
//!
//! Half the lanes of [`super::hevc_avx2`] / [`super::hevc_avx2_u8`] and the
//! same arithmetic. Two things make the narrower vector cost less than the
//! width suggests: at 128 bits `packs_epi32` / `packus_epi16` already land
//! their results in order, so every cross-lane `permute4x64` the 256-bit
//! kernels need to undo per-lane packing disappears; and the AVX2 kernels
//! already fall back to a 128-bit body for `w <= 8`, which is most chroma
//! and every small PU.
//!
//! Both sample widths live in one module so the 8-bit kernels can share the
//! 16-bit ones' deblocking filters, `store_n` / `load_n` and inverse
//! transform, exactly as the AVX2 pair does. The whole set is written once
//! and instantiated twice — SSE4.1 and AVX — by the `kernels_u16!` and
//! `kernels_u8!` macros; see [`super::h264_avx`] for why the second
//! instantiation is worth its code size.

#![cfg(target_arch = "x86_64")]

use super::hevc::HevcDsp;
use crate::hevc::tables::TRANSFORM32;

/// The transform matrix rows for size `n` as interleaved pairs of rows
/// `(j, j+1)`: `[c[j][0], c[j+1][0], c[j][1], c[j+1][1], ...]` (n lanes × 2).
///
/// Shared by both instantiations — it is data, not code.
pub(crate) struct PairRows {
    rows32: [[i16; 64]; 16],
    rows16: [[i16; 32]; 8],
    rows8: [[i16; 16]; 4],
    rows4: [[i16; 8]; 2],
}

const fn build_pairs() -> PairRows {
    let mut p = PairRows { rows32: [[0; 64]; 16], rows16: [[0; 32]; 8], rows8: [[0; 16]; 4], rows4: [[0; 8]; 2] };
    let mut j = 0;
    while j < 16 {
        let mut k = 0;
        while k < 32 {
            p.rows32[j][2 * k] = TRANSFORM32[2 * j][k] as i16;
            p.rows32[j][2 * k + 1] = TRANSFORM32[2 * j + 1][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 8 {
        let mut k = 0;
        while k < 16 {
            p.rows16[j][2 * k] = TRANSFORM32[4 * j][k] as i16;
            p.rows16[j][2 * k + 1] = TRANSFORM32[4 * j + 2][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 4 {
        let mut k = 0;
        while k < 8 {
            p.rows8[j][2 * k] = TRANSFORM32[8 * j][k] as i16;
            p.rows8[j][2 * k + 1] = TRANSFORM32[8 * j + 4][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 2 {
        let mut k = 0;
        while k < 4 {
            p.rows4[j][2 * k] = TRANSFORM32[16 * j][k] as i16;
            p.rows4[j][2 * k + 1] = TRANSFORM32[16 * j + 8][k] as i16;
            k += 1;
        }
        j += 1;
    }
    p
}

pub(crate) static PAIRS: PairRows = build_pairs();

#[inline(always)]
pub(crate) fn pair_row(n: usize, j: usize) -> &'static [i16] {
    match n {
        32 => &PAIRS.rows32[j],
        16 => &PAIRS.rows16[j],
        8 => &PAIRS.rows8[j],
        _ => &PAIRS.rows4[j],
    }
}

/// The 16-bit-sample kernels, plus everything the 8-bit ones share.
macro_rules! kernels_u16 {
    ($feat:literal) => {
        use std::arch::x86_64::*;

        use crate::dsp::hevc::HevcDsp;
        use crate::dsp::hevc_avx::pair_row;
        use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS, TRANSFORM32};

        /// Replace the scalar entries of `d` with these kernels.
        pub fn install_u16(d: &mut HevcDsp<u16>) {
            d.idct = [idct::<4>, idct::<8>, idct::<16>, idct::<32>];
            d.add_residual = add_residual;
            d.qpel_copy = copy_u16;
            d.qpel_h = qpel_h;
            d.qpel_v = qpel_v;
            d.qpel_v2 = qpel_v2;
            d.epel_copy = copy_u16;
            d.epel_h = epel_h;
            d.epel_v = epel_v;
            d.epel_v2 = epel_v2;
            d.uni = uni;
            d.bi = bi;
            d.weighted_uni = weighted_uni;
            d.weighted_bi = weighted_bi;
            d.sao_band = sao_band;
            d.sao_edge = sao_edge;
            d.deblock_luma_v = deblock_luma_v;
            d.deblock_luma_h = deblock_luma_h;
            d.deblock_chroma_v = deblock_chroma_v;
            d.deblock_chroma_h = deblock_chroma_h;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        /// A pair of taps `(a, b)` broadcast as 32-bit lanes `a | b << 16`.
        #[inline(always)]
        fn pair(a: i8, b: i8) -> i32 {
            (a as i16 as u16 as i32) | ((b as i16 as u16 as i32) << 16)
        }

        /// Store the first `n` (≤ 8) lanes of `v` to `dst`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_n(dst: *mut i16, v: __m128i, n: usize) {
            unsafe {
                match n {
                    8 => _mm_storeu_si128(dst as *mut __m128i, v),
                    4 => _mm_storel_epi64(dst as *mut __m128i, v),
                    _ => {
                        let mut t = [0i16; 8];
                        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                        std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
                    }
                }
            }
        }

        /// Store the first `n` (≤ 8) lanes of `v` (u16 samples) to `dst`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_n_u16(dst: *mut u16, v: __m128i, n: usize) {
            unsafe { store_n(dst as *mut i16, v, n) }
        }

        /// Load 8 lanes from `src`, or the first `avail` zero-padded.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_n(src: *const i16, avail: usize) -> __m128i {
            unsafe {
                if avail >= 8 {
                    _mm_loadu_si128(src as *const __m128i)
                } else if avail == 4 {
                    _mm_loadl_epi64(src as *const __m128i)
                } else {
                    let mut t = [0i16; 8];
                    std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
                    _mm_loadu_si128(t.as_ptr() as *const __m128i)
                }
            }
        }

        /// Whether reading `w` lanes into a row of `stride`, for `rows` rows,
        /// plus `extra` samples along, stays inside `len` for an 8-lane load.
        #[inline(always)]
        fn fits(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
            let last_x = if w == 0 { 0 } else { (w - 1) / 8 * 8 };
            (rows - 1) * stride + last_x + extra + 8 <= len
        }

        /// Clip 8 lanes of i16 to `0..=max` (max < 32768) as u16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn clip_u16(v: __m128i, maxv: __m128i) -> __m128i {
            unsafe { _mm_min_epi16(_mm_max_epi16(v, _mm_setzero_si128()), maxv) }
        }

        // ------------------------------------------------------------------
        // Interpolation
        // ------------------------------------------------------------------

        fn copy_u16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
            if !fits(src.len(), src_stride, h, w, 0) {
                return (HevcDsp::<u16>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
            }
            unsafe { copy_u16_impl(dst, src, src_stride, w, h, shift) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn copy_u16_impl(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
            unsafe {
                let sh = _mm_cvtsi32_si128(shift);
                for y in 0..h {
                    let s = src.as_ptr().add(y * src_stride);
                    let d = dst.as_mut_ptr().add(y * w);
                    let mut x = 0;
                    while x < w {
                        let v = _mm_loadu_si128(s.add(x) as *const __m128i);
                        store_n(d.add(x), _mm_sll_epi16(v, sh), (w - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        /// Horizontal FIR with `TAPS` taps over u16 samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fir_h<const TAPS: usize>(dst: *mut i16, src: *const u16, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
            unsafe {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..TAPS / 2 {
                    c[k] = _mm_set1_epi32(pair(taps[2 * k], taps[2 * k + 1]));
                }
                let sh = _mm_cvtsi32_si128(shift);
                for y in 0..h {
                    let s = src.add(y * src_stride);
                    let d = dst.add(y * w);
                    let mut x = 0;
                    while x < w {
                        let mut lo = _mm_setzero_si128();
                        let mut hi = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadu_si128(s.add(x + 2 * k) as *const __m128i);
                            let b = _mm_loadu_si128(s.add(x + 2 * k + 1) as *const __m128i);
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c[k]));
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c[k]));
                        }
                        let r = _mm_packs_epi32(_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
                        store_n(d.add(x), r, (w - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        /// Vertical FIR with `TAPS` taps over u16 or i16 rows (`T` = 2-byte lanes).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fir_v<const TAPS: usize, T>(dst: *mut i16, src: *const T, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
            unsafe {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..TAPS / 2 {
                    c[k] = _mm_set1_epi32(pair(taps[2 * k], taps[2 * k + 1]));
                }
                let sh = _mm_cvtsi32_si128(shift);
                for y in 0..h {
                    let d = dst.add(y * w);
                    let mut x = 0;
                    while x < w {
                        let mut lo = _mm_setzero_si128();
                        let mut hi = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadu_si128(src.add((y + 2 * k) * src_stride + x) as *const __m128i);
                            let b = _mm_loadu_si128(src.add((y + 2 * k + 1) * src_stride + x) as *const __m128i);
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c[k]));
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c[k]));
                        }
                        let r = _mm_packs_epi32(_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
                        store_n(d.add(x), r, (w - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        fn qpel_h(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits(src.len(), src_stride, h, w, 8) {
                return (HevcDsp::<u16>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
            }
            unsafe { fir_h::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
        }

        fn qpel_v(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits(src.len(), src_stride, h + 7, w, 0) {
                return (HevcDsp::<u16>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
            }
            unsafe { fir_v::<8, u16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
        }

        pub(super) fn qpel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
            if !fits(src.len(), src_stride, h + 7, w, 0) {
                return (HevcDsp::<u16>::SCALAR.qpel_v2)(dst, src, src_stride, w, h, frac);
            }
            unsafe { fir_v::<8, i16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], 6) }
        }

        fn epel_h(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits(src.len(), src_stride, h, w, 4) {
                return (HevcDsp::<u16>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
            }
            unsafe { fir_h::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
        }

        fn epel_v(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits(src.len(), src_stride, h + 3, w, 0) {
                return (HevcDsp::<u16>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
            }
            unsafe { fir_v::<4, u16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
        }

        pub(super) fn epel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
            if !fits(src.len(), src_stride, h + 3, w, 0) {
                return (HevcDsp::<u16>::SCALAR.epel_v2)(dst, src, src_stride, w, h, frac);
            }
            unsafe { fir_v::<4, i16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], 6) }
        }

        // ------------------------------------------------------------------
        // Combination / weighting
        // ------------------------------------------------------------------

        fn uni(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            unsafe { uni_impl(dst, stride, src, w, h, shift, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn uni_impl(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            unsafe {
                let round = _mm_set1_epi16(if shift > 0 { 1 << (shift - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(shift);
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let s = load_n(src.as_ptr().add(y * w + x), w - x);
                        // 14-bit + round fits i16 (< 16384 + 8192).
                        let v = _mm_sra_epi16(_mm_adds_epi16(s, round), sh);
                        store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                        x += 8;
                    }
                }
            }
        }

        fn bi(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            unsafe { bi_impl(dst, stride, a, b, w, h, shift, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn bi_impl(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            unsafe {
                let round = _mm_set1_epi32(1 << (shift - 1));
                let sh = _mm_cvtsi32_si128(shift);
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let va = load_n(a.as_ptr().add(y * w + x), w - x);
                        let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                        // Sum in 32 bits (a + b can exceed i16), shift, pack.
                        let wide = |v: __m128i, hi: bool| if hi { _mm_cvtepi16_epi32(_mm_srli_si128(v, 8)) } else { _mm_cvtepi16_epi32(v) };
                        let lo = _mm_add_epi32(_mm_add_epi32(wide(va, false), wide(vb, false)), round);
                        let hi = _mm_add_epi32(_mm_add_epi32(wide(va, true), wide(vb, true)), round);
                        let p = _mm_packs_epi32(_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
                        store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                        x += 8;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_uni(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
            unsafe { weighted_uni_impl(dst, stride, src, w, h, log2_wd, wt, o, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_uni_impl(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
            unsafe {
                let round = _mm_set1_epi32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(log2_wd.max(0));
                let wv = _mm_set1_epi32(wt);
                let ov = _mm_set1_epi32(o);
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let s = load_n(src.as_ptr().add(y * w + x), w - x);
                        let quad = |v: __m128i| _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(v, wv), round), sh), ov);
                        let lo = quad(_mm_cvtepi16_epi32(s));
                        let hi = quad(_mm_cvtepi16_epi32(_mm_srli_si128(s, 8)));
                        let p = _mm_packs_epi32(lo, hi);
                        store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                        x += 8;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_bi(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
            unsafe { weighted_bi_impl(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_bi_impl(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
            unsafe {
                let round = _mm_set1_epi32((o0 + o1 + 1) << log2_wd);
                let sh = _mm_cvtsi32_si128(log2_wd + 1);
                let w0v = _mm_set1_epi32(w0);
                let w1v = _mm_set1_epi32(w1);
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let va = load_n(a.as_ptr().add(y * w + x), w - x);
                        let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                        let quad = |pa: __m128i, pb: __m128i| {
                            _mm_sra_epi32(_mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(pa, w0v), _mm_mullo_epi32(pb, w1v)), round), sh)
                        };
                        let lo = quad(_mm_cvtepi16_epi32(va), _mm_cvtepi16_epi32(vb));
                        let hi = quad(_mm_cvtepi16_epi32(_mm_srli_si128(va, 8)), _mm_cvtepi16_epi32(_mm_srli_si128(vb, 8)));
                        let p = _mm_packs_epi32(lo, hi);
                        store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                        x += 8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Residual add
        // ------------------------------------------------------------------

        fn add_residual(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
            unsafe { add_residual_impl(dst, stride, res, n, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn add_residual_impl(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
            unsafe {
                let maxv = _mm_set1_epi16(max as i16);
                let zero = _mm_setzero_si128();
                if n >= 8 {
                    for y in 0..n {
                        let mut x = 0;
                        while x < n {
                            let d = dst.as_mut_ptr().add(y * stride + x);
                            let p = _mm_loadu_si128(d as *const __m128i);
                            let r = _mm_loadu_si128(res.as_ptr().add(y * n + x) as *const __m128i);
                            // Samples < 4096 and residuals fit: adds saturate correctly.
                            let v = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(p, r), zero), maxv);
                            _mm_storeu_si128(d as *mut __m128i, v);
                            x += 8;
                        }
                    }
                } else {
                    // 4x4: two rows per 128-bit vector.
                    for y in (0..4).step_by(2) {
                        let d0 = dst.as_mut_ptr().add(y * stride);
                        let d1 = dst.as_mut_ptr().add((y + 1) * stride);
                        let p = _mm_unpacklo_epi64(_mm_loadl_epi64(d0 as *const __m128i), _mm_loadl_epi64(d1 as *const __m128i));
                        let r = _mm_loadu_si128(res.as_ptr().add(y * 4) as *const __m128i);
                        let v = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(p, r), zero), maxv);
                        _mm_storel_epi64(d0 as *mut __m128i, v);
                        _mm_storel_epi64(d1 as *mut __m128i, _mm_unpackhi_epi64(v, v));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Inverse DCT
        // ------------------------------------------------------------------

        pub(super) fn idct<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
            if max_x == 0 && max_y == 0 {
                // DC only.
                let round2 = 1i32 << (bd_shift - 1);
                let v = ((coeffs[0] as i32 * 64 + 64) >> 7).clamp(-32768, 32767);
                let r = ((v * 64 + round2) >> bd_shift).clamp(-32768, 32767) as i16;
                coeffs[..N * N].fill(r);
                return;
            }
            if N == 4 {
                // Not worth a vector: the scalar butterfly is 4 lines.
                return (HevcDsp::<u16>::SCALAR.idct[0])(coeffs, bd_shift, max_x, max_y);
            }
            unsafe { idct_impl::<N>(coeffs, bd_shift, max_x, max_y) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn idct_impl<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
            unsafe {
                let mut tmp = [0i16; 32 * 32];
                // Stage 1 (columns): tmp[y][x] = clip((sum_j c[j][y] * coef[j][x] + 64) >> 7),
                // vectorised across x for each y; pairs of input rows (j, j+1).
                let nzy = max_y + 1;
                let npairs = nzy.div_ceil(2);
                let round1 = _mm_set1_epi32(64);
                let step = 32 / N;
                for y in 0..N {
                    let mut x = 0;
                    while x <= max_x {
                        let mut lo = round1;
                        let mut hi = round1;
                        for p in 0..npairs {
                            let j = 2 * p;
                            let a = load_n(coeffs.as_ptr().add(j * N + x), N - x);
                            let b = if j + 1 < nzy { load_n(coeffs.as_ptr().add((j + 1) * N + x), N - x) } else { _mm_setzero_si128() };
                            let c = _mm_set1_epi32(pair(TRANSFORM32[j * step][y], TRANSFORM32[(j + 1) * step][y]));
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c));
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c));
                        }
                        let r = _mm_packs_epi32(_mm_srai_epi32(lo, 7), _mm_srai_epi32(hi, 7));
                        store_n(tmp.as_mut_ptr().add(y * N + x), r, (N - x).min(8));
                        x += 8;
                    }
                }
                // Stage 2 (rows): out[y][x] = clip((sum_j c[j][x] * tmp[y][j] + round) >> shift),
                // vectorised across x with the interleaved pair rows of the matrix.
                let nzx = max_x + 1;
                let npairs = nzx.div_ceil(2);
                let round2 = _mm_set1_epi32(1 << (bd_shift - 1));
                let sh = _mm_cvtsi32_si128(bd_shift);
                for y in 0..N {
                    let row = tmp.as_ptr().add(y * N);
                    let mut x = 0;
                    while x < N {
                        let mut lo = round2;
                        let mut hi = round2;
                        for p in 0..npairs {
                            let j = 2 * p;
                            let t0 = *row.add(j) as i32;
                            let t1 = if j + 1 < nzx { *row.add(j + 1) as i32 } else { 0 };
                            let tv = _mm_set1_epi32((t0 as u16 as i32) | ((t1 as u16 as i32) << 16));
                            let pr = pair_row(N, p);
                            let cl = _mm_loadu_si128(pr.as_ptr().add(2 * x) as *const __m128i); // pairs for x..x+4
                            let ch = if N - x > 4 { _mm_loadu_si128(pr.as_ptr().add(2 * x + 8) as *const __m128i) } else { _mm_setzero_si128() };
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(cl, tv));
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(ch, tv));
                        }
                        // At 128 bits `packs` keeps outputs x..x+7 in order.
                        let r = _mm_packs_epi32(_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
                        store_n(coeffs.as_mut_ptr().add(y * N + x), r, (N - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // SAO
        // ------------------------------------------------------------------

        #[allow(clippy::too_many_arguments)]
        fn sao_band(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
            unsafe { sao_band_impl(dst, dst_stride, src, src_stride, w, h, table, shift, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn sao_band_impl(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
            unsafe {
                // The four consecutive bands (mod 32) with nonzero offsets.
                let mut bands = [0i16; 4];
                let mut offs = [0i16; 4];
                let mut k = 0;
                for b in 0..32 {
                    if table[b] != 0 && k < 4 {
                        bands[k] = b as i16;
                        offs[k] = table[b];
                        k += 1;
                    }
                }
                let sh = _mm_cvtsi32_si128(shift);
                let maxv = _mm_set1_epi16(max as i16);
                let bv: [__m128i; 4] = std::array::from_fn(|i| _mm_set1_epi16(bands[i]));
                let ov: [__m128i; 4] = std::array::from_fn(|i| _mm_set1_epi16(offs[i]));
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let s = src.as_ptr().add(y * src_stride + x);
                        let v = if n == 8 { _mm_loadu_si128(s as *const __m128i) } else { load_n(s as *const i16, n) };
                        let band = _mm_srl_epi16(v, sh);
                        let mut off = _mm_setzero_si128();
                        for i in 0..k {
                            off = _mm_blendv_epi8(off, ov[i], _mm_cmpeq_epi16(band, bv[i]));
                        }
                        let r = clip_u16(_mm_add_epi16(v, off), maxv);
                        store_n_u16(dst.as_mut_ptr().add(y * dst_stride + x), r, n);
                        x += 8;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn sao_edge(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
            unsafe { sao_edge_impl(dst, src, origin, stride, w, h, na, nb, off, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn sao_edge_impl(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
            unsafe {
                let maxv = _mm_set1_epi16(max as i16);
                let one = _mm_set1_epi16(1);
                // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 → offsets via compares.
                let o0 = _mm_set1_epi16(off[0]);
                let o1 = _mm_set1_epi16(off[1]);
                let o3 = _mm_set1_epi16(off[3]);
                let o4 = _mm_set1_epi16(off[4]);
                let two = _mm_set1_epi16(2);
                let three = _mm_set1_epi16(3);
                let four = _mm_set1_epi16(4);
                let lo_reach = na.min(nb).min(0);
                let hi_reach = na.max(nb).max(0);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let i = origin + y * stride + x;
                        // All three loads and the store must stay inside.
                        if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 8 > src.len() || i + 8 > dst.len() {
                            // Tail near the buffer end: scalar.
                            for xx in x..w {
                                let ii = origin + y * stride + xx;
                                let v = src[ii] as i32;
                                let a = src[(ii as isize + na) as usize] as i32;
                                let b = src[(ii as isize + nb) as usize] as i32;
                                let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                                dst[ii] = (v + off[e] as i32).clamp(0, max) as u16;
                            }
                            break;
                        }
                        let v = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                        let a = _mm_loadu_si128(src.as_ptr().offset(i as isize + na) as *const __m128i);
                        let b = _mm_loadu_si128(src.as_ptr().offset(i as isize + nb) as *const __m128i);
                        // sign(v - a) = (v > a) - (v < a); samples < 32768 so signed compares are exact.
                        let sa = _mm_sub_epi16(_mm_and_si128(_mm_cmpgt_epi16(v, a), one), _mm_and_si128(_mm_cmpgt_epi16(a, v), one));
                        let sb = _mm_sub_epi16(_mm_and_si128(_mm_cmpgt_epi16(v, b), one), _mm_and_si128(_mm_cmpgt_epi16(b, v), one));
                        let e = _mm_add_epi16(_mm_add_epi16(sa, sb), two);
                        let mut o = _mm_setzero_si128();
                        o = _mm_blendv_epi8(o, o0, _mm_cmpeq_epi16(e, _mm_setzero_si128()));
                        o = _mm_blendv_epi8(o, o1, _mm_cmpeq_epi16(e, one));
                        o = _mm_blendv_epi8(o, o3, _mm_cmpeq_epi16(e, three));
                        o = _mm_blendv_epi8(o, o4, _mm_cmpeq_epi16(e, four));
                        let r = clip_u16(_mm_add_epi16(v, o), maxv);
                        store_n_u16(dst.as_mut_ptr().add(i), r, n);
                        x += 8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Deblocking
        // ------------------------------------------------------------------
        //
        // Four lines of an edge are four i32 lanes per sample position
        // (p3..q3), which holds every bit depth up to 12 without overflow —
        // and four lines is exactly one luma segment, so where the 256-bit
        // kernel filters two segments at once with per-segment lane masks,
        // this one runs the filter twice and the masks collapse to scalar
        // booleans. Chroma segments are two lines, so a vector still holds
        // two of them.

        /// Eight consecutive u16 as two vectors of 4 x i32 (lines 0..3, 4..7).
        #[target_feature(enable = $feat)]
        #[inline]
        pub(super) unsafe fn widen_u16(v: __m128i) -> (__m128i, __m128i) {
            unsafe { (_mm_cvtepu16_epi32(v), _mm_cvtepu16_epi32(_mm_srli_si128(v, 8))) }
        }

        /// Two vectors of 4 x i32 (each within u16) back to eight u16.
        #[target_feature(enable = $feat)]
        #[inline]
        pub(super) unsafe fn pack8_u16(lo: __m128i, hi: __m128i) -> __m128i {
            unsafe { _mm_packus_epi32(lo, hi) }
        }

        /// Transpose eight 8-lane u16 rows.
        #[target_feature(enable = $feat)]
        #[inline]
        pub(super) unsafe fn transpose8_u16(r: &mut [__m128i; 8]) {
            unsafe {
                let a0 = _mm_unpacklo_epi16(r[0], r[1]);
                let a1 = _mm_unpackhi_epi16(r[0], r[1]);
                let a2 = _mm_unpacklo_epi16(r[2], r[3]);
                let a3 = _mm_unpackhi_epi16(r[2], r[3]);
                let a4 = _mm_unpacklo_epi16(r[4], r[5]);
                let a5 = _mm_unpackhi_epi16(r[4], r[5]);
                let a6 = _mm_unpacklo_epi16(r[6], r[7]);
                let a7 = _mm_unpackhi_epi16(r[6], r[7]);
                let b0 = _mm_unpacklo_epi32(a0, a2);
                let b1 = _mm_unpackhi_epi32(a0, a2);
                let b2 = _mm_unpacklo_epi32(a1, a3);
                let b3 = _mm_unpackhi_epi32(a1, a3);
                let b4 = _mm_unpacklo_epi32(a4, a6);
                let b5 = _mm_unpackhi_epi32(a4, a6);
                let b6 = _mm_unpacklo_epi32(a5, a7);
                let b7 = _mm_unpackhi_epi32(a5, a7);
                r[0] = _mm_unpacklo_epi64(b0, b4);
                r[1] = _mm_unpackhi_epi64(b0, b4);
                r[2] = _mm_unpacklo_epi64(b1, b5);
                r[3] = _mm_unpackhi_epi64(b1, b5);
                r[4] = _mm_unpacklo_epi64(b2, b6);
                r[5] = _mm_unpackhi_epi64(b2, b6);
                r[6] = _mm_unpacklo_epi64(b3, b7);
                r[7] = _mm_unpackhi_epi64(b3, b7);
            }
        }

        /// The luma filter on one four-line segment, in place.
        #[target_feature(enable = $feat)]
        #[inline]
        pub(super) unsafe fn luma_filter4(v: &mut [__m128i; 8], beta: i32, tc: i32, no_p: bool, no_q: bool, max: i32) {
            unsafe {
                if beta == 0 && tc == 0 {
                    return;
                }
                let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
                let add = |a, b| _mm_add_epi32(a, b);
                let sub = |a, b| _mm_sub_epi32(a, b);
                let dbl = |a| _mm_slli_epi32(a, 1);
                let absd = |a, b| _mm_abs_epi32(_mm_sub_epi32(a, b));
                // Lane-wise measures.
                let dpv = _mm_abs_epi32(add(sub(p2, dbl(p1)), p0));
                let dqv = _mm_abs_epi32(add(sub(q2, dbl(q1)), q0));
                let ev = add(absd(p3, p0), absd(q0, q3));
                let fv = absd(p0, q0);
                let mut dp = [0i32; 4];
                let mut dq = [0i32; 4];
                let mut e = [0i32; 4];
                let mut f = [0i32; 4];
                _mm_storeu_si128(dp.as_mut_ptr() as *mut __m128i, dpv);
                _mm_storeu_si128(dq.as_mut_ptr() as *mut __m128i, dqv);
                _mm_storeu_si128(e.as_mut_ptr() as *mut __m128i, ev);
                _mm_storeu_si128(f.as_mut_ptr() as *mut __m128i, fv);
                // The segment's decisions, from its lines 0 and 3.
                let dpq0 = dp[0] + dq[0];
                let dpq3 = dp[3] + dq[3];
                if dpq0 + dpq3 >= beta {
                    return;
                }
                let dsam = |l: usize, dpq: i32| dpq < (beta >> 2) && e[l] < (beta >> 3) && f[l] < ((5 * tc + 1) >> 1);
                let strong = dsam(0, 2 * dpq0) && dsam(3, 2 * dpq3);
                let side = (beta + (beta >> 1)) >> 3;
                let dep = dp[0] + dp[3] < side;
                let deq = dq[0] + dq[3] < side;
                let zero = _mm_setzero_si128();
                let all = _mm_cmpeq_epi32(zero, zero);
                let m = |b: bool| if b { all } else { zero };
                let strong_m = m(strong);
                let dep_m = m(dep);
                let deq_m = m(deq);
                let wp_m = m(!no_p);
                let wq_m = m(!no_q);
                let tcv = _mm_set1_epi32(tc);
                let tc2 = dbl(tcv);
                let tch = _mm_srai_epi32(tcv, 1);
                let tc10 = _mm_mullo_epi32(tcv, _mm_set1_epi32(10));
                let maxv = _mm_set1_epi32(max);
                let clamp = |x, lo, hi| _mm_min_epi32(_mm_max_epi32(x, lo), hi);
                let two = _mm_set1_epi32(2);
                let four = _mm_set1_epi32(4);
                // Strong.
                let p0q0 = add(p0, q0);
                let sp0 = clamp(_mm_srai_epi32(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)), 3), sub(p0, tc2), add(p0, tc2));
                let sp1 = clamp(_mm_srai_epi32(add(add(p2, p1), add(p0q0, two)), 2), sub(p1, tc2), add(p1, tc2));
                let sp2 = clamp(_mm_srai_epi32(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)), 3), sub(p2, tc2), add(p2, tc2));
                let sq0 = clamp(_mm_srai_epi32(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)), 3), sub(q0, tc2), add(q0, tc2));
                let sq1 = clamp(_mm_srai_epi32(add(add(p0q0, q1), add(q2, two)), 2), sub(q1, tc2), add(q1, tc2));
                let sq2 = clamp(_mm_srai_epi32(add(add(p0q0, q1), add(add(q2, dbl(q2)), add(dbl(q3), four))), 3), sub(q2, tc2), add(q2, tc2));
                // Weak.
                let nine = _mm_set1_epi32(9);
                let three = _mm_set1_epi32(3);
                let delta = _mm_srai_epi32(add(sub(_mm_mullo_epi32(sub(q0, p0), nine), _mm_mullo_epi32(sub(q1, p1), three)), _mm_set1_epi32(8)), 4);
                let w_m = _mm_cmpgt_epi32(tc10, _mm_abs_epi32(delta));
                let delta = clamp(delta, sub(zero, tcv), tcv);
                let wp0 = clamp(add(p0, delta), zero, maxv);
                let wq0 = clamp(sub(q0, delta), zero, maxv);
                let one = _mm_set1_epi32(1);
                let dpv2 = clamp(_mm_srai_epi32(add(sub(_mm_srai_epi32(add(add(p2, p0), one), 1), p1), delta), 1), sub(zero, tch), tch);
                let dqv2 = clamp(_mm_srai_epi32(sub(sub(_mm_srai_epi32(add(add(q2, q0), one), 1), q1), delta), 1), sub(zero, tch), tch);
                let wp1 = clamp(add(p1, dpv2), zero, maxv);
                let wq1 = clamp(add(q1, dqv2), zero, maxv);
                // Combine: strong wins over weak; weak needs its per-line test.
                let np0 = _mm_blendv_epi8(_mm_blendv_epi8(p0, wp0, w_m), sp0, strong_m);
                let nq0 = _mm_blendv_epi8(_mm_blendv_epi8(q0, wq0, w_m), sq0, strong_m);
                let np1 = _mm_blendv_epi8(_mm_blendv_epi8(p1, wp1, _mm_and_si128(w_m, dep_m)), sp1, strong_m);
                let nq1 = _mm_blendv_epi8(_mm_blendv_epi8(q1, wq1, _mm_and_si128(w_m, deq_m)), sq1, strong_m);
                let np2 = _mm_blendv_epi8(p2, sp2, strong_m);
                let nq2 = _mm_blendv_epi8(q2, sq2, strong_m);
                v[1] = _mm_blendv_epi8(p2, np2, wp_m);
                v[2] = _mm_blendv_epi8(p1, np1, wp_m);
                v[3] = _mm_blendv_epi8(p0, np0, wp_m);
                v[4] = _mm_blendv_epi8(q0, nq0, wq_m);
                v[5] = _mm_blendv_epi8(q1, nq1, wq_m);
                v[6] = _mm_blendv_epi8(q2, nq2, wq_m);
            }
        }

        /// The chroma filter on four lines (two segments): `[p1, p0, q0, q1]`.
        #[target_feature(enable = $feat)]
        #[inline]
        pub(super) unsafe fn chroma_filter4(v: &mut [__m128i; 4], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            unsafe {
                let [p1, p0, q0, q1] = *v;
                let tcv = _mm_setr_epi32(tc[0], tc[0], tc[1], tc[1]);
                let m = |a: [bool; 2]| {
                    let x = |b: bool| -(b as i32);
                    _mm_setr_epi32(x(a[0]), x(a[0]), x(a[1]), x(a[1]))
                };
                let on = _mm_cmpgt_epi32(tcv, _mm_setzero_si128());
                let wp = _mm_andnot_si128(m(no_p), on);
                let wq = _mm_andnot_si128(m(no_q), on);
                let zero = _mm_setzero_si128();
                let maxv = _mm_set1_epi32(max);
                let d = _mm_srai_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_slli_epi32(_mm_sub_epi32(q0, p0), 2), _mm_sub_epi32(p1, q1)), _mm_set1_epi32(4)),
                    3,
                );
                let d = _mm_min_epi32(_mm_max_epi32(d, _mm_sub_epi32(zero, tcv)), tcv);
                let np0 = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(p0, d), zero), maxv);
                let nq0 = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(q0, d), zero), maxv);
                v[1] = _mm_blendv_epi8(p0, np0, wp);
                v[2] = _mm_blendv_epi8(q0, nq0, wq);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn deblock_luma_v(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
                return;
            }
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn deblock_luma_v_impl(data: *mut u16, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_loadu_si128(data.add(i * stride).sub(4) as *const __m128i);
                }
                transpose8_u16(&mut r);
                let mut v0 = [_mm_setzero_si128(); 8];
                let mut v1 = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let (a, b) = widen_u16(r[k]);
                    v0[k] = a;
                    v1[k] = b;
                }
                luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
                luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
                for k in 0..8 {
                    r[k] = pack8_u16(v0[k], v1[k]);
                }
                transpose8_u16(&mut r);
                for i in 0..8 {
                    _mm_storeu_si128(data.add(i * stride).sub(4) as *mut __m128i, r[i]);
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn deblock_luma_h(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
                return;
            }
            assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
            unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn deblock_luma_h_impl(data: *mut u16, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            unsafe {
                let mut v0 = [_mm_setzero_si128(); 8];
                let mut v1 = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let p = data.offset((k as isize - 4) * stride as isize);
                    v0[k] = _mm_cvtepu16_epi32(_mm_loadl_epi64(p as *const __m128i));
                    v1[k] = _mm_cvtepu16_epi32(_mm_loadl_epi64(p.add(4) as *const __m128i));
                }
                luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
                luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
                for k in 1..7 {
                    _mm_storeu_si128(data.offset((k as isize - 4) * stride as isize) as *mut __m128i, pack8_u16(v0[k], v1[k]));
                }
            }
        }

        fn deblock_chroma_v(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            if tc.iter().all(|&t| t == 0) {
                return;
            }
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_impl(data: *mut u16, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_loadl_epi64(data.add(i * stride).sub(2) as *const __m128i);
                }
                let a0 = _mm_unpacklo_epi16(r[0], r[1]);
                let a1 = _mm_unpacklo_epi16(r[2], r[3]);
                let a2 = _mm_unpacklo_epi16(r[4], r[5]);
                let a3 = _mm_unpacklo_epi16(r[6], r[7]);
                let b0 = _mm_unpacklo_epi32(a0, a1); // p1 r0..3 | p0 r0..3
                let b1 = _mm_unpackhi_epi32(a0, a1); // q0 r0..3 | q1 r0..3
                let b2 = _mm_unpacklo_epi32(a2, a3); // rows 4..7
                let b3 = _mm_unpackhi_epi32(a2, a3);
                let col = |v: __m128i, hi: bool| if hi { _mm_cvtepu16_epi32(_mm_srli_si128(v, 8)) } else { _mm_cvtepu16_epi32(v) };
                let mut v0 = [col(b0, false), col(b0, true), col(b1, false), col(b1, true)];
                let mut v1 = [col(b2, false), col(b2, true), col(b3, false), col(b3, true)];
                chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
                chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
                // (p0, q0) pairs per row, stored as one 32-bit write each.
                let p0 = pack8_u16(v0[1], v1[1]);
                let q0 = pack8_u16(v0[2], v1[2]);
                let lo = _mm_unpacklo_epi16(p0, q0); // rows 0..3
                let hi = _mm_unpackhi_epi16(p0, q0); // rows 4..7
                let mut t = [0u32; 8];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, lo);
                _mm_storeu_si128(t.as_mut_ptr().add(4) as *mut __m128i, hi);
                for i in 0..8 {
                    std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u32, t[i]);
                }
            }
        }

        fn deblock_chroma_h(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            if tc.iter().all(|&t| t == 0) {
                return;
            }
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_impl(data: *mut u16, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            unsafe {
                let ld = |p: *const u16| -> (__m128i, __m128i) {
                    (_mm_cvtepu16_epi32(_mm_loadl_epi64(p as *const __m128i)), _mm_cvtepu16_epi32(_mm_loadl_epi64(p.add(4) as *const __m128i)))
                };
                let (a0, a1) = ld(data.sub(2 * stride));
                let (b0, b1) = ld(data.sub(stride));
                let (c0, c1) = ld(data);
                let (d0, d1) = ld(data.add(stride));
                let mut v0 = [a0, b0, c0, d0];
                let mut v1 = [a1, b1, c1, d1];
                chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
                chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
                _mm_storeu_si128(data.sub(stride) as *mut __m128i, pack8_u16(v0[1], v1[1]));
                _mm_storeu_si128(data as *mut __m128i, pack8_u16(v0[2], v1[2]));
            }
        }
    };
}

/// The 8-bit-sample kernels. Expanded into the same module as
/// [`kernels_u16!`], whose `store_n` / `load_n`, inverse transform and
/// deblocking filters they share — exactly as `hevc_avx2_u8` shares
/// `hevc_avx2`'s.
macro_rules! kernels_u8 {
    ($feat:literal) => {
        /// Replace the scalar entries of `d` with these kernels.
        pub fn install_u8(d: &mut HevcDsp<u8>) {
            d.idct = [idct::<4>, idct::<8>, idct::<16>, idct::<32>];
            d.add_residual = add_residual_u8;
            d.qpel_copy = copy_u8;
            d.qpel_h = qpel_h_u8;
            d.qpel_v = qpel_v_u8;
            d.qpel_v2 = qpel_v2;
            d.epel_copy = copy_u8;
            d.epel_h = epel_h_u8;
            d.epel_v = epel_v_u8;
            d.epel_v2 = epel_v2;
            d.uni = uni_u8;
            d.bi = bi_u8;
            d.weighted_uni = weighted_uni_u8;
            d.weighted_bi = weighted_bi_u8;
            d.qpel_uni = qpel_uni_u8;
            d.epel_uni = epel_uni_u8;
            d.qpel_bi = qpel_bi_u8;
            d.epel_bi = epel_bi_u8;
            d.fused_mc = true;
            d.sao_band = sao_band_u8;
            d.sao_edge = sao_edge_u8;
            d.deblock_luma_v = deblock_luma_v_u8;
            d.deblock_luma_h = deblock_luma_h_u8;
            d.deblock_chroma_v = deblock_chroma_v_u8;
            d.deblock_chroma_h = deblock_chroma_h_u8;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        /// A pair of taps `(a, b)` as one 16-bit lane `a | b << 8` (the low
        /// byte multiplies the even sample of an interleaved pair).
        #[inline(always)]
        fn pair8(a: i8, b: i8) -> i16 {
            (a as u8 as i16) | ((b as i16) << 8)
        }

        /// Store the first `n` (≤ 8) bytes of `v`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_bytes(dst: *mut u8, v: __m128i, n: usize) {
            unsafe {
                match n {
                    8 => _mm_storel_epi64(dst as *mut __m128i, v),
                    4 => std::ptr::write_unaligned(dst as *mut u32, _mm_cvtsi128_si32(v) as u32),
                    2 => std::ptr::write_unaligned(dst as *mut u16, _mm_cvtsi128_si32(v) as u16),
                    _ => {
                        let mut t = [0u8; 16];
                        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                        std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
                    }
                }
            }
        }

        /// Store the first `n` (≤ 16) bytes of `v` (byte-lane kernels: SAO).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_bytes16(dst: *mut u8, v: __m128i, n: usize) {
            unsafe {
                if n == 16 {
                    _mm_storeu_si128(dst as *mut __m128i, v);
                } else {
                    store_bytes(dst, v, n);
                }
            }
        }

        /// Load 16 bytes, or the first `avail` zero-padded.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_bytes16(src: *const u8, avail: usize) -> __m128i {
            unsafe {
                if avail >= 16 {
                    _mm_loadu_si128(src as *const __m128i)
                } else if avail == 8 {
                    _mm_loadl_epi64(src as *const __m128i)
                } else {
                    let mut t = [0u8; 16];
                    std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
                    _mm_loadu_si128(t.as_ptr() as *const __m128i)
                }
            }
        }

        /// 8 i16 lanes to 8 bytes, saturating to `0..=255`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn pack8(v: __m128i) -> __m128i {
            unsafe { _mm_packus_epi16(v, v) }
        }

        /// Whether a block of width `w` is handled as one contiguous run of
        /// samples (the predictions are stored with stride `w`, so a 2- or
        /// 4-wide block is 4 or 2 rows per 8-lane vector instead of a mostly
        /// idle vector per row).
        #[inline(always)]
        fn narrow(w: usize) -> bool {
            w == 4 || w == 2
        }

        /// Store 8 bytes of `p` as `rows` rows of `w` (2 or 4) bytes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn scatter_rows(dst: *mut u8, stride: usize, w: usize, p: __m128i, rows: usize) {
            unsafe {
                if w == 4 {
                    let mut t = [0u32; 4];
                    _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, p);
                    for r in 0..rows.min(2) {
                        std::ptr::write_unaligned(dst.add(r * stride) as *mut u32, t[r]);
                    }
                } else {
                    let mut t = [0u16; 8];
                    _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, p);
                    for r in 0..rows.min(4) {
                        std::ptr::write_unaligned(dst.add(r * stride) as *mut u16, t[r]);
                    }
                }
            }
        }

        /// Whether reading `w` bytes into a row of `stride`, for `rows` rows,
        /// plus `extra` bytes along, stays inside `len` for the vector width
        /// the byte kernels use at that block width.
        #[inline(always)]
        fn fits_b(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
            let (vec, last_x) = if w <= 8 { (8, 0) } else { (16, (w - 1) / 16 * 16) };
            (rows - 1) * stride + last_x + extra + vec <= len
        }

        /// Whether the second stage's `w`-stride 14-bit rows can be read 8
        /// lanes at a time for `rows` rows within `len`.
        #[inline(always)]
        fn fits_i16(len: usize, w: usize, rows: usize) -> bool {
            let last_x = if w <= 8 { 0 } else { (w - 1) / 8 * 8 };
            (rows - 1) * w + last_x + 8 <= len
        }

        // ------------------------------------------------------------------
        // Interpolation
        // ------------------------------------------------------------------

        /// What a FIR stage produces, per output kind (`MODE_*`).
        #[derive(Clone, Copy)]
        struct Out {
            /// `MODE_I16`: 14-bit predictions, stride `w`.
            i16: *mut i16,
            /// `MODE_UNI` / `MODE_BI`: samples, stride `stride`.
            u8: *mut u8,
            /// Sample stride.
            stride: usize,
            /// `MODE_BI`: the other list's 14-bit prediction, stride `w`.
            other: *const i16,
            /// Block width (the stride of `i16` and `other`).
            w: usize,
        }

        /// 14-bit predictions (the two-pass path and the first stage of hv).
        const MODE_I16: u8 = 0;
        /// Default-weighted uni-prediction samples: `(v + 32) >> 6`.
        const MODE_UNI: u8 = 1;
        /// Default-weighted bi-prediction samples: `(v + other + 64) >> 7`.
        const MODE_BI: u8 = 2;

        /// Emit 8 lanes of a stage's output (`v`, 14-bit) at (`row`, `x`), the
        /// first `n` lanes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn emit<const MODE: u8>(out: &Out, row: usize, x: usize, v: __m128i, n: usize) {
            unsafe {
                match MODE {
                    MODE_I16 => store_n(out.i16.add(row * out.w + x), v, n),
                    MODE_UNI => {
                        let r = _mm_srai_epi16(_mm_adds_epi16(v, _mm_set1_epi16(32)), 6);
                        store_bytes(out.u8.add(row * out.stride + x), pack8(r), n);
                    }
                    _ => {
                        // Saturating sums, exact after the clip (see `bi_u8_impl`).
                        let o = load_n(out.other.add(row * out.w + x), n);
                        let r = _mm_srai_epi16(_mm_adds_epi16(_mm_adds_epi16(v, o), _mm_set1_epi16(64)), 7);
                        store_bytes(out.u8.add(row * out.stride + x), pack8(r), n);
                    }
                }
            }
        }

        /// Horizontal FIR with `TAPS` taps over bytes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fir_h_u8<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
            unsafe {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..TAPS / 2 {
                    c[k] = _mm_set1_epi16(pair8(taps[2 * k], taps[2 * k + 1]));
                }
                let sh = _mm_cvtsi32_si128(shift);
                if w <= 8 {
                    // Narrow blocks: 8-byte loads, one vector per row.
                    for y in 0..h {
                        let s = src.add(y * src_stride);
                        let mut acc = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadl_epi64(s.add(2 * k) as *const __m128i);
                            let b = _mm_loadl_epi64(s.add(2 * k + 1) as *const __m128i);
                            acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c[k]));
                        }
                        emit::<MODE>(out, y, 0, _mm_sra_epi16(acc, sh), w);
                    }
                    return;
                }
                for y in 0..h {
                    let s = src.add(y * src_stride);
                    let mut x = 0;
                    while x < w {
                        let mut lo = _mm_setzero_si128();
                        let mut hi = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadu_si128(s.add(x + 2 * k) as *const __m128i);
                            let b = _mm_loadu_si128(s.add(x + 2 * k + 1) as *const __m128i);
                            lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c[k]));
                            hi = _mm_add_epi16(hi, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), c[k]));
                        }
                        let n = w - x;
                        emit::<MODE>(out, y, x, _mm_sra_epi16(lo, sh), n.min(8));
                        if n > 8 {
                            emit::<MODE>(out, y, x + 8, _mm_sra_epi16(hi, sh), (n - 8).min(8));
                        }
                        x += 16;
                    }
                }
            }
        }

        /// Vertical FIR with `TAPS` taps over byte rows.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fir_v_u8<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
            unsafe {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..TAPS / 2 {
                    c[k] = _mm_set1_epi16(pair8(taps[2 * k], taps[2 * k + 1]));
                }
                let sh = _mm_cvtsi32_si128(shift);
                let row = |r: usize| src.add(r * src_stride);
                if w <= 8 {
                    for y in 0..h {
                        let mut acc = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadl_epi64(row(y + 2 * k) as *const __m128i);
                            let b = _mm_loadl_epi64(row(y + 2 * k + 1) as *const __m128i);
                            acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c[k]));
                        }
                        emit::<MODE>(out, y, 0, _mm_sra_epi16(acc, sh), w);
                    }
                    return;
                }
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let mut lo = _mm_setzero_si128();
                        let mut hi = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadu_si128(row(y + 2 * k).add(x) as *const __m128i);
                            let b = _mm_loadu_si128(row(y + 2 * k + 1).add(x) as *const __m128i);
                            lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c[k]));
                            hi = _mm_add_epi16(hi, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), c[k]));
                        }
                        let n = w - x;
                        emit::<MODE>(out, y, x, _mm_sra_epi16(lo, sh), n.min(8));
                        if n > 8 {
                            emit::<MODE>(out, y, x + 8, _mm_sra_epi16(hi, sh), (n - 8).min(8));
                        }
                        x += 16;
                    }
                }
            }
        }

        /// Vertical FIR with `TAPS` taps over 14-bit rows (the second stage of
        /// hv): `pmaddwd` on interleaved row pairs, 32-bit sums, `>> 6`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fir_v2_u8<const TAPS: usize, const MODE: u8>(out: &Out, src: *const i16, src_stride: usize, w: usize, h: usize, taps: &[i8]) {
            unsafe {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..TAPS / 2 {
                    c[k] = _mm_set1_epi32(pair(taps[2 * k], taps[2 * k + 1]));
                }
                let row = |r: usize| src.add(r * src_stride);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let mut lo = _mm_setzero_si128();
                        let mut hi = _mm_setzero_si128();
                        for k in 0..TAPS / 2 {
                            let a = _mm_loadu_si128(row(y + 2 * k).add(x) as *const __m128i);
                            let b = _mm_loadu_si128(row(y + 2 * k + 1).add(x) as *const __m128i);
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c[k]));
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c[k]));
                        }
                        let r = _mm_packs_epi32(_mm_srai_epi32(lo, 6), _mm_srai_epi32(hi, 6));
                        emit::<MODE>(out, y, x, r, (w - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        fn copy_u8(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
            // 8-byte loads at every 8-sample step of each row.
            if (h - 1) * src_stride + (w - 1) / 8 * 8 + 8 > src.len() {
                return (HevcDsp::<u8>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
            }
            unsafe { copy_u8_impl(dst, src, src_stride, w, h, shift) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn copy_u8_impl(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
            unsafe {
                let sh = _mm_cvtsi32_si128(shift);
                for y in 0..h {
                    let s = src.as_ptr().add(y * src_stride);
                    let d = dst.as_mut_ptr().add(y * w);
                    let mut x = 0;
                    while x < w {
                        let v = _mm_cvtepu8_epi16(_mm_loadl_epi64(s.add(x) as *const __m128i));
                        store_n(d.add(x), _mm_sll_epi16(v, sh), (w - x).min(8));
                        x += 8;
                    }
                }
            }
        }

        fn qpel_h_u8(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits_b(src.len(), src_stride, h, w, 7) || dst.len() < w * h {
                return (HevcDsp::<u8>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
            }
            let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
            unsafe { fir_h_u8::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
        }

        fn qpel_v_u8(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits_b(src.len(), src_stride, h + 7, w, 0) || dst.len() < w * h {
                return (HevcDsp::<u8>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
            }
            let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
            unsafe { fir_v_u8::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
        }

        fn epel_h_u8(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits_b(src.len(), src_stride, h, w, 3) || dst.len() < w * h {
                return (HevcDsp::<u8>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
            }
            let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
            unsafe { fir_h_u8::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
        }

        fn epel_v_u8(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
            if !fits_b(src.len(), src_stride, h + 3, w, 0) || dst.len() < w * h {
                return (HevcDsp::<u8>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
            }
            let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
            unsafe { fir_v_u8::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
        }

        // ------------------------------------------------------------------
        // Fused interpolation + prediction
        // ------------------------------------------------------------------

        /// Copy a `w x h` byte block (whole-sample uni-prediction: the
        /// prediction is the reference block).
        #[target_feature(enable = $feat)]
        unsafe fn copy_rows_u8(dst: *mut u8, dst_stride: usize, src: *const u8, src_stride: usize, w: usize, h: usize) {
            unsafe {
                for y in 0..h {
                    let s = src.add(y * src_stride);
                    let d = dst.add(y * dst_stride);
                    let mut x = 0;
                    while x < w {
                        let n = w - x;
                        if n >= 16 {
                            _mm_storeu_si128(d.add(x) as *mut __m128i, _mm_loadu_si128(s.add(x) as *const __m128i));
                            x += 16;
                        } else if n >= 8 {
                            _mm_storel_epi64(d.add(x) as *mut __m128i, _mm_loadl_epi64(s.add(x) as *const __m128i));
                            x += 8;
                        } else if n >= 4 {
                            std::ptr::write_unaligned(d.add(x) as *mut u32, std::ptr::read_unaligned(s.add(x) as *const u32));
                            x += 4;
                        } else {
                            std::ptr::write_unaligned(d.add(x) as *mut u16, std::ptr::read_unaligned(s.add(x) as *const u16));
                            x += 2;
                        }
                    }
                }
            }
        }

        /// The fused kernels: `TAPS` (8 luma / 4 chroma), `MODE_UNI` or `MODE_BI`.
        #[allow(clippy::too_many_arguments)]
        fn fused<const TAPS: usize, const MODE: u8>(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16]) {
            let reach = TAPS / 2 - 1;
            let at_block = reach * src_stride + reach;
            let hh = h + TAPS - 1;
            let ok = w >= 2
                && h >= 1
                && (h - 1) * dst_stride + w <= dst.len()
                && (MODE != MODE_BI || other.len() >= w * h)
                && tmp.len() >= crate::dsp::hevc::MC_TMP_LEN
                && match (fx, fy) {
                    (0, 0) => (h - 1) * src_stride + w + at_block <= src.len(),
                    (_, 0) => src.len() > reach * src_stride && fits_b(src.len() - reach * src_stride, src_stride, h, w, TAPS - 1),
                    (0, _) => src.len() > reach && fits_b(src.len() - reach, src_stride, hh, w, 0),
                    _ => fits_b(src.len(), src_stride, hh, w, TAPS - 1) && fits_i16(crate::dsp::hevc::MC_TMP_LEN, w, hh),
                };
            if !ok {
                let s = HevcDsp::<u8>::SCALAR;
                return match (TAPS, MODE) {
                    (8, MODE_UNI) => (s.qpel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
                    (8, _) => (s.qpel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
                    (_, MODE_UNI) => (s.epel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
                    _ => (s.epel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
                };
            }
            let (tx, ty): (&[i8], &[i8]) = if TAPS == 8 { (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]) } else { (&EPEL_FILTERS[fx], &EPEL_FILTERS[fy]) };
            let out = Out { i16: std::ptr::null_mut(), u8: dst.as_mut_ptr(), stride: dst_stride, other: other.as_ptr(), w };
            unsafe {
                match (fx, fy) {
                    (0, 0) => {
                        if MODE == MODE_UNI {
                            copy_rows_u8(dst.as_mut_ptr(), dst_stride, src.as_ptr().add(at_block), src_stride, w, h);
                        } else {
                            // Whole-sample bi: widen, then the usual average.
                            let (pred, _) = tmp.split_at_mut(w * h);
                            copy_u8(pred, &src[at_block..], src_stride, w, h, 6);
                            bi_u8_impl(dst, dst_stride, other, pred, w, h, 7);
                        }
                    }
                    (_, 0) => fir_h_u8::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, w, h, tx, 0),
                    (0, _) => fir_v_u8::<TAPS, MODE>(&out, src.as_ptr().add(reach), src_stride, w, h, ty, 0),
                    _ => {
                        let mid = Out { i16: tmp.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
                        fir_h_u8::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, 0);
                        fir_v2_u8::<TAPS, MODE>(&out, tmp.as_ptr(), w, w, h, ty);
                    }
                }
            }
        }

        fn qpel_uni_u8(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
            debug_assert_eq!(bit_depth, 8);
            fused::<8, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
        }

        fn epel_uni_u8(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
            debug_assert_eq!(bit_depth, 8);
            fused::<4, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
        }

        #[allow(clippy::too_many_arguments)]
        fn qpel_bi_u8(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
            debug_assert_eq!(bit_depth, 8);
            fused::<8, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
        }

        #[allow(clippy::too_many_arguments)]
        fn epel_bi_u8(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
            debug_assert_eq!(bit_depth, 8);
            fused::<4, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
        }

        // ------------------------------------------------------------------
        // Combination / weighting
        // ------------------------------------------------------------------

        fn uni_u8(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            debug_assert_eq!(max, 255);
            unsafe { uni_u8_impl(dst, stride, src, w, h, shift) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn uni_u8_impl(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32) {
            unsafe {
                let round = _mm_set1_epi16(if shift > 0 { 1 << (shift - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(shift);
                if narrow(w) {
                    let total = w * h;
                    let mut i = 0;
                    while i < total {
                        let n = (total - i).min(8);
                        let s = load_n(src.as_ptr().add(i), total - i);
                        let v = _mm_sra_epi16(_mm_adds_epi16(s, round), sh);
                        scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, pack8(v), n / w);
                        i += 8;
                    }
                    return;
                }
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let s = load_n(src.as_ptr().add(y * w + x), w - x);
                        // 14-bit + round fits i16 (< 16384 + 8192).
                        let v = _mm_sra_epi16(_mm_adds_epi16(s, round), sh);
                        store_bytes(dst.as_mut_ptr().add(y * stride + x), pack8(v), n);
                        x += 8;
                    }
                }
            }
        }

        fn bi_u8(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
            debug_assert_eq!(max, 255);
            unsafe { bi_u8_impl(dst, stride, a, b, w, h, shift) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn bi_u8_impl(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32) {
            unsafe {
                let round = _mm_set1_epi16(1 << (shift - 1));
                let sh = _mm_cvtsi32_si128(shift);
                if narrow(w) {
                    let total = w * h;
                    let mut i = 0;
                    while i < total {
                        let n = (total - i).min(8);
                        let va = load_n(a.as_ptr().add(i), total - i);
                        let vb = load_n(b.as_ptr().add(i), total - i);
                        let v = _mm_sra_epi16(_mm_adds_epi16(_mm_adds_epi16(va, vb), round), sh);
                        scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, pack8(v), n / w);
                        i += 8;
                    }
                    return;
                }
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let va = load_n(a.as_ptr().add(y * w + x), w - x);
                        let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                        // Saturating sums: a + b can exceed i16 only when both
                        // are far above the 8-bit range, and then the clip to
                        // 255 gives the same answer as the exact 32-bit sum.
                        let v = _mm_sra_epi16(_mm_adds_epi16(_mm_adds_epi16(va, vb), round), sh);
                        store_bytes(dst.as_mut_ptr().add(y * stride + x), pack8(v), n);
                        x += 8;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_uni_u8(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
            debug_assert_eq!(max, 255);
            unsafe { weighted_uni_u8_impl(dst, stride, src, w, h, log2_wd, wt, o) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_uni_u8_impl(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32) {
            unsafe {
                let round = _mm_set1_epi32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(log2_wd.max(0));
                let wv = _mm_set1_epi32(wt);
                let ov = _mm_set1_epi32(o);
                let weigh = |s: __m128i| -> __m128i {
                    let quad = |v: __m128i| _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(v, wv), round), sh), ov);
                    let lo = quad(_mm_cvtepi16_epi32(s));
                    let hi = quad(_mm_cvtepi16_epi32(_mm_srli_si128(s, 8)));
                    pack8(_mm_packs_epi32(lo, hi))
                };
                if narrow(w) {
                    let total = w * h;
                    let mut i = 0;
                    while i < total {
                        let n = (total - i).min(8);
                        let s = load_n(src.as_ptr().add(i), total - i);
                        scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, weigh(s), n / w);
                        i += 8;
                    }
                    return;
                }
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let s = load_n(src.as_ptr().add(y * w + x), w - x);
                        store_bytes(dst.as_mut_ptr().add(y * stride + x), weigh(s), n);
                        x += 8;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_bi_u8(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
            debug_assert_eq!(max, 255);
            unsafe { weighted_bi_u8_impl(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_bi_u8_impl(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
            unsafe {
                let round = _mm_set1_epi32((o0 + o1 + 1) << log2_wd);
                let sh = _mm_cvtsi32_si128(log2_wd + 1);
                let w0v = _mm_set1_epi32(w0);
                let w1v = _mm_set1_epi32(w1);
                let weigh = |va: __m128i, vb: __m128i| -> __m128i {
                    let quad = |pa: __m128i, pb: __m128i| {
                        _mm_sra_epi32(_mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(pa, w0v), _mm_mullo_epi32(pb, w1v)), round), sh)
                    };
                    let lo = quad(_mm_cvtepi16_epi32(va), _mm_cvtepi16_epi32(vb));
                    let hi = quad(_mm_cvtepi16_epi32(_mm_srli_si128(va, 8)), _mm_cvtepi16_epi32(_mm_srli_si128(vb, 8)));
                    pack8(_mm_packs_epi32(lo, hi))
                };
                if narrow(w) {
                    let total = w * h;
                    let mut i = 0;
                    while i < total {
                        let n = (total - i).min(8);
                        let va = load_n(a.as_ptr().add(i), total - i);
                        let vb = load_n(b.as_ptr().add(i), total - i);
                        scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, weigh(va, vb), n / w);
                        i += 8;
                    }
                    return;
                }
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(8);
                        let va = load_n(a.as_ptr().add(y * w + x), w - x);
                        let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                        store_bytes(dst.as_mut_ptr().add(y * stride + x), weigh(va, vb), n);
                        x += 8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Residual add
        // ------------------------------------------------------------------

        fn add_residual_u8(dst: &mut [u8], stride: usize, res: &[i16], n: usize, max: i32) {
            debug_assert_eq!(max, 255);
            unsafe { add_residual_u8_impl(dst, stride, res, n) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn add_residual_u8_impl(dst: &mut [u8], stride: usize, res: &[i16], n: usize) {
            unsafe {
                if n == 4 {
                    // 4x4: two rows per vector.
                    let d = dst.as_mut_ptr();
                    for y in (0..4).step_by(2) {
                        let rd = |k: usize| std::ptr::read_unaligned(d.add(k * stride) as *const u32) as i32;
                        let p = _mm_cvtepu8_epi16(_mm_setr_epi32(rd(y), rd(y + 1), 0, 0));
                        let r = _mm_loadu_si128(res.as_ptr().add(y * 4) as *const __m128i);
                        let v = _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128());
                        let mut t = [0u32; 4];
                        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                        std::ptr::write_unaligned(d.add(y * stride) as *mut u32, t[0]);
                        std::ptr::write_unaligned(d.add((y + 1) * stride) as *mut u32, t[1]);
                    }
                    return;
                }
                for y in 0..n {
                    let mut x = 0;
                    while x < n {
                        let d = dst.as_mut_ptr().add(y * stride + x);
                        let p = _mm_cvtepu8_epi16(_mm_loadl_epi64(d as *const __m128i));
                        let r = _mm_loadu_si128(res.as_ptr().add(y * n + x) as *const __m128i);
                        _mm_storel_epi64(d as *mut __m128i, _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128()));
                        x += 8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // SAO
        // ------------------------------------------------------------------

        /// `v + off` on bytes, clipped to `0..=255`, with `off` in `-128..=127`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn add_offset_u8(v: __m128i, off: __m128i) -> __m128i {
            unsafe {
                let zero = _mm_setzero_si128();
                let pos = _mm_max_epi8(off, zero);
                let neg = _mm_max_epi8(_mm_sub_epi8(zero, off), zero);
                _mm_subs_epu8(_mm_adds_epu8(v, pos), neg)
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn sao_band_u8(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
            if shift != 3 || table.iter().any(|&o| !(-128..=127).contains(&o)) {
                return (HevcDsp::<u8>::SCALAR.sao_band)(dst, dst_stride, src, src_stride, w, h, table, shift, max);
            }
            debug_assert_eq!(max, 255);
            unsafe { sao_band_u8_impl(dst, dst_stride, src, src_stride, w, h, table, shift) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn sao_band_u8_impl(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32) {
            unsafe {
                // The four consecutive bands (mod 32) with nonzero offsets.
                let mut bands = [0u8; 4];
                let mut offs = [0i8; 4];
                let mut k = 0;
                for b in 0..32 {
                    if table[b] != 0 && k < 4 {
                        bands[k] = (b as u8) << shift;
                        offs[k] = table[b] as i8;
                        k += 1;
                    }
                }
                let mask = _mm_set1_epi8((0xFFu32 << shift) as u8 as i8);
                let bv: [__m128i; 4] = std::array::from_fn(|i| _mm_set1_epi8(bands[i] as i8));
                let ov: [__m128i; 4] = std::array::from_fn(|i| _mm_set1_epi8(offs[i]));
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(16);
                        let v = load_bytes16(src.as_ptr().add(y * src_stride + x), n);
                        let band = _mm_and_si128(v, mask);
                        let mut off = _mm_setzero_si128();
                        for i in 0..k {
                            off = _mm_blendv_epi8(off, ov[i], _mm_cmpeq_epi8(band, bv[i]));
                        }
                        store_bytes16(dst.as_mut_ptr().add(y * dst_stride + x), add_offset_u8(v, off), n);
                        x += 16;
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn sao_edge_u8(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
            if off.iter().any(|&o| !(-128..=127).contains(&o)) {
                return (HevcDsp::<u8>::SCALAR.sao_edge)(dst, src, origin, stride, w, h, na, nb, off, max);
            }
            debug_assert_eq!(max, 255);
            unsafe { sao_edge_u8_impl(dst, src, origin, stride, w, h, na, nb, off) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn sao_edge_u8_impl(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5]) {
            unsafe {
                // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 indexes the
                // offsets through a byte shuffle.
                let o = |i: usize| off[i] as i8;
                let tab = _mm_setr_epi8(o(0), o(1), o(2), o(3), o(4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                let two = _mm_set1_epi8(2);
                let lo_reach = na.min(nb).min(0);
                let hi_reach = na.max(nb).max(0);
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let n = (w - x).min(16);
                        let i = origin + y * stride + x;
                        if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 16 > src.len() || i + 16 > dst.len() {
                            // Tail near the buffer end: scalar.
                            for xx in x..w {
                                let ii = origin + y * stride + xx;
                                let v = src[ii] as i32;
                                let a = src[(ii as isize + na) as usize] as i32;
                                let b = src[(ii as isize + nb) as usize] as i32;
                                let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                                dst[ii] = (v + off[e] as i32).clamp(0, 255) as u8;
                            }
                            break;
                        }
                        let v = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                        let a = _mm_loadu_si128(src.as_ptr().offset(i as isize + na) as *const __m128i);
                        let b = _mm_loadu_si128(src.as_ptr().offset(i as isize + nb) as *const __m128i);
                        // Unsigned compares: ge = (max(v, a) == v), gt = ge & !eq, lt = !ge.
                        let ge_a = _mm_cmpeq_epi8(_mm_max_epu8(v, a), v);
                        let gt_a = _mm_andnot_si128(_mm_cmpeq_epi8(v, a), ge_a);
                        let ge_b = _mm_cmpeq_epi8(_mm_max_epu8(v, b), v);
                        let gt_b = _mm_andnot_si128(_mm_cmpeq_epi8(v, b), ge_b);
                        // e = 2 + gt_a - lt_a + gt_b - lt_b with masks of -1: 2 - gt + lt.
                        let ones = _mm_cmpeq_epi8(v, v);
                        let lt_a = _mm_xor_si128(ge_a, ones);
                        let lt_b = _mm_xor_si128(ge_b, ones);
                        let e = _mm_add_epi8(_mm_sub_epi8(_mm_sub_epi8(two, gt_a), gt_b), _mm_add_epi8(lt_a, lt_b));
                        let o = _mm_shuffle_epi8(tab, e);
                        store_bytes16(dst.as_mut_ptr().add(i), add_offset_u8(v, o), n);
                        x += 16;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Deblocking — the shared i32-lane filters with byte loads and stores.
        // ------------------------------------------------------------------

        /// Eight consecutive bytes as two vectors of 4 x i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn ld8_u8(p: *const u8) -> (__m128i, __m128i) {
            unsafe {
                let v = _mm_loadl_epi64(p as *const __m128i);
                (_mm_cvtepu8_epi32(v), _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)))
            }
        }

        /// Two vectors of 4 x i32 (each within a byte) to eight bytes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn pack8_u8(lo: __m128i, hi: __m128i) -> __m128i {
            unsafe {
                let p = pack8_u16(lo, hi);
                _mm_packus_epi16(p, p)
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn deblock_luma_v_u8(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
                return;
            }
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_u8_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn deblock_luma_v_u8_impl(data: *mut u8, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_cvtepu8_epi16(_mm_loadl_epi64(data.add(i * stride).sub(4) as *const __m128i));
                }
                transpose8_u16(&mut r);
                let mut v0 = [_mm_setzero_si128(); 8];
                let mut v1 = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let (a, b) = widen_u16(r[k]);
                    v0[k] = a;
                    v1[k] = b;
                }
                luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
                luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
                for k in 0..8 {
                    r[k] = pack8_u16(v0[k], v1[k]);
                }
                transpose8_u16(&mut r);
                for i in 0..8 {
                    _mm_storel_epi64(data.add(i * stride).sub(4) as *mut __m128i, _mm_packus_epi16(r[i], r[i]));
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn deblock_luma_h_u8(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
                return;
            }
            assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
            unsafe { deblock_luma_h_u8_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn deblock_luma_h_u8_impl(data: *mut u8, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
            unsafe {
                let mut v0 = [_mm_setzero_si128(); 8];
                let mut v1 = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let (a, b) = ld8_u8(data.offset((k as isize - 4) * stride as isize));
                    v0[k] = a;
                    v1[k] = b;
                }
                luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
                luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
                for k in 1..7 {
                    _mm_storel_epi64(data.offset((k as isize - 4) * stride as isize) as *mut __m128i, pack8_u8(v0[k], v1[k]));
                }
            }
        }

        fn deblock_chroma_v_u8(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            if tc.iter().all(|&t| t == 0) {
                return;
            }
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_u8_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_u8_impl(data: *mut u8, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    let q = std::ptr::read_unaligned(data.add(i * stride).sub(2) as *const u32);
                    r[i] = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(q as i32));
                }
                let a0 = _mm_unpacklo_epi16(r[0], r[1]);
                let a1 = _mm_unpacklo_epi16(r[2], r[3]);
                let a2 = _mm_unpacklo_epi16(r[4], r[5]);
                let a3 = _mm_unpacklo_epi16(r[6], r[7]);
                let b0 = _mm_unpacklo_epi32(a0, a1); // p1 r0..3 | p0 r0..3
                let b1 = _mm_unpackhi_epi32(a0, a1); // q0 r0..3 | q1 r0..3
                let b2 = _mm_unpacklo_epi32(a2, a3); // rows 4..7
                let b3 = _mm_unpackhi_epi32(a2, a3);
                let col = |v: __m128i, hi: bool| if hi { _mm_cvtepu16_epi32(_mm_srli_si128(v, 8)) } else { _mm_cvtepu16_epi32(v) };
                let mut v0 = [col(b0, false), col(b0, true), col(b1, false), col(b1, true)];
                let mut v1 = [col(b2, false), col(b2, true), col(b3, false), col(b3, true)];
                chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
                chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
                // (p0, q0) byte pairs per row.
                let p0 = pack8_u16(v0[1], v1[1]);
                let q0 = pack8_u16(v0[2], v1[2]);
                let pairs = _mm_packus_epi16(_mm_unpacklo_epi16(p0, q0), _mm_unpackhi_epi16(p0, q0));
                let mut t = [0u16; 8];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, pairs);
                for i in 0..8 {
                    std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u16, t[i]);
                }
            }
        }

        fn deblock_chroma_h_u8(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            if tc.iter().all(|&t| t == 0) {
                return;
            }
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_u8_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_u8_impl(data: *mut u8, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
            unsafe {
                let (a0, a1) = ld8_u8(data.sub(2 * stride));
                let (b0, b1) = ld8_u8(data.sub(stride));
                let (c0, c1) = ld8_u8(data);
                let (d0, d1) = ld8_u8(data.add(stride));
                let mut v0 = [a0, b0, c0, d0];
                let mut v1 = [a1, b1, c1, d1];
                chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
                chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
                _mm_storel_epi64(data.sub(stride) as *mut __m128i, pack8_u8(v0[1], v1[1]));
                _mm_storel_epi64(data as *mut __m128i, pack8_u8(v0[2], v1[2]));
            }
        }
    };
}

/// The kernels compiled for SSE4.1 (legacy two-operand encoding).
pub mod sse41 {
    kernels_u16!("sse4.1");
    kernels_u8!("sse4.1");
}

/// The kernels compiled for AVX: identical algorithms, VEX-encoded.
pub mod avx {
    kernels_u16!("avx");
    kernels_u8!("avx");
}

/// Replace the scalar entries of `d` with the SSE4.1 kernels (16-bit samples).
pub fn install_sse41_u16(d: &mut HevcDsp<u16>) {
    sse41::install_u16(d);
}

/// Replace the scalar entries of `d` with the AVX kernels (16-bit samples).
pub fn install_avx_u16(d: &mut HevcDsp<u16>) {
    avx::install_u16(d);
}

/// Replace the scalar entries of `d` with the SSE4.1 kernels (8-bit samples).
pub fn install_sse41_u8(d: &mut HevcDsp<u8>) {
    sse41::install_u8(d);
}

/// Replace the scalar entries of `d` with the AVX kernels (8-bit samples).
pub fn install_avx_u8(d: &mut HevcDsp<u8>) {
    avx::install_u8(d);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hevc::MC_TMP_LEN;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Both installs, skipped when the CPU cannot run them.
    fn tables_u16() -> Vec<(&'static str, HevcDsp<u16>)> {
        let mut v = Vec::new();
        if std::is_x86_feature_detected!("sse4.1") {
            let mut d = HevcDsp::<u16>::SCALAR;
            install_sse41_u16(&mut d);
            v.push(("sse4.1", d));
        }
        if std::is_x86_feature_detected!("avx") {
            let mut d = HevcDsp::<u16>::SCALAR;
            install_avx_u16(&mut d);
            v.push(("avx", d));
        }
        v
    }

    fn tables_u8() -> Vec<(&'static str, HevcDsp<u8>)> {
        let mut v = Vec::new();
        if std::is_x86_feature_detected!("sse4.1") {
            let mut d = HevcDsp::<u8>::SCALAR;
            install_sse41_u8(&mut d);
            v.push(("sse4.1", d));
        }
        if std::is_x86_feature_detected!("avx") {
            let mut d = HevcDsp::<u8>::SCALAR;
            install_avx_u8(&mut d);
            v.push(("avx", d));
        }
        v
    }

    const SIZES: [(usize, usize); 9] = [(2, 4), (4, 4), (4, 8), (8, 4), (8, 8), (12, 16), (16, 16), (32, 8), (64, 16)];

    #[test]
    fn interp_matches_scalar_u16() {
        let s = HevcDsp::<u16>::SCALAR;
        for (name, d) in tables_u16() {
            let mut seed = 3u64;
            let stride = 96;
            let src: Vec<u16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 1024) as u16).collect();
            for &(w, h) in &SIZES {
                for frac in 0..4 {
                    let mut a = vec![0i16; w * h];
                    let mut b = vec![0i16; w * h];
                    (s.qpel_h)(&mut a, &src, stride, w, h, frac, 2);
                    (d.qpel_h)(&mut b, &src, stride, w, h, frac, 2);
                    assert_eq!(a, b, "{name} qpel_h {w}x{h} frac {frac}");
                    (s.qpel_v)(&mut a, &src, stride, w, h, frac, 2);
                    (d.qpel_v)(&mut b, &src, stride, w, h, frac, 2);
                    assert_eq!(a, b, "{name} qpel_v {w}x{h} frac {frac}");
                }
                for frac in 0..8 {
                    let mut a = vec![0i16; w * h];
                    let mut b = vec![0i16; w * h];
                    (s.epel_h)(&mut a, &src, stride, w, h, frac, 2);
                    (d.epel_h)(&mut b, &src, stride, w, h, frac, 2);
                    assert_eq!(a, b, "{name} epel_h {w}x{h} frac {frac}");
                    (s.epel_v)(&mut a, &src, stride, w, h, frac, 2);
                    (d.epel_v)(&mut b, &src, stride, w, h, frac, 2);
                    assert_eq!(a, b, "{name} epel_v {w}x{h} frac {frac}");
                }
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                (s.qpel_copy)(&mut a, &src, stride, w, h, 2);
                (d.qpel_copy)(&mut b, &src, stride, w, h, 2);
                assert_eq!(a, b, "{name} copy {w}x{h}");
                // Second-stage vertical over 14-bit rows.
                let mid: Vec<i16> = (0..w * (h + 8)).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                for frac in 0..4 {
                    (s.qpel_v2)(&mut a, &mid, w, w, h, frac);
                    (d.qpel_v2)(&mut b, &mid, w, w, h, frac);
                    assert_eq!(a, b, "{name} qpel_v2 {w}x{h} frac {frac}");
                }
                for frac in 0..8 {
                    (s.epel_v2)(&mut a, &mid, w, w, h, frac);
                    (d.epel_v2)(&mut b, &mid, w, w, h, frac);
                    assert_eq!(a, b, "{name} epel_v2 {w}x{h} frac {frac}");
                }
            }
        }
    }

    #[test]
    fn combine_matches_scalar_u16() {
        let s = HevcDsp::<u16>::SCALAR;
        for (name, d) in tables_u16() {
            let mut seed = 7u64;
            for &(w, h) in &SIZES {
                let pa: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                let pb: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                let stride = w + 5;
                let mut a = vec![0u16; stride * h];
                let mut b = vec![0u16; stride * h];
                for &max in &[255i32, 1023, 4095] {
                    (s.uni)(&mut a, stride, &pa, w, h, 6, max);
                    (d.uni)(&mut b, stride, &pa, w, h, 6, max);
                    assert_eq!(a, b, "{name} uni {w}x{h} max {max}");
                    (s.bi)(&mut a, stride, &pa, &pb, w, h, 7, max);
                    (d.bi)(&mut b, stride, &pa, &pb, w, h, 7, max);
                    assert_eq!(a, b, "{name} bi {w}x{h} max {max}");
                    for &(lwd, wt, o) in &[(6i32, 64i32, 0i32), (0, 1, 3), (5, -20, -7), (7, 127, 100)] {
                        (s.weighted_uni)(&mut a, stride, &pa, w, h, lwd, wt, o, max);
                        (d.weighted_uni)(&mut b, stride, &pa, w, h, lwd, wt, o, max);
                        assert_eq!(a, b, "{name} wuni {w}x{h} {lwd} {wt} {o} max {max}");
                        (s.weighted_bi)(&mut a, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, max);
                        (d.weighted_bi)(&mut b, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, max);
                        assert_eq!(a, b, "{name} wbi {w}x{h} {lwd} {wt} {o} max {max}");
                    }
                }
            }
        }
    }

    #[test]
    fn idct_matches_scalar_u16() {
        let s = HevcDsp::<u16>::SCALAR;
        for (name, d) in tables_u16() {
            let mut seed = 13u64;
            for (li, &n) in [4usize, 8, 16, 32].iter().enumerate() {
                for trial in 0..40 {
                    let max_x = (lcg(&mut seed) as usize) % n;
                    let max_y = (lcg(&mut seed) as usize) % n;
                    let mut a = vec![0i16; n * n];
                    for y in 0..=max_y {
                        for x in 0..=max_x {
                            a[y * n + x] = (lcg(&mut seed) % 2048) as i16 - 1024;
                        }
                    }
                    let mut b = a.clone();
                    let bd_shift = 20 - 8;
                    (s.idct[li])(&mut a, bd_shift, max_x, max_y);
                    (d.idct[li])(&mut b, bd_shift, max_x, max_y);
                    assert_eq!(a, b, "{name} idct{n} trial {trial} max {max_x},{max_y}");
                }
            }
            // Residual add.
            let mut seed = 29u64;
            for &n in &[4usize, 8, 16, 32] {
                let stride = n + 7;
                let base: Vec<u16> = (0..stride * n).map(|_| (lcg(&mut seed) % 1024) as u16).collect();
                let res: Vec<i16> = (0..n * n).map(|_| (lcg(&mut seed) % 2048) as i16 - 1024).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                (s.add_residual)(&mut a, stride, &res, n, 1023);
                (d.add_residual)(&mut b, stride, &res, n, 1023);
                assert_eq!(a, b, "{name} add_residual {n}");
            }
        }
    }

    #[test]
    fn sao_matches_scalar_u16() {
        let s = HevcDsp::<u16>::SCALAR;
        for (name, d) in tables_u16() {
            let mut seed = 19u64;
            let stride = 72;
            let src: Vec<u16> = (0..stride * 40).map(|_| (lcg(&mut seed) % 1024) as u16).collect();
            for &(w, h) in &SIZES {
                let mut table = [0i16; 32];
                let start = (lcg(&mut seed) % 28) as usize;
                for k in 0..4 {
                    table[start + k] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                let mut a = vec![0u16; src.len()];
                let mut b = vec![0u16; src.len()];
                (s.sao_band)(&mut a, stride, &src, stride, w, h, &table, 5, 1023);
                (d.sao_band)(&mut b, stride, &src, stride, w, h, &table, 5, 1023);
                assert_eq!(a, b, "{name} sao_band {w}x{h}");
                let mut off = [0i16; 5];
                for k in [0usize, 1, 3, 4] {
                    off[k] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                for &(na, nb) in &[(-1isize, 1isize), (-(stride as isize), stride as isize), (-(stride as isize) - 1, stride as isize + 1)] {
                    let origin = 4 * stride + 4;
                    let mut a = src.clone();
                    let mut b = src.clone();
                    (s.sao_edge)(&mut a, &src, origin, stride, w, h, na, nb, &off, 1023);
                    (d.sao_edge)(&mut b, &src, origin, stride, w, h, na, nb, &off, 1023);
                    assert_eq!(a, b, "{name} sao_edge {w}x{h} {na},{nb}");
                }
            }
        }
    }

    #[test]
    fn deblocking_matches_scalar_u16() {
        let s = HevcDsp::<u16>::SCALAR;
        for (name, d) in tables_u16() {
            let mut seed = 23u64;
            let stride = 48;
            for trial in 0..500 {
                let base = lcg(&mut seed) % 1024;
                let spread = 1 + lcg(&mut seed) % 96;
                let plane: Vec<u16> = (0..stride * 32).map(|_| ((base + lcg(&mut seed) % spread).min(1023)) as u16).collect();
                let beta = [(lcg(&mut seed) % 64) as i32, (lcg(&mut seed) % 64) as i32];
                let tc = [(lcg(&mut seed) % 20) as i32, (lcg(&mut seed) % 20) as i32];
                let bl = |v: u32| v % 2 == 0;
                let no_p = [bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let no_q = [bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let tc4 = [tc[0], tc[1], (lcg(&mut seed) % 20) as i32, (lcg(&mut seed) % 20) as i32];
                let np4 = [no_p[0], no_p[1], bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let nq4 = [no_q[0], no_q[1], bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let off = 8 * stride + 8;
                let mut a = plane.clone();
                let mut b = plane.clone();
                match trial % 4 {
                    0 => {
                        (s.deblock_luma_v)(&mut a, off, stride, beta, tc, no_p, no_q, 1023);
                        (d.deblock_luma_v)(&mut b, off, stride, beta, tc, no_p, no_q, 1023);
                    }
                    1 => {
                        (s.deblock_luma_h)(&mut a, off, stride, beta, tc, no_p, no_q, 1023);
                        (d.deblock_luma_h)(&mut b, off, stride, beta, tc, no_p, no_q, 1023);
                    }
                    2 => {
                        (s.deblock_chroma_v)(&mut a, off, stride, tc4, np4, nq4, 1023);
                        (d.deblock_chroma_v)(&mut b, off, stride, tc4, np4, nq4, 1023);
                    }
                    _ => {
                        (s.deblock_chroma_h)(&mut a, off, stride, tc4, np4, nq4, 1023);
                        (d.deblock_chroma_h)(&mut b, off, stride, tc4, np4, nq4, 1023);
                    }
                }
                assert_eq!(a, b, "{name} deblock kind {} trial {trial}", trial % 4);
            }
        }
    }

    #[test]
    fn interp_matches_scalar_u8() {
        let s = HevcDsp::<u8>::SCALAR;
        for (name, d) in tables_u8() {
            let mut seed = 31u64;
            let stride = 96;
            let src: Vec<u8> = (0..stride * 96).map(|_| lcg(&mut seed) as u8).collect();
            for &(w, h) in &SIZES {
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                for frac in 0..4 {
                    (s.qpel_h)(&mut a, &src, stride, w, h, frac, 0);
                    (d.qpel_h)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "{name} qpel_h u8 {w}x{h} frac {frac}");
                    (s.qpel_v)(&mut a, &src, stride, w, h, frac, 0);
                    (d.qpel_v)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "{name} qpel_v u8 {w}x{h} frac {frac}");
                }
                for frac in 0..8 {
                    (s.epel_h)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_h)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "{name} epel_h u8 {w}x{h} frac {frac}");
                    (s.epel_v)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_v)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "{name} epel_v u8 {w}x{h} frac {frac}");
                }
                (s.qpel_copy)(&mut a, &src, stride, w, h, 6);
                (d.qpel_copy)(&mut b, &src, stride, w, h, 6);
                assert_eq!(a, b, "{name} copy u8 {w}x{h}");
            }
        }
    }

    #[test]
    fn fused_matches_scalar_u8() {
        let s = HevcDsp::<u8>::SCALAR;
        for (name, d) in tables_u8() {
            let mut seed = 37u64;
            let stride = 128;
            let src: Vec<u8> = (0..stride * 128).map(|_| lcg(&mut seed) as u8).collect();
            let mut ta = vec![0i16; MC_TMP_LEN];
            let mut tb = vec![0i16; MC_TMP_LEN];
            for &(w, h) in &SIZES {
                let other: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                let ds = w + 9;
                let mut a = vec![0u8; ds * h];
                let mut b = vec![0u8; ds * h];
                for fx in 0..4 {
                    for fy in 0..4 {
                        (s.qpel_uni)(&mut a, ds, &src, stride, w, h, fx, fy, &mut ta, 8);
                        (d.qpel_uni)(&mut b, ds, &src, stride, w, h, fx, fy, &mut tb, 8);
                        assert_eq!(a, b, "{name} qpel_uni {w}x{h} {fx},{fy}");
                        (s.qpel_bi)(&mut a, ds, &src, stride, w, h, fx, fy, &mut ta, &other, 8);
                        (d.qpel_bi)(&mut b, ds, &src, stride, w, h, fx, fy, &mut tb, &other, 8);
                        assert_eq!(a, b, "{name} qpel_bi {w}x{h} {fx},{fy}");
                    }
                }
                for fx in 0..8 {
                    for fy in 0..8 {
                        (s.epel_uni)(&mut a, ds, &src, stride, w, h, fx, fy, &mut ta, 8);
                        (d.epel_uni)(&mut b, ds, &src, stride, w, h, fx, fy, &mut tb, 8);
                        assert_eq!(a, b, "{name} epel_uni {w}x{h} {fx},{fy}");
                        (s.epel_bi)(&mut a, ds, &src, stride, w, h, fx, fy, &mut ta, &other, 8);
                        (d.epel_bi)(&mut b, ds, &src, stride, w, h, fx, fy, &mut tb, &other, 8);
                        assert_eq!(a, b, "{name} epel_bi {w}x{h} {fx},{fy}");
                    }
                }
            }
        }
    }

    #[test]
    fn combine_matches_scalar_u8() {
        let s = HevcDsp::<u8>::SCALAR;
        for (name, d) in tables_u8() {
            let mut seed = 41u64;
            for &(w, h) in &SIZES {
                let pa: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                let pb: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
                let stride = w + 5;
                let mut a = vec![0u8; stride * h];
                let mut b = vec![0u8; stride * h];
                (s.uni)(&mut a, stride, &pa, w, h, 6, 255);
                (d.uni)(&mut b, stride, &pa, w, h, 6, 255);
                assert_eq!(a, b, "{name} uni u8 {w}x{h}");
                (s.bi)(&mut a, stride, &pa, &pb, w, h, 7, 255);
                (d.bi)(&mut b, stride, &pa, &pb, w, h, 7, 255);
                assert_eq!(a, b, "{name} bi u8 {w}x{h}");
                for &(lwd, wt, o) in &[(6i32, 64i32, 0i32), (0, 1, 3), (5, -20, -7), (7, 127, 100)] {
                    (s.weighted_uni)(&mut a, stride, &pa, w, h, lwd, wt, o, 255);
                    (d.weighted_uni)(&mut b, stride, &pa, w, h, lwd, wt, o, 255);
                    assert_eq!(a, b, "{name} wuni u8 {w}x{h} {lwd} {wt} {o}");
                    (s.weighted_bi)(&mut a, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, 255);
                    (d.weighted_bi)(&mut b, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, 255);
                    assert_eq!(a, b, "{name} wbi u8 {w}x{h} {lwd} {wt} {o}");
                }
            }
            // Residual add.
            let mut seed = 43u64;
            for &n in &[4usize, 8, 16, 32] {
                let stride = n + 7;
                let base: Vec<u8> = (0..stride * n).map(|_| lcg(&mut seed) as u8).collect();
                let res: Vec<i16> = (0..n * n).map(|_| (lcg(&mut seed) % 512) as i16 - 256).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                (s.add_residual)(&mut a, stride, &res, n, 255);
                (d.add_residual)(&mut b, stride, &res, n, 255);
                assert_eq!(a, b, "{name} add_residual u8 {n}");
            }
        }
    }

    #[test]
    fn sao_matches_scalar_u8() {
        let s = HevcDsp::<u8>::SCALAR;
        for (name, d) in tables_u8() {
            let mut seed = 47u64;
            let stride = 72;
            let src: Vec<u8> = (0..stride * 40).map(|_| lcg(&mut seed) as u8).collect();
            for &(w, h) in &SIZES {
                let mut table = [0i16; 32];
                let start = (lcg(&mut seed) % 28) as usize;
                for k in 0..4 {
                    table[start + k] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                let mut a = vec![0u8; src.len()];
                let mut b = vec![0u8; src.len()];
                (s.sao_band)(&mut a, stride, &src, stride, w, h, &table, 3, 255);
                (d.sao_band)(&mut b, stride, &src, stride, w, h, &table, 3, 255);
                assert_eq!(a, b, "{name} sao_band u8 {w}x{h}");
                let mut off = [0i16; 5];
                for k in [0usize, 1, 3, 4] {
                    off[k] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                for &(na, nb) in &[(-1isize, 1isize), (-(stride as isize), stride as isize), (-(stride as isize) - 1, stride as isize + 1)] {
                    let origin = 4 * stride + 4;
                    let mut a = src.clone();
                    let mut b = src.clone();
                    (s.sao_edge)(&mut a, &src, origin, stride, w, h, na, nb, &off, 255);
                    (d.sao_edge)(&mut b, &src, origin, stride, w, h, na, nb, &off, 255);
                    assert_eq!(a, b, "{name} sao_edge u8 {w}x{h} {na},{nb}");
                }
            }
        }
    }

    #[test]
    fn deblocking_matches_scalar_u8() {
        let s = HevcDsp::<u8>::SCALAR;
        for (name, d) in tables_u8() {
            let mut seed = 53u64;
            let stride = 48;
            for trial in 0..500 {
                let base = lcg(&mut seed) % 256;
                let spread = 1 + lcg(&mut seed) % 48;
                let plane: Vec<u8> = (0..stride * 32).map(|_| ((base + lcg(&mut seed) % spread).min(255)) as u8).collect();
                let beta = [(lcg(&mut seed) % 64) as i32, (lcg(&mut seed) % 64) as i32];
                let tc = [(lcg(&mut seed) % 20) as i32, (lcg(&mut seed) % 20) as i32];
                let bl = |v: u32| v % 2 == 0;
                let no_p = [bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let no_q = [bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let tc4 = [tc[0], tc[1], (lcg(&mut seed) % 20) as i32, (lcg(&mut seed) % 20) as i32];
                let np4 = [no_p[0], no_p[1], bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let nq4 = [no_q[0], no_q[1], bl(lcg(&mut seed)), bl(lcg(&mut seed))];
                let off = 8 * stride + 8;
                let mut a = plane.clone();
                let mut b = plane.clone();
                match trial % 4 {
                    0 => {
                        (s.deblock_luma_v)(&mut a, off, stride, beta, tc, no_p, no_q, 255);
                        (d.deblock_luma_v)(&mut b, off, stride, beta, tc, no_p, no_q, 255);
                    }
                    1 => {
                        (s.deblock_luma_h)(&mut a, off, stride, beta, tc, no_p, no_q, 255);
                        (d.deblock_luma_h)(&mut b, off, stride, beta, tc, no_p, no_q, 255);
                    }
                    2 => {
                        (s.deblock_chroma_v)(&mut a, off, stride, tc4, np4, nq4, 255);
                        (d.deblock_chroma_v)(&mut b, off, stride, tc4, np4, nq4, 255);
                    }
                    _ => {
                        (s.deblock_chroma_h)(&mut a, off, stride, tc4, np4, nq4, 255);
                        (d.deblock_chroma_h)(&mut b, off, stride, tc4, np4, nq4, 255);
                    }
                }
                assert_eq!(a, b, "{name} deblock u8 kind {} trial {trial}", trial % 4);
            }
        }
    }
}
