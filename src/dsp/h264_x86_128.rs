//! 128-bit SIMD versions of the H.264 kernels (x86-64), from SSE2 up.
//!
//! Eight 16-bit lanes per vector: half the width of [`super::h264_avx2`], and
//! the same arithmetic. A block row of up to sixteen samples is one or two
//! vectors, and the narrow blocks that dominate H.264 (4- and 8-wide chroma
//! and sub-macroblock partitions) need only one — where the AVX2 kernels
//! compute a full sixteen samples whatever `w` is, these compute
//! `ceil(w / 8)` chunks, so the gap on small blocks is smaller than the
//! vector width suggests.
//!
//! The kernels are written once and compiled four times, one per rung of the
//! x86 ladder, by the `kernels!` macro. SSE2 is baseline on x86-64, so that
//! rung is what makes the scalar reference kernels unreachable on this
//! architecture; the rungs above it swap in better instructions for a few
//! recurring operations (see the `x86_compat` module) and for the two shapes
//! that are specific to this codec:
//!
//! - the six-tap luma filter, which SSSE3 does as three `pmaddubsw` on
//!   interleaved neighbour pairs instead of six widening loads and a
//!   shift-and-add chain — the taps 1, −5, 20 are all `i8` and every partial
//!   sum fits `i16` for 8-bit input, so it is exact;
//! - chroma bilinear, likewise two `pmaddubsw` instead of four `pmullw` and
//!   three adds (the four weights sum to 64, so 255 · 64 = 16320 bounds the
//!   result).
//!
//! [`install`] applies the rungs in order and each one replaces only the
//! kernels it actually improves, so a CPU ends up with the best available
//! version of every kernel and no rung pays for code it did not change.
//! AVX adds no 256-bit *integer* operation — that is AVX2 — so its rung runs
//! the SSE4.1 algorithms recompiled for VEX, whose three-operand form lets
//! the register allocator drop the `movdqa` copies the destructive
//! two-operand encoding forces before every non-commutative op.

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::h264::H264Dsp;

/// The two level-dependent shapes that are particular to H.264: the six-tap
/// luma filter and chroma bilinear. Everything else level-dependent is in
/// [`super::x86_compat`].
macro_rules! codec_compat {
    ($feat:literal, sse2) => {
        /// Six-tap `a - 5b + 20c + 20d - 5e + f` over the eight u8 samples at
        /// `p`, `p + step`, … `p + 5 * step`, as eight i16.
        ///
        /// `t = c + d <= 510` so `20t <= 10200`, `5u <= 2550` and the running
        /// value stays inside i16 throughout.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tap6_u8(p: *const u8, step: usize) -> __m128i {
            unsafe {
                let ld = |k: usize| zx8(_mm_loadl_epi64(p.add(k * step) as *const __m128i));
                let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
                let t = _mm_add_epi16(c, d);
                let u = _mm_add_epi16(b, e);
                let v = _mm_add_epi16(a, f);
                // v + 20t - 5u = v + (t << 4) + (t << 2) - (u << 2) - u
                let t20 = _mm_add_epi16(_mm_slli_epi16(t, 4), _mm_slli_epi16(t, 2));
                let u5 = _mm_add_epi16(_mm_slli_epi16(u, 2), u);
                _mm_sub_epi16(_mm_add_epi16(v, t20), u5)
            }
        }

        /// The four bilinear weights, in the form this level multiplies by.
        struct ChromaW([__m128i; 4]);

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_w(xf: i32, yf: i32) -> ChromaW {
            unsafe {
                ChromaW([
                    _mm_set1_epi16(((8 - xf) * (8 - yf)) as i16),
                    _mm_set1_epi16((xf * (8 - yf)) as i16),
                    _mm_set1_epi16(((8 - xf) * yf) as i16),
                    _mm_set1_epi16((xf * yf) as i16),
                ])
            }
        }

        /// `A·w0 + B·w1 + C·w2 + D·w3` over eight samples, where A/B are the
        /// row at `r0` and the one sample to its right and C/D the same at
        /// `r1`. The weights sum to 64, so the result is at most 16320.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_row(w: &ChromaW, r0: *const u8, r1: *const u8) -> __m128i {
            unsafe {
                let ld = |p: *const u8| zx8(_mm_loadl_epi64(p as *const __m128i));
                _mm_add_epi16(
                    _mm_add_epi16(_mm_mullo_epi16(ld(r0), w.0[0]), _mm_mullo_epi16(ld(r0.add(1)), w.0[1])),
                    _mm_add_epi16(_mm_mullo_epi16(ld(r1), w.0[2]), _mm_mullo_epi16(ld(r1.add(1)), w.0[3])),
                )
            }
        }
    };

    // SSSE3 and everything above it: `pmaddubsw` on interleaved neighbour
    // pairs, which is one instruction per tap pair and needs no widening.
    ($feat:literal, $lvl:tt) => {
        /// A pair of taps `(a, b)` as one 16-bit lane `a | b << 8` (the low
        /// byte multiplies the even sample of an interleaved pair).
        #[inline(always)]
        fn pair8(a: i8, b: i8) -> i16 {
            (a as u8 as i16) | ((b as i16) << 8)
        }

        /// Six-tap `a - 5b + 20c + 20d - 5e + f` over the eight u8 samples at
        /// `p`, `p + step`, … `p + 5 * step`, as eight i16.
        ///
        /// Each `pmaddubsw` saturates its own pair sum, but for 8-bit input
        /// none can reach it: `a - 5b` is in −1275..255, `20c + 20d` in
        /// 0..10200, `-5e + f` in −1275..255, and the total in −2550..10710.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tap6_u8(p: *const u8, step: usize) -> __m128i {
            unsafe {
                let ld = |k: usize| _mm_loadl_epi64(p.add(k * step) as *const __m128i);
                let pr = |k: usize| _mm_unpacklo_epi8(ld(k), ld(k + 1));
                let m = |v, t| _mm_maddubs_epi16(v, _mm_set1_epi16(t));
                _mm_add_epi16(
                    _mm_add_epi16(m(pr(0), pair8(1, -5)), m(pr(2), pair8(20, 20))),
                    m(pr(4), pair8(-5, 1)),
                )
            }
        }

        /// The four bilinear weights, as the two tap pairs `pmaddubsw` wants.
        struct ChromaW([__m128i; 2]);

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_w(xf: i32, yf: i32) -> ChromaW {
            unsafe {
                // Each weight is at most 64, so all four fit i8.
                ChromaW([
                    _mm_set1_epi16(pair8(((8 - xf) * (8 - yf)) as i8, (xf * (8 - yf)) as i8)),
                    _mm_set1_epi16(pair8(((8 - xf) * yf) as i8, (xf * yf) as i8)),
                ])
            }
        }

        /// `A·w0 + B·w1 + C·w2 + D·w3` over eight samples. The weights sum to
        /// 64, so neither `pmaddubsw` nor their sum can leave i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_row(w: &ChromaW, r0: *const u8, r1: *const u8) -> __m128i {
            unsafe {
                let pr = |p: *const u8| {
                    _mm_unpacklo_epi8(_mm_loadl_epi64(p as *const __m128i), _mm_loadl_epi64(p.add(1) as *const __m128i))
                };
                _mm_add_epi16(_mm_maddubs_epi16(pr(r0), w.0[0]), _mm_maddubs_epi16(pr(r1), w.0[1]))
            }
        }
    };
}

/// The whole kernel set, parameterised by the feature it is compiled for and
/// the primitive set that feature makes available.
macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::h264::{H264Dsp, NO_DC, PRED_STRIDE};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);
        codec_compat!($feat, $lvl);

        // ------------------------------------------------------------------
        // Install groups
        // ------------------------------------------------------------------
        //
        // Split so a rung can replace only what it changes: `interp` is the
        // kernels the six-tap and bilinear shapes reach, `deblock` the loop
        // filters, `rest` the transforms, residuals and weighting. A rung
        // that changes none of a group's primitives simply does not call it,
        // and the level below stays installed.

        /// Every kernel — the bottom rung, and the top one.
        pub(crate) fn install_all(d: &mut H264Dsp<u8>) {
            install_interp(d);
            install_deblock(d);
            install_rest(d);
            d.copy = copy;
            d.avg = avg;
        }

        /// Luma quarter-sample interpolation and chroma bilinear.
        pub(crate) fn install_interp(d: &mut H264Dsp<u8>) {
            d.qpel = [
                qpel::<0, 0>,
                qpel::<1, 0>,
                qpel::<2, 0>,
                qpel::<3, 0>,
                qpel::<0, 1>,
                qpel::<1, 1>,
                qpel::<2, 1>,
                qpel::<3, 1>,
                qpel::<0, 2>,
                qpel::<1, 2>,
                qpel::<2, 2>,
                qpel::<3, 2>,
                qpel::<0, 3>,
                qpel::<1, 3>,
                qpel::<2, 3>,
                qpel::<3, 3>,
            ];
            d.chroma = chroma;
        }

        /// The loop-filter entries.
        pub(crate) fn install_deblock(d: &mut H264Dsp<u8>) {
            d.deblock_luma_v = deblock_luma_v;
            d.deblock_luma8_v = deblock_luma8_v;
            d.deblock_luma8_v_intra = deblock_luma8_v_intra;
            d.deblock_luma_h = deblock_luma_h;
            d.deblock_luma_v_intra = deblock_luma_v_intra;
            d.deblock_luma_h_intra = deblock_luma_h_intra;
            d.deblock_chroma_v = deblock_chroma_v;
            d.deblock_chroma_h = deblock_chroma_h;
            d.deblock_chroma_v_intra = deblock_chroma_v_intra;
            d.deblock_chroma_h_intra = deblock_chroma_h_intra;
        }

        /// Weighting, inverse transforms and the residual paths.
        pub(crate) fn install_rest(d: &mut H264Dsp<u8>) {
            d.weighted_uni = weighted_uni;
            d.weighted_bi = weighted_bi;
            d.idct4_add = idct4_add;
            d.idct8_add = idct8_add;
            d.idct4_dc_add = idct4_dc_add;
            d.idct8_dc_add = idct8_dc_add;
            d.residual4 = residual4;
            d.residual8 = residual8;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        /// Store the first `n` (≤ 16) bytes of `v`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_u8_n(dst: *mut u8, v: __m128i, n: usize) {
            unsafe {
                if n == 16 {
                    _mm_storeu_si128(dst as *mut __m128i, v);
                } else if n == 8 {
                    _mm_storel_epi64(dst as *mut __m128i, v);
                } else if n == 4 {
                    // write_unaligned, not a plain store: `dst` is a row of a picture
                    // and is aligned to nothing in particular. A `*mut i32` store
                    // promises 4-byte alignment, which is UB when it is not true —
                    // x86 tolerates it and release builds do not check, so this was
                    // invisible until a debug test run aborted on it.
                    std::ptr::write_unaligned(dst as *mut i32, _mm_cvtsi128_si32(v));
                } else {
                    let mut t = [0u8; 16];
                    _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                    std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
                }
            }
        }

        /// Load 8 bytes as 8 × i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load8(p: *const u8) -> __m128i {
            unsafe { zx8(_mm_loadl_epi64(p as *const __m128i)) }
        }

        /// Store 8 × i16 as 8 bytes, saturating to `0..=255`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store8(p: *mut u8, v: __m128i) {
            unsafe { _mm_storel_epi64(p as *mut __m128i, _mm_packus_epi16(v, v)) }
        }

        /// `clip((v + 16) >> 5)` of 8 i16 lanes packed to 8 u8 (low 64 bits).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn round5_pack(v: __m128i) -> __m128i {
            unsafe {
                let r = _mm_srai_epi16(_mm_add_epi16(v, _mm_set1_epi16(16)), 5);
                _mm_packus_epi16(r, r)
            }
        }

        /// A tap pair as one i32 lane, for `pmaddwd`.
        #[inline(always)]
        fn pair(a: i16, b: i16) -> i32 {
            (a as u16 as i32) | ((b as u16 as i32) << 16)
        }

        // ------------------------------------------------------------------
        // Luma interpolation
        // ------------------------------------------------------------------

        /// Horizontal half-sample intermediate (i16) for the eight output
        /// columns from `x` of window row `row`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn b1_row(src: *const u8, stride: usize, row: usize, x: usize) -> __m128i {
            unsafe { tap6_u8(src.add(row * stride + x), 1) }
        }

        /// Vertical half-sample intermediate (i16) at window column `col`, block row `y`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn h1_row(src: *const u8, stride: usize, col: usize, y: usize) -> __m128i {
            unsafe { tap6_u8(src.add(y * stride + col), stride) }
        }

        /// Centre position, eight columns: vertical six-tap over the six
        /// horizontal intermediates `b1` of window rows `y..y+5`, 32-bit
        /// accumulation, `clip((v + 512) >> 10)`.
        ///
        /// The six are passed in rather than computed here because
        /// consecutive output rows share five of them — see `qpel_impl`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn j_combine(w: &[__m128i; 6]) -> __m128i {
            unsafe {
                let (r0, r1, r2, r3, r4, r5) = (w[0], w[1], w[2], w[3], w[4], w[5]);
                let c01 = _mm_set1_epi32(pair(1, -5));
                let c23 = _mm_set1_epi32(pair(20, 20));
                let c45 = _mm_set1_epi32(pair(-5, 1));
                let round = _mm_set1_epi32(512);
                let lo = _mm_add_epi32(
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(r0, r1), c01), _mm_madd_epi16(_mm_unpacklo_epi16(r2, r3), c23)),
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(r4, r5), c45), round),
                );
                let hi = _mm_add_epi32(
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(r0, r1), c01), _mm_madd_epi16(_mm_unpackhi_epi16(r2, r3), c23)),
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(r4, r5), c45), round),
                );
                // At 128 bits `packs` already lands lanes 0..7 in order — the
                // cross-lane fixup the 256-bit kernel needs has no counterpart.
                let v = _mm_packs_epi32(_mm_srai_epi32(lo, 10), _mm_srai_epi32(hi, 10));
                _mm_packus_epi16(v, v)
            }
        }

        /// Full samples of block row `y` from column `x` (window offset 2, 2).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn g_row(src: *const u8, stride: usize, y: usize, dx: usize, x: usize) -> __m128i {
            unsafe { _mm_loadl_epi64(src.add((y + 2) * stride + 2 + dx + x) as *const __m128i) }
        }

        fn qpel<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, _max: i32) {
            // The window is (w + 5) x (h + 5); an 8-lane load from column
            // x <= 8 reads x + 7 (+5 for taps), so the guard the 16-lane
            // kernel needs covers this one too.
            let need = (h + 5 - 1) * stride + 21;
            if src.len() < need {
                return (H264Dsp::<u8>::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h, 255);
            }
            unsafe { qpel_impl::<XF, YF>(dst, src, stride, w, h) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn qpel_impl<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
            // The five positions whose vertical filter runs over the
            // *horizontal* intermediates rather than over samples. Their row
            // y needs `b1` of window rows y..y+5 and their row y+1 needs
            // y+1..y+6, so five of every six are shared: computed once and
            // slid down, not recomputed. Sliding needs y innermost, which is
            // why that kernel nests its loops the other way round.
            if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
                return unsafe { qpel_centre_impl::<XF, YF>(dst, src, stride, w, h) };
            }
            unsafe {
                let s = src.as_ptr();
                let chunks = w.div_ceil(8);
                for y in 0..h {
                    for c in 0..chunks {
                        let x = c * 8;
                        let d = dst.as_mut_ptr().add(y * PRED_STRIDE + x);
                        let b = || round5_pack(b1_row(s, stride, y + 2, x));
                        let b_below = || round5_pack(b1_row(s, stride, y + 3, x));
                        let hh = || round5_pack(h1_row(s, stride, 2 + x, y));
                        let hh_right = || round5_pack(h1_row(s, stride, 3 + x, y));
                        let v: __m128i = match (XF, YF) {
                            (0, 0) => g_row(s, stride, y, 0, x),
                            (1, 0) => _mm_avg_epu8(g_row(s, stride, y, 0, x), b()),
                            (2, 0) => b(),
                            (3, 0) => _mm_avg_epu8(g_row(s, stride, y, 1, x), b()),
                            (0, 1) => _mm_avg_epu8(g_row(s, stride, y, 0, x), hh()),
                            (0, 2) => hh(),
                            (0, 3) => _mm_avg_epu8(_mm_loadl_epi64(s.add((y + 3) * stride + 2 + x) as *const __m128i), hh()),
                            (1, 1) => _mm_avg_epu8(b(), hh()),
                            (3, 1) => _mm_avg_epu8(b(), hh_right()),
                            (1, 3) => _mm_avg_epu8(hh(), b_below()),
                            (3, 3) => _mm_avg_epu8(hh_right(), b_below()),
                            _ => unreachable!(),
                        };
                        _mm_storel_epi64(d as *mut __m128i, v);
                    }
                }
            }
        }

        /// The centre positions, over a sliding window of `b1` rows.
        ///
        /// `win[k]` holds the horizontal intermediate of window row `y + k`
        /// for the current output row `y`; after each row the window slides
        /// down one and only the new bottom row is filtered. Columns are the
        /// outer loop so that the window belongs to one eight-wide chunk.
        /// `b` and `b_below`, which the half-and-half positions need, are
        /// rows y+2 and y+3 of that same window, so they cost a pack rather
        /// than a filter.
        #[target_feature(enable = $feat)]
        unsafe fn qpel_centre_impl<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
            unsafe {
                let s = src.as_ptr();
                for c in 0..w.div_ceil(8) {
                    let x = c * 8;
                    let mut win = [
                        b1_row(s, stride, 0, x),
                        b1_row(s, stride, 1, x),
                        b1_row(s, stride, 2, x),
                        b1_row(s, stride, 3, x),
                        b1_row(s, stride, 4, x),
                        b1_row(s, stride, 5, x),
                    ];
                    for y in 0..h {
                        let j = j_combine(&win);
                        let v: __m128i = match (XF, YF) {
                            (2, 2) => j,
                            (2, 1) => _mm_avg_epu8(round5_pack(win[2]), j),
                            (2, 3) => _mm_avg_epu8(j, round5_pack(win[3])),
                            (1, 2) => _mm_avg_epu8(round5_pack(h1_row(s, stride, 2 + x, y)), j),
                            (3, 2) => _mm_avg_epu8(j, round5_pack(h1_row(s, stride, 3 + x, y))),
                            _ => unreachable!(),
                        };
                        _mm_storel_epi64(dst.as_mut_ptr().add(y * PRED_STRIDE + x) as *mut __m128i, v);
                        // Not on the last row: the caller's bounds check
                        // covers window rows up to h + 4, and row h + 5 would
                        // read past the block.
                        if y + 1 < h {
                            win = [win[1], win[2], win[3], win[4], win[5], b1_row(s, stride, y + 6, x)];
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Chroma interpolation, combination and weighting
        // ------------------------------------------------------------------

        fn chroma(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
            if src.len() < h * stride + 9 {
                return (H264Dsp::<u8>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
            }
            unsafe { chroma_impl(dst, src, stride, w, h, xf, yf) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn chroma_impl(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
            unsafe {
                // Chroma blocks are at most 8 wide: eight i16 lanes.
                let _ = w;
                let cw = chroma_w(xf, yf);
                let round = _mm_set1_epi16(32);
                let s = src.as_ptr();
                for y in 0..h {
                    let v = chroma_row(&cw, s.add(y * stride), s.add((y + 1) * stride));
                    let v = _mm_srli_epi16(_mm_add_epi16(v, round), 6);
                    _mm_storel_epi64(dst.as_mut_ptr().add(y * PRED_STRIDE) as *mut __m128i, _mm_packus_epi16(v, v));
                }
            }
        }

        fn copy(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
            assert!((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len());
            unsafe { copy_impl(dst.as_mut_ptr(), stride, src.as_ptr(), w, h) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn copy_impl(dst: *mut u8, stride: usize, src: *const u8, w: usize, h: usize) {
            unsafe {
                for y in 0..h {
                    let s = src.add(y * PRED_STRIDE);
                    let d = dst.add(y * stride);
                    match w {
                        16 => _mm_storeu_si128(d as *mut __m128i, _mm_loadu_si128(s as *const __m128i)),
                        8 => std::ptr::write_unaligned(d as *mut u64, std::ptr::read_unaligned(s as *const u64)),
                        4 => std::ptr::write_unaligned(d as *mut u32, std::ptr::read_unaligned(s as *const u32)),
                        _ => std::ptr::copy_nonoverlapping(s, d, w),
                    }
                }
            }
        }

        fn avg(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
            unsafe { avg_impl(dst, stride, a, b, w, h) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn avg_impl(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
            unsafe {
                // The scratch rows are 16 wide, so a full load is always in
                // bounds; only the store into the plane is sized.
                for y in 0..h {
                    let va = _mm_loadu_si128(a.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
                    let vb = _mm_loadu_si128(b.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
                    store_u8_n(dst.as_mut_ptr().add(y * stride), _mm_avg_epu8(va, vb), w);
                }
            }
        }

        fn weighted_uni(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, _max: i32) {
            unsafe { weighted_uni_impl(dst, stride, src, w, h, log_wd, wt, o) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn weighted_uni_impl(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32) {
            unsafe {
                // src * wt + round fits i16 only for |wt| <= 128 (spec: -128..127) ✓.
                let wv = _mm_set1_epi16(wt as i16);
                let ov = _mm_set1_epi16(o as i16);
                let round = _mm_set1_epi16(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(log_wd.max(0));
                let scale = |s: __m128i| _mm_add_epi16(_mm_sra_epi16(_mm_add_epi16(_mm_mullo_epi16(s, wv), round), sh), ov);
                for y in 0..h {
                    let p = src.as_ptr().add(y * PRED_STRIDE);
                    let v0 = scale(load8(p));
                    let v1 = if w > 8 { scale(load8(p.add(8))) } else { v0 };
                    store_u8_n(dst.as_mut_ptr().add(y * stride), _mm_packus_epi16(v0, v1), w);
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_bi(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, _max: i32) {
            unsafe { weighted_bi_impl(dst, stride, a, b, w, h, log_wd, w0, w1, o0, o1) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_bi_impl(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
            unsafe {
                // a * w0 + b * w1 reaches 2 * 255 * 128 = 65280, so it needs
                // 32-bit lanes — but `pmaddwd` on the two predictions
                // interleaved produces exactly that sum in one instruction,
                // for weights in the spec's -128..127 and samples in 0..255.
                let wv = _mm_set1_epi32(pair(w0 as i16, w1 as i16));
                let round = _mm_set1_epi32(1 << log_wd);
                let off = _mm_set1_epi32((o0 + o1 + 1) >> 1);
                let sh = _mm_cvtsi32_si128(log_wd + 1);
                for y in 0..h {
                    let pa = a.as_ptr().add(y * PRED_STRIDE);
                    let pb = b.as_ptr().add(y * PRED_STRIDE);
                    // Eight samples from `x` as one vector of eight i16.
                    let eight = |x: usize| -> __m128i {
                        let va = load8(pa.add(x));
                        let vb = load8(pb.add(x));
                        let quad = |v: __m128i| _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(v, wv), round), sh), off);
                        let lo = quad(_mm_unpacklo_epi16(va, vb));
                        let hi = quad(_mm_unpackhi_epi16(va, vb));
                        _mm_packs_epi32(lo, hi)
                    };
                    let v0 = eight(0);
                    let v1 = if w > 8 { eight(8) } else { v0 };
                    store_u8_n(dst.as_mut_ptr().add(y * stride), _mm_packus_epi16(v0, v1), w);
                }
            }
        }

        // ------------------------------------------------------------------
        // Deblocking
        // ------------------------------------------------------------------
        //
        // Eight lines of an edge are eight i16 lanes of one vector per sample
        // position; a sixteen-line luma edge is two such halves, and its tC0
        // segments (four lines each) fall two per half. A horizontal edge
        // loads a sample position as one row; a vertical edge transposes 8
        // rows x 8 bytes into 8 column vectors, filters, and transposes back.

        /// `|a - b| < t` per i16 lane, as a mask.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn diff_lt(a: __m128i, b: __m128i, t: __m128i) -> __m128i {
            unsafe { _mm_cmpgt_epi16(t, abs16(_mm_sub_epi16(a, b))) }
        }

        /// The eight positions of eight luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
        type LumaLines = [__m128i; 8];

        /// bS < 4 luma filter on eight lines (8.7.2.3), in place on the vectors.
        /// `tc0v` holds the line's tC0 (−1 = bS 0).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: __m128i) {
            unsafe {
                let [_, p2, p1, p0, q0, q1, q2, _] = *v;
                let alpha = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let zero = _mm_setzero_si128();
                let bs_on = _mm_cmpgt_epi16(tc0v, _mm_set1_epi16(-1));
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), _mm_and_si128(diff_lt(q1, q0, beta), bs_on));
                let ap = diff_lt(p2, p0, beta);
                let aq = diff_lt(q2, q0, beta);
                // tc = tc0 + (ap < beta) + (aq < beta); masks are -1.
                let tc = _mm_sub_epi16(_mm_sub_epi16(tc0v, ap), aq);
                // delta = clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3)
                let d = _mm_srai_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(q0, p0), 2), _mm_sub_epi16(p1, q1)), _mm_set1_epi16(4)),
                    3,
                );
                let d = _mm_min_epi16(_mm_max_epi16(d, _mm_sub_epi16(zero, tc)), tc);
                let np0 = _mm_add_epi16(p0, d);
                let nq0 = _mm_sub_epi16(q0, d);
                // p1' = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - 2 p1) >> 1), when ap
                let avg = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(p0, q0), _mm_set1_epi16(1)), 1);
                let ntc0 = _mm_sub_epi16(zero, tc0v);
                let dp1 = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(p2, avg), _mm_slli_epi16(p1, 1)), 1);
                let dp1 = _mm_min_epi16(_mm_max_epi16(dp1, ntc0), tc0v);
                let np1 = _mm_add_epi16(p1, _mm_and_si128(dp1, ap));
                let dq1 = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(q2, avg), _mm_slli_epi16(q1, 1)), 1);
                let dq1 = _mm_min_epi16(_mm_max_epi16(dq1, ntc0), tc0v);
                let nq1 = _mm_add_epi16(q1, _mm_and_si128(dq1, aq));
                // Clip to 8 bits (p1'/q1' cannot leave the range; p0'/q0' can).
                let clip = |x: __m128i| _mm_min_epi16(_mm_max_epi16(x, zero), _mm_set1_epi16(255));
                v[2] = sel(p1, np1, mask);
                v[3] = sel(p0, clip(np0), mask);
                v[4] = sel(q0, clip(nq0), mask);
                v[5] = sel(q1, nq1, mask);
            }
        }

        /// bS 4 luma filter on eight lines (8.7.2.4).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn luma_filter_intra(v: &mut LumaLines, alpha: i32, beta: i32) {
            unsafe {
                let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
                let alphav = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
                let strong = diff_lt(p0, q0, _mm_set1_epi16(((alpha >> 2) + 2) as i16));
                let ap = _mm_and_si128(diff_lt(p2, p0, beta), strong);
                let aq = _mm_and_si128(diff_lt(q2, q0, beta), strong);
                let two = _mm_set1_epi16(2);
                let four = _mm_set1_epi16(4);
                let add = |a, b| _mm_add_epi16(a, b);
                let dbl = |a| _mm_slli_epi16(a, 1);
                // Weak: p0' = (2 p1 + p0 + q1 + 2) >> 2, q0' = (2 q1 + q0 + p1 + 2) >> 2.
                let wp0 = _mm_srai_epi16(add(add(dbl(p1), p0), add(q1, two)), 2);
                let wq0 = _mm_srai_epi16(add(add(dbl(q1), q0), add(p1, two)), 2);
                // Strong p side.
                let p0q0 = add(p0, q0);
                let sp0 = _mm_srai_epi16(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)), 3);
                let sp1 = _mm_srai_epi16(add(add(p2, p1), add(p0q0, two)), 2);
                let sp2 = _mm_srai_epi16(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)), 3);
                // Strong q side.
                let sq0 = _mm_srai_epi16(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)), 3);
                let sq1 = _mm_srai_epi16(add(add(p0q0, q1), add(q2, two)), 2);
                let sq2 = _mm_srai_epi16(add(add(dbl(q3), add(q2, dbl(q2))), add(add(q1, p0q0), four)), 3);
                let np0 = sel(wp0, sp0, ap);
                let np1 = sel(p1, sp1, ap);
                let np2 = sel(p2, sp2, ap);
                let nq0 = sel(wq0, sq0, aq);
                let nq1 = sel(q1, sq1, aq);
                let nq2 = sel(q2, sq2, aq);
                v[1] = sel(p2, np2, mask);
                v[2] = sel(p1, np1, mask);
                v[3] = sel(p0, np0, mask);
                v[4] = sel(q0, nq0, mask);
                v[5] = sel(q1, nq1, mask);
                v[6] = sel(q2, nq2, mask);
            }
        }

        /// tC0 per lane for the eight luma lines of half `half` (four per segment).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tc0_luma(tc0: &[i16; 4], half: usize) -> __m128i {
            unsafe {
                let (a, b) = (tc0[2 * half], tc0[2 * half + 1]);
                _mm_setr_epi16(a, a, a, a, b, b, b, b)
            }
        }

        /// Load the eight rows x 8 bytes around a vertical edge (`q0` at
        /// `data`) as eight column vectors p3..q3.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_transposed_8x8(data: *const u8, stride: usize) -> LumaLines {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_loadl_epi64(data.add(i * stride).sub(4) as *const __m128i);
                }
                // Bytes: pairs of rows.
                let a0 = _mm_unpacklo_epi8(r[0], r[1]);
                let a1 = _mm_unpacklo_epi8(r[2], r[3]);
                let a2 = _mm_unpacklo_epi8(r[4], r[5]);
                let a3 = _mm_unpacklo_epi8(r[6], r[7]);
                // Words: quads of rows; lo = columns 0..3, hi = columns 4..7.
                let b0 = _mm_unpacklo_epi16(a0, a1); // cols 0..3, rows 0..3
                let b1 = _mm_unpackhi_epi16(a0, a1); // cols 4..7, rows 0..3
                let b2 = _mm_unpacklo_epi16(a2, a3); // cols 0..3, rows 4..7
                let b3 = _mm_unpackhi_epi16(a2, a3); // cols 4..7, rows 4..7
                // Dwords: a column pair's eight rows per vector.
                let c0 = _mm_unpacklo_epi32(b0, b2); // col0 rows 0..7 | col1
                let c1 = _mm_unpackhi_epi32(b0, b2); // col2 | col3
                let c2 = _mm_unpacklo_epi32(b1, b3); // col4 | col5
                let c3 = _mm_unpackhi_epi32(b1, b3); // col6 | col7
                [zx8(c0), zx8h(c0), zx8(c1), zx8h(c1), zx8(c2), zx8h(c2), zx8(c3), zx8h(c3)]
            }
        }

        /// Store eight column vectors back as eight rows x 8 bytes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_transposed_8x8(data: *mut u8, stride: usize, v: &LumaLines) {
            unsafe {
                let p = |x: __m128i| _mm_packus_epi16(x, x);
                let (c0, c1, c2, c3) = (p(v[0]), p(v[1]), p(v[2]), p(v[3]));
                let (c4, c5, c6, c7) = (p(v[4]), p(v[5]), p(v[6]), p(v[7]));
                // Bytes: column pairs -> rows interleaved.
                let a01 = _mm_unpacklo_epi8(c0, c1);
                let a23 = _mm_unpacklo_epi8(c2, c3);
                let a45 = _mm_unpacklo_epi8(c4, c5);
                let a67 = _mm_unpacklo_epi8(c6, c7);
                // Words: rows with p3..p0 / q0..q3.
                let bp_lo = _mm_unpacklo_epi16(a01, a23); // rows 0..3
                let bp_hi = _mm_unpackhi_epi16(a01, a23); // rows 4..7
                let bq_lo = _mm_unpacklo_epi16(a45, a67);
                let bq_hi = _mm_unpackhi_epi16(a45, a67);
                // Dwords: whole rows (8 bytes), two per vector.
                let rows = [
                    _mm_unpacklo_epi32(bp_lo, bq_lo), // rows 0,1
                    _mm_unpackhi_epi32(bp_lo, bq_lo), // rows 2,3
                    _mm_unpacklo_epi32(bp_hi, bq_hi), // rows 4,5
                    _mm_unpackhi_epi32(bp_hi, bq_hi), // rows 6,7
                ];
                for (k, pair) in rows.iter().enumerate() {
                    _mm_storel_epi64(data.add(2 * k * stride).sub(4) as *mut __m128i, *pair);
                    _mm_storel_epi64(data.add((2 * k + 1) * stride).sub(4) as *mut __m128i, _mm_srli_si128(*pair, 8));
                }
            }
        }

        fn deblock_luma_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
            unsafe {
                for half in 0..2 {
                    let d = data.add(half * 8 * stride);
                    let mut v = load_transposed_8x8(d, stride);
                    luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half));
                    store_transposed_8x8(d, stride, &v);
                }
            }
        }

        /// tC0 per lane for an eight-line luma edge: `tc0[i / 2]`, an MBAFF
        /// mixed edge's strength changing every two lines rather than every
        /// four.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tc0_luma8(tc0: &[i16; 4]) -> __m128i {
            unsafe {
                let t = |k: usize| tc0[k];
                _mm_setr_epi16(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
            }
        }

        fn deblock_luma8_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma8_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
        }

        /// Eight lines is exactly one half of the sixteen-line kernel's
        /// loop; only the tC0 lanes differ.
        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma8_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
            unsafe {
                let mut v = load_transposed_8x8(data, stride);
                luma_filter_normal(&mut v, alpha, beta, tc0_luma8(tc0));
                store_transposed_8x8(data, stride, &v);
            }
        }

        fn deblock_luma8_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma8_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma8_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                let mut v = load_transposed_8x8(data, stride);
                luma_filter_intra(&mut v, alpha, beta);
                store_transposed_8x8(data, stride, &v);
            }
        }

        fn deblock_luma_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
            assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                for half in 0..2 {
                    let d = data.add(half * 8 * stride);
                    let mut v = load_transposed_8x8(d, stride);
                    luma_filter_intra(&mut v, alpha, beta);
                    store_transposed_8x8(d, stride, &v);
                }
            }
        }

        fn deblock_luma_h(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            assert!(off >= 3 * stride && off + 2 * stride + 16 <= data.len());
            unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
            unsafe {
                let zero = _mm_setzero_si128();
                for half in 0..2 {
                    let d = data.add(half * 8);
                    let ld = |k: isize| load8(d.offset(k * stride as isize));
                    let mut v: LumaLines = [zero, ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), zero];
                    luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half));
                    store8(d.offset(-2 * stride as isize), v[2]);
                    store8(d.offset(-(stride as isize)), v[3]);
                    store8(d, v[4]);
                    store8(d.add(stride), v[5]);
                }
            }
        }

        fn deblock_luma_h_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
            assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
            unsafe { deblock_luma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                for half in 0..2 {
                    let d = data.add(half * 8);
                    let ld = |k: isize| load8(d.offset(k * stride as isize));
                    let mut v: LumaLines = [ld(-4), ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), ld(3)];
                    luma_filter_intra(&mut v, alpha, beta);
                    for k in 1..7 {
                        store8(d.offset((k as isize - 4) * stride as isize), v[k]);
                    }
                }
            }
        }

        // Chroma: eight lines, positions [p1, p0, q0, q1] as 8 x i16.
        type ChromaLines = [__m128i; 4];

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: __m128i) {
            unsafe {
                let [p1, p0, q0, q1] = *v;
                let alpha = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let zero = _mm_setzero_si128();
                let bs_on = _mm_cmpgt_epi16(tc0v, _mm_set1_epi16(-1));
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), _mm_and_si128(diff_lt(q1, q0, beta), bs_on));
                let tc = _mm_add_epi16(tc0v, _mm_set1_epi16(1));
                let d = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(q0, p0), 2), _mm_sub_epi16(p1, q1)), _mm_set1_epi16(4)), 3);
                let d = _mm_min_epi16(_mm_max_epi16(d, _mm_sub_epi16(zero, tc)), tc);
                let clip = |x: __m128i| _mm_min_epi16(_mm_max_epi16(x, zero), _mm_set1_epi16(255));
                v[1] = sel(p0, clip(_mm_add_epi16(p0, d)), mask);
                v[2] = sel(q0, clip(_mm_sub_epi16(q0, d)), mask);
            }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
            unsafe {
                let [p1, p0, q0, q1] = *v;
                let alpha = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
                let two = _mm_set1_epi16(2);
                let np0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(p1, 1), p0), _mm_add_epi16(q1, two)), 2);
                let nq0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(q1, 1), q0), _mm_add_epi16(p1, two)), 2);
                v[1] = sel(p0, np0, mask);
                v[2] = sel(q0, nq0, mask);
            }
        }

        /// tC0 per lane for eight chroma lines (two per segment).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tc0_chroma(tc0: &[i16; 4]) -> __m128i {
            unsafe {
                let t = |k: usize| tc0[k];
                _mm_setr_epi16(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
            }
        }

        /// Load 8 rows x 4 bytes (p1 p0 q0 q1) around a vertical chroma edge
        /// as four column vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_transposed_8x4(data: *const u8, stride: usize) -> ChromaLines {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_cvtsi32_si128(std::ptr::read_unaligned(data.add(i * stride).sub(2) as *const i32));
                }
                let a0 = _mm_unpacklo_epi8(r[0], r[1]); // p1r0 p1r1 p0r0 p0r1 q0r0 q0r1 q1r0 q1r1
                let a1 = _mm_unpacklo_epi8(r[2], r[3]);
                let a2 = _mm_unpacklo_epi8(r[4], r[5]);
                let a3 = _mm_unpacklo_epi8(r[6], r[7]);
                let b0 = _mm_unpacklo_epi16(a0, a1); // p1 r0..3, p0 r0..3, q0 r0..3, q1 r0..3
                let b1 = _mm_unpacklo_epi16(a2, a3); // rows 4..7
                let c0 = _mm_unpacklo_epi32(b0, b1); // p1 r0..7 | p0 r0..7
                let c1 = _mm_unpackhi_epi32(b0, b1); // q0 r0..7 | q1 r0..7
                [zx8(c0), zx8h(c0), zx8(c1), zx8h(c1)]
            }
        }

        /// Store the p0 / q0 columns of eight rows back (p1, q1 are unchanged).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_transposed_8x4(data: *mut u8, stride: usize, v: &ChromaLines) {
            unsafe {
                // Interleave p0 and q0 bytes per row and store two bytes at x-1, x.
                let p0 = _mm_packus_epi16(v[1], v[1]);
                let q0 = _mm_packus_epi16(v[2], v[2]);
                let pq = _mm_unpacklo_epi8(p0, q0); // p0r0 q0r0 p0r1 q0r1 ...
                let mut t = [0u8; 16];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, pq);
                for i in 0..8 {
                    let d = data.add(i * stride).sub(1);
                    *d = t[2 * i];
                    *d.add(1) = t[2 * i + 1];
                }
            }
        }

        fn deblock_chroma_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
            unsafe {
                let mut v = load_transposed_8x4(data, stride);
                chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
                store_transposed_8x4(data, stride, &v);
            }
        }

        fn deblock_chroma_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                let mut v = load_transposed_8x4(data, stride);
                chroma_filter_intra(&mut v, alpha, beta);
                store_transposed_8x4(data, stride, &v);
            }
        }

        fn deblock_chroma_h(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
            unsafe {
                let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
                chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
                store8(data.sub(stride), v[1]);
                store8(data, v[2]);
            }
        }

        fn deblock_chroma_h_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
                chroma_filter_intra(&mut v, alpha, beta);
                store8(data.sub(stride), v[1]);
                store8(data, v[2]);
            }
        }

        // ------------------------------------------------------------------
        // Inverse transforms
        // ------------------------------------------------------------------
        //
        // Identical to the AVX2 kernels, which are 128-bit throughout: a 4x4
        // or 8x8 block's row is four or eight i16 lanes whatever the vector
        // width, so there was never anything for an upper half to do.

        /// Add `(v + 32) >> 6` rows to `dst`, clipping, `n` = 4 or 8 samples per row.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn add_row(dst: *mut u8, v: __m128i, n: usize) {
            unsafe {
                let r = _mm_srai_epi16(_mm_add_epi16(v, _mm_set1_epi16(32)), 6);
                if n == 4 {
                    let p = zx8(_mm_cvtsi32_si128(std::ptr::read_unaligned(dst as *const i32)));
                    let s = _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128());
                    std::ptr::write_unaligned(dst as *mut i32, _mm_cvtsi128_si32(s));
                } else {
                    let p = load8(dst);
                    let s = _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128());
                    _mm_storel_epi64(dst as *mut __m128i, s);
                }
            }
        }

        fn idct4_add(dst: &mut [u8], stride: usize, coeffs: &[i16; 16], _max: i32) {
            assert!(3 * stride + 4 <= dst.len());
            unsafe { idct4_add_impl(dst.as_mut_ptr(), stride, coeffs) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn idct4_add_impl(dst: *mut u8, stride: usize, c: &[i16; 16]) {
            unsafe {
                let r0 = _mm_loadl_epi64(c.as_ptr() as *const __m128i);
                let r1 = _mm_loadl_epi64(c.as_ptr().add(4) as *const __m128i);
                let r2 = _mm_loadl_epi64(c.as_ptr().add(8) as *const __m128i);
                let r3 = _mm_loadl_epi64(c.as_ptr().add(12) as *const __m128i);
                // Columns of the block, four lanes each.
                let t0 = _mm_unpacklo_epi16(r0, r1);
                let t1 = _mm_unpacklo_epi16(r2, r3);
                let c01 = _mm_unpacklo_epi32(t0, t1);
                let c23 = _mm_unpackhi_epi32(t0, t1);
                let (c0, c1, c2, c3) = (c01, _mm_srli_si128(c01, 8), c23, _mm_srli_si128(c23, 8));
                // Row pass.
                let e0 = _mm_add_epi16(c0, c2);
                let e1 = _mm_sub_epi16(c0, c2);
                let e2 = _mm_sub_epi16(_mm_srai_epi16(c1, 1), c3);
                let e3 = _mm_add_epi16(c1, _mm_srai_epi16(c3, 1));
                let f0 = _mm_add_epi16(e0, e3);
                let f1 = _mm_add_epi16(e1, e2);
                let f2 = _mm_sub_epi16(e1, e2);
                let f3 = _mm_sub_epi16(e0, e3);
                // Back to rows.
                let u0 = _mm_unpacklo_epi16(f0, f1);
                let u1 = _mm_unpacklo_epi16(f2, f3);
                let r01 = _mm_unpacklo_epi32(u0, u1);
                let r23 = _mm_unpackhi_epi32(u0, u1);
                let (row0, row1, row2, row3) = (r01, _mm_srli_si128(r01, 8), r23, _mm_srli_si128(r23, 8));
                // Column pass.
                let g0 = _mm_add_epi16(row0, row2);
                let g1 = _mm_sub_epi16(row0, row2);
                let g2 = _mm_sub_epi16(_mm_srai_epi16(row1, 1), row3);
                let g3 = _mm_add_epi16(row1, _mm_srai_epi16(row3, 1));
                add_row(dst, _mm_add_epi16(g0, g3), 4);
                add_row(dst.add(stride), _mm_add_epi16(g1, g2), 4);
                add_row(dst.add(2 * stride), _mm_sub_epi16(g1, g2), 4);
                add_row(dst.add(3 * stride), _mm_sub_epi16(g0, g3), 4);
            }
        }

        /// Transpose eight 8-lane i16 rows.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn transpose8(r: &mut [__m128i; 8]) {
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

        /// One 8-point pass (8.5.13.2) across eight registers.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn idct8_pass(d: &[__m128i; 8]) -> [__m128i; 8] {
            unsafe {
                let add = |a, b| _mm_add_epi16(a, b);
                let sub = |a, b| _mm_sub_epi16(a, b);
                let sh1 = |a| _mm_srai_epi16(a, 1);
                let sh2 = |a| _mm_srai_epi16(a, 2);
                let a0 = add(d[0], d[4]);
                let a4 = sub(d[0], d[4]);
                let a2 = sub(sh1(d[2]), d[6]);
                let a6 = add(d[2], sh1(d[6]));
                let b0 = add(a0, a6);
                let b2 = add(a4, a2);
                let b4 = sub(a4, a2);
                let b6 = sub(a0, a6);
                // a1 = -d3 + d5 - d7 - (d7 >> 1)
                let a1 = sub(sub(sub(d[5], d[3]), d[7]), sh1(d[7]));
                // a3 = d1 + d7 - d3 - (d3 >> 1)
                let a3 = sub(sub(add(d[1], d[7]), d[3]), sh1(d[3]));
                // a5 = -d1 + d7 + d5 + (d5 >> 1)
                let a5 = add(add(sub(d[7], d[1]), d[5]), sh1(d[5]));
                // a7 = d3 + d5 + d1 + (d1 >> 1)
                let a7 = add(add(add(d[3], d[5]), d[1]), sh1(d[1]));
                let b1 = add(a1, sh2(a7));
                let b7 = sub(a7, sh2(a1));
                let b3 = add(a3, sh2(a5));
                let b5 = sub(sh2(a3), a5);
                [add(b0, b7), add(b2, b5), add(b4, b3), add(b6, b1), sub(b6, b1), sub(b4, b3), sub(b2, b5), sub(b0, b7)]
            }
        }

        fn idct8_add(dst: &mut [u8], stride: usize, coeffs: &[i16; 64], _max: i32) {
            assert!(7 * stride + 8 <= dst.len());
            unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn idct8_add_impl(dst: *mut u8, stride: usize, c: &[i16; 64]) {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for i in 0..8 {
                    r[i] = _mm_loadu_si128(c.as_ptr().add(i * 8) as *const __m128i);
                }
                transpose8(&mut r);
                let mut f = idct8_pass(&r);
                transpose8(&mut f);
                let h = idct8_pass(&f);
                for i in 0..8 {
                    add_row(dst.add(i * stride), h[i], 8);
                }
            }
        }

        fn idct4_dc_add(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
            assert!(3 * stride + 4 <= dst.len());
            unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 4) }
        }

        fn idct8_dc_add(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
            assert!(7 * stride + 8 <= dst.len());
            unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 8) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn dc_add_impl(dst: *mut u8, stride: usize, dc: i32, n: usize) {
            unsafe {
                let v = _mm_set1_epi16(dc as i16);
                for i in 0..n {
                    add_row(dst.add(i * stride), v, n);
                }
            }
        }

        /// Eight dequantised coefficients (two vectors of four i32) as one
        /// vector of eight i16, saturating.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn coefs16(coefs: *const i32) -> __m128i {
            unsafe { _mm_packs_epi32(_mm_loadu_si128(coefs as *const __m128i), _mm_loadu_si128(coefs.add(4) as *const __m128i)) }
        }

        fn residual4(dst: &mut [u8], stride: usize, coefs: &[i32; 16], dc: i32, _max: i32) {
            assert!(3 * stride + 4 <= dst.len());
            unsafe { residual4_impl(dst.as_mut_ptr(), stride, coefs, dc) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn residual4_impl(dst: *mut u8, stride: usize, coefs: &[i32; 16], dc: i32) {
            unsafe {
                let mut c0 = coefs16(coefs.as_ptr());
                let c1 = coefs16(coefs.as_ptr().add(8));
                if dc != NO_DC {
                    c0 = _mm_insert_epi16(c0, dc as i16 as i32, 0);
                }
                // Any AC nonzero? Zero lane 0, compare the rest with zero.
                let ac = _mm_or_si128(_mm_andnot_si128(_mm_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0), c0), c1);
                if is_zero(ac) {
                    let d = _mm_extract_epi16(c0, 0) as i16 as i32;
                    if d != 0 {
                        dc_add_impl(dst, stride, d, 4);
                    }
                    return;
                }
                let mut coeffs = [0i16; 16];
                _mm_storeu_si128(coeffs.as_mut_ptr() as *mut __m128i, c0);
                _mm_storeu_si128(coeffs.as_mut_ptr().add(8) as *mut __m128i, c1);
                idct4_add_impl(dst, stride, &coeffs);
            }
        }

        fn residual8(dst: &mut [u8], stride: usize, coefs: &[i32; 64], _max: i32) {
            assert!(7 * stride + 8 <= dst.len());
            unsafe { residual8_impl(dst.as_mut_ptr(), stride, coefs) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn residual8_impl(dst: *mut u8, stride: usize, coefs: &[i32; 64]) {
            unsafe {
                let mut coeffs = [0i16; 64];
                let mut ac = _mm_setzero_si128();
                for k in 0..8 {
                    let c = coefs16(coefs.as_ptr().add(8 * k));
                    _mm_storeu_si128(coeffs.as_mut_ptr().add(8 * k) as *mut __m128i, c);
                    let masked = if k == 0 { _mm_andnot_si128(_mm_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0), c) } else { c };
                    ac = _mm_or_si128(ac, masked);
                }
                if is_zero(ac) {
                    let d = coeffs[0] as i32;
                    if d != 0 {
                        dc_add_impl(dst, stride, d, 8);
                    }
                    return;
                }
                idct8_add_impl(dst, stride, &coeffs);
            }
        }
    };
}

// Each rung is a full compilation of the kernels, but the ladder calls only
// the install groups that rung improves, so the kernels it did not change
// are unreachable and are dropped. Nothing here is public: were these
// modules part of the crate's API, every rung would be retained in full.
// `dead_code` is allowed because "unused" is the intended state for most
// of three of the four rungs, not because anything is unreachable by
// mistake.

/// SSE2: baseline on x86-64, so this rung is the one that makes the scalar
/// kernels unreachable on this architecture.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSSE3: `pmaddubsw` for the six-tap and bilinear filters, `pabsw` for the
/// loop filters' `|p - q|`.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// SSE4.1: `pblendvb` for the loop filters' lane selects, `pmovzxbw` for the
/// widening loads, `ptest` for the all-zero residual test.
pub(crate) mod sse41 {
    #![allow(dead_code)]
    kernels!("sse4.1", sse41);
}

/// AVX: the SSE4.1 algorithms, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// Install the best kernels `cpu` can run, one rung at a time.
///
/// Each rung replaces only the kernels whose code it changes, so the table
/// ends up with the best available version of every kernel: an SSE4.1 CPU,
/// for instance, keeps the SSSE3 interpolation (SSE4.1 adds nothing the
/// six-tap filter can use) and takes the SSE4.1 loop filters and transforms.
pub fn install(d: &mut H264Dsp<u8>, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_interp(d);
        ssse3::install_deblock(d);
    }
    if cpu.sse41 {
        sse41::install_deblock(d);
        sse41::install_rest(d);
    }
    if cpu.avx {
        avx::install_all(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::h264::PRED_STRIDE;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Every rung the host can run, as it would be installed in the field:
    /// cumulatively, so each table is exactly what a CPU of that generation
    /// would get.
    fn tables() -> Vec<(&'static str, H264Dsp<u8>)> {
        let mut v = Vec::new();
        let base = Cpu::SCALAR;
        for (name, cpu) in [
            ("sse2", Cpu { sse2: true, ..base }),
            ("ssse3", Cpu { sse2: true, ssse3: true, ..base }),
            ("sse4.1", Cpu { sse2: true, ssse3: true, sse41: true, ..base }),
            ("avx", Cpu { sse2: true, ssse3: true, sse41: true, avx: true, ..base }),
        ] {
            // Skip rungs this host cannot execute.
            let have = match name {
                "sse2" => std::is_x86_feature_detected!("sse2"),
                "ssse3" => std::is_x86_feature_detected!("ssse3"),
                "sse4.1" => std::is_x86_feature_detected!("sse4.1"),
                _ => std::is_x86_feature_detected!("avx"),
            };
            if !have {
                continue;
            }
            let mut d = H264Dsp::<u8>::SCALAR;
            install(&mut d, cpu);
            v.push((name, d));
        }
        v
    }

    #[test]
    fn qpel_matches_scalar() {
        let s = H264Dsp::<u8>::SCALAR;
        for (name, d) in tables() {
            let mut seed = 5u64;
            let stride = 64;
            let src: Vec<u8> = (0..stride * 64).map(|_| lcg(&mut seed) as u8).collect();
            for &(w, h) in &[(4usize, 4usize), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)] {
                // Only the w x h block of the stride-16 scratch is compared:
                // the SIMD kernels may write the rest of each row.
                let block = |v: &[u8]| -> Vec<u8> { (0..h).flat_map(|y| v[y * PRED_STRIDE..y * PRED_STRIDE + w].to_vec()).collect() };
                for pos in 0..16 {
                    let mut a = vec![0u8; 16 * PRED_STRIDE];
                    let mut b = vec![0u8; 16 * PRED_STRIDE];
                    (s.qpel[pos])(&mut a, &src[stride * 3 + 3..], stride, w, h, 255);
                    (d.qpel[pos])(&mut b, &src[stride * 3 + 3..], stride, w, h, 255);
                    assert_eq!(block(&a), block(&b), "{name} qpel pos={pos} {w}x{h}");
                }
                for xf in 0..8 {
                    for yf in 0..8 {
                        let (cw, ch) = (w / 2, h / 2);
                        let mut a = vec![0u8; 16 * PRED_STRIDE];
                        let mut b = vec![0u8; 16 * PRED_STRIDE];
                        (s.chroma)(&mut a, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                        (d.chroma)(&mut b, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                        let cb = |v: &[u8]| -> Vec<u8> { (0..ch).flat_map(|y| v[y * PRED_STRIDE..y * PRED_STRIDE + cw].to_vec()).collect() };
                        assert_eq!(cb(&a), cb(&b), "{name} chroma {xf},{yf} {cw}x{ch}");
                    }
                }
                let a: Vec<u8> = (0..16 * PRED_STRIDE).map(|_| lcg(&mut seed) as u8).collect();
                let b: Vec<u8> = (0..16 * PRED_STRIDE).map(|_| lcg(&mut seed) as u8).collect();
                let ds = w + 3;
                let mut d1 = vec![0u8; ds * h];
                let mut d2 = vec![0u8; ds * h];
                (s.avg)(&mut d1, ds, &a, &b, w, h);
                (d.avg)(&mut d2, ds, &a, &b, w, h);
                assert_eq!(d1, d2, "{name} avg {w}x{h}");
                (s.copy)(&mut d1, ds, &a, w, h);
                (d.copy)(&mut d2, ds, &a, w, h);
                assert_eq!(d1, d2, "{name} copy {w}x{h}");
                for &(lwd, wt, o) in &[(6, 64, 0), (0, 1, 3), (5, -20, -7), (7, 127, 127), (2, 33, -128)] {
                    (s.weighted_uni)(&mut d1, ds, &a, w, h, lwd, wt, o, 255);
                    (d.weighted_uni)(&mut d2, ds, &a, w, h, lwd, wt, o, 255);
                    assert_eq!(d1, d2, "{name} wuni {w}x{h} {lwd} {wt} {o}");
                    (s.weighted_bi)(&mut d1, ds, &a, &b, w, h, lwd, wt, 64 - wt, o, -o, 255);
                    (d.weighted_bi)(&mut d2, ds, &a, &b, w, h, lwd, wt, 64 - wt, o, -o, 255);
                    assert_eq!(d1, d2, "{name} wbi {w}x{h} {lwd} {wt} {o}");
                }
            }
        }
    }

    #[test]
    fn deblocking_matches_scalar() {
        let s = H264Dsp::<u8>::SCALAR;
        for (name, d) in tables() {
            let mut seed = 11u64;
            let stride = 48;
            for trial in 0..400 {
                // Smooth-ish content so the alpha/beta tests pass often.
                let base = lcg(&mut seed) % 256;
                let spread = 1 + lcg(&mut seed) % 64;
                let plane: Vec<u8> = (0..stride * 40).map(|_| (base + lcg(&mut seed) % spread).min(255) as u8).collect();
                let alpha = (lcg(&mut seed) % 256) as i32;
                let beta = (lcg(&mut seed) % 20) as i32;
                let mut tc0 = [0i16; 4];
                for t in tc0.iter_mut() {
                    *t = (lcg(&mut seed) % 6) as i16 - 1;
                }
                let off = 8 * stride + 8;
                let mut a = plane.clone();
                let mut b = plane.clone();
                match trial % 10 {
                    8 => {
                        (s.deblock_luma8_v)(&mut a, off, stride, alpha, beta, &tc0, 255);
                        (d.deblock_luma8_v)(&mut b, off, stride, alpha, beta, &tc0, 255);
                    }
                    9 => {
                        (s.deblock_luma8_v_intra)(&mut a, off, stride, alpha, beta, 255);
                        (d.deblock_luma8_v_intra)(&mut b, off, stride, alpha, beta, 255);
                    }
                    0 => {
                        (s.deblock_luma_v)(&mut a, off, stride, alpha, beta, &tc0, 255);
                        (d.deblock_luma_v)(&mut b, off, stride, alpha, beta, &tc0, 255);
                    }
                    1 => {
                        (s.deblock_luma_h)(&mut a, off, stride, alpha, beta, &tc0, 255);
                        (d.deblock_luma_h)(&mut b, off, stride, alpha, beta, &tc0, 255);
                    }
                    2 => {
                        (s.deblock_luma_v_intra)(&mut a, off, stride, alpha, beta, 255);
                        (d.deblock_luma_v_intra)(&mut b, off, stride, alpha, beta, 255);
                    }
                    3 => {
                        (s.deblock_luma_h_intra)(&mut a, off, stride, alpha, beta, 255);
                        (d.deblock_luma_h_intra)(&mut b, off, stride, alpha, beta, 255);
                    }
                    4 => {
                        (s.deblock_chroma_v)(&mut a, off, stride, alpha, beta, &tc0, 255);
                        (d.deblock_chroma_v)(&mut b, off, stride, alpha, beta, &tc0, 255);
                    }
                    5 => {
                        (s.deblock_chroma_h)(&mut a, off, stride, alpha, beta, &tc0, 255);
                        (d.deblock_chroma_h)(&mut b, off, stride, alpha, beta, &tc0, 255);
                    }
                    6 => {
                        (s.deblock_chroma_v_intra)(&mut a, off, stride, alpha, beta, 255);
                        (d.deblock_chroma_v_intra)(&mut b, off, stride, alpha, beta, 255);
                    }
                    _ => {
                        (s.deblock_chroma_h_intra)(&mut a, off, stride, alpha, beta, 255);
                        (d.deblock_chroma_h_intra)(&mut b, off, stride, alpha, beta, 255);
                    }
                }
                assert_eq!(a, b, "{name} deblock kind {} trial {trial} alpha {alpha} beta {beta} tc0 {tc0:?}", trial % 8);
            }
        }
    }

    #[test]
    fn transforms_match_scalar() {
        let s = H264Dsp::<u8>::SCALAR;
        for (name, d) in tables() {
            let mut seed = 17u64;
            let stride = 24;
            for trial in 0..500 {
                let base: Vec<u8> = (0..stride * 8).map(|_| lcg(&mut seed) as u8).collect();
                // Coefficients small enough that the transform stays in range.
                let mut c16 = [0i16; 64];
                let mut c32 = [0i32; 64];
                let nz = 1 + lcg(&mut seed) % 64;
                for k in 0..64 {
                    let v = if k < nz as usize { (lcg(&mut seed) % 512) as i32 - 256 } else { 0 };
                    c16[k] = v as i16;
                    c32[k] = v;
                }
                let mut a = base.clone();
                let mut b = base.clone();
                match trial % 6 {
                    0 => {
                        let c: [i16; 16] = c16[0..16].try_into().unwrap();
                        (s.idct4_add)(&mut a, stride, &c, 255);
                        (d.idct4_add)(&mut b, stride, &c, 255);
                    }
                    1 => {
                        (s.idct8_add)(&mut a, stride, &c16, 255);
                        (d.idct8_add)(&mut b, stride, &c16, 255);
                    }
                    2 => {
                        let dc = c32[0];
                        (s.idct4_dc_add)(&mut a, stride, dc, 255);
                        (d.idct4_dc_add)(&mut b, stride, dc, 255);
                    }
                    3 => {
                        let dc = c32[0];
                        (s.idct8_dc_add)(&mut a, stride, dc, 255);
                        (d.idct8_dc_add)(&mut b, stride, dc, 255);
                    }
                    4 => {
                        let c: [i32; 16] = c32[0..16].try_into().unwrap();
                        let dc = if trial % 12 == 4 { crate::dsp::h264::NO_DC } else { c32[17] };
                        (s.residual4)(&mut a, stride, &c, dc, 255);
                        (d.residual4)(&mut b, stride, &c, dc, 255);
                    }
                    _ => {
                        (s.residual8)(&mut a, stride, &c32, 255);
                        (d.residual8)(&mut b, stride, &c32, 255);
                    }
                }
                assert_eq!(a, b, "{name} transform kind {} trial {trial}", trial % 6);
            }
        }
    }
}
