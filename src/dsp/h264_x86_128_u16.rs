//! 128-bit SIMD versions of the H.264 kernels for 16-bit sample planes
//! (x86-64), from SSE2 up — what High 10, High 4:2:2 and High 4:4:4 decode
//! run, and the deep encoder's reconstruction.
//!
//! Eight u16 lanes per vector, the lane layout of [`super::h264_x86_128`],
//! and the same ladder: written once and compiled once per rung by
//! `kernels!`, each rung installing only the groups whose instructions it
//! improves.
//!
//! What does not carry over from eight bits is the arithmetic. H.264 allows
//! samples of up to 14 bits, and most of the 8-bit kernels' 16-bit
//! intermediates stop fitting well before that. Kernel by kernel, what each
//! does instead, and how far it is exact:
//!
//! - **Luma six-tap.** `a − 5b + 20c + 20d − 5e + f` reaches `42 · max`,
//!   which leaves i16 at 10 bits already. Two paths, chosen per call by the
//!   `max` the caller passes. Up to 10 bits the tap is computed in wrapping
//!   i16 *offset by −16384*: its true value spans −10230..=42966 there, so
//!   the offset one lands inside i16 exactly, and because 16384 is a multiple
//!   of 32 the rounded half-sample `(v + 16) >> 5` is the offset one's plus
//!   512. The centre position filters those offset intermediates with
//!   `pmaddwd` and puts `32 · 16384` back before its `>> 10`. Above 10 bits
//!   the tap is `pmaddwd` of `(c + d, b + e)` against `(20, −5)` plus the
//!   widened `a + f`, in i32 lanes — `c + d` is still an i16 at 14 bits — and
//!   the centre is the six-tap over those i32 intermediates by shifts and
//!   adds, which peak near 2^25.
//! - **Chroma bilinear** is given no `max`, so it decides per row from the
//!   samples it loaded. All of at most ten bits: the weighted sum is at most
//!   `64 · 1023` and four `pmullw` make it in u16. Below 2^15: `pmaddwd` of
//!   the interleaved neighbours makes it in i32. A row with a larger sample
//!   (16-bit content, which no H.264 stream has) is left to the scalar
//!   reference.
//! - **Deblocking.** The bS < 4 delta `(((q0 − p0) << 2) + (p1 − q1) + 4) >> 3`
//!   overflows i16 above 12 bits in that form. It is computed as
//!   `((q0 − p0) + ((p1 − q1 + 4) >> 2)) >> 1`, the same integer — both are
//!   the floor of one quotient, taken in one step or two — which stays
//!   inside ±20479 at 14 bits. The bS 4 filter's eight-sample sums reach
//!   `8 · max`; they are kept in the u16 domain (wrapping adds, logical
//!   shifts) with the division split the same way so that no partial sum
//!   passes 65534: `(p2 + 2p1 + 2p0 + 2q0 + q1 + 4) >> 3` as
//!   `(((p2 + q1 + 4) >> 1) + p1 + p0 + q0) >> 2`, and so on.
//! - **Inverse transforms** run in i32 lanes: the standard bounds their
//!   intermediates by `2^(7 + BitDepth)`, which is i16 only at 8 bits. The
//!   rounded result is narrowed with `packs` and added with `paddsw`, and the
//!   clip to `0..=max` that follows absorbs both saturations exactly.
//! - **Weighting** multiplies in i32 (`pmullw` + `pmulhw` for one reference,
//!   `pmaddwd` for the two-reference sum), narrows with `packs` and clips.
//! - `avg` is `pavgw` and `copy` a copy, exact for any u16.
//!
//! A kernel that is given `max` hands anything deeper than 14 bits, which
//! H.264 does not have, to the scalar reference rather than guess; so does
//! one whose other arguments leave the range a stream can produce. The tests
//! check every rung against the scalar reference at every depth from 9 to 14,
//! over uniform samples and over inputs built to reach each kernel's widest
//! intermediate (`super::u16_sweep`).

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::h264::H264Dsp;

// The range every 16-bit tier is exact over, and refuses calls outside of.
use super::h264::simd16::{DEEPEST, normal_in_range, strong_in_range, weights_in_range};

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use super::{DEEPEST, normal_in_range, strong_in_range, weights_in_range};
        use crate::dsp::h264::{H264Dsp, NO_DC, PRED_STRIDE};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        // ------------------------------------------------------------------
        // Install groups
        // ------------------------------------------------------------------
        //
        // As in the 8-bit file, a rung calls only the groups whose
        // primitives it improves: `interp` uses the zero extension and the
        // all-zero test, `deblock` the absolute value and the lane select,
        // `rest` the sign extension and the all-zero test. Weighting, `copy`
        // and `avg` are plain SSE2 and change only with VEX.

        /// Every kernel — the bottom rung, and the top one.
        pub(crate) fn install_all(d: &mut H264Dsp<u16>) {
            install_interp(d);
            install_deblock(d);
            install_rest(d);
            d.copy = copy;
            d.avg = avg;
            d.weighted_uni = weighted_uni;
            d.weighted_bi = weighted_bi;
        }

        /// Luma quarter-sample interpolation and chroma bilinear.
        pub(crate) fn install_interp(d: &mut H264Dsp<u16>) {
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
        pub(crate) fn install_deblock(d: &mut H264Dsp<u16>) {
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

        /// Inverse transforms, the DC-only adds and the residual paths.
        pub(crate) fn install_rest(d: &mut H264Dsp<u16>) {
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

        /// Eight samples at `p` as eight u16 lanes.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load8(p: *const u16) -> __m128i {
            unsafe { _mm_loadu_si128(p as *const __m128i) }
        }

        /// Four samples at `p` in the low lanes, the high four zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load4(p: *const u16) -> __m128i {
            unsafe { _mm_loadl_epi64(p as *const __m128i) }
        }

        /// Store eight lanes as eight samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store8(p: *mut u16, v: __m128i) {
            unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
        }

        /// Store the first `n` (≤ 8) lanes of `v` as samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_n(dst: *mut u16, v: __m128i, n: usize) {
            unsafe {
                match n {
                    8 => _mm_storeu_si128(dst as *mut __m128i, v),
                    4 => _mm_storel_epi64(dst as *mut __m128i, v),
                    _ => {
                        let mut t = [0u16; 8];
                        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                        std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
                    }
                }
            }
        }

        /// Signed i16 lanes clipped to `0..=max` (`max` ≤ 32767).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn clip(v: __m128i, maxv: __m128i) -> __m128i {
            unsafe { _mm_min_epi16(_mm_max_epi16(v, _mm_setzero_si128()), maxv) }
        }

        /// A tap pair as one i32 lane, for `pmaddwd`.
        #[inline(always)]
        fn pair(a: i16, b: i16) -> i32 {
            (a as u16 as i32) | ((b as u16 as i32) << 16)
        }

        // ------------------------------------------------------------------
        // Luma interpolation
        // ------------------------------------------------------------------

        /// Six-tap over the eight samples at `p`, `p + step`, …
        /// `p + 5 · step`, minus 16384, in wrapping i16. Exact for samples of
        /// at most ten bits, whose six-tap lies in −10230..=42966: the
        /// wrapped partial sums do not matter, only that the final value is
        /// representable.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tap6_narrow(p: *const u16, step: usize) -> __m128i {
            unsafe {
                let ld = |k: usize| load8(p.add(k * step));
                let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
                let t = _mm_add_epi16(c, d);
                let u = _mm_add_epi16(b, e);
                let v = _mm_add_epi16(_mm_add_epi16(a, f), _mm_set1_epi16(-16384));
                let t20 = _mm_add_epi16(_mm_slli_epi16(t, 4), _mm_slli_epi16(t, 2));
                let u5 = _mm_add_epi16(_mm_slli_epi16(u, 2), u);
                _mm_sub_epi16(_mm_add_epi16(v, t20), u5)
            }
        }

        /// The half-sample value `clip((tap + 16) >> 5)` of an offset tap:
        /// 16384 is 512 · 32, so the offset comes out of the shift whole.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn half_narrow(v: __m128i, maxv: __m128i) -> __m128i {
            unsafe {
                let r = _mm_srai_epi16(_mm_add_epi16(v, _mm_set1_epi16(16)), 5);
                clip(_mm_add_epi16(r, _mm_set1_epi16(512)), maxv)
            }
        }

        /// The centre value from the offset intermediates of window rows
        /// `y..y+5`: their vertical six-tap is the true one minus
        /// `32 · 16384` (the taps sum to 32), put back with the rounding.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn j_narrow(w: &[__m128i; 6], maxv: __m128i) -> __m128i {
            unsafe {
                let (r0, r1, r2, r3, r4, r5) = (w[0], w[1], w[2], w[3], w[4], w[5]);
                let c01 = _mm_set1_epi32(pair(1, -5));
                let c23 = _mm_set1_epi32(pair(20, 20));
                let c45 = _mm_set1_epi32(pair(-5, 1));
                let round = _mm_set1_epi32(512 + 32 * 16384);
                let lo = _mm_add_epi32(
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(r0, r1), c01), _mm_madd_epi16(_mm_unpacklo_epi16(r2, r3), c23)),
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(r4, r5), c45), round),
                );
                let hi = _mm_add_epi32(
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(r0, r1), c01), _mm_madd_epi16(_mm_unpackhi_epi16(r2, r3), c23)),
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(r4, r5), c45), round),
                );
                clip(_mm_packs_epi32(_mm_srai_epi32(lo, 10), _mm_srai_epi32(hi, 10)), maxv)
            }
        }

        /// A six-tap as two vectors of four i32: lanes 0..4 and 4..8.
        type Wide = [__m128i; 2];

        /// Six-tap over the eight samples at `p`, … `p + 5 · step`, in i32.
        /// Exact for samples of at most 14 bits, where the pair sums `c + d`
        /// and `b + e` are still positive i16 for `pmaddwd`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tap6_wide(p: *const u16, step: usize) -> Wide {
            unsafe {
                let ld = |k: usize| load8(p.add(k * step));
                let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
                let t = _mm_add_epi16(c, d);
                let u = _mm_add_epi16(b, e);
                let v = _mm_add_epi16(a, f);
                let k = _mm_set1_epi32(pair(20, -5));
                [
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(t, u), k), zx16(v)),
                    _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(t, u), k), zx16h(v)),
                ]
            }
        }

        /// The half-sample value of a wide tap.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn half_wide(v: Wide, maxv: __m128i) -> __m128i {
            unsafe {
                let r = _mm_set1_epi32(16);
                clip(_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(v[0], r), 5), _mm_srai_epi32(_mm_add_epi32(v[1], r), 5)), maxv)
            }
        }

        /// Six-tap over six i32 vectors, by shifts and adds. The wide
        /// intermediates lie in −163830..=688086 at 14 bits, so the largest
        /// partial sum is under 2^25.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tap6_i32(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i, r4: __m128i, r5: __m128i) -> __m128i {
            unsafe {
                let t = _mm_add_epi32(r2, r3);
                let u = _mm_add_epi32(r1, r4);
                let v = _mm_add_epi32(r0, r5);
                let t20 = _mm_add_epi32(_mm_slli_epi32(t, 4), _mm_slli_epi32(t, 2));
                let u5 = _mm_add_epi32(_mm_slli_epi32(u, 2), u);
                _mm_sub_epi32(_mm_add_epi32(v, t20), u5)
            }
        }

        /// The centre value from the wide intermediates of window rows
        /// `y..y+5`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn j_wide(w: &[Wide; 6], maxv: __m128i) -> __m128i {
            unsafe {
                let round = _mm_set1_epi32(512);
                let half = |k: usize| _mm_srai_epi32(_mm_add_epi32(tap6_i32(w[0][k], w[1][k], w[2][k], w[3][k], w[4][k], w[5][k]), round), 10);
                clip(_mm_packs_epi32(half(0), half(1)), maxv)
            }
        }

        fn qpel<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
            // The window is (w + 5) x (h + 5); an eight-lane load from column
            // x ≤ 8 reads x + 7 (+5 for the taps): the 8-bit kernels' bound,
            // in samples.
            let need = (h + 5 - 1) * stride + 21;
            if src.len() < need || dst.len() < h * PRED_STRIDE || w > 16 || !(1..=DEEPEST).contains(&max) {
                return (H264Dsp::<u16>::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h, max);
            }
            unsafe {
                if max <= 1023 {
                    qpel_narrow::<XF, YF>(dst, src, stride, w, h, max)
                } else {
                    qpel_wide::<XF, YF>(dst, src, stride, w, h, max)
                }
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn qpel_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
            // The five positions whose vertical filter runs over horizontal
            // intermediates slide a window of them instead (see the 8-bit
            // file), which wants the loops the other way round.
            if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
                return unsafe { qpel_centre_narrow::<XF, YF>(dst, src, stride, w, h, max) };
            }
            unsafe {
                let s = src.as_ptr();
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    for c in 0..w.div_ceil(8) {
                        let x = c * 8;
                        let g = |dx: usize, dy: usize| load8(s.add((y + 2 + dy) * stride + 2 + dx + x));
                        let b = || half_narrow(tap6_narrow(s.add((y + 2) * stride + x), 1), maxv);
                        let b_below = || half_narrow(tap6_narrow(s.add((y + 3) * stride + x), 1), maxv);
                        let hh = || half_narrow(tap6_narrow(s.add(y * stride + 2 + x), stride), maxv);
                        let hh_right = || half_narrow(tap6_narrow(s.add(y * stride + 3 + x), stride), maxv);
                        let v: __m128i = match (XF, YF) {
                            (0, 0) => g(0, 0),
                            (1, 0) => _mm_avg_epu16(g(0, 0), b()),
                            (2, 0) => b(),
                            (3, 0) => _mm_avg_epu16(g(1, 0), b()),
                            (0, 1) => _mm_avg_epu16(g(0, 0), hh()),
                            (0, 2) => hh(),
                            (0, 3) => _mm_avg_epu16(g(0, 1), hh()),
                            (1, 1) => _mm_avg_epu16(b(), hh()),
                            (3, 1) => _mm_avg_epu16(b(), hh_right()),
                            (1, 3) => _mm_avg_epu16(hh(), b_below()),
                            (3, 3) => _mm_avg_epu16(hh_right(), b_below()),
                            _ => unreachable!(),
                        };
                        store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                    }
                }
            }
        }

        /// The centre positions up to ten bits, over a sliding window of
        /// offset horizontal intermediates: `win[k]` is window row `y + k`.
        #[target_feature(enable = $feat)]
        unsafe fn qpel_centre_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
            unsafe {
                let s = src.as_ptr();
                let maxv = _mm_set1_epi16(max as i16);
                let row = |r: usize, x: usize| tap6_narrow(s.add(r * stride + x), 1);
                for c in 0..w.div_ceil(8) {
                    let x = c * 8;
                    let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
                    for y in 0..h {
                        let j = j_narrow(&win, maxv);
                        let hh = |col: usize| half_narrow(tap6_narrow(s.add(y * stride + col + x), stride), maxv);
                        let v: __m128i = match (XF, YF) {
                            (2, 2) => j,
                            (2, 1) => _mm_avg_epu16(half_narrow(win[2], maxv), j),
                            (2, 3) => _mm_avg_epu16(j, half_narrow(win[3], maxv)),
                            (1, 2) => _mm_avg_epu16(hh(2), j),
                            (3, 2) => _mm_avg_epu16(j, hh(3)),
                            _ => unreachable!(),
                        };
                        store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                        // Not on the last row: the caller's bounds check
                        // covers window rows up to h + 4.
                        if y + 1 < h {
                            win = [win[1], win[2], win[3], win[4], win[5], row(y + 6, x)];
                        }
                    }
                }
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn qpel_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
            if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
                return unsafe { qpel_centre_wide::<XF, YF>(dst, src, stride, w, h, max) };
            }
            unsafe {
                let s = src.as_ptr();
                let maxv = _mm_set1_epi16(max as i16);
                for y in 0..h {
                    for c in 0..w.div_ceil(8) {
                        let x = c * 8;
                        let g = |dx: usize, dy: usize| load8(s.add((y + 2 + dy) * stride + 2 + dx + x));
                        let b = || half_wide(tap6_wide(s.add((y + 2) * stride + x), 1), maxv);
                        let b_below = || half_wide(tap6_wide(s.add((y + 3) * stride + x), 1), maxv);
                        let hh = || half_wide(tap6_wide(s.add(y * stride + 2 + x), stride), maxv);
                        let hh_right = || half_wide(tap6_wide(s.add(y * stride + 3 + x), stride), maxv);
                        let v: __m128i = match (XF, YF) {
                            (0, 0) => g(0, 0),
                            (1, 0) => _mm_avg_epu16(g(0, 0), b()),
                            (2, 0) => b(),
                            (3, 0) => _mm_avg_epu16(g(1, 0), b()),
                            (0, 1) => _mm_avg_epu16(g(0, 0), hh()),
                            (0, 2) => hh(),
                            (0, 3) => _mm_avg_epu16(g(0, 1), hh()),
                            (1, 1) => _mm_avg_epu16(b(), hh()),
                            (3, 1) => _mm_avg_epu16(b(), hh_right()),
                            (1, 3) => _mm_avg_epu16(hh(), b_below()),
                            (3, 3) => _mm_avg_epu16(hh_right(), b_below()),
                            _ => unreachable!(),
                        };
                        store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                    }
                }
            }
        }

        /// The centre positions above ten bits, over a sliding window of wide
        /// horizontal intermediates.
        #[target_feature(enable = $feat)]
        unsafe fn qpel_centre_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
            unsafe {
                let s = src.as_ptr();
                let maxv = _mm_set1_epi16(max as i16);
                let row = |r: usize, x: usize| tap6_wide(s.add(r * stride + x), 1);
                for c in 0..w.div_ceil(8) {
                    let x = c * 8;
                    let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
                    for y in 0..h {
                        let j = j_wide(&win, maxv);
                        let hh = |col: usize| half_wide(tap6_wide(s.add(y * stride + col + x), stride), maxv);
                        let v: __m128i = match (XF, YF) {
                            (2, 2) => j,
                            (2, 1) => _mm_avg_epu16(half_wide(win[2], maxv), j),
                            (2, 3) => _mm_avg_epu16(j, half_wide(win[3], maxv)),
                            (1, 2) => _mm_avg_epu16(hh(2), j),
                            (3, 2) => _mm_avg_epu16(j, hh(3)),
                            _ => unreachable!(),
                        };
                        store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                        if y + 1 < h {
                            win = [win[1], win[2], win[3], win[4], win[5], row(y + 6, x)];
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Chroma interpolation, combination and weighting
        // ------------------------------------------------------------------

        fn chroma(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
            if src.len() < h * stride + 9 || dst.len() < h * PRED_STRIDE || w > 8 || !(0..8).contains(&xf) || !(0..8).contains(&yf) {
                return (H264Dsp::<u16>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
            }
            unsafe { chroma_impl(dst, src, stride, w, h, xf, yf) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn chroma_impl(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
            unsafe {
                let (wa, wb, wc, wd) = (((8 - xf) * (8 - yf)) as i16, (xf * (8 - yf)) as i16, ((8 - xf) * yf) as i16, (xf * yf) as i16);
                let mul = [_mm_set1_epi16(wa), _mm_set1_epi16(wb), _mm_set1_epi16(wc), _mm_set1_epi16(wd)];
                let k0 = _mm_set1_epi32(pair(wa, wb));
                let k1 = _mm_set1_epi32(pair(wc, wd));
                let above_ten = _mm_set1_epi16(!1023);
                let sign = _mm_set1_epi16(i16::MIN);
                let s = src.as_ptr();
                for y in 0..h {
                    let r0 = s.add(y * stride);
                    let r1 = s.add((y + 1) * stride);
                    let (a, b, c, d) = (load8(r0), load8(r0.add(1)), load8(r1), load8(r1.add(1)));
                    let seen = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
                    let v = if is_zero(_mm_and_si128(seen, above_ten)) {
                        // Every product and their sum ≤ 64 · 1023: u16.
                        let sum = _mm_add_epi16(
                            _mm_add_epi16(_mm_mullo_epi16(a, mul[0]), _mm_mullo_epi16(b, mul[1])),
                            _mm_add_epi16(_mm_mullo_epi16(c, mul[2]), _mm_mullo_epi16(d, mul[3])),
                        );
                        _mm_srli_epi16(_mm_add_epi16(sum, _mm_set1_epi16(32)), 6)
                    } else if is_zero(_mm_and_si128(seen, sign)) {
                        let r = _mm_set1_epi32(32);
                        let lo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(a, b), k0), _mm_madd_epi16(_mm_unpacklo_epi16(c, d), k1));
                        let hi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(a, b), k0), _mm_madd_epi16(_mm_unpackhi_epi16(c, d), k1));
                        _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(lo, r), 6), _mm_srai_epi32(_mm_add_epi32(hi, r), 6))
                    } else {
                        (H264Dsp::<u16>::SCALAR.chroma)(&mut dst[y * PRED_STRIDE..], &src[y * stride..], stride, w, 1, xf, yf);
                        continue;
                    };
                    store8(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
                }
            }
        }

        fn copy(dst: &mut [u16], stride: usize, src: &[u16], w: usize, h: usize) {
            assert!((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len());
            unsafe { copy_impl(dst.as_mut_ptr(), stride, src.as_ptr(), w, h) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn copy_impl(dst: *mut u16, stride: usize, src: *const u16, w: usize, h: usize) {
            unsafe {
                for y in 0..h {
                    let s = src.add(y * PRED_STRIDE);
                    let d = dst.add(y * stride);
                    match w {
                        16 => {
                            store8(d, load8(s));
                            store8(d.add(8), load8(s.add(8)));
                        }
                        8 => store8(d, load8(s)),
                        4 => std::ptr::write_unaligned(d as *mut u64, std::ptr::read_unaligned(s as *const u64)),
                        _ => std::ptr::copy_nonoverlapping(s, d, w),
                    }
                }
            }
        }

        fn avg(dst: &mut [u16], stride: usize, a: &[u16], b: &[u16], w: usize, h: usize) {
            assert!(h == 0 || ((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= a.len().min(b.len()) && w <= 16));
            unsafe { avg_impl(dst.as_mut_ptr(), stride, a.as_ptr(), b.as_ptr(), w, h) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn avg_impl(dst: *mut u16, stride: usize, a: *const u16, b: *const u16, w: usize, h: usize) {
            unsafe {
                for y in 0..h {
                    let (pa, pb, d) = (a.add(y * PRED_STRIDE), b.add(y * PRED_STRIDE), dst.add(y * stride));
                    store_n(d, _mm_avg_epu16(load8(pa), load8(pb)), w.min(8));
                    if w > 8 {
                        store_n(d.add(8), _mm_avg_epu16(load8(pa.add(8)), load8(pb.add(8))), w - 8);
                    }
                }
            }
        }

        /// Whether a combiner's buffers hold what it reads and writes.
        #[inline(always)]
        fn combine_fits(dst: &[u16], stride: usize, src: &[u16], w: usize, h: usize) -> bool {
            w <= 16 && (h == 0 || ((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len()))
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_uni(dst: &mut [u16], stride: usize, src: &[u16], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32) {
            if !weights_in_range(log_wd, [wt, 0], [o, 0], max) || !combine_fits(dst, stride, src, w, h) {
                return (H264Dsp::<u16>::SCALAR.weighted_uni)(dst, stride, src, w, h, log_wd, wt, o, max);
            }
            unsafe { weighted_uni_impl(dst.as_mut_ptr(), stride, src.as_ptr(), w, h, log_wd, wt, o, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_uni_impl(dst: *mut u16, stride: usize, src: *const u16, w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32) {
            unsafe {
                let wv = _mm_set1_epi16(wt as i16);
                let round = _mm_set1_epi32(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
                let sh = _mm_cvtsi32_si128(log_wd);
                let ov = _mm_set1_epi32(o);
                let maxv = _mm_set1_epi16(max as i16);
                let scale = |s: __m128i| {
                    // `s · wt` in i32: the low and high halves of the signed
                    // product, interleaved.
                    let lo = _mm_mullo_epi16(s, wv);
                    let hi = _mm_mulhi_epi16(s, wv);
                    let q = |p: __m128i| _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(p, round), sh), ov);
                    clip(_mm_packs_epi32(q(_mm_unpacklo_epi16(lo, hi)), q(_mm_unpackhi_epi16(lo, hi))), maxv)
                };
                for y in 0..h {
                    let p = src.add(y * PRED_STRIDE);
                    let d = dst.add(y * stride);
                    store_n(d, scale(load8(p)), w.min(8));
                    if w > 8 {
                        store_n(d.add(8), scale(load8(p.add(8))), w - 8);
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn weighted_bi(dst: &mut [u16], stride: usize, a: &[u16], b: &[u16], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
            if !weights_in_range(log_wd, [w0, w1], [o0, o1], max) || !combine_fits(dst, stride, a, w, h) || !combine_fits(dst, stride, b, w, h) {
                return (H264Dsp::<u16>::SCALAR.weighted_bi)(dst, stride, a, b, w, h, log_wd, w0, w1, o0, o1, max);
            }
            unsafe { weighted_bi_impl(dst.as_mut_ptr(), stride, a.as_ptr(), b.as_ptr(), w, h, log_wd, w0, w1, o0, o1, max) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn weighted_bi_impl(dst: *mut u16, stride: usize, a: *const u16, b: *const u16, w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
            unsafe {
                // `a · w0 + b · w1` by `pmaddwd` over the two predictions
                // interleaved: exact for positive i16 samples and i16 weights,
                // whose largest sum is under 2^31.
                let wv = _mm_set1_epi32(pair(w0 as i16, w1 as i16));
                let round = _mm_set1_epi32(1 << log_wd);
                let off = _mm_set1_epi32((o0 + o1 + 1) >> 1);
                let sh = _mm_cvtsi32_si128(log_wd + 1);
                let maxv = _mm_set1_epi16(max as i16);
                let eight = |va: __m128i, vb: __m128i| {
                    let q = |v: __m128i| _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(v, wv), round), sh), off);
                    clip(_mm_packs_epi32(q(_mm_unpacklo_epi16(va, vb)), q(_mm_unpackhi_epi16(va, vb))), maxv)
                };
                for y in 0..h {
                    let (pa, pb, d) = (a.add(y * PRED_STRIDE), b.add(y * PRED_STRIDE), dst.add(y * stride));
                    store_n(d, eight(load8(pa), load8(pb)), w.min(8));
                    if w > 8 {
                        store_n(d.add(8), eight(load8(pa.add(8)), load8(pb.add(8))), w - 8);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Deblocking
        // ------------------------------------------------------------------
        //
        // The 8-bit file's shapes: eight lines of an edge are eight lanes of
        // one vector per sample position, a sixteen-line luma edge two such
        // halves. A vertical edge transposes eight rows of eight samples,
        // which for 16-bit lanes is the transform's `transpose8` exactly.

        /// `|a − b| < t` per lane, as a mask. Differences of 14-bit samples
        /// are i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn diff_lt(a: __m128i, b: __m128i, t: __m128i) -> __m128i {
            unsafe { _mm_cmpgt_epi16(t, abs16(_mm_sub_epi16(a, b))) }
        }

        /// The eight positions of eight luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
        type LumaLines = [__m128i; 8];

        /// bS < 4 luma filter on eight lines (8.7.2.3), in place on the
        /// vectors. `tc0v` holds each line's tC0 (−1 = bS 0).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: __m128i, maxv: __m128i) {
            unsafe {
                let [_, p2, p1, p0, q0, q1, q2, _] = *v;
                let alpha = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let zero = _mm_setzero_si128();
                let bs_on = _mm_cmpgt_epi16(tc0v, _mm_set1_epi16(-1));
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), _mm_and_si128(diff_lt(q1, q0, beta), bs_on));
                let ap = diff_lt(p2, p0, beta);
                let aq = diff_lt(q2, q0, beta);
                // tc = tc0 + (ap < beta) + (aq < beta); masks are −1.
                let tc = _mm_sub_epi16(_mm_sub_epi16(tc0v, ap), aq);
                // delta = clip3(−tc, tc, (((q0 − p0) << 2) + (p1 − q1) + 4) >> 3),
                // as ((q0 − p0) + ((p1 − q1 + 4) >> 2)) >> 1, which cannot
                // leave i16 (see the module documentation).
                let d = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(q0, p0), _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(p1, q1), _mm_set1_epi16(4)), 2)), 1);
                let d = _mm_min_epi16(_mm_max_epi16(d, _mm_sub_epi16(zero, tc)), tc);
                let np0 = _mm_add_epi16(p0, d);
                let nq0 = _mm_sub_epi16(q0, d);
                // p1' = p1 + clip3(−tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) − 2 p1) >> 1), when ap.
                let avg = _mm_avg_epu16(p0, q0);
                let ntc0 = _mm_sub_epi16(zero, tc0v);
                let dp1 = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(p2, avg), _mm_slli_epi16(p1, 1)), 1);
                let dp1 = _mm_min_epi16(_mm_max_epi16(dp1, ntc0), tc0v);
                let np1 = _mm_add_epi16(p1, _mm_and_si128(dp1, ap));
                let dq1 = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(q2, avg), _mm_slli_epi16(q1, 1)), 1);
                let dq1 = _mm_min_epi16(_mm_max_epi16(dq1, ntc0), tc0v);
                let nq1 = _mm_add_epi16(q1, _mm_and_si128(dq1, aq));
                v[2] = sel(p1, np1, mask);
                v[3] = sel(p0, clip(np0, maxv), mask);
                v[4] = sel(q0, clip(nq0, maxv), mask);
                v[5] = sel(q1, nq1, mask);
            }
        }

        /// bS 4 luma filter on eight lines (8.7.2.4). The sums are taken in
        /// the u16 domain, each at most 65534 for 14-bit samples.
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
                let one = _mm_set1_epi16(1);
                let two = _mm_set1_epi16(2);
                let four = _mm_set1_epi16(4);
                let add = |a, b| _mm_add_epi16(a, b);
                let dbl = |a| _mm_slli_epi16(a, 1);
                // Weak: p0' = (2 p1 + p0 + q1 + 2) >> 2, q0' = (2 q1 + q0 + p1 + 2) >> 2.
                let wp0 = _mm_srli_epi16(add(add(dbl(p1), p0), add(q1, two)), 2);
                let wq0 = _mm_srli_epi16(add(add(dbl(q1), q0), add(p1, two)), 2);
                let p0q0 = add(p0, q0);
                // p0' = (p2 + 2 p1 + 2 p0 + 2 q0 + q1 + 4) >> 3
                //     = (((p2 + q1 + 4) >> 1) + p1 + p0 + q0) >> 2.
                let sp0 = _mm_srli_epi16(add(_mm_srli_epi16(add(add(p2, q1), four), 1), add(p1, p0q0)), 2);
                // p1' = (p2 + p1 + p0 + q0 + 2) >> 2, and
                // p2' = (2 p3 + 3 p2 + p1 + p0 + q0 + 4) >> 3
                //     = ((((p2 + p1 + p0 + q0 + 2) >> 1) + 1) + p3 + p2) >> 2.
                let tp = add(add(p2, p1), add(p0q0, two));
                let sp1 = _mm_srli_epi16(tp, 2);
                let sp2 = _mm_srli_epi16(add(add(_mm_srli_epi16(tp, 1), one), add(p3, p2)), 2);
                // The q side, mirrored.
                let sq0 = _mm_srli_epi16(add(_mm_srli_epi16(add(add(q2, p1), four), 1), add(q1, p0q0)), 2);
                let tq = add(add(q2, q1), add(p0q0, two));
                let sq1 = _mm_srli_epi16(tq, 2);
                let sq2 = _mm_srli_epi16(add(add(_mm_srli_epi16(tq, 1), one), add(q3, q2)), 2);
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

        /// tC0 per lane for eight lines two to a segment: an MBAFF mixed
        /// luma edge, and chroma.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tc0_pairs(tc0: &[i16; 4]) -> __m128i {
            unsafe {
                let t = |k: usize| tc0[k];
                _mm_setr_epi16(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
            }
        }

        /// Transpose eight 8-lane rows.
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

        /// The eight rows × eight samples around a vertical edge (`q0` at
        /// `data`) as eight column vectors p3..q3.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_transposed_8x8(data: *const u16, stride: usize) -> LumaLines {
            unsafe {
                let mut r = [_mm_setzero_si128(); 8];
                for (i, v) in r.iter_mut().enumerate() {
                    *v = load8(data.add(i * stride).sub(4));
                }
                transpose8(&mut r);
                r
            }
        }

        /// Eight column vectors back as eight rows × eight samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_transposed_8x8(data: *mut u16, stride: usize, v: &LumaLines) {
            unsafe {
                let mut r = *v;
                transpose8(&mut r);
                for (i, v) in r.iter().enumerate() {
                    store8(data.add(i * stride).sub(4), *v);
                }
            }
        }

        fn deblock_luma_v(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            if !normal_in_range(alpha, beta, tc0, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma_v)(data, off, stride, alpha, beta, tc0, max);
            }
            assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_v_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            unsafe {
                let maxv = _mm_set1_epi16(max as i16);
                for half in 0..2 {
                    let d = data.add(half * 8 * stride);
                    let mut v = load_transposed_8x8(d, stride);
                    luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half), maxv);
                    store_transposed_8x8(d, stride, &v);
                }
            }
        }

        fn deblock_luma8_v(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            if !normal_in_range(alpha, beta, tc0, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma8_v)(data, off, stride, alpha, beta, tc0, max);
            }
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma8_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma8_v_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            unsafe {
                let mut v = load_transposed_8x8(data, stride);
                luma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), _mm_set1_epi16(max as i16));
                store_transposed_8x8(data, stride, &v);
            }
        }

        fn deblock_luma8_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
            if !strong_in_range(alpha, beta, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma8_v_intra)(data, off, stride, alpha, beta, max);
            }
            assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
            unsafe { deblock_luma8_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma8_v_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                let mut v = load_transposed_8x8(data, stride);
                luma_filter_intra(&mut v, alpha, beta);
                store_transposed_8x8(data, stride, &v);
            }
        }

        fn deblock_luma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
            if !strong_in_range(alpha, beta, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma_v_intra)(data, off, stride, alpha, beta, max);
            }
            assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
            unsafe { deblock_luma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_v_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                for half in 0..2 {
                    let d = data.add(half * 8 * stride);
                    let mut v = load_transposed_8x8(d, stride);
                    luma_filter_intra(&mut v, alpha, beta);
                    store_transposed_8x8(d, stride, &v);
                }
            }
        }

        fn deblock_luma_h(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            if !normal_in_range(alpha, beta, tc0, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma_h)(data, off, stride, alpha, beta, tc0, max);
            }
            assert!(off >= 3 * stride && off + 2 * stride + 16 <= data.len());
            unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_h_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            unsafe {
                let zero = _mm_setzero_si128();
                let maxv = _mm_set1_epi16(max as i16);
                for half in 0..2 {
                    let d = data.add(half * 8);
                    let ld = |k: isize| load8(d.offset(k * stride as isize));
                    let mut v: LumaLines = [zero, ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), zero];
                    luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half), maxv);
                    store8(d.offset(-2 * stride as isize), v[2]);
                    store8(d.offset(-(stride as isize)), v[3]);
                    store8(d, v[4]);
                    store8(d.add(stride), v[5]);
                }
            }
        }

        fn deblock_luma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
            if !strong_in_range(alpha, beta, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_luma_h_intra)(data, off, stride, alpha, beta, max);
            }
            assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
            unsafe { deblock_luma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_luma_h_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
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

        /// The four positions of eight chroma lines: `[p1, p0, q0, q1]`.
        type ChromaLines = [__m128i; 4];

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: __m128i, maxv: __m128i) {
            unsafe {
                let [p1, p0, q0, q1] = *v;
                let alpha = _mm_set1_epi16(alpha as i16);
                let beta = _mm_set1_epi16(beta as i16);
                let zero = _mm_setzero_si128();
                let bs_on = _mm_cmpgt_epi16(tc0v, _mm_set1_epi16(-1));
                let mask = _mm_and_si128(_mm_and_si128(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), _mm_and_si128(diff_lt(q1, q0, beta), bs_on));
                let tc = _mm_add_epi16(tc0v, _mm_set1_epi16(1));
                let d = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(q0, p0), _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(p1, q1), _mm_set1_epi16(4)), 2)), 1);
                let d = _mm_min_epi16(_mm_max_epi16(d, _mm_sub_epi16(zero, tc)), tc);
                v[1] = sel(p0, clip(_mm_add_epi16(p0, d), maxv), mask);
                v[2] = sel(q0, clip(_mm_sub_epi16(q0, d), maxv), mask);
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
                let np0 = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(p1, 1), p0), _mm_add_epi16(q1, two)), 2);
                let nq0 = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(q1, 1), q0), _mm_add_epi16(p1, two)), 2);
                v[1] = sel(p0, np0, mask);
                v[2] = sel(q0, nq0, mask);
            }
        }

        /// Eight rows × four samples (p1 p0 q0 q1) around a vertical chroma
        /// edge as four column vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load_transposed_8x4(data: *const u16, stride: usize) -> ChromaLines {
            unsafe {
                let r = |i: usize| load4(data.add(i * stride).sub(2));
                let a0 = _mm_unpacklo_epi16(r(0), r(1)); // p1r0 p1r1 p0r0 p0r1 q0r0 q0r1 q1r0 q1r1
                let a1 = _mm_unpacklo_epi16(r(2), r(3));
                let a2 = _mm_unpacklo_epi16(r(4), r(5));
                let a3 = _mm_unpacklo_epi16(r(6), r(7));
                let b0 = _mm_unpacklo_epi32(a0, a1); // p1 rows 0..4 | p0 rows 0..4
                let b1 = _mm_unpackhi_epi32(a0, a1); // q0 rows 0..4 | q1 rows 0..4
                let b2 = _mm_unpacklo_epi32(a2, a3); // the same, rows 4..8
                let b3 = _mm_unpackhi_epi32(a2, a3);
                [_mm_unpacklo_epi64(b0, b2), _mm_unpackhi_epi64(b0, b2), _mm_unpacklo_epi64(b1, b3), _mm_unpackhi_epi64(b1, b3)]
            }
        }

        /// The p0 / q0 columns of eight rows back (p1, q1 are unchanged).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn store_transposed_8x4(data: *mut u16, stride: usize, v: &ChromaLines) {
            unsafe {
                // p0 and q0 of each row interleaved: one 32-bit lane a row.
                let mut t = [0u32; 8];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, _mm_unpacklo_epi16(v[1], v[2]));
                _mm_storeu_si128(t.as_mut_ptr().add(4) as *mut __m128i, _mm_unpackhi_epi16(v[1], v[2]));
                for (i, pq) in t.iter().enumerate() {
                    std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u32, *pq);
                }
            }
        }

        fn deblock_chroma_v(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            if !normal_in_range(alpha, beta, tc0, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_chroma_v)(data, off, stride, alpha, beta, tc0, max);
            }
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            unsafe {
                let mut v = load_transposed_8x4(data, stride);
                chroma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), _mm_set1_epi16(max as i16));
                store_transposed_8x4(data, stride, &v);
            }
        }

        fn deblock_chroma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
            if !strong_in_range(alpha, beta, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_chroma_v_intra)(data, off, stride, alpha, beta, max);
            }
            assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
            unsafe { deblock_chroma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_v_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
            unsafe {
                let mut v = load_transposed_8x4(data, stride);
                chroma_filter_intra(&mut v, alpha, beta);
                store_transposed_8x4(data, stride, &v);
            }
        }

        fn deblock_chroma_h(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            if tc0.iter().all(|&t| t < 0) {
                return;
            }
            if !normal_in_range(alpha, beta, tc0, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_chroma_h)(data, off, stride, alpha, beta, tc0, max);
            }
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
            unsafe {
                let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
                chroma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), _mm_set1_epi16(max as i16));
                store8(data.sub(stride), v[1]);
                store8(data, v[2]);
            }
        }

        fn deblock_chroma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
            if !strong_in_range(alpha, beta, max) {
                return (H264Dsp::<u16>::SCALAR.deblock_chroma_h_intra)(data, off, stride, alpha, beta, max);
            }
            assert!(off >= 2 * stride && off + stride + 8 <= data.len());
            unsafe { deblock_chroma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn deblock_chroma_h_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
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
        // Rows first, then columns, as the standard orders them (the `>> 1`
        // inside each pass makes the order matter), in i32 lanes: four to a
        // vector, so a 4x4 block's row is one vector and an 8x8 block's two.

        /// `(v + 32) >> 6` of eight i32 lanes (`lo`, `hi`) added to the eight
        /// samples at `dst`, clipped to `0..=max`. `packs` and `paddsw`
        /// saturate only where the exact sum is already outside `0..=max`,
        /// so the clip gives the exact answer (samples ≤ `max` ≤ 32767).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn add8(dst: *mut u16, lo: __m128i, hi: __m128i, maxv: __m128i) {
            unsafe {
                let r = _mm_set1_epi32(32);
                let v = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(lo, r), 6), _mm_srai_epi32(_mm_add_epi32(hi, r), 6));
                store8(dst, clip(_mm_adds_epi16(load8(dst), v), maxv));
            }
        }

        /// The same for four lanes and four samples.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn add4(dst: *mut u16, v: __m128i, maxv: __m128i) {
            unsafe {
                let r = _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(32)), 6);
                let s = _mm_adds_epi16(load4(dst), _mm_packs_epi32(r, _mm_setzero_si128()));
                _mm_storel_epi64(dst as *mut __m128i, clip(s, maxv));
            }
        }

        /// Transpose a 4x4 block of i32 held as four row vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn transpose4(r: [__m128i; 4]) -> [__m128i; 4] {
            unsafe {
                let t0 = _mm_unpacklo_epi32(r[0], r[1]); // r0c0 r1c0 r0c1 r1c1
                let t1 = _mm_unpacklo_epi32(r[2], r[3]); // r2c0 r3c0 r2c1 r3c1
                let t2 = _mm_unpackhi_epi32(r[0], r[1]); // r0c2 r1c2 r0c3 r1c3
                let t3 = _mm_unpackhi_epi32(r[2], r[3]);
                [_mm_unpacklo_epi64(t0, t1), _mm_unpackhi_epi64(t0, t1), _mm_unpacklo_epi64(t2, t3), _mm_unpackhi_epi64(t2, t3)]
            }
        }

        /// The 4x4 inverse transform (8.5.12.2) of four rows of i32
        /// coefficients, added to `dst`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn idct4_rows(dst: *mut u16, stride: usize, rows: [__m128i; 4], maxv: __m128i) {
            unsafe {
                // Row pass, across the columns: lane i is row i.
                let [c0, c1, c2, c3] = transpose4(rows);
                let e0 = _mm_add_epi32(c0, c2);
                let e1 = _mm_sub_epi32(c0, c2);
                let e2 = _mm_sub_epi32(_mm_srai_epi32(c1, 1), c3);
                let e3 = _mm_add_epi32(c1, _mm_srai_epi32(c3, 1));
                let f = [_mm_add_epi32(e0, e3), _mm_add_epi32(e1, e2), _mm_sub_epi32(e1, e2), _mm_sub_epi32(e0, e3)];
                // Column pass, across the rows: lane j is column j.
                let [r0, r1, r2, r3] = transpose4(f);
                let g0 = _mm_add_epi32(r0, r2);
                let g1 = _mm_sub_epi32(r0, r2);
                let g2 = _mm_sub_epi32(_mm_srai_epi32(r1, 1), r3);
                let g3 = _mm_add_epi32(r1, _mm_srai_epi32(r3, 1));
                add4(dst, _mm_add_epi32(g0, g3), maxv);
                add4(dst.add(stride), _mm_add_epi32(g1, g2), maxv);
                add4(dst.add(2 * stride), _mm_sub_epi32(g1, g2), maxv);
                add4(dst.add(3 * stride), _mm_sub_epi32(g0, g3), maxv);
            }
        }

        /// One 8-point pass (8.5.13.2) across eight vectors of i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn idct8_pass(d: &[__m128i; 8]) -> [__m128i; 8] {
            unsafe {
                let add = |a, b| _mm_add_epi32(a, b);
                let sub = |a, b| _mm_sub_epi32(a, b);
                let sh1 = |a| _mm_srai_epi32(a, 1);
                let sh2 = |a| _mm_srai_epi32(a, 2);
                let a0 = add(d[0], d[4]);
                let a4 = sub(d[0], d[4]);
                let a2 = sub(sh1(d[2]), d[6]);
                let a6 = add(d[2], sh1(d[6]));
                let b0 = add(a0, a6);
                let b2 = add(a4, a2);
                let b4 = sub(a4, a2);
                let b6 = sub(a0, a6);
                let a1 = sub(sub(sub(d[5], d[3]), d[7]), sh1(d[7]));
                let a3 = sub(sub(add(d[1], d[7]), d[3]), sh1(d[3]));
                let a5 = add(add(sub(d[7], d[1]), d[5]), sh1(d[5]));
                let a7 = add(add(add(d[3], d[5]), d[1]), sh1(d[1]));
                let b1 = add(a1, sh2(a7));
                let b7 = sub(a7, sh2(a1));
                let b3 = add(a3, sh2(a5));
                let b5 = sub(sh2(a3), a5);
                [add(b0, b7), add(b2, b5), add(b4, b3), add(b6, b1), sub(b6, b1), sub(b4, b3), sub(b2, b5), sub(b0, b7)]
            }
        }

        /// The 8x8 inverse transform (8.5.13.2) of eight rows of i32
        /// coefficients (columns 0..4 and 4..8 of each), added to `dst`.
        #[target_feature(enable = $feat)]
        unsafe fn idct8_rows(dst: *mut u16, stride: usize, rows: &[[__m128i; 2]; 8], maxv: __m128i) {
            unsafe {
                let zero = _mm_setzero_si128();
                // Row pass, four rows at a time: transposing a group's two
                // halves gives its eight columns, lane i being row i of the
                // group. `tmp[k][g]` is column k of group g's result.
                let mut tmp = [[zero; 2]; 8];
                for g in 0..2 {
                    let r = &rows[4 * g..4 * g + 4];
                    let lo = transpose4([r[0][0], r[1][0], r[2][0], r[3][0]]);
                    let hi = transpose4([r[0][1], r[1][1], r[2][1], r[3][1]]);
                    let f = idct8_pass(&[lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]);
                    for k in 0..8 {
                        tmp[k][g] = f[k];
                    }
                }
                // Column pass: back to rows, lane j being column j of one
                // half, and the eight rows through the pass one half at a time.
                let mut back = [[zero; 2]; 8];
                for g in 0..2 {
                    let lo = transpose4([tmp[0][g], tmp[1][g], tmp[2][g], tmp[3][g]]);
                    let hi = transpose4([tmp[4][g], tmp[5][g], tmp[6][g], tmp[7][g]]);
                    for i in 0..4 {
                        back[4 * g + i] = [lo[i], hi[i]];
                    }
                }
                let out_lo = idct8_pass(&[back[0][0], back[1][0], back[2][0], back[3][0], back[4][0], back[5][0], back[6][0], back[7][0]]);
                let out_hi = idct8_pass(&[back[0][1], back[1][1], back[2][1], back[3][1], back[4][1], back[5][1], back[6][1], back[7][1]]);
                for i in 0..8 {
                    add8(dst.add(i * stride), out_lo[i], out_hi[i], maxv);
                }
            }
        }

        /// Whether a transform call's samples and clip are inside the range
        /// the saturating add is exact over.
        #[inline(always)]
        fn add_in_range(max: i32) -> bool {
            (1..=32767).contains(&max)
        }

        fn idct4_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 16], max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.idct4_add)(dst, stride, coeffs, max);
            }
            assert!(3 * stride + 4 <= dst.len());
            unsafe { idct4_add_impl(dst.as_mut_ptr(), stride, coeffs, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn idct4_add_impl(dst: *mut u16, stride: usize, c: &[i16; 16], max: i32) {
            unsafe {
                let ld = |i: usize| sx16(_mm_loadl_epi64(c.as_ptr().add(4 * i) as *const __m128i));
                idct4_rows(dst, stride, [ld(0), ld(1), ld(2), ld(3)], _mm_set1_epi16(max as i16));
            }
        }

        fn idct8_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 64], max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.idct8_add)(dst, stride, coeffs, max);
            }
            assert!(7 * stride + 8 <= dst.len());
            unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn idct8_add_impl(dst: *mut u16, stride: usize, c: &[i16; 64], max: i32) {
            unsafe {
                let mut rows = [[_mm_setzero_si128(); 2]; 8];
                for (i, r) in rows.iter_mut().enumerate() {
                    let v = _mm_loadu_si128(c.as_ptr().add(8 * i) as *const __m128i);
                    *r = [sx16(v), sx16h(v)];
                }
                idct8_rows(dst, stride, &rows, _mm_set1_epi16(max as i16));
            }
        }

        fn idct4_dc_add(dst: &mut [u16], stride: usize, dc: i32, max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.idct4_dc_add)(dst, stride, dc, max);
            }
            assert!(3 * stride + 4 <= dst.len());
            unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 4, max) }
        }

        fn idct8_dc_add(dst: &mut [u16], stride: usize, dc: i32, max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.idct8_dc_add)(dst, stride, dc, max);
            }
            assert!(7 * stride + 8 <= dst.len());
            unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 8, max) }
        }

        /// `(dc + 32) >> 6` added to every sample of the `n x n` block, the
        /// value saturated to i16 like the transforms' (and for the same
        /// reason exact after the clip).
        #[target_feature(enable = $feat)]
        unsafe fn dc_add_impl(dst: *mut u16, stride: usize, dc: i32, n: usize, max: i32) {
            unsafe {
                let v = _mm_set1_epi16((dc.wrapping_add(32) >> 6).clamp(-32768, 32767) as i16);
                let maxv = _mm_set1_epi16(max as i16);
                for i in 0..n {
                    let p = dst.add(i * stride);
                    if n == 4 {
                        _mm_storel_epi64(p as *mut __m128i, clip(_mm_adds_epi16(load4(p), v), maxv));
                    } else {
                        store8(p, clip(_mm_adds_epi16(load8(p), v), maxv));
                    }
                }
            }
        }

        fn residual4(dst: &mut [u16], stride: usize, coefs: &[i32; 16], dc: i32, max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.residual4)(dst, stride, coefs, dc, max);
            }
            assert!(3 * stride + 4 <= dst.len());
            unsafe { residual4_impl(dst.as_mut_ptr(), stride, coefs, dc, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn residual4_impl(dst: *mut u16, stride: usize, coefs: &[i32; 16], dc: i32, max: i32) {
            unsafe {
                let mut c = *coefs;
                if dc != NO_DC {
                    c[0] = dc;
                }
                let p = c.as_ptr();
                let rows = [
                    _mm_loadu_si128(p as *const __m128i),
                    _mm_loadu_si128(p.add(4) as *const __m128i),
                    _mm_loadu_si128(p.add(8) as *const __m128i),
                    _mm_loadu_si128(p.add(12) as *const __m128i),
                ];
                // Any AC nonzero? Lane 0 of the first row masked out.
                let ac = _mm_or_si128(_mm_or_si128(_mm_andnot_si128(_mm_setr_epi32(-1, 0, 0, 0), rows[0]), rows[1]), _mm_or_si128(rows[2], rows[3]));
                if is_zero(ac) {
                    if c[0] != 0 {
                        dc_add_impl(dst, stride, c[0], 4, max);
                    }
                    return;
                }
                idct4_rows(dst, stride, rows, _mm_set1_epi16(max as i16));
            }
        }

        fn residual8(dst: &mut [u16], stride: usize, coefs: &[i32; 64], max: i32) {
            if !add_in_range(max) {
                return (H264Dsp::<u16>::SCALAR.residual8)(dst, stride, coefs, max);
            }
            assert!(7 * stride + 8 <= dst.len());
            unsafe { residual8_impl(dst.as_mut_ptr(), stride, coefs, max) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn residual8_impl(dst: *mut u16, stride: usize, coefs: &[i32; 64], max: i32) {
            unsafe {
                let p = coefs.as_ptr();
                let mut rows = [[_mm_setzero_si128(); 2]; 8];
                let mut ac = _mm_andnot_si128(_mm_setr_epi32(-1, 0, 0, 0), _mm_loadu_si128(p as *const __m128i));
                for (i, r) in rows.iter_mut().enumerate() {
                    *r = [_mm_loadu_si128(p.add(8 * i) as *const __m128i), _mm_loadu_si128(p.add(8 * i + 4) as *const __m128i)];
                    ac = _mm_or_si128(ac, if i == 0 { r[1] } else { _mm_or_si128(r[0], r[1]) });
                }
                if is_zero(ac) {
                    if coefs[0] != 0 {
                        dc_add_impl(dst, stride, coefs[0], 8, max);
                    }
                    return;
                }
                idct8_rows(dst, stride, &rows, _mm_set1_epi16(max as i16));
            }
        }
    };
}

// As in the 8-bit file: each rung is a full compilation of the kernels, the
// ladder calls only the groups a rung improves, and `dead_code` is allowed
// because "unused" is the intended state for most of three of the four.

/// SSE2: baseline on x86-64.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSSE3: `pabsw` for the loop filters' `|p − q|`.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// SSE4.1: `pblendvb` for the loop filters' lane selects, `pmovsxwd` /
/// `pmovzxwd` for the widenings, `ptest` for the all-zero tests.
pub(crate) mod sse41 {
    #![allow(dead_code)]
    kernels!("sse4.1", sse41);
}

/// AVX: the SSE4.1 algorithms, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// Install the best 16-bit-sample kernels `cpu` can run, one rung at a time.
pub fn install(d: &mut H264Dsp<u16>, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_deblock(d);
    }
    if cpu.sse41 {
        sse41::install_interp(d);
        sse41::install_deblock(d);
        sse41::install_rest(d);
    }
    if cpu.avx {
        avx::install_all(d);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dsp::u16_sweep;

    /// Every x86 rung the host can run, installed cumulatively as in the
    /// field — the 128-bit ladder here, and AVX2 over it.
    pub(crate) fn tables() -> Vec<(&'static str, H264Dsp<u16>)> {
        let base = Cpu::SCALAR;
        let ladder = [
            ("sse2", Cpu { sse2: true, ..base }, std::is_x86_feature_detected!("sse2")),
            ("ssse3", Cpu { sse2: true, ssse3: true, ..base }, std::is_x86_feature_detected!("ssse3")),
            ("sse4.1", Cpu { sse2: true, ssse3: true, sse41: true, ..base }, std::is_x86_feature_detected!("sse4.1")),
            ("avx", Cpu { sse2: true, ssse3: true, sse41: true, avx: true, ..base }, std::is_x86_feature_detected!("avx")),
            ("avx2", Cpu { sse2: true, ssse3: true, sse41: true, avx: true, avx2: true, ..base }, std::is_x86_feature_detected!("avx2")),
        ];
        ladder
            .into_iter()
            .filter(|&(_, _, have)| have)
            .map(|(name, cpu, _)| {
                let mut d = H264Dsp::<u16>::SCALAR;
                crate::dsp::h264::install_simd_u16(&mut d, cpu);
                (name, d)
            })
            .collect()
    }

    fn run(sweep: fn(&[(&str, H264Dsp<u16>)]) -> Result<u64, String>) {
        let t = tables();
        assert!(!t.is_empty(), "no x86 rung to test");
        match sweep(&t) {
            Ok(n) => assert!(n > 0, "the sweep compared nothing"),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn interpolation_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_interp);
    }

    #[test]
    fn chroma_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_chroma);
    }

    #[test]
    fn combination_and_weighting_match_scalar_at_every_depth() {
        run(u16_sweep::h264_combine);
    }

    #[test]
    fn deblocking_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_deblock);
    }

    #[test]
    fn transforms_match_scalar_at_every_depth() {
        run(u16_sweep::h264_transforms);
    }

    /// `H264Dsp::<u16>::new` reaches the tiers through the sample type's
    /// install, and takes them.
    #[test]
    fn new_installs_the_u16_tiers() {
        let cpu = Cpu::detect();
        if !cpu.sse2 {
            return;
        }
        let d = H264Dsp::<u16>::new(cpu);
        let s = H264Dsp::<u16>::SCALAR;
        assert!(d.qpel[10] as usize != s.qpel[10] as usize, "u16 qpel still scalar");
        assert!(d.residual4 as usize != s.residual4 as usize, "u16 residual4 still scalar");
        assert!(d.deblock_luma_v as usize != s.deblock_luma_v as usize, "u16 deblock still scalar");
    }

    /// Nanoseconds per call per kernel family, scalar against every rung,
    /// at 10 bits (and the six-tap at 12, whose path differs). Not a
    /// correctness test: `cargo test --release --lib h264_x86_128_u16 --
    /// --ignored --nocapture` prints it. The two scalar rows are the
    /// same-table control.
    #[test]
    #[ignore]
    fn kernel_bench() {
        use std::time::Instant;
        let mut seed = 0xbe9c_u64;
        let mut lcg = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let plane10: Vec<u16> = (0..64 * 64).map(|_| (lcg() % 1024) as u16).collect();
        let plane12: Vec<u16> = (0..64 * 64).map(|_| (lcg() % 4096) as u16).collect();
        // Smooth content, so that the loop filters' alpha / beta tests pass
        // and every call does the filtering; the buffer is filtered in place
        // call after call, and stays smooth.
        let smooth: Vec<u16> = (0..48 * 40).map(|_| 500 + (lcg() % 12) as u16).collect();
        let coefs: [i32; 64] = std::array::from_fn(|_| (lcg() % 801) as i32 - 400);
        let c4: [i32; 16] = coefs[..16].try_into().unwrap();
        let a: Vec<u16> = plane10[..256].to_vec();
        let b: Vec<u16> = plane10[256..512].to_vec();
        let edge = 8 * 48 + 8;
        let tc0 = [2i16 << 2, 3 << 2, 4 << 2, 5 << 2];
        type Call<'a> = &'a dyn Fn(&H264Dsp<u16>, &mut [u16]) -> u64;
        let families: [(&str, u32, Call); 14] = [
            ("qpel (2,2) 16x16 @10", 20_000, &|d, o| {
                (d.qpel[10])(o, &plane10[3 * 64 + 3..], 64, 16, 16, 1023);
                o[37] as u64
            }),
            ("qpel (1,0) 8x8 @10", 100_000, &|d, o| {
                (d.qpel[1])(o, &plane10[3 * 64 + 3..], 64, 8, 8, 1023);
                o[9] as u64
            }),
            ("qpel (2,2) 16x16 @12", 20_000, &|d, o| {
                (d.qpel[10])(o, &plane12[3 * 64 + 3..], 64, 16, 16, 4095);
                o[37] as u64
            }),
            ("qpel (1,0) 8x8 @12", 100_000, &|d, o| {
                (d.qpel[1])(o, &plane12[3 * 64 + 3..], 64, 8, 8, 4095);
                o[9] as u64
            }),
            ("chroma 8x8 (3,5) @10", 100_000, &|d, o| {
                (d.chroma)(o, &plane10[5 * 64 + 5..], 64, 8, 8, 3, 5);
                o[9] as u64
            }),
            ("chroma 8x8 (3,5) @12", 100_000, &|d, o| {
                (d.chroma)(o, &plane12[5 * 64 + 5..], 64, 8, 8, 3, 5);
                o[9] as u64
            }),
            ("avg 16x16", 100_000, &|d, o| {
                (d.avg)(o, 16, &a, &b, 16, 16);
                o[9] as u64
            }),
            ("weighted_bi 16x16 @10", 100_000, &|d, o| {
                (d.weighted_bi)(o, 16, &a, &b, 16, 16, 5, 20, 44, 4, -4, 1023);
                o[9] as u64
            }),
            ("deblock luma_v bS<4 @10", 100_000, &|d, o| {
                (d.deblock_luma_v)(o, edge, 48, 40 << 2, 12 << 2, &tc0, 1023);
                o[edge] as u64
            }),
            ("deblock luma_h bS4 @10", 100_000, &|d, o| {
                (d.deblock_luma_h_intra)(o, edge, 48, 40 << 2, 12 << 2, 1023);
                o[edge] as u64
            }),
            ("deblock chroma_v bS<4 @10", 100_000, &|d, o| {
                (d.deblock_chroma_v)(o, edge, 48, 40 << 2, 12 << 2, &tc0, 1023);
                o[edge] as u64
            }),
            ("idct8 residual8 @10", 100_000, &|d, o| {
                (d.residual8)(o, 48, &coefs, 1023);
                o[9] as u64
            }),
            ("idct4 residual4 @10", 200_000, &|d, o| {
                (d.residual4)(o, 48, &c4, NO_DC_BENCH, 1023);
                o[9] as u64
            }),
            ("dc residual4 @10", 200_000, &|d, o| {
                (d.idct4_dc_add)(o, 48, 300, 1023);
                o[9] as u64
            }),
        ];
        let s = H264Dsp::<u16>::SCALAR;
        let mut tabs = vec![("scalar", s), ("scalar-again", s)];
        tabs.extend(tables());
        // Every table back to back within a round, seven rounds: whatever
        // else the machine does drifts across all of them, and the ratio to
        // scalar is taken per round and its median reported.
        const ROUNDS: usize = 7;
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        for (label, iters, f) in &families {
            let per = (*iters as usize / ROUNDS).max(1);
            let mut ns = vec![[0f64; ROUNDS]; tabs.len()];
            let mut bufs: Vec<Vec<u16>> = tabs.iter().map(|_| smooth.clone()).collect();
            let mut sink = 0u64;
            for r in 0..ROUNDS {
                for (t, (_, d)) in tabs.iter().enumerate() {
                    let start = Instant::now();
                    for _ in 0..per {
                        sink = sink.wrapping_add(f(d, &mut bufs[t]));
                    }
                    ns[t][r] = start.elapsed().as_nanos() as f64 / per as f64;
                }
            }
            for (t, (name, _)) in tabs.iter().enumerate() {
                let own = median(ns[t].to_vec());
                let ratio = median((0..ROUNDS).map(|r| ns[0][r] / ns[t][r]).collect());
                println!("{label:26} {name:13} {own:9.1} ns/call  {ratio:6.2}x scalar (median of {ROUNDS} paired rounds) [{}]", sink & 1);
            }
        }
    }

    /// `NO_DC` for the bench: a block whose position 0 is a level like the rest.
    const NO_DC_BENCH: i32 = crate::dsp::h264::NO_DC;
}
