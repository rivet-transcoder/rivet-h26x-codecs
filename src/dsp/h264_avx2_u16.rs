//! AVX2 versions of the 16-bit-sample H.264 kernels (x86-64).
//!
//! Sixteen u16 lanes per vector, so a sixteen-sample luma row is one vector.
//! The arithmetic is [`super::h264_x86_128_u16`]'s lane for lane — the same
//! narrow and wide six-tap paths, the same loop-filter reformulations, the
//! same i32 transforms, and that file's documentation says why each is
//! exact — at twice the width. `unpack` and `packs` work per 128-bit lane,
//! and each `pmaddwd` over an `unpacklo` / `unpackhi` pair leaves outputs
//! 0..4 and 8..12 in one register and 4..8 and 12..16 in the other, so the
//! `packs` that narrows them puts all sixteen back in order with no permute.
//!
//! Installed over the AVX rung, replacing only the shapes where 256 bits
//! have something to do: sixteen-sample luma rows (interpolation, the
//! combiners, the sixteen-line loop filters) and the 8x8 transform, whose
//! rows are eight i32. Chroma interpolation and loop filtering, the 4x4
//! transform, the DC adds, `copy` and the MBAFF eight-line edges keep the
//! VEX-encoded 128-bit kernels: one row of those is one 128-bit vector
//! already. The rungs' tests (`super::h264_x86_128_u16`) include this one.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};
use super::h264::simd16::{DEEPEST, normal_in_range, strong_in_range, weights_in_range};

/// Replace the entries of `d` that 256-bit lanes improve.
pub fn install(d: &mut H264Dsp<u16>) {
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
    d.avg = avg;
    d.weighted_uni = weighted_uni;
    d.weighted_bi = weighted_bi;
    d.deblock_luma_v = deblock_luma_v;
    d.deblock_luma_h = deblock_luma_h;
    d.deblock_luma_v_intra = deblock_luma_v_intra;
    d.deblock_luma_h_intra = deblock_luma_h_intra;
    d.idct8_add = idct8_add;
    d.residual8 = residual8;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Sixteen samples at `p` as sixteen u16 lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load16(p: *const u16) -> __m256i {
    unsafe { _mm256_loadu_si256(p as *const __m256i) }
}

/// Eight samples at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load8(p: *const u16) -> __m128i {
    unsafe { _mm_loadu_si128(p as *const __m128i) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store16(p: *mut u16, v: __m256i) {
    unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store8(p: *mut u16, v: __m128i) {
    unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
}

/// Store the first `n` (≤ 16) lanes of `v` as samples.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_n(dst: *mut u16, v: __m256i, n: usize) {
    unsafe {
        match n {
            16 => store16(dst, v),
            8 => store8(dst, _mm256_castsi256_si128(v)),
            4 => _mm_storel_epi64(dst as *mut __m128i, _mm256_castsi256_si128(v)),
            _ => {
                let mut t = [0u16; 16];
                store16(t.as_mut_ptr(), v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Signed i16 lanes clipped to `0..=max` (`max` ≤ 32767).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn clip(v: __m256i, maxv: __m256i) -> __m256i {
    unsafe { _mm256_min_epi16(_mm256_max_epi16(v, _mm256_setzero_si256()), maxv) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn clip128(v: __m128i, maxv: __m128i) -> __m128i {
    unsafe { _mm_min_epi16(_mm_max_epi16(v, _mm_setzero_si128()), maxv) }
}

/// A tap pair as one i32 lane, for `pmaddwd`.
#[inline(always)]
fn pair(a: i16, b: i16) -> i32 {
    (a as u16 as i32) | ((b as u16 as i32) << 16)
}

// ----------------------------------------------------------------------
// Luma interpolation
// ----------------------------------------------------------------------

/// Six-tap over sixteen samples, minus 16384, in wrapping i16 (exact to ten bits).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tap6_narrow(p: *const u16, step: usize) -> __m256i {
    unsafe {
        let ld = |k: usize| load16(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = _mm256_add_epi16(c, d);
        let u = _mm256_add_epi16(b, e);
        let v = _mm256_add_epi16(_mm256_add_epi16(a, f), _mm256_set1_epi16(-16384));
        let t20 = _mm256_add_epi16(_mm256_slli_epi16(t, 4), _mm256_slli_epi16(t, 2));
        let u5 = _mm256_add_epi16(_mm256_slli_epi16(u, 2), u);
        _mm256_sub_epi16(_mm256_add_epi16(v, t20), u5)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn half_narrow(v: __m256i, maxv: __m256i) -> __m256i {
    unsafe {
        let r = _mm256_srai_epi16(_mm256_add_epi16(v, _mm256_set1_epi16(16)), 5);
        clip(_mm256_add_epi16(r, _mm256_set1_epi16(512)), maxv)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn j_narrow(w: &[__m256i; 6], maxv: __m256i) -> __m256i {
    unsafe {
        let (r0, r1, r2, r3, r4, r5) = (w[0], w[1], w[2], w[3], w[4], w[5]);
        let c01 = _mm256_set1_epi32(pair(1, -5));
        let c23 = _mm256_set1_epi32(pair(20, 20));
        let c45 = _mm256_set1_epi32(pair(-5, 1));
        let round = _mm256_set1_epi32(512 + 32 * 16384);
        let lo = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), c01), _mm256_madd_epi16(_mm256_unpacklo_epi16(r2, r3), c23)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(r4, r5), c45), round),
        );
        let hi = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), c01), _mm256_madd_epi16(_mm256_unpackhi_epi16(r2, r3), c23)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(r4, r5), c45), round),
        );
        clip(_mm256_packs_epi32(_mm256_srai_epi32(lo, 10), _mm256_srai_epi32(hi, 10)), maxv)
    }
}

/// A six-tap as two vectors of i32: lanes 0..4 | 8..12, and 4..8 | 12..16.
type Wide = [__m256i; 2];

/// Six-tap over sixteen samples in i32 (exact to 14 bits).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tap6_wide(p: *const u16, step: usize) -> Wide {
    unsafe {
        let ld = |k: usize| load16(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = _mm256_add_epi16(c, d);
        let u = _mm256_add_epi16(b, e);
        let v = _mm256_add_epi16(a, f);
        let k = _mm256_set1_epi32(pair(20, -5));
        let zero = _mm256_setzero_si256();
        [
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(t, u), k), _mm256_unpacklo_epi16(v, zero)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(t, u), k), _mm256_unpackhi_epi16(v, zero)),
        ]
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn half_wide(v: Wide, maxv: __m256i) -> __m256i {
    unsafe {
        let r = _mm256_set1_epi32(16);
        clip(_mm256_packs_epi32(_mm256_srai_epi32(_mm256_add_epi32(v[0], r), 5), _mm256_srai_epi32(_mm256_add_epi32(v[1], r), 5)), maxv)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tap6_i32(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i, r4: __m256i, r5: __m256i) -> __m256i {
    unsafe {
        let t = _mm256_add_epi32(r2, r3);
        let u = _mm256_add_epi32(r1, r4);
        let v = _mm256_add_epi32(r0, r5);
        let t20 = _mm256_add_epi32(_mm256_slli_epi32(t, 4), _mm256_slli_epi32(t, 2));
        let u5 = _mm256_add_epi32(_mm256_slli_epi32(u, 2), u);
        _mm256_sub_epi32(_mm256_add_epi32(v, t20), u5)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn j_wide(w: &[Wide; 6], maxv: __m256i) -> __m256i {
    unsafe {
        let round = _mm256_set1_epi32(512);
        let half = |k: usize| _mm256_srai_epi32(_mm256_add_epi32(tap6_i32(w[0][k], w[1][k], w[2][k], w[3][k], w[4][k], w[5][k]), round), 10);
        clip(_mm256_packs_epi32(half(0), half(1)), maxv)
    }
}

fn qpel<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    // A sixteen-lane load from column 0 reads column 15, +5 for the taps.
    let need = (h + 5 - 1) * stride + 21;
    if src.len() < need || dst.len() < h * PRED_STRIDE || w > 16 || !(1..=DEEPEST).contains(&max) {
        return (H264Dsp::<u16>::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h, max);
    }
    unsafe {
        if max <= 1023 {
            qpel_narrow::<XF, YF>(dst, src, stride, h, max)
        } else {
            qpel_wide::<XF, YF>(dst, src, stride, h, max)
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn qpel_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, h: usize, max: i32) {
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_narrow::<XF, YF>(dst, src, stride, h, max) };
    }
    unsafe {
        let s = src.as_ptr();
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let g = |dx: usize, dy: usize| load16(s.add((y + 2 + dy) * stride + 2 + dx));
            let b = || half_narrow(tap6_narrow(s.add((y + 2) * stride), 1), maxv);
            let b_below = || half_narrow(tap6_narrow(s.add((y + 3) * stride), 1), maxv);
            let hh = || half_narrow(tap6_narrow(s.add(y * stride + 2), stride), maxv);
            let hh_right = || half_narrow(tap6_narrow(s.add(y * stride + 3), stride), maxv);
            let v: __m256i = match (XF, YF) {
                (0, 0) => g(0, 0),
                (1, 0) => _mm256_avg_epu16(g(0, 0), b()),
                (2, 0) => b(),
                (3, 0) => _mm256_avg_epu16(g(1, 0), b()),
                (0, 1) => _mm256_avg_epu16(g(0, 0), hh()),
                (0, 2) => hh(),
                (0, 3) => _mm256_avg_epu16(g(0, 1), hh()),
                (1, 1) => _mm256_avg_epu16(b(), hh()),
                (3, 1) => _mm256_avg_epu16(b(), hh_right()),
                (1, 3) => _mm256_avg_epu16(hh(), b_below()),
                (3, 3) => _mm256_avg_epu16(hh_right(), b_below()),
                _ => unreachable!(),
            };
            // The scratch row is sixteen wide whatever `w` is.
            store16(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn qpel_centre_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = _mm256_set1_epi16(max as i16);
        let row = |r: usize| tap6_narrow(s.add(r * stride), 1);
        let mut win = [row(0), row(1), row(2), row(3), row(4), row(5)];
        for y in 0..h {
            let j = j_narrow(&win, maxv);
            let hh = |col: usize| half_narrow(tap6_narrow(s.add(y * stride + col), stride), maxv);
            let v: __m256i = match (XF, YF) {
                (2, 2) => j,
                (2, 1) => _mm256_avg_epu16(half_narrow(win[2], maxv), j),
                (2, 3) => _mm256_avg_epu16(j, half_narrow(win[3], maxv)),
                (1, 2) => _mm256_avg_epu16(hh(2), j),
                (3, 2) => _mm256_avg_epu16(j, hh(3)),
                _ => unreachable!(),
            };
            store16(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
            if y + 1 < h {
                win = [win[1], win[2], win[3], win[4], win[5], row(y + 6)];
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn qpel_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, h: usize, max: i32) {
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_wide::<XF, YF>(dst, src, stride, h, max) };
    }
    unsafe {
        let s = src.as_ptr();
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let g = |dx: usize, dy: usize| load16(s.add((y + 2 + dy) * stride + 2 + dx));
            let b = || half_wide(tap6_wide(s.add((y + 2) * stride), 1), maxv);
            let b_below = || half_wide(tap6_wide(s.add((y + 3) * stride), 1), maxv);
            let hh = || half_wide(tap6_wide(s.add(y * stride + 2), stride), maxv);
            let hh_right = || half_wide(tap6_wide(s.add(y * stride + 3), stride), maxv);
            let v: __m256i = match (XF, YF) {
                (0, 0) => g(0, 0),
                (1, 0) => _mm256_avg_epu16(g(0, 0), b()),
                (2, 0) => b(),
                (3, 0) => _mm256_avg_epu16(g(1, 0), b()),
                (0, 1) => _mm256_avg_epu16(g(0, 0), hh()),
                (0, 2) => hh(),
                (0, 3) => _mm256_avg_epu16(g(0, 1), hh()),
                (1, 1) => _mm256_avg_epu16(b(), hh()),
                (3, 1) => _mm256_avg_epu16(b(), hh_right()),
                (1, 3) => _mm256_avg_epu16(hh(), b_below()),
                (3, 3) => _mm256_avg_epu16(hh_right(), b_below()),
                _ => unreachable!(),
            };
            store16(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn qpel_centre_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = _mm256_set1_epi16(max as i16);
        let row = |r: usize| tap6_wide(s.add(r * stride), 1);
        let mut win = [row(0), row(1), row(2), row(3), row(4), row(5)];
        for y in 0..h {
            let j = j_wide(&win, maxv);
            let hh = |col: usize| half_wide(tap6_wide(s.add(y * stride + col), stride), maxv);
            let v: __m256i = match (XF, YF) {
                (2, 2) => j,
                (2, 1) => _mm256_avg_epu16(half_wide(win[2], maxv), j),
                (2, 3) => _mm256_avg_epu16(j, half_wide(win[3], maxv)),
                (1, 2) => _mm256_avg_epu16(hh(2), j),
                (3, 2) => _mm256_avg_epu16(j, hh(3)),
                _ => unreachable!(),
            };
            store16(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
            if y + 1 < h {
                win = [win[1], win[2], win[3], win[4], win[5], row(y + 6)];
            }
        }
    }
}

// ----------------------------------------------------------------------
// Combination and weighting
// ----------------------------------------------------------------------

fn avg(dst: &mut [u16], stride: usize, a: &[u16], b: &[u16], w: usize, h: usize) {
    assert!(h == 0 || ((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= a.len().min(b.len()) && w <= 16));
    unsafe { avg_impl(dst.as_mut_ptr(), stride, a.as_ptr(), b.as_ptr(), w, h) }
}

#[target_feature(enable = "avx2")]
unsafe fn avg_impl(dst: *mut u16, stride: usize, a: *const u16, b: *const u16, w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let v = _mm256_avg_epu16(load16(a.add(y * PRED_STRIDE)), load16(b.add(y * PRED_STRIDE)));
            store_n(dst.add(y * stride), v, w);
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

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_uni_impl(dst: *mut u16, stride: usize, src: *const u16, w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32) {
    unsafe {
        let wv = _mm256_set1_epi16(wt as i16);
        let round = _mm256_set1_epi32(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(log_wd);
        let ov = _mm256_set1_epi32(o);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let s = load16(src.add(y * PRED_STRIDE));
            let lo = _mm256_mullo_epi16(s, wv);
            let hi = _mm256_mulhi_epi16(s, wv);
            let q = |p: __m256i| _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(p, round), sh), ov);
            let v = clip(_mm256_packs_epi32(q(_mm256_unpacklo_epi16(lo, hi)), q(_mm256_unpackhi_epi16(lo, hi))), maxv);
            store_n(dst.add(y * stride), v, w);
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

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_bi_impl(dst: *mut u16, stride: usize, a: *const u16, b: *const u16, w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    unsafe {
        let wv = _mm256_set1_epi32(pair(w0 as i16, w1 as i16));
        let round = _mm256_set1_epi32(1 << log_wd);
        let off = _mm256_set1_epi32((o0 + o1 + 1) >> 1);
        let sh = _mm_cvtsi32_si128(log_wd + 1);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let va = load16(a.add(y * PRED_STRIDE));
            let vb = load16(b.add(y * PRED_STRIDE));
            let q = |v: __m256i| _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_madd_epi16(v, wv), round), sh), off);
            let v = clip(_mm256_packs_epi32(q(_mm256_unpacklo_epi16(va, vb)), q(_mm256_unpackhi_epi16(va, vb))), maxv);
            store_n(dst.add(y * stride), v, w);
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking: the sixteen-line luma edges
// ----------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn diff_lt(a: __m256i, b: __m256i, t: __m256i) -> __m256i {
    unsafe { _mm256_cmpgt_epi16(t, _mm256_abs_epi16(_mm256_sub_epi16(a, b))) }
}

/// The eight positions of sixteen luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
type LumaLines = [__m256i; 8];

/// bS < 4 luma filter on sixteen lines (8.7.2.3), in place.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: __m256i, maxv: __m256i) {
    unsafe {
        let [_, p2, p1, p0, q0, q1, q2, _] = *v;
        let alpha = _mm256_set1_epi16(alpha as i16);
        let beta = _mm256_set1_epi16(beta as i16);
        let zero = _mm256_setzero_si256();
        let bs_on = _mm256_cmpgt_epi16(tc0v, _mm256_set1_epi16(-1));
        let mask = _mm256_and_si256(_mm256_and_si256(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), _mm256_and_si256(diff_lt(q1, q0, beta), bs_on));
        let ap = diff_lt(p2, p0, beta);
        let aq = diff_lt(q2, q0, beta);
        let tc = _mm256_sub_epi16(_mm256_sub_epi16(tc0v, ap), aq);
        // ((q0 − p0) + ((p1 − q1 + 4) >> 2)) >> 1: the standard's delta, inside i16.
        let d = _mm256_srai_epi16(
            _mm256_add_epi16(_mm256_sub_epi16(q0, p0), _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(p1, q1), _mm256_set1_epi16(4)), 2)),
            1,
        );
        let d = _mm256_min_epi16(_mm256_max_epi16(d, _mm256_sub_epi16(zero, tc)), tc);
        let np0 = _mm256_add_epi16(p0, d);
        let nq0 = _mm256_sub_epi16(q0, d);
        let avg = _mm256_avg_epu16(p0, q0);
        let ntc0 = _mm256_sub_epi16(zero, tc0v);
        let dp1 = _mm256_srai_epi16(_mm256_sub_epi16(_mm256_add_epi16(p2, avg), _mm256_slli_epi16(p1, 1)), 1);
        let dp1 = _mm256_min_epi16(_mm256_max_epi16(dp1, ntc0), tc0v);
        let np1 = _mm256_add_epi16(p1, _mm256_and_si256(dp1, ap));
        let dq1 = _mm256_srai_epi16(_mm256_sub_epi16(_mm256_add_epi16(q2, avg), _mm256_slli_epi16(q1, 1)), 1);
        let dq1 = _mm256_min_epi16(_mm256_max_epi16(dq1, ntc0), tc0v);
        let nq1 = _mm256_add_epi16(q1, _mm256_and_si256(dq1, aq));
        v[2] = _mm256_blendv_epi8(p1, np1, mask);
        v[3] = _mm256_blendv_epi8(p0, clip(np0, maxv), mask);
        v[4] = _mm256_blendv_epi8(q0, clip(nq0, maxv), mask);
        v[5] = _mm256_blendv_epi8(q1, nq1, mask);
    }
}

/// bS 4 luma filter on sixteen lines (8.7.2.4), the sums in the u16 domain.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn luma_filter_intra(v: &mut LumaLines, alpha: i32, beta: i32) {
    unsafe {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
        let alphav = _mm256_set1_epi16(alpha as i16);
        let beta = _mm256_set1_epi16(beta as i16);
        let mask = _mm256_and_si256(_mm256_and_si256(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
        let strong = diff_lt(p0, q0, _mm256_set1_epi16(((alpha >> 2) + 2) as i16));
        let ap = _mm256_and_si256(diff_lt(p2, p0, beta), strong);
        let aq = _mm256_and_si256(diff_lt(q2, q0, beta), strong);
        let one = _mm256_set1_epi16(1);
        let two = _mm256_set1_epi16(2);
        let four = _mm256_set1_epi16(4);
        let add = |a, b| _mm256_add_epi16(a, b);
        let dbl = |a| _mm256_slli_epi16(a, 1);
        let wp0 = _mm256_srli_epi16(add(add(dbl(p1), p0), add(q1, two)), 2);
        let wq0 = _mm256_srli_epi16(add(add(dbl(q1), q0), add(p1, two)), 2);
        let p0q0 = add(p0, q0);
        let sp0 = _mm256_srli_epi16(add(_mm256_srli_epi16(add(add(p2, q1), four), 1), add(p1, p0q0)), 2);
        let tp = add(add(p2, p1), add(p0q0, two));
        let sp1 = _mm256_srli_epi16(tp, 2);
        let sp2 = _mm256_srli_epi16(add(add(_mm256_srli_epi16(tp, 1), one), add(p3, p2)), 2);
        let sq0 = _mm256_srli_epi16(add(_mm256_srli_epi16(add(add(q2, p1), four), 1), add(q1, p0q0)), 2);
        let tq = add(add(q2, q1), add(p0q0, two));
        let sq1 = _mm256_srli_epi16(tq, 2);
        let sq2 = _mm256_srli_epi16(add(add(_mm256_srli_epi16(tq, 1), one), add(q3, q2)), 2);
        let np0 = _mm256_blendv_epi8(wp0, sp0, ap);
        let np1 = _mm256_blendv_epi8(p1, sp1, ap);
        let np2 = _mm256_blendv_epi8(p2, sp2, ap);
        let nq0 = _mm256_blendv_epi8(wq0, sq0, aq);
        let nq1 = _mm256_blendv_epi8(q1, sq1, aq);
        let nq2 = _mm256_blendv_epi8(q2, sq2, aq);
        v[1] = _mm256_blendv_epi8(p2, np2, mask);
        v[2] = _mm256_blendv_epi8(p1, np1, mask);
        v[3] = _mm256_blendv_epi8(p0, np0, mask);
        v[4] = _mm256_blendv_epi8(q0, nq0, mask);
        v[5] = _mm256_blendv_epi8(q1, nq1, mask);
        v[6] = _mm256_blendv_epi8(q2, nq2, mask);
    }
}

/// tC0 per lane for sixteen luma lines (four per segment).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tc0_luma(tc0: &[i16; 4]) -> __m256i {
    unsafe {
        let t = |k: usize| tc0[k];
        _mm256_setr_epi16(t(0), t(0), t(0), t(0), t(1), t(1), t(1), t(1), t(2), t(2), t(2), t(2), t(3), t(3), t(3), t(3))
    }
}

/// Transpose eight 8-lane rows.
#[target_feature(enable = "avx2")]
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

/// The sixteen rows × eight samples around a vertical edge (`q0` at `data`)
/// as eight column vectors of sixteen lanes: each half transposed at 128
/// bits, the two halves then joined per column.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_transposed_16x8(data: *const u16, stride: usize) -> LumaLines {
    unsafe {
        let mut top = [_mm_setzero_si128(); 8];
        let mut bottom = [_mm_setzero_si128(); 8];
        for i in 0..8 {
            top[i] = load8(data.add(i * stride).sub(4));
            bottom[i] = load8(data.add((i + 8) * stride).sub(4));
        }
        transpose8(&mut top);
        transpose8(&mut bottom);
        std::array::from_fn(|k| _mm256_inserti128_si256(_mm256_castsi128_si256(top[k]), bottom[k], 1))
    }
}

/// Eight sixteen-lane column vectors back as sixteen rows × eight samples.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_transposed_16x8(data: *mut u16, stride: usize, v: &LumaLines) {
    unsafe {
        let mut top: [__m128i; 8] = std::array::from_fn(|k| _mm256_castsi256_si128(v[k]));
        let mut bottom: [__m128i; 8] = std::array::from_fn(|k| _mm256_extracti128_si256(v[k], 1));
        transpose8(&mut top);
        transpose8(&mut bottom);
        for i in 0..8 {
            store8(data.add(i * stride).sub(4), top[i]);
            store8(data.add((i + 8) * stride).sub(4), bottom[i]);
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

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_v_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
    unsafe {
        let mut v = load_transposed_16x8(data, stride);
        luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0), _mm256_set1_epi16(max as i16));
        store_transposed_16x8(data, stride, &v);
    }
}

fn deblock_luma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_v_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v = load_transposed_16x8(data, stride);
        luma_filter_intra(&mut v, alpha, beta);
        store_transposed_16x8(data, stride, &v);
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

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_h_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32) {
    unsafe {
        let zero = _mm256_setzero_si256();
        let ld = |k: isize| load16(data.offset(k * stride as isize));
        let mut v: LumaLines = [zero, ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), zero];
        luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0), _mm256_set1_epi16(max as i16));
        store16(data.offset(-2 * stride as isize), v[2]);
        store16(data.offset(-(stride as isize)), v[3]);
        store16(data, v[4]);
        store16(data.add(stride), v[5]);
    }
}

fn deblock_luma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma_h_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
    unsafe { deblock_luma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_h_intra_impl(data: *mut u16, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let ld = |k: isize| load16(data.offset(k * stride as isize));
        let mut v: LumaLines = [ld(-4), ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), ld(3)];
        luma_filter_intra(&mut v, alpha, beta);
        for k in 1..7 {
            store16(data.offset((k as isize - 4) * stride as isize), v[k]);
        }
    }
}

// ----------------------------------------------------------------------
// The 8x8 inverse transform
// ----------------------------------------------------------------------

/// Transpose an 8x8 block of i32 held as eight row vectors: in-lane unpacks
/// gather each half's columns, and `vperm2i128` joins the halves.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose8_32(r: &[__m256i; 8]) -> [__m256i; 8] {
    unsafe {
        let t = |i: usize| (_mm256_unpacklo_epi32(r[i], r[i + 1]), _mm256_unpackhi_epi32(r[i], r[i + 1]));
        let (t0, t1) = t(0);
        let (t2, t3) = t(2);
        let (t4, t5) = t(4);
        let (t6, t7) = t(6);
        // u0 = columns 0 | 4 of rows 0..4, u1 = 1 | 5, u2 = 2 | 6, u3 = 3 | 7;
        // u4..u8 the same for rows 4..8.
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        [
            _mm256_permute2x128_si256(u0, u4, 0x20),
            _mm256_permute2x128_si256(u1, u5, 0x20),
            _mm256_permute2x128_si256(u2, u6, 0x20),
            _mm256_permute2x128_si256(u3, u7, 0x20),
            _mm256_permute2x128_si256(u0, u4, 0x31),
            _mm256_permute2x128_si256(u1, u5, 0x31),
            _mm256_permute2x128_si256(u2, u6, 0x31),
            _mm256_permute2x128_si256(u3, u7, 0x31),
        ]
    }
}

/// One 8-point pass (8.5.13.2) across eight vectors of i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct8_pass(d: &[__m256i; 8]) -> [__m256i; 8] {
    unsafe {
        let add = |a, b| _mm256_add_epi32(a, b);
        let sub = |a, b| _mm256_sub_epi32(a, b);
        let sh1 = |a| _mm256_srai_epi32(a, 1);
        let sh2 = |a| _mm256_srai_epi32(a, 2);
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

/// The transform of eight rows of i32 coefficients, added to `dst`: rows
/// first, then columns, each pass across the transposed block.
#[target_feature(enable = "avx2")]
unsafe fn idct8_rows(dst: *mut u16, stride: usize, rows: &[__m256i; 8], max: i32) {
    unsafe {
        let f = idct8_pass(&transpose8_32(rows));
        let out = idct8_pass(&transpose8_32(&f));
        let maxv = _mm_set1_epi16(max as i16);
        let r = _mm256_set1_epi32(32);
        for (i, o) in out.iter().enumerate() {
            let v = _mm256_srai_epi32(_mm256_add_epi32(*o, r), 6);
            let v16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            let p = dst.add(i * stride);
            store8(p, clip128(_mm_adds_epi16(load8(p), v16), maxv));
        }
    }
}

fn idct8_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 64], max: i32) {
    if !(1..=32767).contains(&max) {
        return (H264Dsp::<u16>::SCALAR.idct8_add)(dst, stride, coeffs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn idct8_add_impl(dst: *mut u16, stride: usize, c: &[i16; 64], max: i32) {
    unsafe {
        let rows: [__m256i; 8] = std::array::from_fn(|i| _mm256_cvtepi16_epi32(_mm_loadu_si128(c.as_ptr().add(8 * i) as *const __m128i)));
        idct8_rows(dst, stride, &rows, max);
    }
}

fn residual8(dst: &mut [u16], stride: usize, coefs: &[i32; 64], max: i32) {
    if !(1..=32767).contains(&max) {
        return (H264Dsp::<u16>::SCALAR.residual8)(dst, stride, coefs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe { residual8_impl(dst.as_mut_ptr(), stride, coefs, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn residual8_impl(dst: *mut u16, stride: usize, coefs: &[i32; 64], max: i32) {
    unsafe {
        let p = coefs.as_ptr();
        let rows: [__m256i; 8] = std::array::from_fn(|i| _mm256_loadu_si256(p.add(8 * i) as *const __m256i));
        let mut ac = _mm256_andnot_si256(_mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0), rows[0]);
        for r in &rows[1..] {
            ac = _mm256_or_si256(ac, *r);
        }
        if _mm256_testz_si256(ac, ac) != 0 {
            if coefs[0] != 0 {
                // DC only: `(dc + 32) >> 6` saturated to i16, as the 128-bit
                // rung adds it (and exact after the clip for the same reason).
                let v = _mm_set1_epi16((coefs[0].wrapping_add(32) >> 6).clamp(-32768, 32767) as i16);
                let maxv = _mm_set1_epi16(max as i16);
                for i in 0..8 {
                    let q = dst.add(i * stride);
                    store8(q, clip128(_mm_adds_epi16(load8(q), v), maxv));
                }
            }
            return;
        }
        idct8_rows(dst, stride, &rows, max);
    }
}

// `NO_DC` is the residual paths' marker, used by the 4x4 path this file
// leaves to the 128-bit rung; imported so the two files read alike.
const _: i32 = NO_DC;
