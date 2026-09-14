//! NEON versions of the 16-bit-sample H.264 kernels (AArch64).
//!
//! Eight u16 lanes per vector and the arithmetic of
//! [`super::h264_x86_128_u16`], whose documentation says why each kernel is
//! exact, on NEON's instructions. Where NEON has a widening instruction that
//! x86 lacks it is used, and a depth switch disappears with it:
//!
//! - chroma's weighted sum above ten bits is `umlal` into u32 lanes, exact
//!   for any u16 (rows of ten-bit samples keep four `mla` in u16, chosen by
//!   `umaxv` over what the row loaded);
//! - the inverse transforms add in i32 and narrow with `uqxtn`, which
//!   saturates to 0..=65535, before the `min` with `max` — exact for any
//!   depth, where the x86 rungs saturate to i16 and rely on `max` ≤ 32767;
//! - weighting multiplies with `smull` straight into i32 and narrows the
//!   same way.
//!
//! The six-tap keeps both of the x86 paths (offset i16 up to ten bits, i32
//! above), and the loop filters the same reformulations.
//!
//! The same set of entries as the 8-bit NEON table. Written on x86 and
//! checked for compilation against `aarch64-unknown-linux-gnu`; the sweeps
//! below (`super::u16_sweep`) run on the CI arm64 runners.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::h264::simd16::{DEEPEST, normal_in_range, strong_in_range, weights_in_range};
use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// Replace the scalar entries of `d` with the NEON kernels.
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
    d.chroma = chroma;
    d.avg = avg;
    d.weighted_uni = weighted_uni;
    d.weighted_bi = weighted_bi;
    d.deblock_luma_v = deblock_luma_v;
    d.deblock_luma_h = deblock_luma_h;
    d.deblock_luma_v_intra = deblock_luma_v_intra;
    d.deblock_luma_h_intra = deblock_luma_h_intra;
    d.deblock_chroma_v = deblock_chroma_v;
    d.deblock_chroma_h = deblock_chroma_h;
    d.deblock_chroma_v_intra = deblock_chroma_v_intra;
    d.deblock_chroma_h_intra = deblock_chroma_h_intra;
    d.idct4_add = idct4_add;
    d.idct8_add = idct8_add;
    d.idct4_dc_add = idct4_dc_add;
    d.idct8_dc_add = idct8_dc_add;
    d.residual4 = residual4;
    d.residual8 = residual8;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Eight samples as eight i16 lanes (14-bit samples are positive i16).
#[inline(always)]
unsafe fn load_s(p: *const u16) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vld1q_u16(p)) }
}

/// Store the first `n` (≤ 8) lanes of `v` as samples.
#[inline(always)]
unsafe fn store_n(dst: *mut u16, v: uint16x8_t, n: usize) {
    unsafe {
        match n {
            8 => vst1q_u16(dst, v),
            4 => vst1_u16(dst, vget_low_u16(v)),
            _ => {
                let mut t = [0u16; 8];
                vst1q_u16(t.as_mut_ptr(), v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Signed i16 lanes clipped to `0..=max`.
#[inline(always)]
unsafe fn clip(v: int16x8_t, maxv: int16x8_t) -> int16x8_t {
    unsafe { vminq_s16(vmaxq_s16(v, vdupq_n_s16(0)), maxv) }
}

// ----------------------------------------------------------------------
// Luma interpolation
// ----------------------------------------------------------------------

/// Six-tap over the eight samples at `p`, … `p + 5 · step`, minus 16384, in
/// wrapping i16 (exact to ten bits).
#[inline(always)]
unsafe fn tap6_narrow(p: *const u16, step: usize) -> int16x8_t {
    unsafe {
        let ld = |k: usize| load_s(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = vaddq_s16(c, d);
        let u = vaddq_s16(b, e);
        let v = vaddq_s16(vaddq_s16(a, f), vdupq_n_s16(-16384));
        let t20 = vaddq_s16(vshlq_n_s16::<4>(t), vshlq_n_s16::<2>(t));
        let u5 = vaddq_s16(vshlq_n_s16::<2>(u), u);
        vsubq_s16(vaddq_s16(v, t20), u5)
    }
}

#[inline(always)]
unsafe fn half_narrow(v: int16x8_t, maxv: int16x8_t) -> uint16x8_t {
    unsafe {
        let r = vshrq_n_s16::<5>(vaddq_s16(v, vdupq_n_s16(16)));
        vreinterpretq_u16_s16(clip(vaddq_s16(r, vdupq_n_s16(512)), maxv))
    }
}

/// The centre value from six offset intermediates: the offset comes back as
/// `32 · 16384` with the rounding.
#[inline(always)]
unsafe fn j_narrow(r: &[int16x8_t; 6], maxv: int16x8_t) -> uint16x8_t {
    unsafe {
        let taps: [i16; 6] = [1, -5, 20, 20, -5, 1];
        let mut lo = vdupq_n_s32(512 + 32 * 16384);
        let mut hi = lo;
        for k in 0..6 {
            lo = vmlal_n_s16(lo, vget_low_s16(r[k]), taps[k]);
            hi = vmlal_high_n_s16(hi, r[k], taps[k]);
        }
        vreinterpretq_u16_s16(clip(vcombine_s16(vqmovn_s32(vshrq_n_s32::<10>(lo)), vqmovn_s32(vshrq_n_s32::<10>(hi))), maxv))
    }
}

/// A six-tap as two vectors of four i32.
type Wide = [int32x4_t; 2];

/// Six-tap in i32 (exact to 14 bits): `20 (c + d) − 5 (b + e)` by widening
/// multiplies of the u16 pair sums, `a + f` widened on.
#[inline(always)]
unsafe fn tap6_wide(p: *const u16, step: usize) -> Wide {
    unsafe {
        let ld = |k: usize| vld1q_u16(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = vaddq_u16(c, d);
        let u = vaddq_u16(b, e);
        let v = vaddq_u16(a, f);
        // Modulo 2^32 the partial differences do not matter; the sum is an i32.
        let lo = vaddw_u16(vmlsl_n_u16(vmull_n_u16(vget_low_u16(t), 20), vget_low_u16(u), 5), vget_low_u16(v));
        let hi = vaddw_high_u16(vmlsl_high_n_u16(vmull_high_n_u16(t, 20), u, 5), v);
        [vreinterpretq_s32_u32(lo), vreinterpretq_s32_u32(hi)]
    }
}

#[inline(always)]
unsafe fn half_wide(v: Wide, maxv: int16x8_t) -> uint16x8_t {
    unsafe { vreinterpretq_u16_s16(clip(vcombine_s16(vqmovn_s32(vrshrq_n_s32::<5>(v[0])), vqmovn_s32(vrshrq_n_s32::<5>(v[1]))), maxv)) }
}

#[inline(always)]
unsafe fn tap6_i32(r0: int32x4_t, r1: int32x4_t, r2: int32x4_t, r3: int32x4_t, r4: int32x4_t, r5: int32x4_t) -> int32x4_t {
    unsafe {
        let t = vaddq_s32(r2, r3);
        let u = vaddq_s32(r1, r4);
        let v = vaddq_s32(r0, r5);
        let t20 = vaddq_s32(vshlq_n_s32::<4>(t), vshlq_n_s32::<2>(t));
        let u5 = vaddq_s32(vshlq_n_s32::<2>(u), u);
        vsubq_s32(vaddq_s32(v, t20), u5)
    }
}

#[inline(always)]
unsafe fn j_wide(w: &[Wide; 6], maxv: int16x8_t) -> uint16x8_t {
    unsafe {
        let half = |k: usize| vqmovn_s32(vrshrq_n_s32::<10>(tap6_i32(w[0][k], w[1][k], w[2][k], w[3][k], w[4][k], w[5][k])));
        vreinterpretq_u16_s16(clip(vcombine_s16(half(0), half(1)), maxv))
    }
}

fn qpel<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    // An eight-lane load from column x ≤ 8 reads x + 7, +5 for the taps.
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

unsafe fn qpel_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_narrow::<XF, YF>(dst, src, stride, w, h, max) };
    }
    unsafe {
        let s = src.as_ptr();
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let g = |dx: usize, dy: usize| vld1q_u16(s.add((y + 2 + dy) * stride + 2 + dx + x));
                let b = || half_narrow(tap6_narrow(s.add((y + 2) * stride + x), 1), maxv);
                let b_below = || half_narrow(tap6_narrow(s.add((y + 3) * stride + x), 1), maxv);
                let hh = || half_narrow(tap6_narrow(s.add(y * stride + 2 + x), stride), maxv);
                let hh_right = || half_narrow(tap6_narrow(s.add(y * stride + 3 + x), stride), maxv);
                let v = match (XF, YF) {
                    (0, 0) => g(0, 0),
                    (1, 0) => vrhaddq_u16(g(0, 0), b()),
                    (2, 0) => b(),
                    (3, 0) => vrhaddq_u16(g(1, 0), b()),
                    (0, 1) => vrhaddq_u16(g(0, 0), hh()),
                    (0, 2) => hh(),
                    (0, 3) => vrhaddq_u16(g(0, 1), hh()),
                    (1, 1) => vrhaddq_u16(b(), hh()),
                    (3, 1) => vrhaddq_u16(b(), hh_right()),
                    (1, 3) => vrhaddq_u16(hh(), b_below()),
                    (3, 3) => vrhaddq_u16(hh_right(), b_below()),
                    _ => unreachable!(),
                };
                vst1q_u16(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                x += 8;
            }
        }
    }
}

unsafe fn qpel_centre_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = vdupq_n_s16(max as i16);
        let row = |r: usize, x: usize| tap6_narrow(s.add(r * stride + x), 1);
        let mut x = 0;
        while x < w {
            let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
            for y in 0..h {
                let j = j_narrow(&win, maxv);
                let hh = |col: usize| half_narrow(tap6_narrow(s.add(y * stride + col + x), stride), maxv);
                let v = match (XF, YF) {
                    (2, 2) => j,
                    (2, 1) => vrhaddq_u16(half_narrow(win[2], maxv), j),
                    (2, 3) => vrhaddq_u16(j, half_narrow(win[3], maxv)),
                    (1, 2) => vrhaddq_u16(hh(2), j),
                    (3, 2) => vrhaddq_u16(j, hh(3)),
                    _ => unreachable!(),
                };
                vst1q_u16(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                if y + 1 < h {
                    win = [win[1], win[2], win[3], win[4], win[5], row(y + 6, x)];
                }
            }
            x += 8;
        }
    }
}

unsafe fn qpel_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_wide::<XF, YF>(dst, src, stride, w, h, max) };
    }
    unsafe {
        let s = src.as_ptr();
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let g = |dx: usize, dy: usize| vld1q_u16(s.add((y + 2 + dy) * stride + 2 + dx + x));
                let b = || half_wide(tap6_wide(s.add((y + 2) * stride + x), 1), maxv);
                let b_below = || half_wide(tap6_wide(s.add((y + 3) * stride + x), 1), maxv);
                let hh = || half_wide(tap6_wide(s.add(y * stride + 2 + x), stride), maxv);
                let hh_right = || half_wide(tap6_wide(s.add(y * stride + 3 + x), stride), maxv);
                let v = match (XF, YF) {
                    (0, 0) => g(0, 0),
                    (1, 0) => vrhaddq_u16(g(0, 0), b()),
                    (2, 0) => b(),
                    (3, 0) => vrhaddq_u16(g(1, 0), b()),
                    (0, 1) => vrhaddq_u16(g(0, 0), hh()),
                    (0, 2) => hh(),
                    (0, 3) => vrhaddq_u16(g(0, 1), hh()),
                    (1, 1) => vrhaddq_u16(b(), hh()),
                    (3, 1) => vrhaddq_u16(b(), hh_right()),
                    (1, 3) => vrhaddq_u16(hh(), b_below()),
                    (3, 3) => vrhaddq_u16(hh_right(), b_below()),
                    _ => unreachable!(),
                };
                vst1q_u16(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                x += 8;
            }
        }
    }
}

unsafe fn qpel_centre_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = vdupq_n_s16(max as i16);
        let row = |r: usize, x: usize| tap6_wide(s.add(r * stride + x), 1);
        let mut x = 0;
        while x < w {
            let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
            for y in 0..h {
                let j = j_wide(&win, maxv);
                let hh = |col: usize| half_wide(tap6_wide(s.add(y * stride + col + x), stride), maxv);
                let v = match (XF, YF) {
                    (2, 2) => j,
                    (2, 1) => vrhaddq_u16(half_wide(win[2], maxv), j),
                    (2, 3) => vrhaddq_u16(j, half_wide(win[3], maxv)),
                    (1, 2) => vrhaddq_u16(hh(2), j),
                    (3, 2) => vrhaddq_u16(j, hh(3)),
                    _ => unreachable!(),
                };
                vst1q_u16(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                if y + 1 < h {
                    win = [win[1], win[2], win[3], win[4], win[5], row(y + 6, x)];
                }
            }
            x += 8;
        }
    }
}

// ----------------------------------------------------------------------
// Chroma interpolation, combination and weighting
// ----------------------------------------------------------------------

fn chroma(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + 9 || dst.len() < h * PRED_STRIDE || w > 8 || !(0..8).contains(&xf) || !(0..8).contains(&yf) {
        return (H264Dsp::<u16>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe {
        let (wa, wb, wc, wd) = (((8 - xf) * (8 - yf)) as u16, (xf * (8 - yf)) as u16, ((8 - xf) * yf) as u16, (xf * yf) as u16);
        let s = src.as_ptr();
        for y in 0..h {
            let r0 = s.add(y * stride);
            let r1 = s.add((y + 1) * stride);
            let (a, b, c, d) = (vld1q_u16(r0), vld1q_u16(r0.add(1)), vld1q_u16(r1), vld1q_u16(r1.add(1)));
            // The or of samples of at most ten bits has at most ten bits.
            let v = if vmaxvq_u16(vorrq_u16(vorrq_u16(a, b), vorrq_u16(c, d))) <= 1023 {
                // The weighted sum ≤ 64 · 1023: u16.
                vrshrq_n_u16::<6>(vmlaq_n_u16(vmlaq_n_u16(vmlaq_n_u16(vmulq_n_u16(a, wa), b, wb), c, wc), d, wd))
            } else {
                // Any u16: ≤ 64 · 65535 in u32 lanes.
                let lo = vmlal_n_u16(vmlal_n_u16(vmlal_n_u16(vmull_n_u16(vget_low_u16(a), wa), vget_low_u16(b), wb), vget_low_u16(c), wc), vget_low_u16(d), wd);
                let hi = vmlal_high_n_u16(vmlal_high_n_u16(vmlal_high_n_u16(vmull_high_n_u16(a, wa), b, wb), c, wc), d, wd);
                vcombine_u16(vrshrn_n_u32::<6>(lo), vrshrn_n_u32::<6>(hi))
            };
            vst1q_u16(dst.as_mut_ptr().add(y * PRED_STRIDE), v);
        }
    }
}

fn avg(dst: &mut [u16], stride: usize, a: &[u16], b: &[u16], w: usize, h: usize) {
    assert!(h == 0 || ((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= a.len().min(b.len()) && w <= 16));
    unsafe {
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let v = vrhaddq_u16(vld1q_u16(a.as_ptr().add(y * PRED_STRIDE + x)), vld1q_u16(b.as_ptr().add(y * PRED_STRIDE + x)));
                store_n(dst.as_mut_ptr().add(y * stride + x), v, (w - x).min(8));
                x += 8;
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
    unsafe {
        let round = vdupq_n_s32(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = vdupq_n_s32(-log_wd);
        let ov = vdupq_n_s32(o);
        let maxv = vdupq_n_u16(max as u16);
        let q = |p: int32x4_t| vqmovun_s32(vaddq_s32(vshlq_s32(vaddq_s32(p, round), sh), ov));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let s = load_s(src.as_ptr().add(y * PRED_STRIDE + x));
                let v = vminq_u16(vcombine_u16(q(vmull_n_s16(vget_low_s16(s), wt as i16)), q(vmull_high_n_s16(s, wt as i16))), maxv);
                store_n(dst.as_mut_ptr().add(y * stride + x), v, (w - x).min(8));
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi(dst: &mut [u16], stride: usize, a: &[u16], b: &[u16], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    if !weights_in_range(log_wd, [w0, w1], [o0, o1], max) || !combine_fits(dst, stride, a, w, h) || !combine_fits(dst, stride, b, w, h) {
        return (H264Dsp::<u16>::SCALAR.weighted_bi)(dst, stride, a, b, w, h, log_wd, w0, w1, o0, o1, max);
    }
    unsafe {
        let round = vdupq_n_s32(1 << log_wd);
        let off = vdupq_n_s32((o0 + o1 + 1) >> 1);
        let sh = vdupq_n_s32(-(log_wd + 1));
        let maxv = vdupq_n_u16(max as u16);
        let q = |p: int32x4_t| vqmovun_s32(vaddq_s32(vshlq_s32(vaddq_s32(p, round), sh), off));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let va = load_s(a.as_ptr().add(y * PRED_STRIDE + x));
                let vb = load_s(b.as_ptr().add(y * PRED_STRIDE + x));
                let lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(va), w0 as i16), vget_low_s16(vb), w1 as i16);
                let hi = vmlal_high_n_s16(vmull_high_n_s16(va, w0 as i16), vb, w1 as i16);
                let v = vminq_u16(vcombine_u16(q(lo), q(hi)), maxv);
                store_n(dst.as_mut_ptr().add(y * stride + x), v, (w - x).min(8));
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------

/// The eight positions of eight lines, `[p3, p2, p1, p0, q0, q1, q2, q3]`, as samples.
type Lines8 = [uint16x8_t; 8];

/// `|a − b| < t` per lane (differences of 14-bit samples are i16).
#[inline(always)]
unsafe fn diff_lt(a: int16x8_t, b: int16x8_t, t: int16x8_t) -> uint16x8_t {
    unsafe { vcltq_s16(vabdq_s16(a, b), t) }
}

/// bS < 4 luma filter on eight lines (8.7.2.3).
#[inline(always)]
unsafe fn luma_filter_normal(v: &mut Lines8, alpha: i32, beta: i32, tc0v: int16x8_t, maxv: int16x8_t) {
    unsafe {
        let s = |x: uint16x8_t| vreinterpretq_s16_u16(x);
        let (p2, p1, p0, q0, q1, q2) = (s(v[1]), s(v[2]), s(v[3]), s(v[4]), s(v[5]), s(v[6]));
        let alpha = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let bs_on = vcgtq_s16(tc0v, vdupq_n_s16(-1));
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), vandq_u16(diff_lt(q1, q0, beta), bs_on));
        let ap = diff_lt(p2, p0, beta);
        let aq = diff_lt(q2, q0, beta);
        let tc = vsubq_s16(vsubq_s16(tc0v, vreinterpretq_s16_u16(ap)), vreinterpretq_s16_u16(aq));
        // ((q0 − p0) + ((p1 − q1 + 4) >> 2)) >> 1: the standard's delta, inside i16.
        let d = vshrq_n_s16::<1>(vaddq_s16(vsubq_s16(q0, p0), vshrq_n_s16::<2>(vaddq_s16(vsubq_s16(p1, q1), vdupq_n_s16(4)))));
        let d = vminq_s16(vmaxq_s16(d, vnegq_s16(tc)), tc);
        let np0 = vaddq_s16(p0, d);
        let nq0 = vsubq_s16(q0, d);
        let avg = vreinterpretq_s16_u16(vrhaddq_u16(v[3], v[4]));
        let ntc0 = vnegq_s16(tc0v);
        let dp1 = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(p2, avg), vshlq_n_s16::<1>(p1)));
        let dp1 = vminq_s16(vmaxq_s16(dp1, ntc0), tc0v);
        let np1 = vaddq_s16(p1, vandq_s16(dp1, vreinterpretq_s16_u16(ap)));
        let dq1 = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(q2, avg), vshlq_n_s16::<1>(q1)));
        let dq1 = vminq_s16(vmaxq_s16(dq1, ntc0), tc0v);
        let nq1 = vaddq_s16(q1, vandq_s16(dq1, vreinterpretq_s16_u16(aq)));
        let u = |x: int16x8_t| vreinterpretq_u16_s16(x);
        v[2] = vbslq_u16(mask, u(np1), v[2]);
        v[3] = vbslq_u16(mask, u(clip(np0, maxv)), v[3]);
        v[4] = vbslq_u16(mask, u(clip(nq0, maxv)), v[4]);
        v[5] = vbslq_u16(mask, u(nq1), v[5]);
    }
}

/// bS 4 luma filter on eight lines (8.7.2.4), the sums in u16.
#[inline(always)]
unsafe fn luma_filter_intra(v: &mut Lines8, alpha: i32, beta: i32) {
    unsafe {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
        let lt = |a: uint16x8_t, b: uint16x8_t, t: uint16x8_t| vcltq_u16(vabdq_u16(a, b), t);
        let alphav = vdupq_n_u16(alpha as u16);
        let beta = vdupq_n_u16(beta as u16);
        let mask = vandq_u16(vandq_u16(lt(p0, q0, alphav), lt(p1, p0, beta)), lt(q1, q0, beta));
        let strong = lt(p0, q0, vdupq_n_u16(((alpha >> 2) + 2) as u16));
        let ap = vandq_u16(lt(p2, p0, beta), strong);
        let aq = vandq_u16(lt(q2, q0, beta), strong);
        let one = vdupq_n_u16(1);
        let two = vdupq_n_u16(2);
        let four = vdupq_n_u16(4);
        let add = |a, b| vaddq_u16(a, b);
        let dbl = |a| vshlq_n_u16::<1>(a);
        let wp0 = vshrq_n_u16::<2>(add(add(dbl(p1), p0), add(q1, two)));
        let wq0 = vshrq_n_u16::<2>(add(add(dbl(q1), q0), add(p1, two)));
        let p0q0 = add(p0, q0);
        let sp0 = vshrq_n_u16::<2>(add(vshrq_n_u16::<1>(add(add(p2, q1), four)), add(p1, p0q0)));
        let tp = add(add(p2, p1), add(p0q0, two));
        let sp1 = vshrq_n_u16::<2>(tp);
        let sp2 = vshrq_n_u16::<2>(add(add(vshrq_n_u16::<1>(tp), one), add(p3, p2)));
        let sq0 = vshrq_n_u16::<2>(add(vshrq_n_u16::<1>(add(add(q2, p1), four)), add(q1, p0q0)));
        let tq = add(add(q2, q1), add(p0q0, two));
        let sq1 = vshrq_n_u16::<2>(tq);
        let sq2 = vshrq_n_u16::<2>(add(add(vshrq_n_u16::<1>(tq), one), add(q3, q2)));
        let np0 = vbslq_u16(ap, sp0, wp0);
        let np1 = vbslq_u16(ap, sp1, p1);
        let np2 = vbslq_u16(ap, sp2, p2);
        let nq0 = vbslq_u16(aq, sq0, wq0);
        let nq1 = vbslq_u16(aq, sq1, q1);
        let nq2 = vbslq_u16(aq, sq2, q2);
        v[1] = vbslq_u16(mask, np2, p2);
        v[2] = vbslq_u16(mask, np1, p1);
        v[3] = vbslq_u16(mask, np0, p0);
        v[4] = vbslq_u16(mask, nq0, q0);
        v[5] = vbslq_u16(mask, nq1, q1);
        v[6] = vbslq_u16(mask, nq2, q2);
    }
}

/// tC0 per lane for lines `8 · half ..` of a sixteen-line edge (four per segment).
#[inline(always)]
unsafe fn tc0_luma(tc0: &[i16; 4], half: usize) -> int16x8_t {
    unsafe {
        let (a, b) = (tc0[2 * half], tc0[2 * half + 1]);
        let t = [a, a, a, a, b, b, b, b];
        vld1q_s16(t.as_ptr())
    }
}

/// tC0 per lane for eight chroma lines (two per segment).
#[inline(always)]
unsafe fn tc0_chroma(tc0: &[i16; 4]) -> int16x8_t {
    unsafe {
        let t = [tc0[0], tc0[0], tc0[1], tc0[1], tc0[2], tc0[2], tc0[3], tc0[3]];
        vld1q_s16(t.as_ptr())
    }
}

/// Transpose eight 8-lane rows of samples.
#[inline(always)]
unsafe fn transpose8(r: &mut Lines8) {
    unsafe {
        let s = |x: uint16x8_t| vreinterpretq_s16_u16(x);
        let a0 = vtrnq_s16(s(r[0]), s(r[1]));
        let a1 = vtrnq_s16(s(r[2]), s(r[3]));
        let a2 = vtrnq_s16(s(r[4]), s(r[5]));
        let a3 = vtrnq_s16(s(r[6]), s(r[7]));
        let b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.0), vreinterpretq_s32_s16(a1.0));
        let b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.1), vreinterpretq_s32_s16(a1.1));
        let b2 = vtrnq_s32(vreinterpretq_s32_s16(a2.0), vreinterpretq_s32_s16(a3.0));
        let b3 = vtrnq_s32(vreinterpretq_s32_s16(a2.1), vreinterpretq_s32_s16(a3.1));
        let lo = |x: int32x4_t| vget_low_s32(x);
        let hi = |x: int32x4_t| vget_high_s32(x);
        let u = |a: int32x2_t, b: int32x2_t| vreinterpretq_u16_s32(vcombine_s32(a, b));
        r[0] = u(lo(b0.0), lo(b2.0));
        r[1] = u(lo(b1.0), lo(b3.0));
        r[2] = u(lo(b0.1), lo(b2.1));
        r[3] = u(lo(b1.1), lo(b3.1));
        r[4] = u(hi(b0.0), hi(b2.0));
        r[5] = u(hi(b1.0), hi(b3.0));
        r[6] = u(hi(b0.1), hi(b2.1));
        r[7] = u(hi(b1.1), hi(b3.1));
    }
}

/// The eight rows × eight samples around a vertical edge (`q0` at `data`)
/// as eight column vectors.
#[inline(always)]
unsafe fn load_transposed_8x8(data: *const u16, stride: usize) -> Lines8 {
    unsafe {
        let mut r: Lines8 = std::array::from_fn(|i| vld1q_u16(data.add(i * stride).sub(4)));
        transpose8(&mut r);
        r
    }
}

#[inline(always)]
unsafe fn store_transposed_8x8(data: *mut u16, stride: usize, v: &Lines8) {
    unsafe {
        let mut r = *v;
        transpose8(&mut r);
        for (i, row) in r.iter().enumerate() {
            vst1q_u16(data.add(i * stride).sub(4), *row);
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
    unsafe {
        let maxv = vdupq_n_s16(max as i16);
        for half in 0..2 {
            let p = data.as_mut_ptr().add(off + half * 8 * stride);
            let mut v = load_transposed_8x8(p, stride);
            luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half), maxv);
            store_transposed_8x8(p, stride, &v);
        }
    }
}

fn deblock_luma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe {
        for half in 0..2 {
            let p = data.as_mut_ptr().add(off + half * 8 * stride);
            let mut v = load_transposed_8x8(p, stride);
            luma_filter_intra(&mut v, alpha, beta);
            store_transposed_8x8(p, stride, &v);
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
    unsafe {
        let maxv = vdupq_n_s16(max as i16);
        for half in 0..2 {
            let p = data.as_mut_ptr().add(off + half * 8);
            let ld = |k: isize| vld1q_u16(p.offset(k * stride as isize));
            let zero = vdupq_n_u16(0);
            let mut v: Lines8 = [zero, ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), zero];
            luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half), maxv);
            for k in 2..6 {
                vst1q_u16(p.offset((k as isize - 4) * stride as isize), v[k]);
            }
        }
    }
}

fn deblock_luma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma_h_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
    unsafe {
        for half in 0..2 {
            let p = data.as_mut_ptr().add(off + half * 8);
            let mut v: Lines8 = std::array::from_fn(|k| vld1q_u16(p.offset((k as isize - 4) * stride as isize)));
            luma_filter_intra(&mut v, alpha, beta);
            for k in 1..7 {
                vst1q_u16(p.offset((k as isize - 4) * stride as isize), v[k]);
            }
        }
    }
}

/// The four positions of eight chroma lines, `[p1, p0, q0, q1]`.
type ChromaLines = [uint16x8_t; 4];

#[inline(always)]
unsafe fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: int16x8_t, maxv: int16x8_t) {
    unsafe {
        let s = |x: uint16x8_t| vreinterpretq_s16_u16(x);
        let (p1, p0, q0, q1) = (s(v[0]), s(v[1]), s(v[2]), s(v[3]));
        let alpha = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let bs_on = vcgtq_s16(tc0v, vdupq_n_s16(-1));
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), vandq_u16(diff_lt(q1, q0, beta), bs_on));
        let tc = vaddq_s16(tc0v, vdupq_n_s16(1));
        let d = vshrq_n_s16::<1>(vaddq_s16(vsubq_s16(q0, p0), vshrq_n_s16::<2>(vaddq_s16(vsubq_s16(p1, q1), vdupq_n_s16(4)))));
        let d = vminq_s16(vmaxq_s16(d, vnegq_s16(tc)), tc);
        v[1] = vbslq_u16(mask, vreinterpretq_u16_s16(clip(vaddq_s16(p0, d), maxv)), v[1]);
        v[2] = vbslq_u16(mask, vreinterpretq_u16_s16(clip(vsubq_s16(q0, d), maxv)), v[2]);
    }
}

#[inline(always)]
unsafe fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let lt = |a: uint16x8_t, b: uint16x8_t, t: uint16x8_t| vcltq_u16(vabdq_u16(a, b), t);
        let alpha = vdupq_n_u16(alpha as u16);
        let beta = vdupq_n_u16(beta as u16);
        let mask = vandq_u16(vandq_u16(lt(p0, q0, alpha), lt(p1, p0, beta)), lt(q1, q0, beta));
        let two = vdupq_n_u16(2);
        let np0 = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(p1), p0), vaddq_u16(q1, two)));
        let nq0 = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(q1), q0), vaddq_u16(p1, two)));
        v[1] = vbslq_u16(mask, np0, p0);
        v[2] = vbslq_u16(mask, nq0, q0);
    }
}

/// Eight rows × four samples (p1 p0 q0 q1) around a vertical chroma edge as
/// four column vectors: `vld4` de-interleaves the four positions.
#[inline(always)]
unsafe fn load_transposed_8x4(data: *const u16, stride: usize) -> ChromaLines {
    unsafe {
        let mut rows = [0u16; 32];
        for i in 0..8 {
            std::ptr::copy_nonoverlapping(data.add(i * stride).sub(2), rows.as_mut_ptr().add(4 * i), 4);
        }
        let c = vld4q_u16(rows.as_ptr());
        [c.0, c.1, c.2, c.3]
    }
}

/// The p0 / q0 columns of eight rows back (p1, q1 are unchanged).
#[inline(always)]
unsafe fn store_transposed_8x4(data: *mut u16, stride: usize, v: &ChromaLines) {
    unsafe {
        let pq = vzipq_u16(v[1], v[2]);
        let mut t = [0u16; 16];
        vst1q_u16(t.as_mut_ptr(), pq.0);
        vst1q_u16(t.as_mut_ptr().add(8), pq.1);
        for i in 0..8 {
            let d = data.add(i * stride).sub(1);
            *d = t[2 * i];
            *d.add(1) = t[2 * i + 1];
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
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(p, stride);
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0), vdupq_n_s16(max as i16));
        store_transposed_8x4(p, stride, &v);
    }
}

fn deblock_chroma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_chroma_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(p, stride);
        chroma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x4(p, stride, &v);
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
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [vld1q_u16(p.sub(2 * stride)), vld1q_u16(p.sub(stride)), vld1q_u16(p), vld1q_u16(p.add(stride))];
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0), vdupq_n_s16(max as i16));
        vst1q_u16(p.sub(stride), v[1]);
        vst1q_u16(p, v[2]);
    }
}

fn deblock_chroma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_chroma_h_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [vld1q_u16(p.sub(2 * stride)), vld1q_u16(p.sub(stride)), vld1q_u16(p), vld1q_u16(p.add(stride))];
        chroma_filter_intra(&mut v, alpha, beta);
        vst1q_u16(p.sub(stride), v[1]);
        vst1q_u16(p, v[2]);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------

/// `(v + 32) >> 6` of four i32 lanes added to the four samples at `dst` in
/// i32, saturated to u16 and clipped to `max`: exact for any depth.
#[inline(always)]
unsafe fn add4(dst: *mut u16, v: int32x4_t, maxv: uint16x4_t) {
    unsafe {
        let p = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(dst)));
        vst1_u16(dst, vmin_u16(vqmovun_s32(vaddq_s32(p, vrshrq_n_s32::<6>(v))), maxv));
    }
}

/// The same for eight lanes (`lo`, `hi`) and eight samples.
#[inline(always)]
unsafe fn add8(dst: *mut u16, lo: int32x4_t, hi: int32x4_t, maxv: uint16x8_t) {
    unsafe {
        let p = vld1q_u16(dst);
        let l = vqmovun_s32(vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(p))), vrshrq_n_s32::<6>(lo)));
        let h = vqmovun_s32(vaddq_s32(vreinterpretq_s32_u32(vmovl_high_u16(p)), vrshrq_n_s32::<6>(hi)));
        vst1q_u16(dst, vminq_u16(vcombine_u16(l, h), maxv));
    }
}

/// Transpose a 4x4 block of i32 held as four row vectors.
#[inline(always)]
unsafe fn transpose4(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let a = vtrnq_s32(r[0], r[1]); // [r0c0 r1c0 r0c2 r1c2], [r0c1 r1c1 r0c3 r1c3]
        let b = vtrnq_s32(r[2], r[3]);
        [
            vcombine_s32(vget_low_s32(a.0), vget_low_s32(b.0)),
            vcombine_s32(vget_low_s32(a.1), vget_low_s32(b.1)),
            vcombine_s32(vget_high_s32(a.0), vget_high_s32(b.0)),
            vcombine_s32(vget_high_s32(a.1), vget_high_s32(b.1)),
        ]
    }
}

/// The 4x4 inverse transform (8.5.12.2) of four rows of i32, added to `dst`.
#[inline(always)]
unsafe fn idct4_rows(dst: *mut u16, stride: usize, rows: [int32x4_t; 4], max: i32) {
    unsafe {
        let [c0, c1, c2, c3] = transpose4(rows);
        let e0 = vaddq_s32(c0, c2);
        let e1 = vsubq_s32(c0, c2);
        let e2 = vsubq_s32(vshrq_n_s32::<1>(c1), c3);
        let e3 = vaddq_s32(c1, vshrq_n_s32::<1>(c3));
        let [r0, r1, r2, r3] = transpose4([vaddq_s32(e0, e3), vaddq_s32(e1, e2), vsubq_s32(e1, e2), vsubq_s32(e0, e3)]);
        let g0 = vaddq_s32(r0, r2);
        let g1 = vsubq_s32(r0, r2);
        let g2 = vsubq_s32(vshrq_n_s32::<1>(r1), r3);
        let g3 = vaddq_s32(r1, vshrq_n_s32::<1>(r3));
        let maxv = vdup_n_u16(max as u16);
        add4(dst, vaddq_s32(g0, g3), maxv);
        add4(dst.add(stride), vaddq_s32(g1, g2), maxv);
        add4(dst.add(2 * stride), vsubq_s32(g1, g2), maxv);
        add4(dst.add(3 * stride), vsubq_s32(g0, g3), maxv);
    }
}

/// One 8-point pass (8.5.13.2) across eight vectors of i32.
#[inline(always)]
unsafe fn idct8_pass(d: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    unsafe {
        let add = |a, b| vaddq_s32(a, b);
        let sub = |a, b| vsubq_s32(a, b);
        let sh1 = |a| vshrq_n_s32::<1>(a);
        let sh2 = |a| vshrq_n_s32::<2>(a);
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

/// The 8x8 inverse transform of eight rows of i32 (columns 0..4 and 4..8),
/// added to `dst` — the x86 128-bit rungs' arrangement: each pass across
/// four rows (or four columns) at a time.
#[inline(always)]
unsafe fn idct8_rows(dst: *mut u16, stride: usize, rows: &[[int32x4_t; 2]; 8], max: i32) {
    unsafe {
        let zero = vdupq_n_s32(0);
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
        let mut back = [[zero; 2]; 8];
        for g in 0..2 {
            let lo = transpose4([tmp[0][g], tmp[1][g], tmp[2][g], tmp[3][g]]);
            let hi = transpose4([tmp[4][g], tmp[5][g], tmp[6][g], tmp[7][g]]);
            for i in 0..4 {
                back[4 * g + i] = [lo[i], hi[i]];
            }
        }
        let out_lo = idct8_pass(&std::array::from_fn(|i| back[i][0]));
        let out_hi = idct8_pass(&std::array::from_fn(|i| back[i][1]));
        let maxv = vdupq_n_u16(max as u16);
        for i in 0..8 {
            add8(dst.add(i * stride), out_lo[i], out_hi[i], maxv);
        }
    }
}

/// Whether `max` is a u16.
#[inline(always)]
fn max_in_range(max: i32) -> bool {
    (0..=65535).contains(&max)
}

fn idct4_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 16], max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct4_add)(dst, stride, coeffs, max);
    }
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let ld = |i: usize| vmovl_s16(vld1_s16(coeffs.as_ptr().add(4 * i)));
        idct4_rows(dst.as_mut_ptr(), stride, [ld(0), ld(1), ld(2), ld(3)], max);
    }
}

fn idct8_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 64], max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct8_add)(dst, stride, coeffs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let rows: [[int32x4_t; 2]; 8] = std::array::from_fn(|i| {
            let v = vld1q_s16(coeffs.as_ptr().add(8 * i));
            [vmovl_s16(vget_low_s16(v)), vmovl_high_s16(v)]
        });
        idct8_rows(dst.as_mut_ptr(), stride, &rows, max);
    }
}

/// `(dc + 32) >> 6` added to every sample of the `n x n` block, in i32 and
/// narrowed as the transforms narrow.
#[inline(always)]
unsafe fn dc_add_impl(dst: *mut u16, stride: usize, dc: i32, n: usize, max: i32) {
    unsafe {
        let v = vdupq_n_s32(dc.wrapping_add(32) >> 6);
        for i in 0..n {
            let p = dst.add(i * stride);
            if n == 4 {
                let s = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p)));
                vst1_u16(p, vmin_u16(vqmovun_s32(vaddq_s32(s, v)), vdup_n_u16(max as u16)));
            } else {
                let s = vld1q_u16(p);
                let l = vqmovun_s32(vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s))), v));
                let h = vqmovun_s32(vaddq_s32(vreinterpretq_s32_u32(vmovl_high_u16(s)), v));
                vst1q_u16(p, vminq_u16(vcombine_u16(l, h), vdupq_n_u16(max as u16)));
            }
        }
    }
}

fn idct4_dc_add(dst: &mut [u16], stride: usize, dc: i32, max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct4_dc_add)(dst, stride, dc, max);
    }
    assert!(3 * stride + 4 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 4, max) }
}

fn idct8_dc_add(dst: &mut [u16], stride: usize, dc: i32, max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct8_dc_add)(dst, stride, dc, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 8, max) }
}

fn residual4(dst: &mut [u16], stride: usize, coefs: &[i32; 16], dc: i32, max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.residual4)(dst, stride, coefs, dc, max);
    }
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let mut c = *coefs;
        if dc != NO_DC {
            c[0] = dc;
        }
        let p = c.as_ptr();
        let rows = [vld1q_s32(p), vld1q_s32(p.add(4)), vld1q_s32(p.add(8)), vld1q_s32(p.add(12))];
        let ac = vorrq_s32(vorrq_s32(vsetq_lane_s32::<0>(0, rows[0]), rows[1]), vorrq_s32(rows[2], rows[3]));
        if vmaxvq_u32(vreinterpretq_u32_s32(ac)) == 0 {
            if c[0] != 0 {
                dc_add_impl(dst.as_mut_ptr(), stride, c[0], 4, max);
            }
            return;
        }
        idct4_rows(dst.as_mut_ptr(), stride, rows, max);
    }
}

fn residual8(dst: &mut [u16], stride: usize, coefs: &[i32; 64], max: i32) {
    if !max_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.residual8)(dst, stride, coefs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let p = coefs.as_ptr();
        let rows: [[int32x4_t; 2]; 8] = std::array::from_fn(|i| [vld1q_s32(p.add(8 * i)), vld1q_s32(p.add(8 * i + 4))]);
        let mut ac = vorrq_s32(vsetq_lane_s32::<0>(0, rows[0][0]), rows[0][1]);
        for r in &rows[1..] {
            ac = vorrq_s32(ac, vorrq_s32(r[0], r[1]));
        }
        if vmaxvq_u32(vreinterpretq_u32_s32(ac)) == 0 {
            if coefs[0] != 0 {
                dc_add_impl(dst.as_mut_ptr(), stride, coefs[0], 8, max);
            }
            return;
        }
        idct8_rows(dst.as_mut_ptr(), stride, &rows, max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::u16_sweep;

    fn run(sweep: fn(&[(&str, H264Dsp<u16>)]) -> Result<u64, String>) {
        let mut d = H264Dsp::<u16>::SCALAR;
        install(&mut d);
        match sweep(&[("neon", d)]) {
            Ok(n) => assert!(n > 0, "the sweep compared nothing"),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn neon_u16_interpolation_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_interp);
    }

    #[test]
    fn neon_u16_chroma_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_chroma);
    }

    #[test]
    fn neon_u16_combination_and_weighting_match_scalar_at_every_depth() {
        run(u16_sweep::h264_combine);
    }

    #[test]
    fn neon_u16_deblocking_matches_scalar_at_every_depth() {
        run(u16_sweep::h264_deblock);
    }

    #[test]
    fn neon_u16_transforms_match_scalar_at_every_depth() {
        run(u16_sweep::h264_transforms);
    }

    /// `H264Dsp::<u16>::new` takes the tier on every AArch64 CPU.
    #[test]
    fn neon_u16_new_installs_the_tier() {
        let d = H264Dsp::<u16>::new(crate::dsp::Cpu::detect());
        assert!(d.qpel[10] as usize != H264Dsp::<u16>::SCALAR.qpel[10] as usize, "u16 qpel still scalar");
    }
}
