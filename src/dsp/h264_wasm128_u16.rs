//! 128-bit SIMD versions of the 16-bit-sample H.264 kernels for
//! WebAssembly.
//!
//! A port of [`super::h264_x86_128_u16`]'s SSE4.1 rung — the same lane
//! layout, the same depth paths and reformulations, and that file's
//! documentation says why each is exact — onto `simd128`, which has the
//! SSE4.1 primitives natively (`v128_bitselect`, `i16x8_abs`,
//! `v128_any_true`, the widening extends). Where the x86 file makes a
//! product in i32 from `pmullw` and `pmulhw`, wasm has the widening multiply
//! itself (`i32x4_extmul_*_i16x8`), and it is used.
//!
//! One rung, compiled only with `+simd128`. `wasm32-unknown-unknown` has no
//! test harness, so the sweeps that check this tier (`super::u16_sweep`) run
//! inside the module: `examples/wasm_probe.rs` calls them and
//! `tools/wasm.sh` drives it.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::h264::simd16::{DEEPEST, normal_in_range, strong_in_range, weights_in_range};
use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// Replace the scalar entries of `d` with the simd128 kernels.
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
    d.copy = copy;
    d.avg = avg;
    d.weighted_uni = weighted_uni;
    d.weighted_bi = weighted_bi;
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

/// Eight samples at `p`.
#[inline]
unsafe fn load8(p: *const u16) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

/// Four samples at `p` in the low lanes, the high four zero.
#[inline]
unsafe fn load4(p: *const u16) -> v128 {
    unsafe { v128_load64_zero(p as *const u64) }
}

#[inline]
unsafe fn store8(p: *mut u16, v: v128) {
    unsafe { v128_store(p as *mut v128, v) }
}

/// Store the first `n` (≤ 8) lanes of `v` as samples.
#[inline]
unsafe fn store_n(dst: *mut u16, v: v128, n: usize) {
    unsafe {
        match n {
            8 => v128_store(dst as *mut v128, v),
            4 => v128_store64_lane::<0>(v, dst as *mut u64),
            _ => {
                let mut t = [0u16; 8];
                v128_store(t.as_mut_ptr() as *mut v128, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Signed i16 lanes clipped to `0..=max`.
#[inline]
fn clip(v: v128, maxv: v128) -> v128 {
    i16x8_min(i16x8_max(v, i16x8_splat(0)), maxv)
}

/// A tap pair as one i32 lane, for `i32x4_dot_i16x8`.
#[inline]
const fn pair(a: i16, b: i16) -> i32 {
    (a as u16 as i32) | ((b as u16 as i32) << 16)
}

#[inline]
fn zip_lo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}
#[inline]
fn zip_hi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}
#[inline]
fn zip_lo32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 4, 1, 5>(a, b)
}
#[inline]
fn zip_hi32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<2, 6, 3, 7>(a, b)
}
#[inline]
fn zip_lo64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<0, 2>(a, b)
}
#[inline]
fn zip_hi64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<1, 3>(a, b)
}

/// `b` where `m`'s lanes are all-ones, `a` where they are all-zeros.
#[inline]
fn sel(a: v128, b: v128, m: v128) -> v128 {
    v128_bitselect(b, a, m)
}

// ----------------------------------------------------------------------
// Luma interpolation
// ----------------------------------------------------------------------

/// Six-tap over eight samples, minus 16384, in wrapping i16 (exact to ten bits).
#[inline]
unsafe fn tap6_narrow(p: *const u16, step: usize) -> v128 {
    unsafe {
        let ld = |k: usize| load8(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = i16x8_add(c, d);
        let u = i16x8_add(b, e);
        let v = i16x8_add(i16x8_add(a, f), i16x8_splat(-16384));
        let t20 = i16x8_add(i16x8_shl(t, 4), i16x8_shl(t, 2));
        let u5 = i16x8_add(i16x8_shl(u, 2), u);
        i16x8_sub(i16x8_add(v, t20), u5)
    }
}

#[inline]
fn half_narrow(v: v128, maxv: v128) -> v128 {
    let r = i16x8_shr(i16x8_add(v, i16x8_splat(16)), 5);
    clip(i16x8_add(r, i16x8_splat(512)), maxv)
}

#[inline]
fn j_narrow(w: &[v128; 6], maxv: v128) -> v128 {
    let (r0, r1, r2, r3, r4, r5) = (w[0], w[1], w[2], w[3], w[4], w[5]);
    let c01 = i32x4_splat(pair(1, -5));
    let c23 = i32x4_splat(pair(20, 20));
    let c45 = i32x4_splat(pair(-5, 1));
    let round = i32x4_splat(512 + 32 * 16384);
    let half = |a: v128, b: v128, c: v128| i32x4_add(i32x4_add(i32x4_dot_i16x8(a, c01), i32x4_dot_i16x8(b, c23)), i32x4_add(i32x4_dot_i16x8(c, c45), round));
    let lo = half(zip_lo16(r0, r1), zip_lo16(r2, r3), zip_lo16(r4, r5));
    let hi = half(zip_hi16(r0, r1), zip_hi16(r2, r3), zip_hi16(r4, r5));
    clip(i16x8_narrow_i32x4(i32x4_shr(lo, 10), i32x4_shr(hi, 10)), maxv)
}

/// A six-tap as two vectors of four i32.
type Wide = [v128; 2];

/// Six-tap in i32 (exact to 14 bits).
#[inline]
unsafe fn tap6_wide(p: *const u16, step: usize) -> Wide {
    unsafe {
        let ld = |k: usize| load8(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = i16x8_add(c, d);
        let u = i16x8_add(b, e);
        let v = i16x8_add(a, f);
        let k = i32x4_splat(pair(20, -5));
        [
            i32x4_add(i32x4_dot_i16x8(zip_lo16(t, u), k), u32x4_extend_low_u16x8(v)),
            i32x4_add(i32x4_dot_i16x8(zip_hi16(t, u), k), u32x4_extend_high_u16x8(v)),
        ]
    }
}

#[inline]
fn half_wide(v: Wide, maxv: v128) -> v128 {
    let r = i32x4_splat(16);
    clip(i16x8_narrow_i32x4(i32x4_shr(i32x4_add(v[0], r), 5), i32x4_shr(i32x4_add(v[1], r), 5)), maxv)
}

#[inline]
fn tap6_i32(r0: v128, r1: v128, r2: v128, r3: v128, r4: v128, r5: v128) -> v128 {
    let t = i32x4_add(r2, r3);
    let u = i32x4_add(r1, r4);
    let v = i32x4_add(r0, r5);
    let t20 = i32x4_add(i32x4_shl(t, 4), i32x4_shl(t, 2));
    let u5 = i32x4_add(i32x4_shl(u, 2), u);
    i32x4_sub(i32x4_add(v, t20), u5)
}

#[inline]
fn j_wide(w: &[Wide; 6], maxv: v128) -> v128 {
    let round = i32x4_splat(512);
    let half = |k: usize| i32x4_shr(i32x4_add(tap6_i32(w[0][k], w[1][k], w[2][k], w[3][k], w[4][k], w[5][k]), round), 10);
    clip(i16x8_narrow_i32x4(half(0), half(1)), maxv)
}

fn qpel<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
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
        let maxv = i16x8_splat(max as i16);
        for y in 0..h {
            for c in 0..w.div_ceil(8) {
                let x = c * 8;
                let g = |dx: usize, dy: usize| load8(s.add((y + 2 + dy) * stride + 2 + dx + x));
                let b = || half_narrow(tap6_narrow(s.add((y + 2) * stride + x), 1), maxv);
                let b_below = || half_narrow(tap6_narrow(s.add((y + 3) * stride + x), 1), maxv);
                let hh = || half_narrow(tap6_narrow(s.add(y * stride + 2 + x), stride), maxv);
                let hh_right = || half_narrow(tap6_narrow(s.add(y * stride + 3 + x), stride), maxv);
                let v = match (XF, YF) {
                    (0, 0) => g(0, 0),
                    (1, 0) => u16x8_avgr(g(0, 0), b()),
                    (2, 0) => b(),
                    (3, 0) => u16x8_avgr(g(1, 0), b()),
                    (0, 1) => u16x8_avgr(g(0, 0), hh()),
                    (0, 2) => hh(),
                    (0, 3) => u16x8_avgr(g(0, 1), hh()),
                    (1, 1) => u16x8_avgr(b(), hh()),
                    (3, 1) => u16x8_avgr(b(), hh_right()),
                    (1, 3) => u16x8_avgr(hh(), b_below()),
                    (3, 3) => u16x8_avgr(hh_right(), b_below()),
                    _ => unreachable!(),
                };
                store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
            }
        }
    }
}

unsafe fn qpel_centre_narrow<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = i16x8_splat(max as i16);
        let row = |r: usize, x: usize| tap6_narrow(s.add(r * stride + x), 1);
        for c in 0..w.div_ceil(8) {
            let x = c * 8;
            let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
            for y in 0..h {
                let j = j_narrow(&win, maxv);
                let hh = |col: usize| half_narrow(tap6_narrow(s.add(y * stride + col + x), stride), maxv);
                let v = match (XF, YF) {
                    (2, 2) => j,
                    (2, 1) => u16x8_avgr(half_narrow(win[2], maxv), j),
                    (2, 3) => u16x8_avgr(j, half_narrow(win[3], maxv)),
                    (1, 2) => u16x8_avgr(hh(2), j),
                    (3, 2) => u16x8_avgr(j, hh(3)),
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

unsafe fn qpel_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_wide::<XF, YF>(dst, src, stride, w, h, max) };
    }
    unsafe {
        let s = src.as_ptr();
        let maxv = i16x8_splat(max as i16);
        for y in 0..h {
            for c in 0..w.div_ceil(8) {
                let x = c * 8;
                let g = |dx: usize, dy: usize| load8(s.add((y + 2 + dy) * stride + 2 + dx + x));
                let b = || half_wide(tap6_wide(s.add((y + 2) * stride + x), 1), maxv);
                let b_below = || half_wide(tap6_wide(s.add((y + 3) * stride + x), 1), maxv);
                let hh = || half_wide(tap6_wide(s.add(y * stride + 2 + x), stride), maxv);
                let hh_right = || half_wide(tap6_wide(s.add(y * stride + 3 + x), stride), maxv);
                let v = match (XF, YF) {
                    (0, 0) => g(0, 0),
                    (1, 0) => u16x8_avgr(g(0, 0), b()),
                    (2, 0) => b(),
                    (3, 0) => u16x8_avgr(g(1, 0), b()),
                    (0, 1) => u16x8_avgr(g(0, 0), hh()),
                    (0, 2) => hh(),
                    (0, 3) => u16x8_avgr(g(0, 1), hh()),
                    (1, 1) => u16x8_avgr(b(), hh()),
                    (3, 1) => u16x8_avgr(b(), hh_right()),
                    (1, 3) => u16x8_avgr(hh(), b_below()),
                    (3, 3) => u16x8_avgr(hh_right(), b_below()),
                    _ => unreachable!(),
                };
                store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
            }
        }
    }
}

unsafe fn qpel_centre_wide<const XF: usize, const YF: usize>(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, max: i32) {
    unsafe {
        let s = src.as_ptr();
        let maxv = i16x8_splat(max as i16);
        let row = |r: usize, x: usize| tap6_wide(s.add(r * stride + x), 1);
        for c in 0..w.div_ceil(8) {
            let x = c * 8;
            let mut win = [row(0, x), row(1, x), row(2, x), row(3, x), row(4, x), row(5, x)];
            for y in 0..h {
                let j = j_wide(&win, maxv);
                let hh = |col: usize| half_wide(tap6_wide(s.add(y * stride + col + x), stride), maxv);
                let v = match (XF, YF) {
                    (2, 2) => j,
                    (2, 1) => u16x8_avgr(half_wide(win[2], maxv), j),
                    (2, 3) => u16x8_avgr(j, half_wide(win[3], maxv)),
                    (1, 2) => u16x8_avgr(hh(2), j),
                    (3, 2) => u16x8_avgr(j, hh(3)),
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

// ----------------------------------------------------------------------
// Chroma interpolation, combination and weighting
// ----------------------------------------------------------------------

fn chroma(dst: &mut [u16], src: &[u16], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + 9 || dst.len() < h * PRED_STRIDE || w > 8 || !(0..8).contains(&xf) || !(0..8).contains(&yf) {
        return (H264Dsp::<u16>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe {
        let (wa, wb, wc, wd) = (((8 - xf) * (8 - yf)) as i16, (xf * (8 - yf)) as i16, ((8 - xf) * yf) as i16, (xf * yf) as i16);
        let mul = [i16x8_splat(wa), i16x8_splat(wb), i16x8_splat(wc), i16x8_splat(wd)];
        let k0 = i32x4_splat(pair(wa, wb));
        let k1 = i32x4_splat(pair(wc, wd));
        let s = src.as_ptr();
        for y in 0..h {
            let r0 = s.add(y * stride);
            let r1 = s.add((y + 1) * stride);
            let (a, b, c, d) = (load8(r0), load8(r0.add(1)), load8(r1), load8(r1.add(1)));
            let seen = v128_or(v128_or(a, b), v128_or(c, d));
            let v = if !v128_any_true(v128_and(seen, i16x8_splat(!1023))) {
                let sum = i16x8_add(i16x8_add(i16x8_mul(a, mul[0]), i16x8_mul(b, mul[1])), i16x8_add(i16x8_mul(c, mul[2]), i16x8_mul(d, mul[3])));
                u16x8_shr(i16x8_add(sum, i16x8_splat(32)), 6)
            } else if !v128_any_true(v128_and(seen, i16x8_splat(i16::MIN))) {
                let r = i32x4_splat(32);
                let lo = i32x4_add(i32x4_dot_i16x8(zip_lo16(a, b), k0), i32x4_dot_i16x8(zip_lo16(c, d), k1));
                let hi = i32x4_add(i32x4_dot_i16x8(zip_hi16(a, b), k0), i32x4_dot_i16x8(zip_hi16(c, d), k1));
                i16x8_narrow_i32x4(i32x4_shr(i32x4_add(lo, r), 6), i32x4_shr(i32x4_add(hi, r), 6))
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
    unsafe {
        for y in 0..h {
            let s = src.as_ptr().add(y * PRED_STRIDE);
            let d = dst.as_mut_ptr().add(y * stride);
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
    unsafe {
        for y in 0..h {
            let (pa, pb, d) = (a.as_ptr().add(y * PRED_STRIDE), b.as_ptr().add(y * PRED_STRIDE), dst.as_mut_ptr().add(y * stride));
            store_n(d, u16x8_avgr(load8(pa), load8(pb)), w.min(8));
            if w > 8 {
                store_n(d.add(8), u16x8_avgr(load8(pa.add(8)), load8(pb.add(8))), w - 8);
            }
        }
    }
}

/// Whether a combiner's buffers hold what it reads and writes.
#[inline]
fn combine_fits(dst: &[u16], stride: usize, src: &[u16], w: usize, h: usize) -> bool {
    w <= 16 && (h == 0 || ((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len()))
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni(dst: &mut [u16], stride: usize, src: &[u16], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32) {
    if !weights_in_range(log_wd, [wt, 0], [o, 0], max) || !combine_fits(dst, stride, src, w, h) {
        return (H264Dsp::<u16>::SCALAR.weighted_uni)(dst, stride, src, w, h, log_wd, wt, o, max);
    }
    unsafe {
        let wv = i16x8_splat(wt as i16);
        let round = i32x4_splat(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = log_wd as u32;
        let ov = i32x4_splat(o);
        let maxv = i16x8_splat(max as i16);
        let scale = |s: v128| {
            let q = |p: v128| i32x4_add(i32x4_shr(i32x4_add(p, round), sh), ov);
            clip(i16x8_narrow_i32x4(q(i32x4_extmul_low_i16x8(s, wv)), q(i32x4_extmul_high_i16x8(s, wv))), maxv)
        };
        for y in 0..h {
            let p = src.as_ptr().add(y * PRED_STRIDE);
            let d = dst.as_mut_ptr().add(y * stride);
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
    unsafe {
        let wv = i32x4_splat(pair(w0 as i16, w1 as i16));
        let round = i32x4_splat(1 << log_wd);
        let off = i32x4_splat((o0 + o1 + 1) >> 1);
        let sh = (log_wd + 1) as u32;
        let maxv = i16x8_splat(max as i16);
        let eight = |va: v128, vb: v128| {
            let q = |v: v128| i32x4_add(i32x4_shr(i32x4_add(i32x4_dot_i16x8(v, wv), round), sh), off);
            clip(i16x8_narrow_i32x4(q(zip_lo16(va, vb)), q(zip_hi16(va, vb))), maxv)
        };
        for y in 0..h {
            let (pa, pb, d) = (a.as_ptr().add(y * PRED_STRIDE), b.as_ptr().add(y * PRED_STRIDE), dst.as_mut_ptr().add(y * stride));
            store_n(d, eight(load8(pa), load8(pb)), w.min(8));
            if w > 8 {
                store_n(d.add(8), eight(load8(pa.add(8)), load8(pb.add(8))), w - 8);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------

#[inline]
fn diff_lt(a: v128, b: v128, t: v128) -> v128 {
    i16x8_gt(t, i16x8_abs(i16x8_sub(a, b)))
}

/// The eight positions of eight luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
type LumaLines = [v128; 8];

#[inline]
fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: v128, maxv: v128) {
    let [_, p2, p1, p0, q0, q1, q2, _] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let zero = i16x8_splat(0);
    let bs_on = i16x8_gt(tc0v, i16x8_splat(-1));
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), v128_and(diff_lt(q1, q0, beta), bs_on));
    let ap = diff_lt(p2, p0, beta);
    let aq = diff_lt(q2, q0, beta);
    let tc = i16x8_sub(i16x8_sub(tc0v, ap), aq);
    // ((q0 − p0) + ((p1 − q1 + 4) >> 2)) >> 1: the standard's delta, inside i16.
    let d = i16x8_shr(i16x8_add(i16x8_sub(q0, p0), i16x8_shr(i16x8_add(i16x8_sub(p1, q1), i16x8_splat(4)), 2)), 1);
    let d = i16x8_min(i16x8_max(d, i16x8_sub(zero, tc)), tc);
    let np0 = i16x8_add(p0, d);
    let nq0 = i16x8_sub(q0, d);
    let avg = u16x8_avgr(p0, q0);
    let ntc0 = i16x8_sub(zero, tc0v);
    let dp1 = i16x8_shr(i16x8_sub(i16x8_add(p2, avg), i16x8_shl(p1, 1)), 1);
    let dp1 = i16x8_min(i16x8_max(dp1, ntc0), tc0v);
    let np1 = i16x8_add(p1, v128_and(dp1, ap));
    let dq1 = i16x8_shr(i16x8_sub(i16x8_add(q2, avg), i16x8_shl(q1, 1)), 1);
    let dq1 = i16x8_min(i16x8_max(dq1, ntc0), tc0v);
    let nq1 = i16x8_add(q1, v128_and(dq1, aq));
    v[2] = sel(p1, np1, mask);
    v[3] = sel(p0, clip(np0, maxv), mask);
    v[4] = sel(q0, clip(nq0, maxv), mask);
    v[5] = sel(q1, nq1, mask);
}

#[inline]
fn luma_filter_intra(v: &mut LumaLines, alpha: i32, beta: i32) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let alphav = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let mask = v128_and(v128_and(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
    let strong = diff_lt(p0, q0, i16x8_splat(((alpha >> 2) + 2) as i16));
    let ap = v128_and(diff_lt(p2, p0, beta), strong);
    let aq = v128_and(diff_lt(q2, q0, beta), strong);
    let one = i16x8_splat(1);
    let two = i16x8_splat(2);
    let four = i16x8_splat(4);
    let add = i16x8_add;
    let dbl = |a| i16x8_shl(a, 1);
    let shr = u16x8_shr;
    let wp0 = shr(add(add(dbl(p1), p0), add(q1, two)), 2);
    let wq0 = shr(add(add(dbl(q1), q0), add(p1, two)), 2);
    let p0q0 = add(p0, q0);
    let sp0 = shr(add(shr(add(add(p2, q1), four), 1), add(p1, p0q0)), 2);
    let tp = add(add(p2, p1), add(p0q0, two));
    let sp1 = shr(tp, 2);
    let sp2 = shr(add(add(shr(tp, 1), one), add(p3, p2)), 2);
    let sq0 = shr(add(shr(add(add(q2, p1), four), 1), add(q1, p0q0)), 2);
    let tq = add(add(q2, q1), add(p0q0, two));
    let sq1 = shr(tq, 2);
    let sq2 = shr(add(add(shr(tq, 1), one), add(q3, q2)), 2);
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

#[inline]
fn tc0_luma(tc0: &[i16; 4], half: usize) -> v128 {
    let (a, b) = (tc0[2 * half], tc0[2 * half + 1]);
    i16x8(a, a, a, a, b, b, b, b)
}

#[inline]
fn tc0_pairs(tc0: &[i16; 4]) -> v128 {
    let t = |k: usize| tc0[k];
    i16x8(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
}

/// Transpose eight 8-lane rows.
#[inline]
fn transpose8(r: &mut [v128; 8]) {
    let a0 = zip_lo16(r[0], r[1]);
    let a1 = zip_hi16(r[0], r[1]);
    let a2 = zip_lo16(r[2], r[3]);
    let a3 = zip_hi16(r[2], r[3]);
    let a4 = zip_lo16(r[4], r[5]);
    let a5 = zip_hi16(r[4], r[5]);
    let a6 = zip_lo16(r[6], r[7]);
    let a7 = zip_hi16(r[6], r[7]);
    let b0 = zip_lo32(a0, a2);
    let b1 = zip_hi32(a0, a2);
    let b2 = zip_lo32(a1, a3);
    let b3 = zip_hi32(a1, a3);
    let b4 = zip_lo32(a4, a6);
    let b5 = zip_hi32(a4, a6);
    let b6 = zip_lo32(a5, a7);
    let b7 = zip_hi32(a5, a7);
    r[0] = zip_lo64(b0, b4);
    r[1] = zip_hi64(b0, b4);
    r[2] = zip_lo64(b1, b5);
    r[3] = zip_hi64(b1, b5);
    r[4] = zip_lo64(b2, b6);
    r[5] = zip_hi64(b2, b6);
    r[6] = zip_lo64(b3, b7);
    r[7] = zip_hi64(b3, b7);
}

#[inline]
unsafe fn load_transposed_8x8(data: *const u16, stride: usize) -> LumaLines {
    unsafe {
        let mut r: LumaLines = std::array::from_fn(|i| load8(data.add(i * stride).sub(4)));
        transpose8(&mut r);
        r
    }
}

#[inline]
unsafe fn store_transposed_8x8(data: *mut u16, stride: usize, v: &LumaLines) {
    unsafe {
        let mut r = *v;
        transpose8(&mut r);
        for (i, row) in r.iter().enumerate() {
            store8(data.add(i * stride).sub(4), *row);
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
        let maxv = i16x8_splat(max as i16);
        for half in 0..2 {
            let d = data.as_mut_ptr().add(off + half * 8 * stride);
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
    unsafe {
        let d = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x8(d, stride);
        luma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), i16x8_splat(max as i16));
        store_transposed_8x8(d, stride, &v);
    }
}

fn deblock_luma8_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma8_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe {
        let d = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x8(d, stride);
        luma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x8(d, stride, &v);
    }
}

fn deblock_luma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_luma_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe {
        for half in 0..2 {
            let d = data.as_mut_ptr().add(off + half * 8 * stride);
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
    unsafe {
        let zero = i16x8_splat(0);
        let maxv = i16x8_splat(max as i16);
        for half in 0..2 {
            let d = data.as_mut_ptr().add(off + half * 8);
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
    unsafe {
        for half in 0..2 {
            let d = data.as_mut_ptr().add(off + half * 8);
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
type ChromaLines = [v128; 4];

#[inline]
fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: v128, maxv: v128) {
    let [p1, p0, q0, q1] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let zero = i16x8_splat(0);
    let bs_on = i16x8_gt(tc0v, i16x8_splat(-1));
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), v128_and(diff_lt(q1, q0, beta), bs_on));
    let tc = i16x8_add(tc0v, i16x8_splat(1));
    let d = i16x8_shr(i16x8_add(i16x8_sub(q0, p0), i16x8_shr(i16x8_add(i16x8_sub(p1, q1), i16x8_splat(4)), 2)), 1);
    let d = i16x8_min(i16x8_max(d, i16x8_sub(zero, tc)), tc);
    v[1] = sel(p0, clip(i16x8_add(p0, d), maxv), mask);
    v[2] = sel(q0, clip(i16x8_sub(q0, d), maxv), mask);
}

#[inline]
fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
    let [p1, p0, q0, q1] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
    let two = i16x8_splat(2);
    let np0 = u16x8_shr(i16x8_add(i16x8_add(i16x8_shl(p1, 1), p0), i16x8_add(q1, two)), 2);
    let nq0 = u16x8_shr(i16x8_add(i16x8_add(i16x8_shl(q1, 1), q0), i16x8_add(p1, two)), 2);
    v[1] = sel(p0, np0, mask);
    v[2] = sel(q0, nq0, mask);
}

#[inline]
unsafe fn load_transposed_8x4(data: *const u16, stride: usize) -> ChromaLines {
    unsafe {
        let r = |i: usize| load4(data.add(i * stride).sub(2));
        let a0 = zip_lo16(r(0), r(1));
        let a1 = zip_lo16(r(2), r(3));
        let a2 = zip_lo16(r(4), r(5));
        let a3 = zip_lo16(r(6), r(7));
        let b0 = zip_lo32(a0, a1);
        let b1 = zip_hi32(a0, a1);
        let b2 = zip_lo32(a2, a3);
        let b3 = zip_hi32(a2, a3);
        [zip_lo64(b0, b2), zip_hi64(b0, b2), zip_lo64(b1, b3), zip_hi64(b1, b3)]
    }
}

#[inline]
unsafe fn store_transposed_8x4(data: *mut u16, stride: usize, v: &ChromaLines) {
    unsafe {
        let mut t = [0u32; 8];
        v128_store(t.as_mut_ptr() as *mut v128, zip_lo16(v[1], v[2]));
        v128_store(t.as_mut_ptr().add(4) as *mut v128, zip_hi16(v[1], v[2]));
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
    unsafe {
        let d = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(d, stride);
        chroma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), i16x8_splat(max as i16));
        store_transposed_8x4(d, stride, &v);
    }
}

fn deblock_chroma_v_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_chroma_v_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let d = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(d, stride);
        chroma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x4(d, stride, &v);
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
        let d = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [load8(d.sub(2 * stride)), load8(d.sub(stride)), load8(d), load8(d.add(stride))];
        chroma_filter_normal(&mut v, alpha, beta, tc0_pairs(tc0), i16x8_splat(max as i16));
        store8(d.sub(stride), v[1]);
        store8(d, v[2]);
    }
}

fn deblock_chroma_h_intra(data: &mut [u16], off: usize, stride: usize, alpha: i32, beta: i32, max: i32) {
    if !strong_in_range(alpha, beta, max) {
        return (H264Dsp::<u16>::SCALAR.deblock_chroma_h_intra)(data, off, stride, alpha, beta, max);
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let d = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [load8(d.sub(2 * stride)), load8(d.sub(stride)), load8(d), load8(d.add(stride))];
        chroma_filter_intra(&mut v, alpha, beta);
        store8(d.sub(stride), v[1]);
        store8(d, v[2]);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------

#[inline]
unsafe fn add8(dst: *mut u16, lo: v128, hi: v128, maxv: v128) {
    unsafe {
        let r = i32x4_splat(32);
        let v = i16x8_narrow_i32x4(i32x4_shr(i32x4_add(lo, r), 6), i32x4_shr(i32x4_add(hi, r), 6));
        store8(dst, clip(i16x8_add_sat(load8(dst), v), maxv));
    }
}

#[inline]
unsafe fn add4(dst: *mut u16, v: v128, maxv: v128) {
    unsafe {
        let r = i32x4_shr(i32x4_add(v, i32x4_splat(32)), 6);
        let s = i16x8_add_sat(load4(dst), i16x8_narrow_i32x4(r, i32x4_splat(0)));
        v128_store64_lane::<0>(clip(s, maxv), dst as *mut u64);
    }
}

#[inline]
fn transpose4(r: [v128; 4]) -> [v128; 4] {
    let t0 = zip_lo32(r[0], r[1]);
    let t1 = zip_lo32(r[2], r[3]);
    let t2 = zip_hi32(r[0], r[1]);
    let t3 = zip_hi32(r[2], r[3]);
    [zip_lo64(t0, t1), zip_hi64(t0, t1), zip_lo64(t2, t3), zip_hi64(t2, t3)]
}

#[inline]
unsafe fn idct4_rows(dst: *mut u16, stride: usize, rows: [v128; 4], maxv: v128) {
    unsafe {
        let [c0, c1, c2, c3] = transpose4(rows);
        let e0 = i32x4_add(c0, c2);
        let e1 = i32x4_sub(c0, c2);
        let e2 = i32x4_sub(i32x4_shr(c1, 1), c3);
        let e3 = i32x4_add(c1, i32x4_shr(c3, 1));
        let [r0, r1, r2, r3] = transpose4([i32x4_add(e0, e3), i32x4_add(e1, e2), i32x4_sub(e1, e2), i32x4_sub(e0, e3)]);
        let g0 = i32x4_add(r0, r2);
        let g1 = i32x4_sub(r0, r2);
        let g2 = i32x4_sub(i32x4_shr(r1, 1), r3);
        let g3 = i32x4_add(r1, i32x4_shr(r3, 1));
        add4(dst, i32x4_add(g0, g3), maxv);
        add4(dst.add(stride), i32x4_add(g1, g2), maxv);
        add4(dst.add(2 * stride), i32x4_sub(g1, g2), maxv);
        add4(dst.add(3 * stride), i32x4_sub(g0, g3), maxv);
    }
}

#[inline]
fn idct8_pass(d: &[v128; 8]) -> [v128; 8] {
    let add = i32x4_add;
    let sub = i32x4_sub;
    let sh1 = |a| i32x4_shr(a, 1);
    let sh2 = |a| i32x4_shr(a, 2);
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

unsafe fn idct8_rows(dst: *mut u16, stride: usize, rows: &[[v128; 2]; 8], maxv: v128) {
    unsafe {
        let zero = i32x4_splat(0);
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
        for i in 0..8 {
            add8(dst.add(i * stride), out_lo[i], out_hi[i], maxv);
        }
    }
}

#[inline]
fn add_in_range(max: i32) -> bool {
    (1..=32767).contains(&max)
}

fn idct4_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 16], max: i32) {
    if !add_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct4_add)(dst, stride, coeffs, max);
    }
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let ld = |i: usize| i32x4_extend_low_i16x8(v128_load64_zero(coeffs.as_ptr().add(4 * i) as *const u64));
        idct4_rows(dst.as_mut_ptr(), stride, [ld(0), ld(1), ld(2), ld(3)], i16x8_splat(max as i16));
    }
}

fn idct8_add(dst: &mut [u16], stride: usize, coeffs: &[i16; 64], max: i32) {
    if !add_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.idct8_add)(dst, stride, coeffs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let rows: [[v128; 2]; 8] = std::array::from_fn(|i| {
            let v = v128_load(coeffs.as_ptr().add(8 * i) as *const v128);
            [i32x4_extend_low_i16x8(v), i32x4_extend_high_i16x8(v)]
        });
        idct8_rows(dst.as_mut_ptr(), stride, &rows, i16x8_splat(max as i16));
    }
}

unsafe fn dc_add_impl(dst: *mut u16, stride: usize, dc: i32, n: usize, max: i32) {
    unsafe {
        let v = i16x8_splat((dc.wrapping_add(32) >> 6).clamp(-32768, 32767) as i16);
        let maxv = i16x8_splat(max as i16);
        for i in 0..n {
            let p = dst.add(i * stride);
            if n == 4 {
                v128_store64_lane::<0>(clip(i16x8_add_sat(load4(p), v), maxv), p as *mut u64);
            } else {
                store8(p, clip(i16x8_add_sat(load8(p), v), maxv));
            }
        }
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

fn residual4(dst: &mut [u16], stride: usize, coefs: &[i32; 16], dc: i32, max: i32) {
    if !add_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.residual4)(dst, stride, coefs, dc, max);
    }
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let mut c = *coefs;
        if dc != NO_DC {
            c[0] = dc;
        }
        let p = c.as_ptr();
        let rows = [v128_load(p as *const v128), v128_load(p.add(4) as *const v128), v128_load(p.add(8) as *const v128), v128_load(p.add(12) as *const v128)];
        // `v128_andnot(a, b)` is `a & !b`.
        let ac = v128_or(v128_or(v128_andnot(rows[0], i32x4(-1, 0, 0, 0)), rows[1]), v128_or(rows[2], rows[3]));
        if !v128_any_true(ac) {
            if c[0] != 0 {
                dc_add_impl(dst.as_mut_ptr(), stride, c[0], 4, max);
            }
            return;
        }
        idct4_rows(dst.as_mut_ptr(), stride, rows, i16x8_splat(max as i16));
    }
}

fn residual8(dst: &mut [u16], stride: usize, coefs: &[i32; 64], max: i32) {
    if !add_in_range(max) {
        return (H264Dsp::<u16>::SCALAR.residual8)(dst, stride, coefs, max);
    }
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let p = coefs.as_ptr();
        let rows: [[v128; 2]; 8] = std::array::from_fn(|i| [v128_load(p.add(8 * i) as *const v128), v128_load(p.add(8 * i + 4) as *const v128)]);
        let mut ac = v128_or(v128_andnot(rows[0][0], i32x4(-1, 0, 0, 0)), rows[0][1]);
        for r in &rows[1..] {
            ac = v128_or(ac, v128_or(r[0], r[1]));
        }
        if !v128_any_true(ac) {
            if coefs[0] != 0 {
                dc_add_impl(dst.as_mut_ptr(), stride, coefs[0], 8, max);
            }
            return;
        }
        idct8_rows(dst.as_mut_ptr(), stride, &rows, i16x8_splat(max as i16));
    }
}
