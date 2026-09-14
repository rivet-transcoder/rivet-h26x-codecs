//! 128-bit SIMD versions of the distortion metrics for WebAssembly, 16-bit
//! samples — the kernels of [`super::distortion_x86_u16`], exact for every
//! `u16` input as those are, on what `simd128` has for them. Two of the x86
//! file's devices are unnecessary here, because wasm widens:
//!
//! - SAD folds its absolute differences (`u16x8_sub_sat` both ways, or'd)
//!   with `u32x4_extadd_pairwise_u16x8`, which is exact where the x86 file
//!   flips a sign bit for `pmaddwd`.
//! - SSD squares them with `u32x4_extmul_*_u16x8` (65535² fits u32) and
//!   widens the squares to the 64-bit accumulator directly, where the x86
//!   file splits each square into halves.
//! - SATD keeps the x86 choice per tile pair: i16 for samples below 2048
//!   (the per-column sums of four absolute values, at most 65504, widened
//!   by the pairwise add), i32 one tile a vector otherwise.
//!
//! One rung, compiled only with `+simd128`; the sweep that checks it
//! (`super::u16_sweep`) runs inside the module, from `examples/wasm_probe.rs`.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::distortion::{DistortionDsp, sad_scalar, satd_scalar, ssd_scalar};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut DistortionDsp<u16>) {
    d.sad = sad;
    d.satd = satd;
    d.ssd = ssd;
}

#[inline]
unsafe fn load8(p: *const u16) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

#[inline]
unsafe fn load4(p: *const u16) -> v128 {
    unsafe { v128_load64_zero(p as *const u64) }
}

#[inline]
fn absdiff(a: v128, b: v128) -> v128 {
    v128_or(u16x8_sub_sat(a, b), u16x8_sub_sat(b, a))
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

/// The four i32 lanes of `v`, summed with wrap.
#[inline]
fn lanes(v: v128) -> u32 {
    (i32x4_extract_lane::<0>(v) as u32)
        .wrapping_add(i32x4_extract_lane::<1>(v) as u32)
        .wrapping_add(i32x4_extract_lane::<2>(v) as u32)
        .wrapping_add(i32x4_extract_lane::<3>(v) as u32)
}

// ----------------------------------------------------------------------
// SAD
// ----------------------------------------------------------------------

unsafe fn sad_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = i32x4_splat(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut x = 0;
            while x + 8 <= w {
                acc = i32x4_add(acc, u32x4_extadd_pairwise_u16x8(absdiff(load8(ra.add(x)), load8(rb.add(x)))));
                x += 8;
            }
            if x + 4 <= w {
                acc = i32x4_add(acc, u32x4_extadd_pairwise_u16x8(absdiff(load4(ra.add(x)), load4(rb.add(x)))));
            }
        }
        // Wrapping, as the scalar sum's u32 does.
        lanes(acc)
    }
}

fn sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h == 0 {
        return sad_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

// ----------------------------------------------------------------------
// SSD
// ----------------------------------------------------------------------

unsafe fn ssd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u64 {
    unsafe {
        let mut acc = i64x2_splat(0);
        let mut sq = |d: v128| {
            let lo = u32x4_extmul_low_u16x8(d, d);
            let hi = u32x4_extmul_high_u16x8(d, d);
            acc = i64x2_add(acc, i64x2_add(u64x2_extend_low_u32x4(lo), u64x2_extend_high_u32x4(lo)));
            acc = i64x2_add(acc, i64x2_add(u64x2_extend_low_u32x4(hi), u64x2_extend_high_u32x4(hi)));
        };
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut x = 0;
            while x + 8 <= w {
                sq(absdiff(load8(ra.add(x)), load8(rb.add(x))));
                x += 8;
            }
            if x + 4 <= w {
                sq(absdiff(load4(ra.add(x)), load4(rb.add(x))));
            }
        }
        (i64x2_extract_lane::<0>(acc) as u64).wrapping_add(i64x2_extract_lane::<1>(acc) as u64)
    }
}

fn ssd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u64 {
    if w % 4 != 0 || h == 0 {
        return ssd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

// ----------------------------------------------------------------------
// SATD
// ----------------------------------------------------------------------

#[inline]
fn butterfly16(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let s0 = i16x8_add(r0, r3);
    let s1 = i16x8_add(r1, r2);
    let s2 = i16x8_sub(r1, r2);
    let s3 = i16x8_sub(r0, r3);
    [i16x8_add(s0, s1), i16x8_add(s3, s2), i16x8_sub(s0, s1), i16x8_sub(s3, s2)]
}

#[inline]
fn butterfly32(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let s0 = i32x4_add(r0, r3);
    let s1 = i32x4_add(r1, r2);
    let s2 = i32x4_sub(r1, r2);
    let s3 = i32x4_sub(r0, r3);
    [i32x4_add(s0, s1), i32x4_add(s3, s2), i32x4_sub(s0, s1), i32x4_sub(s3, s2)]
}

/// SATD of the two tiles of i16 differences in `r0..r3` (tile A low, B
/// high), for samples below 2048, as `[A, A, B, B]` rounded.
#[inline]
fn pair16(r0: v128, r1: v128, r2: v128, r3: v128) -> v128 {
    let [t0, t1, t2, t3] = butterfly16(r0, r1, r2, r3);
    let u0 = zip_lo16(t0, t1);
    let u1 = zip_lo16(t2, t3);
    let u2 = zip_hi16(t0, t1);
    let u3 = zip_hi16(t2, t3);
    let v0 = zip_lo32(u0, u1);
    let v1 = zip_hi32(u0, u1);
    let v2 = zip_lo32(u2, u3);
    let v3 = zip_hi32(u2, u3);
    let [w0, w1, w2, w3] = butterfly16(zip_lo64(v0, v2), zip_hi64(v0, v2), zip_lo64(v1, v3), zip_hi64(v1, v3));
    // Four absolute values a column: at most 65504, a u16.
    let s = i16x8_add(i16x8_add(i16x8_abs(w0), i16x8_abs(w1)), i16x8_add(i16x8_abs(w2), i16x8_abs(w3)));
    // [A01, A23, B01, B23] -> [A, A, B, B], then the tile rounding.
    let p = u32x4_extadd_pairwise_u16x8(s);
    let q = i32x4_add(p, i32x4_shuffle::<1, 0, 3, 2>(p, p));
    u32x4_shr(i32x4_add(q, i32x4_splat(1)), 1)
}

/// SATD of one tile of i32 differences, as four equal lanes, rounded.
#[inline]
fn tile32(r: [v128; 4]) -> v128 {
    let [t0, t1, t2, t3] = butterfly32(r[0], r[1], r[2], r[3]);
    let u0 = zip_lo32(t0, t1);
    let u1 = zip_lo32(t2, t3);
    let u2 = zip_hi32(t0, t1);
    let u3 = zip_hi32(t2, t3);
    let [w0, w1, w2, w3] = butterfly32(zip_lo64(u0, u1), zip_hi64(u0, u1), zip_lo64(u2, u3), zip_hi64(u2, u3));
    let s = i32x4_add(i32x4_add(i32x4_abs(w0), i32x4_abs(w1)), i32x4_add(i32x4_abs(w2), i32x4_abs(w3)));
    let x = i32x4_add(s, i32x4_shuffle::<2, 3, 0, 1>(s, s));
    let t = i32x4_add(x, i32x4_shuffle::<1, 0, 3, 2>(x, x));
    u32x4_shr(i32x4_add(t, i32x4_splat(1)), 1)
}

/// SATD of the two tiles whose sample rows are `ra` and `rb` (A in the low
/// four lanes, B in the high four), as `[A, A, B, B]`: in i16 if every sample
/// is below 2048, in i32 otherwise.
#[inline]
fn pair(ra: [v128; 4], rb: [v128; 4]) -> v128 {
    let seen = v128_or(v128_or(v128_or(ra[0], ra[1]), v128_or(ra[2], ra[3])), v128_or(v128_or(rb[0], rb[1]), v128_or(rb[2], rb[3])));
    if !v128_any_true(v128_and(seen, i16x8_splat(!2047))) {
        let d = |k: usize| i16x8_sub(ra[k], rb[k]);
        pair16(d(0), d(1), d(2), d(3))
    } else {
        let lo = std::array::from_fn(|k| i32x4_sub(u32x4_extend_low_u16x8(ra[k]), u32x4_extend_low_u16x8(rb[k])));
        let hi = std::array::from_fn(|k| i32x4_sub(u32x4_extend_high_u16x8(ra[k]), u32x4_extend_high_u16x8(rb[k])));
        zip_lo64(tile32(lo), tile32(hi))
    }
}

unsafe fn satd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = i32x4_splat(0);
        if w == 4 {
            let mut y = 0;
            while y + 8 <= h {
                let rows = |p: *const u16, s: usize| std::array::from_fn(|r| zip_lo64(load4(p.add((y + r) * s)), load4(p.add((y + r + 4) * s))));
                acc = i32x4_add(acc, pair(rows(a, sa), rows(b, sb)));
                y += 8;
            }
            if y < h {
                let rows = |p: *const u16, s: usize| std::array::from_fn(|r| load4(p.add((y + r) * s)));
                acc = i32x4_add(acc, pair(rows(a, sa), rows(b, sb)));
            }
        } else {
            let mut y = 0;
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x + 8 <= w {
                    let rows = |p: *const u16, s: usize| std::array::from_fn(|r| load8(p.add(r * s + x)));
                    acc = i32x4_add(acc, pair(rows(ra, sa), rows(rb, sb)));
                    x += 8;
                }
                if x < w {
                    let rows = |p: *const u16, s: usize| std::array::from_fn(|r| load4(p.add(r * s + x)));
                    acc = i32x4_add(acc, pair(rows(ra, sa), rows(rb, sb)));
                }
                y += 4;
            }
        }
        // Lanes are [A, A, B, B] sums: one of each.
        (i32x4_extract_lane::<0>(acc) as u32).wrapping_add(i32x4_extract_lane::<2>(acc) as u32)
    }
}

fn satd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h % 4 != 0 || h == 0 {
        return satd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}
