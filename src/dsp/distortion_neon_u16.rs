//! AArch64 NEON versions of the distortion metrics for 16-bit samples — the
//! kernels of [`super::distortion_x86_u16`], exact for every `u16` input as
//! those are, on the instructions this architecture has for them. Two of the
//! three need none of the x86 file's folding tricks, because NEON widens:
//!
//! - SAD: `uabd` gives the absolute differences, `uadalp` folds them into a
//!   u32 accumulator — at most 2 · 65535 a lane per vector.
//! - SSD: `uabd` again, `umull` squares each half of the vector into u32
//!   (65535² fits), `uadalp` folds the squares into a u64 accumulator.
//! - SATD: the x86 file's choice per tile pair. Samples all below 2048
//!   (`umaxv` over the rows' or): the 8-bit NEON kernel's shape in i16, the
//!   per-column sums of four absolute values (at most 65504) widened by
//!   `uaddlp`. Otherwise: `usubl` into i32 lanes and the same transform one
//!   tile a vector.
//!
//! Written on x86 and checked for compilation against
//! `aarch64-unknown-linux-gnu`; the sweep below (`super::u16_sweep`) runs
//! on the CI arm64 runners.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::distortion::{DistortionDsp, sad_scalar, satd_scalar, ssd_scalar};

#[target_feature(enable = "neon")]
unsafe fn sad_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = vdupq_n_u32(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut x = 0;
            while x + 8 <= w {
                acc = vpadalq_u16(acc, vabdq_u16(vld1q_u16(ra.add(x)), vld1q_u16(rb.add(x))));
                x += 8;
            }
            if x + 4 <= w {
                acc = vaddq_u32(acc, vabdl_u16(vld1_u16(ra.add(x)), vld1_u16(rb.add(x))));
            }
        }
        // Wrapping, as the scalar sum's u32 does.
        vaddvq_u32(acc)
    }
}

pub(crate) fn sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h == 0 {
        return sad_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

#[target_feature(enable = "neon")]
unsafe fn ssd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u64 {
    unsafe {
        let mut acc = vdupq_n_u64(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut x = 0;
            while x + 8 <= w {
                let d = vabdq_u16(vld1q_u16(ra.add(x)), vld1q_u16(rb.add(x)));
                acc = vpadalq_u32(acc, vmull_u16(vget_low_u16(d), vget_low_u16(d)));
                acc = vpadalq_u32(acc, vmull_high_u16(d, d));
                x += 8;
            }
            if x + 4 <= w {
                let d = vabd_u16(vld1_u16(ra.add(x)), vld1_u16(rb.add(x)));
                acc = vpadalq_u32(acc, vmull_u16(d, d));
            }
        }
        vaddvq_u64(acc)
    }
}

pub(crate) fn ssd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u64 {
    if w % 4 != 0 || h == 0 {
        return ssd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

/// The 4-point Hadamard butterfly, lane-wise across four i16 vectors.
#[inline(always)]
unsafe fn butterfly16(r0: int16x8_t, r1: int16x8_t, r2: int16x8_t, r3: int16x8_t) -> [int16x8_t; 4] {
    unsafe {
        let s0 = vaddq_s16(r0, r3);
        let s1 = vaddq_s16(r1, r2);
        let s2 = vsubq_s16(r1, r2);
        let s3 = vsubq_s16(r0, r3);
        [vaddq_s16(s0, s1), vaddq_s16(s3, s2), vsubq_s16(s0, s1), vsubq_s16(s3, s2)]
    }
}

/// The same across four i32 vectors.
#[inline(always)]
unsafe fn butterfly32(r0: int32x4_t, r1: int32x4_t, r2: int32x4_t, r3: int32x4_t) -> [int32x4_t; 4] {
    unsafe {
        let s0 = vaddq_s32(r0, r3);
        let s1 = vaddq_s32(r1, r2);
        let s2 = vsubq_s32(r1, r2);
        let s3 = vsubq_s32(r0, r3);
        [vaddq_s32(s0, s1), vaddq_s32(s3, s2), vsubq_s32(s0, s1), vsubq_s32(s3, s2)]
    }
}

/// SATD of the two tiles of i16 differences in `r` (one row each, A low, B
/// high), for samples below 2048: `A + B`, each rounded.
#[inline(always)]
unsafe fn pair16(r: [int16x8_t; 4]) -> u32 {
    unsafe {
        let [t0, t1, t2, t3] = butterfly16(r[0], r[1], r[2], r[3]);
        // Transpose each 4x4 tile: 16-bit pairs, then 32-bit pairs.
        let a0 = vtrn1q_s16(t0, t1);
        let a1 = vtrn2q_s16(t0, t1);
        let a2 = vtrn1q_s16(t2, t3);
        let a3 = vtrn2q_s16(t2, t3);
        let c0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2)));
        let c2 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2)));
        let c1 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3)));
        let c3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3)));
        let [w0, w1, w2, w3] = butterfly16(c0, c1, c2, c3);
        let u = |x: int16x8_t| vreinterpretq_u16_s16(vabsq_s16(x));
        // Four absolute values a column: at most 65504, a u16.
        let s = vaddq_u16(vaddq_u16(u(w0), u(w1)), vaddq_u16(u(w2), u(w3)));
        // [A01, A23, B01, B23] -> [A, B, A, B], then the tile rounding.
        let p = vpaddlq_u16(s);
        let q = vshrq_n_u32::<1>(vaddq_u32(vpaddq_u32(p, p), vdupq_n_u32(1)));
        vgetq_lane_u32::<0>(q).wrapping_add(vgetq_lane_u32::<1>(q))
    }
}

/// Transpose a 4x4 block of i32 held as four row vectors.
#[inline(always)]
unsafe fn transpose4(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let a = vtrnq_s32(r[0], r[1]);
        let b = vtrnq_s32(r[2], r[3]);
        [
            vcombine_s32(vget_low_s32(a.0), vget_low_s32(b.0)),
            vcombine_s32(vget_low_s32(a.1), vget_low_s32(b.1)),
            vcombine_s32(vget_high_s32(a.0), vget_high_s32(b.0)),
            vcombine_s32(vget_high_s32(a.1), vget_high_s32(b.1)),
        ]
    }
}

/// SATD of one tile of i32 differences (four rows of four), rounded: exact
/// for any u16 samples, whose coefficients reach 16 · 65535.
#[inline(always)]
unsafe fn tile32(r: [int32x4_t; 4]) -> u32 {
    unsafe {
        let [t0, t1, t2, t3] = butterfly32(r[0], r[1], r[2], r[3]);
        let [c0, c1, c2, c3] = transpose4([t0, t1, t2, t3]);
        let [w0, w1, w2, w3] = butterfly32(c0, c1, c2, c3);
        let s = vaddq_s32(vaddq_s32(vabsq_s32(w0), vabsq_s32(w1)), vaddq_s32(vabsq_s32(w2), vabsq_s32(w3)));
        (vaddvq_s32(s) as u32 + 1) >> 1
    }
}

/// SATD of the two tiles whose sample rows are `ra` and `rb` (tile A in the
/// low four lanes, B in the high four): in i16 if every sample is below
/// 2048, in i32 otherwise.
#[inline(always)]
unsafe fn pair(ra: [uint16x8_t; 4], rb: [uint16x8_t; 4]) -> u32 {
    unsafe {
        let seen = vorrq_u16(
            vorrq_u16(vorrq_u16(ra[0], ra[1]), vorrq_u16(ra[2], ra[3])),
            vorrq_u16(vorrq_u16(rb[0], rb[1]), vorrq_u16(rb[2], rb[3])),
        );
        if vmaxvq_u16(seen) < 2048 {
            pair16(std::array::from_fn(|k| vreinterpretq_s16_u16(vsubq_u16(ra[k], rb[k]))))
        } else {
            let lo = std::array::from_fn(|k| vreinterpretq_s32_u32(vsubl_u16(vget_low_u16(ra[k]), vget_low_u16(rb[k]))));
            let hi = std::array::from_fn(|k| vreinterpretq_s32_u32(vsubl_high_u16(ra[k], rb[k])));
            tile32(lo).wrapping_add(tile32(hi))
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn satd_impl(a: *const u16, sa: usize, b: *const u16, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut total = 0u32;
        let zero = vdup_n_u16(0);
        if w == 4 {
            let mut y = 0;
            while y + 8 <= h {
                // Two tiles one above the other: rows y and y + 4 share a vector.
                let rows = |p: *const u16, s: usize| std::array::from_fn(|r| vcombine_u16(vld1_u16(p.add((y + r) * s)), vld1_u16(p.add((y + r + 4) * s))));
                total = total.wrapping_add(pair(rows(a, sa), rows(b, sb)));
                y += 8;
            }
            if y < h {
                let rows = |p: *const u16, s: usize| std::array::from_fn(|r| vcombine_u16(vld1_u16(p.add((y + r) * s)), zero));
                total = total.wrapping_add(pair(rows(a, sa), rows(b, sb)));
            }
        } else {
            let mut y = 0;
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x + 8 <= w {
                    let rows = |p: *const u16, s: usize| std::array::from_fn(|r| vld1q_u16(p.add(r * s + x)));
                    total = total.wrapping_add(pair(rows(ra, sa), rows(rb, sb)));
                    x += 8;
                }
                if x < w {
                    let rows = |p: *const u16, s: usize| std::array::from_fn(|r| vcombine_u16(vld1_u16(p.add(r * s + x)), zero));
                    total = total.wrapping_add(pair(rows(ra, sa), rows(rb, sb)));
                }
                y += 4;
            }
        }
        total
    }
}

pub(crate) fn satd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h % 4 != 0 || h == 0 {
        return satd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

/// Install the NEON kernels.
pub fn install(d: &mut DistortionDsp<u16>, cpu: Cpu) {
    if cpu.neon {
        d.sad = sad;
        d.satd = satd;
        d.ssd = ssd;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::u16_sweep;

    #[test]
    fn neon_u16_distortion_matches_scalar_at_every_depth() {
        let mut d = DistortionDsp::<u16>::scalar();
        install(&mut d, Cpu { neon: true, ..Cpu::SCALAR });
        match u16_sweep::distortion(&[("neon", d)]) {
            Ok(n) => assert!(n > 0, "the sweep compared nothing"),
            Err(e) => panic!("{e}"),
        }
    }
}
