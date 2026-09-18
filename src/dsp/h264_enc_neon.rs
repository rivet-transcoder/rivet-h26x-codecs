//! AArch64 NEON versions of the H.264 forward transforms and quantisers —
//! [`super::h264_enc_x86`]'s 128-bit kernels, on the instructions this
//! architecture has for them.
//!
//! The transforms are that file's lane-wise butterflies on `int32x4_t`,
//! with the 4x4 transposes as `trn1` / `trn2` at 32 and then 64 bits. The
//! quantisers compute the reference's `(|c| * mf + offset) >> qbits` in 64
//! bits: `umull` / `umull2` are the four 32 x 32 products, `ushl` by a
//! negative count the logical shift, `uzp1` the low halves back in lane
//! order. The level is truncated to i16 the way the reference's casts
//! truncate (`xtn`, which keeps the low half rather than saturating), and
//! a negative multiplier or offset, or a shift past 62, is the
//! reference's.
//!
//! NEON is baseline on AArch64. Checked bit-exact by the tests below, which
//! run under arm64 emulation (Docker) as well as on arm64 hardware.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::h264_enc::{H264EncDsp, quant4_scalar, quant8_scalar};

/// Install the NEON kernels.
pub fn install(d: &mut H264EncDsp, cpu: Cpu) {
    if cpu.neon {
        d.fdct4 = fdct4;
        d.fdct8 = fdct8;
        d.hadamard4 = hadamard4;
        d.quant4 = quant4;
        d.quant8 = quant8;
    }
}

/// Transpose a 4x4 block of i32, one row a vector.
#[inline(always)]
unsafe fn transpose4(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let t0 = vtrn1q_s32(r[0], r[1]);
        let t1 = vtrn2q_s32(r[0], r[1]);
        let t2 = vtrn1q_s32(r[2], r[3]);
        let t3 = vtrn2q_s32(r[2], r[3]);
        let w = |v: int32x4_t| vreinterpretq_s64_s32(v);
        let n = |v: int64x2_t| vreinterpretq_s32_s64(v);
        [n(vtrn1q_s64(w(t0), w(t2))), n(vtrn1q_s64(w(t1), w(t3))), n(vtrn2q_s64(w(t0), w(t2))), n(vtrn2q_s64(w(t1), w(t3)))]
    }
}

/// `fdct4_1d`, lane-wise.
#[inline(always)]
unsafe fn fdct4_lanes(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let s0 = vaddq_s32(x[0], x[3]);
        let s1 = vaddq_s32(x[1], x[2]);
        let s2 = vsubq_s32(x[1], x[2]);
        let s3 = vsubq_s32(x[0], x[3]);
        [vaddq_s32(s0, s1), vaddq_s32(vaddq_s32(s3, s3), s2), vsubq_s32(s0, s1), vsubq_s32(s3, vaddq_s32(s2, s2))]
    }
}

/// `had4_1d`, lane-wise.
#[inline(always)]
unsafe fn had4_lanes(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let s0 = vaddq_s32(x[0], x[3]);
        let s1 = vaddq_s32(x[1], x[2]);
        let s2 = vsubq_s32(x[1], x[2]);
        let s3 = vsubq_s32(x[0], x[3]);
        [vaddq_s32(s0, s1), vaddq_s32(s3, s2), vsubq_s32(s0, s1), vsubq_s32(s3, s2)]
    }
}

fn fdct4(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
    unsafe {
        let p = residual.as_ptr();
        let row = |i: usize| vmovl_s16(vld1_s16(p.add(4 * i)));
        // Columns as vectors for the row pass, rows for the column pass.
        let t = fdct4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = fdct4_lanes(transpose4(t));
        let q = coeffs.as_mut_ptr();
        for (i, v) in o.iter().enumerate() {
            vst1q_s32(q.add(4 * i), *v);
        }
    }
}

fn hadamard4(dc: &mut [i32; 16]) {
    unsafe {
        let p = dc.as_mut_ptr();
        let row = |i: usize| vld1q_s32(p.add(4 * i));
        let t = had4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = had4_lanes(transpose4(t));
        for (i, v) in o.iter().enumerate() {
            vst1q_s32(p.add(4 * i), *v);
        }
    }
}

/// `fdct8_1d`, lane-wise.
#[inline(always)]
unsafe fn fdct8_lanes(x: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    unsafe {
        let a0 = vaddq_s32(x[0], x[7]);
        let a1 = vaddq_s32(x[1], x[6]);
        let a2 = vaddq_s32(x[2], x[5]);
        let a3 = vaddq_s32(x[3], x[4]);
        let a4 = vsubq_s32(x[0], x[7]);
        let a5 = vsubq_s32(x[1], x[6]);
        let a6 = vsubq_s32(x[2], x[5]);
        let a7 = vsubq_s32(x[3], x[4]);
        let b0 = vaddq_s32(a0, a3);
        let b1 = vaddq_s32(a1, a2);
        let b2 = vsubq_s32(a0, a3);
        let b3 = vsubq_s32(a1, a2);
        let h = |v: int32x4_t| vaddq_s32(vshrq_n_s32::<1>(v), v);
        let b4 = vaddq_s32(vaddq_s32(a5, a6), h(a4));
        let b5 = vsubq_s32(vsubq_s32(a4, a7), h(a6));
        let b6 = vsubq_s32(vaddq_s32(a4, a7), h(a5));
        let b7 = vaddq_s32(vsubq_s32(a5, a6), h(a7));
        [
            vaddq_s32(b0, b1),
            vaddq_s32(b4, vshrq_n_s32::<2>(b7)),
            vaddq_s32(b2, vshrq_n_s32::<1>(b3)),
            vaddq_s32(b5, vshrq_n_s32::<2>(b6)),
            vsubq_s32(b0, b1),
            vsubq_s32(b6, vshrq_n_s32::<2>(b5)),
            vsubq_s32(vshrq_n_s32::<1>(b2), b3),
            vsubq_s32(vshrq_n_s32::<2>(b4), b7),
        ]
    }
}

/// The 8x8 transpose of `q[half][k]` (row `4 * half + lane`, column `k`)
/// into the same layout with rows and columns swapped.
#[inline(always)]
unsafe fn transpose8x8(q: &[[int32x4_t; 8]; 2]) -> [[int32x4_t; 8]; 2] {
    unsafe {
        let mut out = [[vdupq_n_s32(0); 8]; 2];
        for rh in 0..2 {
            for cb in 0..2 {
                let t = transpose4([q[rh][4 * cb], q[rh][4 * cb + 1], q[rh][4 * cb + 2], q[rh][4 * cb + 3]]);
                out[cb][4 * rh..4 * rh + 4].copy_from_slice(&t);
            }
        }
        out
    }
}

fn fdct8(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
    unsafe {
        let p = residual.as_ptr();
        // `c[half][k]`: column `k` of rows `4 * half..`, sign-extended.
        let mut c = [[vdupq_n_s32(0); 8]; 2];
        for (half, ch) in c.iter_mut().enumerate() {
            let rows: [int16x8_t; 4] = std::array::from_fn(|i| vld1q_s16(p.add(8 * (4 * half + i))));
            let lo = transpose4(std::array::from_fn(|i| vmovl_s16(vget_low_s16(rows[i]))));
            let hi = transpose4(std::array::from_fn(|i| vmovl_high_s16(rows[i])));
            ch[..4].copy_from_slice(&lo);
            ch[4..].copy_from_slice(&hi);
        }
        let t = [fdct8_lanes(&c[0]), fdct8_lanes(&c[1])];
        let u = transpose8x8(&t);
        let o = [fdct8_lanes(&u[0]), fdct8_lanes(&u[1])];
        let q = coeffs.as_mut_ptr();
        for (i, (lo, hi)) in o[0].iter().zip(&o[1]).enumerate() {
            vst1q_s32(q.add(8 * i), *lo);
            vst1q_s32(q.add(8 * i + 4), *hi);
        }
    }
}

/// Quantise four coefficients: the levels, still i32.
#[inline(always)]
unsafe fn quant_lanes(c: int32x4_t, mf: uint32x4_t, off: uint64x2_t, sh: int64x2_t) -> int32x4_t {
    unsafe {
        // `abs` of i32::MIN wraps to 0x8000_0000: 2^31 as u32, `unsigned_abs`.
        let a = vreinterpretq_u32_s32(vabsq_s32(c));
        let lo = vshlq_u64(vaddq_u64(vmull_u32(vget_low_u32(a), vget_low_u32(mf)), off), sh);
        let hi = vshlq_u64(vaddq_u64(vmull_high_u32(a, mf), off), sh);
        // The low 32 bits of each 64-bit result, in lane order.
        let m = vreinterpretq_s32_u32(vuzp1q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi)));
        let s = vshrq_n_s32::<31>(c);
        vsubq_s32(veorq_s32(m, s), s)
    }
}

/// `None` when a multiplier is negative (not a u32 one): the caller redoes
/// the block in the reference.
unsafe fn quant_impl(coeffs: *const i32, levels: *mut i16, mf: *const i32, n: usize, qbits: u32, offset: i32) -> Option<u32> {
    unsafe {
        let off = vdupq_n_u64(offset as u64);
        let sh = vdupq_n_s64(-(qbits as i64));
        let mut signs = vdupq_n_s32(0);
        let mut zeros = vdupq_n_u32(0);
        let mut i = 0;
        while i < n {
            let (m0, m1) = (vld1q_s32(mf.add(i)), vld1q_s32(mf.add(i + 4)));
            signs = vorrq_s32(signs, vorrq_s32(m0, m1));
            let v0 = quant_lanes(vld1q_s32(coeffs.add(i)), vreinterpretq_u32_s32(m0), off, sh);
            let v1 = quant_lanes(vld1q_s32(coeffs.add(i + 4)), vreinterpretq_u32_s32(m1), off, sh);
            // `xtn` keeps each lane's low half: the reference's `as i16`.
            vst1q_s16(levels.add(i), vcombine_s16(vmovn_s32(v0), vmovn_s32(v1)));
            // One per zero level: subtract the all-ones masks.
            zeros = vsubq_u32(zeros, vceqzq_s32(v0));
            zeros = vsubq_u32(zeros, vceqzq_s32(v1));
            i += 8;
        }
        if vmaxvq_u32(vreinterpretq_u32_s32(vshrq_n_s32::<31>(signs))) != 0 { None } else { Some(n as u32 - vaddvq_u32(zeros)) }
    }
}

fn quant4(coeffs: &[i32; 16], levels: &mut [i16; 16], mf: &[i32; 16], qbits: u32, offset: i32) -> u32 {
    if offset < 0 || qbits > 62 {
        return quant4_scalar(coeffs, levels, mf, qbits, offset);
    }
    match unsafe { quant_impl(coeffs.as_ptr(), levels.as_mut_ptr(), mf.as_ptr(), 16, qbits, offset) } {
        Some(nz) => nz,
        None => quant4_scalar(coeffs, levels, mf, qbits, offset),
    }
}

fn quant8(coeffs: &[i32; 64], levels: &mut [i16; 64], mf: &[i32; 64], qbits: u32, offset: i32) -> u32 {
    if offset < 0 || qbits > 62 {
        return quant8_scalar(coeffs, levels, mf, qbits, offset);
    }
    match unsafe { quant_impl(coeffs.as_ptr(), levels.as_mut_ptr(), mf.as_ptr(), 64, qbits, offset) } {
        Some(nz) => nz,
        None => quant8_scalar(coeffs, levels, mf, qbits, offset),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::h264_enc::{Quant, qbits4, qbits8, quant_offset};
    use crate::h264::sps::ScalingLists;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// The NEON table. NEON is baseline on AArch64, so this cannot be
    /// vacuous on the architecture it compiles for.
    fn neon() -> H264EncDsp {
        let mut d = H264EncDsp::SCALAR;
        install(&mut d, Cpu { neon: true, ..Cpu::SCALAR });
        assert!(d.quant4 as usize != H264EncDsp::SCALAR.quant4 as usize, "NEON installed nothing");
        d
    }

    #[test]
    fn transforms_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let d = neon();
        let mut seed = 0x4dc7_u64;
        for bit_depth in [8u32, 9, 10, 12, 14] {
            let span = 1i32 << bit_depth;
            for round in 0..300 {
                let mut r = || match lcg(&mut seed) % 8 {
                    0 => 32767i16,
                    1 => -32768,
                    2 => lcg(&mut seed) as i16,
                    _ => ((lcg(&mut seed) as i32 % (2 * span - 1)) - (span - 1)) as i16,
                };
                let r4: [i16; 16] = std::array::from_fn(|_| r());
                let r8: [i16; 64] = std::array::from_fn(|_| r());
                let (mut w4, mut g4, mut w8, mut g8) = ([0i32; 16], [0i32; 16], [0i32; 64], [0i32; 64]);
                (s.fdct4)(&r4, &mut w4);
                (d.fdct4)(&r4, &mut g4);
                (s.fdct8)(&r8, &mut w8);
                (d.fdct8)(&r8, &mut g8);
                assert_eq!(g4, w4, "fdct4, {bit_depth} bits, round {round}");
                assert_eq!(g8, w8, "fdct8, {bit_depth} bits, round {round}");
                let dc: [i32; 16] = std::array::from_fn(|i| w4[i] * if round % 2 == 0 { 1 } else { 16 });
                let (mut wh, mut gh) = (dc, dc);
                (s.hadamard4)(&mut wh);
                (d.hadamard4)(&mut gh);
                assert_eq!(gh, wh, "hadamard4, round {round}");
            }
        }
    }

    #[test]
    fn quantisers_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let d = neon();
        let mut seed = 0x9a41_u64;
        let flat = ScalingLists { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] };
        let small = ScalingLists { list4x4: [[4; 16]; 6], list8x8: [[4; 64]; 6] };
        for lists in [flat, small] {
            let q = Quant::new(&lists);
            for qp in 0..=87 {
                for intra in [true, false] {
                    let (qb4, qb8) = (qbits4(qp), qbits8(qp));
                    let (o4, o8) = (quant_offset(qb4, intra), quant_offset(qb8, intra));
                    let (mf4, mf8) = (&q.mf4[0][(qp % 6) as usize], &q.mf8[0][(qp % 6) as usize]);
                    for span in [255 * 36, 1023 * 64, 16383 * 256] {
                        let mut c = || match lcg(&mut seed) % 16 {
                            0 => 0,
                            1 => i32::MAX,
                            2 => i32::MIN + 1,
                            5 => lcg(&mut seed) as i32,
                            _ => (lcg(&mut seed) as i32 % (2 * span)) - span,
                        };
                        let c4: [i32; 16] = std::array::from_fn(|_| c());
                        let c8: [i32; 64] = std::array::from_fn(|_| c());
                        let (mut w4, mut g4, mut w8, mut g8) = ([0i16; 16], [0i16; 16], [0i16; 64], [0i16; 64]);
                        let nw4 = (s.quant4)(&c4, &mut w4, mf4, qb4, o4);
                        let ng4 = (d.quant4)(&c4, &mut g4, mf4, qb4, o4);
                        let nw8 = (s.quant8)(&c8, &mut w8, mf8, qb8, o8);
                        let ng8 = (d.quant8)(&c8, &mut g8, mf8, qb8, o8);
                        assert_eq!((g4, ng4), (w4, nw4), "quant4 qp {qp} intra {intra} span {span}");
                        assert_eq!((g8, ng8), (w8, nw8), "quant8 qp {qp} intra {intra} span {span}");
                    }
                }
            }
        }
        // The calls handed back to the reference.
        let c: [i32; 16] = std::array::from_fn(|i| i as i32 * 977 - 7000);
        let mut neg = [13107i32; 16];
        neg[5] = -3;
        for (mf, qb, off) in [(neg, 15, 100), ([13107; 16], 15, -5), ([13107; 16], 63, 0)] {
            let (mut want, mut got) = ([0i16; 16], [0i16; 16]);
            let nw = (s.quant4)(&c, &mut want, &mf, qb, off);
            let ng = (d.quant4)(&c, &mut got, &mf, qb, off);
            assert_eq!((got, ng), (want, nw), "fallback qbits {qb} offset {off}");
        }
    }
}
