//! 128-bit SIMD versions of the H.264 forward transforms and quantisers for
//! WebAssembly — [`super::h264_enc_x86`]'s 128-bit kernels on `simd128`.
//!
//! The transforms are that file's lane-wise butterflies with a transpose on
//! each side of the first pass, on `i32x4`. The quantisers compute the
//! reference's `(|c| * mf + offset) >> qbits` in 64 bits as x86 does, and
//! more directly: `u64x2_extmul_{low,high}_u32x4` are the four 32 x 32
//! products, `u64x2_shr` the shift, and one shuffle takes each product's
//! low 32 bits back into lane order. The level is truncated to i16 the way
//! the reference's casts truncate (a shift pair per lane before the
//! saturating narrow); a negative multiplier or offset, or a shift past
//! 62, is the reference's.
//!
//! One rung, compiled only with `+simd128`; `H26X_NO_SIMD=1` still selects
//! the scalar reference, which is what `tools/wasm.sh` compares against.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::h264_enc::{H264EncDsp, quant4_scalar, quant8_scalar};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut H264EncDsp) {
    d.fdct4 = fdct4;
    d.fdct8 = fdct8;
    d.hadamard4 = hadamard4;
    d.quant4 = quant4;
    d.quant8 = quant8;
}

/// Transpose a 4x4 block of i32, one row a vector.
#[inline]
fn transpose4(r: [v128; 4]) -> [v128; 4] {
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(r[0], r[1]);
    let t1 = i32x4_shuffle::<0, 4, 1, 5>(r[2], r[3]);
    let t2 = i32x4_shuffle::<2, 6, 3, 7>(r[0], r[1]);
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(r[2], r[3]);
    [
        i64x2_shuffle::<0, 2>(t0, t1),
        i64x2_shuffle::<1, 3>(t0, t1),
        i64x2_shuffle::<0, 2>(t2, t3),
        i64x2_shuffle::<1, 3>(t2, t3),
    ]
}

/// `fdct4_1d`, lane-wise.
#[inline]
fn fdct4_lanes(x: [v128; 4]) -> [v128; 4] {
    let s0 = i32x4_add(x[0], x[3]);
    let s1 = i32x4_add(x[1], x[2]);
    let s2 = i32x4_sub(x[1], x[2]);
    let s3 = i32x4_sub(x[0], x[3]);
    [
        i32x4_add(s0, s1),
        i32x4_add(i32x4_add(s3, s3), s2),
        i32x4_sub(s0, s1),
        i32x4_sub(s3, i32x4_add(s2, s2)),
    ]
}

/// `had4_1d`, lane-wise.
#[inline]
fn had4_lanes(x: [v128; 4]) -> [v128; 4] {
    let s0 = i32x4_add(x[0], x[3]);
    let s1 = i32x4_add(x[1], x[2]);
    let s2 = i32x4_sub(x[1], x[2]);
    let s3 = i32x4_sub(x[0], x[3]);
    [
        i32x4_add(s0, s1),
        i32x4_add(s3, s2),
        i32x4_sub(s0, s1),
        i32x4_sub(s3, s2),
    ]
}

fn fdct4(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
    unsafe {
        let p = residual.as_ptr();
        let row = |i: usize| i32x4_extend_low_i16x8(v128_load64_zero(p.add(4 * i) as *const u64));
        // Columns as vectors for the row pass, rows for the column pass.
        let t = fdct4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = fdct4_lanes(transpose4(t));
        let q = coeffs.as_mut_ptr();
        for (i, v) in o.iter().enumerate() {
            v128_store(q.add(4 * i) as *mut v128, *v);
        }
    }
}

fn hadamard4(dc: &mut [i32; 16]) {
    unsafe {
        let p = dc.as_mut_ptr();
        let row = |i: usize| v128_load(p.add(4 * i) as *const v128);
        let t = had4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = had4_lanes(transpose4(t));
        for (i, v) in o.iter().enumerate() {
            v128_store(p.add(4 * i) as *mut v128, *v);
        }
    }
}

/// `fdct8_1d`, lane-wise.
#[inline]
fn fdct8_lanes(x: &[v128; 8]) -> [v128; 8] {
    let a0 = i32x4_add(x[0], x[7]);
    let a1 = i32x4_add(x[1], x[6]);
    let a2 = i32x4_add(x[2], x[5]);
    let a3 = i32x4_add(x[3], x[4]);
    let a4 = i32x4_sub(x[0], x[7]);
    let a5 = i32x4_sub(x[1], x[6]);
    let a6 = i32x4_sub(x[2], x[5]);
    let a7 = i32x4_sub(x[3], x[4]);
    let b0 = i32x4_add(a0, a3);
    let b1 = i32x4_add(a1, a2);
    let b2 = i32x4_sub(a0, a3);
    let b3 = i32x4_sub(a1, a2);
    let h = |v: v128| i32x4_add(i32x4_shr(v, 1), v);
    let b4 = i32x4_add(i32x4_add(a5, a6), h(a4));
    let b5 = i32x4_sub(i32x4_sub(a4, a7), h(a6));
    let b6 = i32x4_sub(i32x4_add(a4, a7), h(a5));
    let b7 = i32x4_add(i32x4_sub(a5, a6), h(a7));
    [
        i32x4_add(b0, b1),
        i32x4_add(b4, i32x4_shr(b7, 2)),
        i32x4_add(b2, i32x4_shr(b3, 1)),
        i32x4_add(b5, i32x4_shr(b6, 2)),
        i32x4_sub(b0, b1),
        i32x4_sub(b6, i32x4_shr(b5, 2)),
        i32x4_sub(i32x4_shr(b2, 1), b3),
        i32x4_sub(i32x4_shr(b4, 2), b7),
    ]
}

/// The 8x8 transpose of `q[half][k]` (row `4 * half + lane`, column `k`)
/// into the same layout with rows and columns swapped.
#[inline]
fn transpose8x8(q: &[[v128; 8]; 2]) -> [[v128; 8]; 2] {
    let mut out = [[i32x4_splat(0); 8]; 2];
    for rh in 0..2 {
        for cb in 0..2 {
            let t = transpose4([q[rh][4 * cb], q[rh][4 * cb + 1], q[rh][4 * cb + 2], q[rh][4 * cb + 3]]);
            out[cb][4 * rh..4 * rh + 4].copy_from_slice(&t);
        }
    }
    out
}

fn fdct8(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
    unsafe {
        let p = residual.as_ptr();
        // `c[half][k]`: column `k` of rows `4 * half..`, sign-extended.
        let mut c = [[i32x4_splat(0); 8]; 2];
        for (half, ch) in c.iter_mut().enumerate() {
            let rows: [v128; 4] = std::array::from_fn(|i| v128_load(p.add(8 * (4 * half + i)) as *const v128));
            let lo = transpose4(std::array::from_fn(|i| i32x4_extend_low_i16x8(rows[i])));
            let hi = transpose4(std::array::from_fn(|i| i32x4_extend_high_i16x8(rows[i])));
            ch[..4].copy_from_slice(&lo);
            ch[4..].copy_from_slice(&hi);
        }
        let t = [fdct8_lanes(&c[0]), fdct8_lanes(&c[1])];
        let u = transpose8x8(&t);
        let o = [fdct8_lanes(&u[0]), fdct8_lanes(&u[1])];
        let q = coeffs.as_mut_ptr();
        for (i, (lo, hi)) in o[0].iter().zip(&o[1]).enumerate() {
            v128_store(q.add(8 * i) as *mut v128, *lo);
            v128_store(q.add(8 * i + 4) as *mut v128, *hi);
        }
    }
}

/// Quantise four coefficients: the levels, still i32.
#[inline]
fn quant_lanes(c: v128, mf: v128, off: v128, qbits: u32) -> v128 {
    let a = i32x4_abs(c);
    let lo = u64x2_shr(i64x2_add(u64x2_extmul_low_u32x4(a, mf), off), qbits);
    let hi = u64x2_shr(i64x2_add(u64x2_extmul_high_u32x4(a, mf), off), qbits);
    // The low 32 bits of each 64-bit result, in lane order.
    let m = i32x4_shuffle::<0, 2, 4, 6>(lo, hi);
    let s = i32x4_shr(c, 31);
    i32x4_sub(v128_xor(m, s), s)
}

/// Eight i32 levels to eight i16, truncating as `as i16` does.
#[inline]
fn narrow(a: v128, b: v128) -> v128 {
    let t = |v: v128| i32x4_shr(i32x4_shl(v, 16), 16);
    i16x8_narrow_i32x4(t(a), t(b))
}

/// `None` when a multiplier is negative (not a u32 one): the caller redoes
/// the block in the reference.
unsafe fn quant_impl(coeffs: *const i32, levels: *mut i16, mf: *const i32, n: usize, qbits: u32, offset: i32) -> Option<u32> {
    unsafe {
        let off = i64x2_splat(offset as i64);
        let zero = i32x4_splat(0);
        let mut signs = zero;
        let mut zeros = 0u32;
        let mut i = 0;
        while i < n {
            let ld = |k: usize| v128_load(coeffs.add(i + k) as *const v128);
            let lm = |k: usize| v128_load(mf.add(i + k) as *const v128);
            let (m0, m1) = (lm(0), lm(4));
            signs = v128_or(signs, v128_or(m0, m1));
            let v0 = quant_lanes(ld(0), m0, off, qbits);
            let v1 = quant_lanes(ld(4), m1, off, qbits);
            v128_store(levels.add(i) as *mut v128, narrow(v0, v1));
            zeros += (i32x4_bitmask(i32x4_eq(v0, zero)) | (i32x4_bitmask(i32x4_eq(v1, zero)) << 4)).count_ones();
            i += 8;
        }
        if i32x4_bitmask(signs) != 0 { None } else { Some(n as u32 - zeros) }
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
