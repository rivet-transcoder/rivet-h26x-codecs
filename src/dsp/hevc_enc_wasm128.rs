//! 128-bit SIMD versions of the H.265 forward transforms and quantiser for
//! WebAssembly — [`super::hevc_enc_x86`]'s 128-bit kernels on `simd128`.
//!
//! The forward transform is that file's two `pmaddwd` shapes exactly, on
//! `i32x4_dot_i16x8`, reading the same pair table (`hevc_enc::fp`, built
//! from the decoder's matrix): stage 1 broadcasts an input pair against
//! eight consecutive outputs' weights and stores its results interleaved
//! in row pairs, which is the operand stage 2 reads with the weight pair
//! broadcast instead. The roundings are the reference's `(sum + (1 << (s
//! - 1))) >> s` as an arithmetic shift (`i32x4_shr`), the clamp to i16 is
//! `i16x8_narrow_i32x4`'s saturation, and every product is `i16 x i16`
//! summed in i32 — at most 32 terms of `90 x 32767`, under 2^27.
//!
//! The quantiser differs in one instruction. wasm has no `pmulhuw`, so the
//! exact 32-bit product of the u16 magnitude and the u16 scale is not two
//! halves but a widening (`u32x4_extend_low/high_u16x8`) and an
//! `i32x4_mul`, whose low 32 bits are the whole product because it is at
//! most `32768 x 32767 < 2^31`. The magnitude of `-32768` is `0x8000`,
//! which the widening reads as 32768, so the one value an i16 absolute
//! cannot represent is still right. The shift is logical (`u32x4_shr`) on
//! a value nonnegative by construction; the sign goes back with `(m ^ s)
//! - s`; the nonzero count is `8 - popcount(i16x8_bitmask(v == 0))`.
//!
//! One rung, compiled only with `+simd128`; `H26X_NO_SIMD=1` still selects
//! the scalar reference, which is what `tools/wasm.sh` compares against.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::hevc_enc::layouts::{FDST, fp};
use super::hevc_enc::{HevcEncDsp, fdct_scalar, fdst4_scalar, quant_scalar};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut HevcEncDsp) {
    d.fdct = [fdct::<4>, fdct::<8>, fdct::<16>, fdct::<32>];
    d.fdst4 = fdst4;
    d.quant = quant;
}

/// `unpacklo_epi16` / `unpackhi_epi16`.
#[inline]
fn zip_lo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}
#[inline]
fn zip_hi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}

/// The 32-bit lane at `p` (an adjacent i16 pair), broadcast.
#[inline]
unsafe fn splat_pair(p: *const i16) -> v128 {
    unsafe { i32x4_splat((p as *const i32).read_unaligned()) }
}

/// Both stages of an `N`-point forward transform over `block` (raster,
/// in place), with `table` the pair table of its matrix. `s1 >= 1`.
unsafe fn fwd_impl<const N: usize>(block: *mut i16, s1: u32, s2: u32, table: *const i16) {
    unsafe {
        // Stage 1's output, rows interleaved in pairs.
        let mut inter = [0i16; 32 * 32];
        let r1 = i32x4_splat(1 << (s1 - 1));
        let r2 = i32x4_splat(1 << (s2 - 1));
        let np = N / 2;
        for p in 0..np {
            let re = block.add(2 * p * N);
            let ro = block.add((2 * p + 1) * N);
            let mut j = 0;
            while j < N {
                let (mut e0, mut e1, mut o0, mut o1) = (r1, r1, r1, r1);
                for q in 0..np {
                    let te = splat_pair(re.add(2 * q));
                    let to = splat_pair(ro.add(2 * q));
                    let c0 = v128_load(table.add(q * 2 * N + 2 * j) as *const v128);
                    e0 = i32x4_add(e0, i32x4_dot_i16x8(c0, te));
                    o0 = i32x4_add(o0, i32x4_dot_i16x8(c0, to));
                    if N > 4 {
                        let c1 = v128_load(table.add(q * 2 * N + 2 * j + 8) as *const v128);
                        e1 = i32x4_add(e1, i32x4_dot_i16x8(c1, te));
                        o1 = i32x4_add(o1, i32x4_dot_i16x8(c1, to));
                    }
                }
                let ve = i16x8_narrow_i32x4(i32x4_shr(e0, s1), i32x4_shr(e1, s1));
                let vo = i16x8_narrow_i32x4(i32x4_shr(o0, s1), i32x4_shr(o1, s1));
                let dst = inter.as_mut_ptr().add(p * 2 * N + 2 * j);
                v128_store(dst as *mut v128, zip_lo16(ve, vo));
                if N > 4 {
                    v128_store(dst.add(8) as *mut v128, zip_hi16(ve, vo));
                }
                j += 8;
            }
        }
        for j in 0..N {
            let mut x = 0;
            while x < N {
                let (mut a0, mut a1) = (r2, r2);
                for q in 0..np {
                    let c = splat_pair(table.add(q * 2 * N + 2 * j));
                    let src = inter.as_ptr().add(q * 2 * N + 2 * x);
                    a0 = i32x4_add(a0, i32x4_dot_i16x8(v128_load(src as *const v128), c));
                    if N > 4 {
                        a1 = i32x4_add(a1, i32x4_dot_i16x8(v128_load(src.add(8) as *const v128), c));
                    }
                }
                let v = i16x8_narrow_i32x4(i32x4_shr(a0, s2), i32x4_shr(a1, s2));
                let dst = block.add(j * N + x);
                if N > 4 {
                    v128_store(dst as *mut v128, v);
                } else {
                    v128_store64_lane::<0>(v, dst as *mut u64);
                }
                x += 8;
            }
        }
    }
}

fn fdct<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
    debug_assert_eq!(1usize << log2, N);
    let s1 = log2 as i32 + bit_depth as i32 - 9;
    let s2 = log2 + 6;
    // A first-stage shift of zero has no rounding term; only a bit depth
    // below eight gets there, and the reference handles it.
    if s1 <= 0 {
        return fdct_scalar::<N>(block, log2, bit_depth);
    }
    assert!(block.len() >= N * N, "block too small");
    unsafe { fwd_impl::<N>(block.as_mut_ptr(), s1 as u32, s2, fp::<N>().as_ptr()) }
}

fn fdst4(block: &mut [i16], bit_depth: u32) {
    let s1 = 2 + bit_depth as i32 - 9;
    if s1 <= 0 {
        return fdst4_scalar(block, bit_depth);
    }
    assert!(block.len() >= 16, "block too small");
    unsafe { fwd_impl::<4>(block.as_mut_ptr(), s1 as u32, 8, FDST.as_ptr()) }
}

unsafe fn quant_impl(coeffs: *const i16, levels: *mut i16, n2: usize, scale: i32, qbits: u32, offset: i32) -> u32 {
    unsafe {
        let vs = i32x4_splat(scale);
        let vo = i32x4_splat(offset);
        let zero = i16x8_splat(0);
        let mut nz = 0u32;
        let mut i = 0;
        while i < n2 {
            let c = v128_load(coeffs.add(i) as *const v128);
            let a = i16x8_abs(c);
            let m0 = u32x4_shr(i32x4_add(i32x4_mul(u32x4_extend_low_u16x8(a), vs), vo), qbits);
            let m1 = u32x4_shr(i32x4_add(i32x4_mul(u32x4_extend_high_u16x8(a), vs), vo), qbits);
            let s = i16x8_shr(c, 15);
            let s0 = i32x4_extend_low_i16x8(s);
            let s1 = i32x4_extend_high_i16x8(s);
            let v0 = i32x4_sub(v128_xor(m0, s0), s0);
            let v1 = i32x4_sub(v128_xor(m1, s1), s1);
            let v = i16x8_narrow_i32x4(v0, v1);
            v128_store(levels.add(i) as *mut v128, v);
            let z = i16x8_bitmask(i16x8_eq(v, zero)) as u32;
            nz += 8 - z.count_ones();
            i += 8;
        }
        nz
    }
}

fn quant(coeffs: &[i16], levels: &mut [i16], n: usize, scale: i32, qbits: u32, offset: i32) -> u32 {
    let n2 = n * n;
    // The product needs `scale` in u16 and the sum in i32, which
    // `quant_scale` (at most 26214) and a shift of at least 14 guarantee;
    // anything else is the reference's.
    if n2 % 8 != 0 || !(0..=32767).contains(&scale) || offset < 0 || qbits > 31 {
        return quant_scalar(coeffs, levels, n, scale, qbits, offset);
    }
    assert!(coeffs.len() >= n2 && levels.len() >= n2, "block too small");
    unsafe { quant_impl(coeffs.as_ptr(), levels.as_mut_ptr(), n2, scale, qbits, offset) }
}
