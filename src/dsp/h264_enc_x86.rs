//! x86-64 SIMD versions of the H.264 forward transforms and quantisers.
//!
//! The profile that asked for these (1080p, one thread, 10 frames IP at QP
//! 27 with the 8x8 transform and sub-partitions) put the scalar `quant4`,
//! `quant8`, `fdct4` and `fdct8` at 6.9% of an 8-bit encode's self time
//! and 5.9% of a 10-bit one's.
//!
//! **The transforms** are the scalar reference's integer butterflies run on
//! i32 lanes, one lane per row (first pass) or per column (second pass),
//! with a transpose on each side of the first. Nothing is approximated: the
//! 4x4 transform and the Hadamard have no rounding at all, and the 8x8's
//! `>> 1` and `>> 2` are `psrad` on exactly the values the reference shifts.
//! Every intermediate is i32 as in the reference, which no i16 residual can
//! overflow.
//!
//! **The quantisers** compute the reference's `(|c| * mf + offset) >>
//! qbits` in 64 bits, as it does: `pmuludq` takes the even and odd lanes'
//! 32 x 32 products, and the shift is `psrlq` on values that are
//! nonnegative by construction. `|i32::MIN|` is `0x8000_0000` in a u32 lane,
//! which is what `unsigned_abs` gives it. The level keeps the low 32 bits
//! and then the low 16, as the reference's `as i32` / `as i16` casts do —
//! a shift pair truncates each lane before `packssdw`, whose saturation
//! would otherwise clamp instead — and `nz` counts the 32-bit values. A
//! negative multiplier or offset, or a shift past 62, is the reference's.
//!
//! The two chroma DC Hadamards are left scalar: four or eight values once
//! a macroblock is not a vector's worth of work.

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::h264_enc::H264EncDsp;

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::h264_enc::{H264EncDsp, quant4_scalar, quant8_scalar};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        /// Every kernel this rung carries.
        pub(crate) fn install_all(d: &mut H264EncDsp) {
            d.fdct4 = fdct4;
            d.fdct8 = fdct8;
            d.hadamard4 = hadamard4;
            d.quant4 = quant4;
            d.quant8 = quant8;
        }

        /// Transpose a 4x4 block of i32, one row a vector.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn transpose4(r: [__m128i; 4]) -> [__m128i; 4] {
            let t0 = _mm_unpacklo_epi32(r[0], r[1]);
            let t1 = _mm_unpacklo_epi32(r[2], r[3]);
            let t2 = _mm_unpackhi_epi32(r[0], r[1]);
            let t3 = _mm_unpackhi_epi32(r[2], r[3]);
            [
                _mm_unpacklo_epi64(t0, t1),
                _mm_unpackhi_epi64(t0, t1),
                _mm_unpacklo_epi64(t2, t3),
                _mm_unpackhi_epi64(t2, t3),
            ]
        }

        /// `fdct4_1d`, lane-wise.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fdct4_lanes(x: [__m128i; 4]) -> [__m128i; 4] {
            let s0 = _mm_add_epi32(x[0], x[3]);
            let s1 = _mm_add_epi32(x[1], x[2]);
            let s2 = _mm_sub_epi32(x[1], x[2]);
            let s3 = _mm_sub_epi32(x[0], x[3]);
            [
                _mm_add_epi32(s0, s1),
                _mm_add_epi32(_mm_add_epi32(s3, s3), s2),
                _mm_sub_epi32(s0, s1),
                _mm_sub_epi32(s3, _mm_add_epi32(s2, s2)),
            ]
        }

        /// `had4_1d`, lane-wise.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn had4_lanes(x: [__m128i; 4]) -> [__m128i; 4] {
            let s0 = _mm_add_epi32(x[0], x[3]);
            let s1 = _mm_add_epi32(x[1], x[2]);
            let s2 = _mm_sub_epi32(x[1], x[2]);
            let s3 = _mm_sub_epi32(x[0], x[3]);
            [
                _mm_add_epi32(s0, s1),
                _mm_add_epi32(s3, s2),
                _mm_sub_epi32(s0, s1),
                _mm_sub_epi32(s3, s2),
            ]
        }

        #[target_feature(enable = $feat)]
        unsafe fn fdct4_impl(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
            unsafe {
                let p = residual.as_ptr();
                let row = |i: usize| sx16(_mm_loadl_epi64(p.add(4 * i) as *const __m128i));
                // Columns as vectors: the row pass runs across them.
                let t = fdct4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
                // `t[k]` holds output `k` of every row; transposed, a vector
                // is a row of the intermediate and the column pass runs
                // across those.
                let o = fdct4_lanes(transpose4(t));
                let q = coeffs.as_mut_ptr();
                for (i, v) in o.iter().enumerate() {
                    _mm_storeu_si128(q.add(4 * i) as *mut __m128i, *v);
                }
            }
        }

        fn fdct4(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
            unsafe { fdct4_impl(residual, coeffs) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn hadamard4_impl(dc: &mut [i32; 16]) {
            unsafe {
                let p = dc.as_mut_ptr();
                let row = |i: usize| _mm_loadu_si128(p.add(4 * i) as *const __m128i);
                let t = had4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
                let o = had4_lanes(transpose4(t));
                for (i, v) in o.iter().enumerate() {
                    _mm_storeu_si128(p.add(4 * i) as *mut __m128i, *v);
                }
            }
        }

        fn hadamard4(dc: &mut [i32; 16]) {
            unsafe { hadamard4_impl(dc) }
        }

        /// `fdct8_1d`, lane-wise.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn fdct8_lanes(x: &[__m128i; 8]) -> [__m128i; 8] {
            let a0 = _mm_add_epi32(x[0], x[7]);
            let a1 = _mm_add_epi32(x[1], x[6]);
            let a2 = _mm_add_epi32(x[2], x[5]);
            let a3 = _mm_add_epi32(x[3], x[4]);
            let a4 = _mm_sub_epi32(x[0], x[7]);
            let a5 = _mm_sub_epi32(x[1], x[6]);
            let a6 = _mm_sub_epi32(x[2], x[5]);
            let a7 = _mm_sub_epi32(x[3], x[4]);
            let b0 = _mm_add_epi32(a0, a3);
            let b1 = _mm_add_epi32(a1, a2);
            let b2 = _mm_sub_epi32(a0, a3);
            let b3 = _mm_sub_epi32(a1, a2);
            let h = |v: __m128i| _mm_add_epi32(_mm_srai_epi32(v, 1), v);
            let b4 = _mm_add_epi32(_mm_add_epi32(a5, a6), h(a4));
            let b5 = _mm_sub_epi32(_mm_sub_epi32(a4, a7), h(a6));
            let b6 = _mm_sub_epi32(_mm_add_epi32(a4, a7), h(a5));
            let b7 = _mm_add_epi32(_mm_sub_epi32(a5, a6), h(a7));
            [
                _mm_add_epi32(b0, b1),
                _mm_add_epi32(b4, _mm_srai_epi32(b7, 2)),
                _mm_add_epi32(b2, _mm_srai_epi32(b3, 1)),
                _mm_add_epi32(b5, _mm_srai_epi32(b6, 2)),
                _mm_sub_epi32(b0, b1),
                _mm_sub_epi32(b6, _mm_srai_epi32(b5, 2)),
                _mm_sub_epi32(_mm_srai_epi32(b2, 1), b3),
                _mm_sub_epi32(_mm_srai_epi32(b4, 2), b7),
            ]
        }

        /// The 8x8 transpose of `q[half][k]` (row `4 * half + lane`,
        /// column `k`) into the same layout with rows and columns swapped.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn transpose8x8(q: &[[__m128i; 8]; 2]) -> [[__m128i; 8]; 2] {
            unsafe {
                let mut out = [[_mm_setzero_si128(); 8]; 2];
                // Four 4x4 blocks: block (rh, cb) holds rows 4rh.., columns
                // 4cb..; transposed, it lands at (cb, rh).
                for rh in 0..2 {
                    for cb in 0..2 {
                        let t = transpose4([
                            q[rh][4 * cb],
                            q[rh][4 * cb + 1],
                            q[rh][4 * cb + 2],
                            q[rh][4 * cb + 3],
                        ]);
                        out[cb][4 * rh..4 * rh + 4].copy_from_slice(&t);
                    }
                }
                out
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn fdct8_impl(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
            unsafe {
                let p = residual.as_ptr();
                // `c[half][k]`: column `k` of rows `4 * half..`, sign-extended.
                let mut c = [[_mm_setzero_si128(); 8]; 2];
                for half in 0..2 {
                    let r = |i: usize| _mm_loadu_si128(p.add(8 * (4 * half + i)) as *const __m128i);
                    let rows = [r(0), r(1), r(2), r(3)];
                    let lo =
                        transpose4([sx16(rows[0]), sx16(rows[1]), sx16(rows[2]), sx16(rows[3])]);
                    let hi = transpose4([
                        sx16h(rows[0]),
                        sx16h(rows[1]),
                        sx16h(rows[2]),
                        sx16h(rows[3]),
                    ]);
                    c[half][..4].copy_from_slice(&lo);
                    c[half][4..].copy_from_slice(&hi);
                }
                // Row pass: `t[half][k]` is output `k` of rows `4 * half..`.
                let t = [fdct8_lanes(&c[0]), fdct8_lanes(&c[1])];
                // Transposed, `u[half][i]` is row `i` of the intermediate at
                // columns `4 * half..`, and the column pass runs across rows.
                let u = transpose8x8(&t);
                let o = [fdct8_lanes(&u[0]), fdct8_lanes(&u[1])];
                let q = coeffs.as_mut_ptr();
                for i in 0..8 {
                    _mm_storeu_si128(q.add(8 * i) as *mut __m128i, o[0][i]);
                    _mm_storeu_si128(q.add(8 * i + 4) as *mut __m128i, o[1][i]);
                }
            }
        }

        fn fdct8(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
            unsafe { fdct8_impl(residual, coeffs) }
        }

        /// Quantise four coefficients: the levels, still i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn quant_lanes(c: __m128i, mf: __m128i, off: __m128i, sh: __m128i) -> __m128i {
            unsafe {
                let a = abs32(c);
                let even = _mm_srl_epi64(_mm_add_epi64(_mm_mul_epu32(a, mf), off), sh);
                let odd = _mm_srl_epi64(
                    _mm_add_epi64(
                        _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(mf, 32)),
                        off,
                    ),
                    sh,
                );
                // The low halves of [e0, e1] and [o0, o1], interleaved back
                // into lane order.
                let m = _mm_unpacklo_epi32(
                    _mm_shuffle_epi32(even, 0b10_00_10_00),
                    _mm_shuffle_epi32(odd, 0b10_00_10_00),
                );
                let s = _mm_srai_epi32(c, 31);
                _mm_sub_epi32(_mm_xor_si128(m, s), s)
            }
        }

        /// Eight i32 levels to eight i16, truncating as `as i16` does.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn narrow(a: __m128i, b: __m128i) -> __m128i {
            let t = |v: __m128i| _mm_srai_epi32(_mm_slli_epi32(v, 16), 16);
            _mm_packs_epi32(t(a), t(b))
        }

        #[target_feature(enable = $feat)]
        unsafe fn quant_impl(
            coeffs: *const i32,
            levels: *mut i16,
            mf: *const i32,
            n: usize,
            qbits: u32,
            offset: i32,
        ) -> Option<u32> {
            unsafe {
                let off = _mm_set1_epi64x(offset as i64);
                let sh = _mm_cvtsi32_si128(qbits as i32);
                let zero = _mm_setzero_si128();
                let mut signs = zero;
                let mut zeros = 0u32;
                let mut i = 0;
                while i < n {
                    let ld = |k: usize| _mm_loadu_si128(coeffs.add(i + k) as *const __m128i);
                    let lm = |k: usize| _mm_loadu_si128(mf.add(i + k) as *const __m128i);
                    let (m0, m1) = (lm(0), lm(4));
                    signs = _mm_or_si128(signs, _mm_or_si128(m0, m1));
                    let v0 = quant_lanes(ld(0), m0, off, sh);
                    let v1 = quant_lanes(ld(4), m1, off, sh);
                    _mm_storeu_si128(levels.add(i) as *mut __m128i, narrow(v0, v1));
                    let z = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v0, zero)))
                        | (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v1, zero))) << 4);
                    zeros += (z as u32).count_ones();
                    i += 8;
                }
                // A negative multiplier is not a u32 one: the caller redoes
                // the block in the reference.
                if _mm_movemask_ps(_mm_castsi128_ps(signs)) != 0 {
                    None
                } else {
                    Some(n as u32 - zeros)
                }
            }
        }

        fn quant4(
            coeffs: &[i32; 16],
            levels: &mut [i16; 16],
            mf: &[i32; 16],
            qbits: u32,
            offset: i32,
        ) -> u32 {
            if offset < 0 || qbits > 62 {
                return quant4_scalar(coeffs, levels, mf, qbits, offset);
            }
            match unsafe {
                quant_impl(
                    coeffs.as_ptr(),
                    levels.as_mut_ptr(),
                    mf.as_ptr(),
                    16,
                    qbits,
                    offset,
                )
            } {
                Some(nz) => nz,
                None => quant4_scalar(coeffs, levels, mf, qbits, offset),
            }
        }

        fn quant8(
            coeffs: &[i32; 64],
            levels: &mut [i16; 64],
            mf: &[i32; 64],
            qbits: u32,
            offset: i32,
        ) -> u32 {
            if offset < 0 || qbits > 62 {
                return quant8_scalar(coeffs, levels, mf, qbits, offset);
            }
            match unsafe {
                quant_impl(
                    coeffs.as_ptr(),
                    levels.as_mut_ptr(),
                    mf.as_ptr(),
                    64,
                    qbits,
                    offset,
                )
            } {
                Some(nz) => nz,
                None => quant8_scalar(coeffs, levels, mf, qbits, offset),
            }
        }
    };
}

/// SSE2: baseline on x86-64, so this rung is the one that makes the scalar
/// kernels unreachable on this architecture.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSSE3: `pabsd` in the quantisers.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// SSE4.1: `pmovsxwd` for the 8x8 transform's widening loads.
pub(crate) mod sse41 {
    #![allow(dead_code)]
    kernels!("sse4.1", sse41);
}

/// AVX: the SSE4.1 primitive set, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// AVX2: the 8x8 transform with a whole row or column of i32 a vector, and
/// the quantisers eight coefficients at a time. The 4x4 transform and the
/// Hadamard keep the [`avx`] kernels: their rows are one 128-bit vector.
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    use crate::dsp::h264_enc::{H264EncDsp, quant4_scalar, quant8_scalar};

    pub(crate) fn install(d: &mut H264EncDsp) {
        d.fdct8 = fdct8;
        d.quant4 = quant4;
        d.quant8 = quant8;
    }

    /// Transpose an 8x8 block of i32, one row a vector.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose8(r: &[__m256i; 8]) -> [__m256i; 8] {
        let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
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

    /// `fdct8_1d`, lane-wise.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fdct8_lanes(x: &[__m256i; 8]) -> [__m256i; 8] {
        let a0 = _mm256_add_epi32(x[0], x[7]);
        let a1 = _mm256_add_epi32(x[1], x[6]);
        let a2 = _mm256_add_epi32(x[2], x[5]);
        let a3 = _mm256_add_epi32(x[3], x[4]);
        let a4 = _mm256_sub_epi32(x[0], x[7]);
        let a5 = _mm256_sub_epi32(x[1], x[6]);
        let a6 = _mm256_sub_epi32(x[2], x[5]);
        let a7 = _mm256_sub_epi32(x[3], x[4]);
        let b0 = _mm256_add_epi32(a0, a3);
        let b1 = _mm256_add_epi32(a1, a2);
        let b2 = _mm256_sub_epi32(a0, a3);
        let b3 = _mm256_sub_epi32(a1, a2);
        let h = |v: __m256i| _mm256_add_epi32(_mm256_srai_epi32(v, 1), v);
        let b4 = _mm256_add_epi32(_mm256_add_epi32(a5, a6), h(a4));
        let b5 = _mm256_sub_epi32(_mm256_sub_epi32(a4, a7), h(a6));
        let b6 = _mm256_sub_epi32(_mm256_add_epi32(a4, a7), h(a5));
        let b7 = _mm256_add_epi32(_mm256_sub_epi32(a5, a6), h(a7));
        [
            _mm256_add_epi32(b0, b1),
            _mm256_add_epi32(b4, _mm256_srai_epi32(b7, 2)),
            _mm256_add_epi32(b2, _mm256_srai_epi32(b3, 1)),
            _mm256_add_epi32(b5, _mm256_srai_epi32(b6, 2)),
            _mm256_sub_epi32(b0, b1),
            _mm256_sub_epi32(b6, _mm256_srai_epi32(b5, 2)),
            _mm256_sub_epi32(_mm256_srai_epi32(b2, 1), b3),
            _mm256_sub_epi32(_mm256_srai_epi32(b4, 2), b7),
        ]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fdct8_impl(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
        unsafe {
            let p = residual.as_ptr();
            let rows: [__m256i; 8] = std::array::from_fn(|i| {
                _mm256_cvtepi16_epi32(_mm_loadu_si128(p.add(8 * i) as *const __m128i))
            });
            // Columns as vectors for the row pass, rows for the column pass.
            let t = fdct8_lanes(&transpose8(&rows));
            let o = fdct8_lanes(&transpose8(&t));
            let q = coeffs.as_mut_ptr();
            for (i, v) in o.iter().enumerate() {
                _mm256_storeu_si256(q.add(8 * i) as *mut __m256i, *v);
            }
        }
    }

    fn fdct8(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
        unsafe { fdct8_impl(residual, coeffs) }
    }

    /// The 128-bit rung's `quant_lanes`, eight lanes at a time.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_lanes(c: __m256i, mf: __m256i, off: __m256i, sh: __m128i) -> __m256i {
        let a = _mm256_abs_epi32(c);
        let even = _mm256_srl_epi64(_mm256_add_epi64(_mm256_mul_epu32(a, mf), off), sh);
        let odd = _mm256_srl_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(mf, 32)),
                off,
            ),
            sh,
        );
        let m = _mm256_unpacklo_epi32(
            _mm256_shuffle_epi32(even, 0b10_00_10_00),
            _mm256_shuffle_epi32(odd, 0b10_00_10_00),
        );
        let s = _mm256_srai_epi32(c, 31);
        _mm256_sub_epi32(_mm256_xor_si256(m, s), s)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quant_impl(
        coeffs: *const i32,
        levels: *mut i16,
        mf: *const i32,
        n: usize,
        qbits: u32,
        offset: i32,
    ) -> Option<u32> {
        unsafe {
            let off = _mm256_set1_epi64x(offset as i64);
            let sh = _mm_cvtsi32_si128(qbits as i32);
            let zero = _mm256_setzero_si256();
            let mut signs = zero;
            let mut zeros = 0u32;
            let t = |v: __m256i| _mm256_srai_epi32(_mm256_slli_epi32(v, 16), 16);
            let mut i = 0;
            while i < n {
                let ld = |k: usize| _mm256_loadu_si256(coeffs.add(i + k) as *const __m256i);
                let lm = |k: usize| _mm256_loadu_si256(mf.add(i + k) as *const __m256i);
                let (m0, m1) = (lm(0), lm(8));
                signs = _mm256_or_si256(signs, _mm256_or_si256(m0, m1));
                let v0 = quant_lanes(ld(0), m0, off, sh);
                let v1 = quant_lanes(ld(8), m1, off, sh);
                // `packs` interleaves the two 128-bit lanes; the permute puts
                // the sixteen levels back in order.
                let v = _mm256_permute4x64_epi64(_mm256_packs_epi32(t(v0), t(v1)), 0b11_01_10_00);
                _mm256_storeu_si256(levels.add(i) as *mut __m256i, v);
                let z = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v0, zero)))
                    | (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v1, zero))) << 8);
                zeros += (z as u32).count_ones();
                i += 16;
            }
            if _mm256_movemask_ps(_mm256_castsi256_ps(signs)) != 0 {
                None
            } else {
                Some(n as u32 - zeros)
            }
        }
    }

    fn quant4(
        coeffs: &[i32; 16],
        levels: &mut [i16; 16],
        mf: &[i32; 16],
        qbits: u32,
        offset: i32,
    ) -> u32 {
        if offset < 0 || qbits > 62 {
            return quant4_scalar(coeffs, levels, mf, qbits, offset);
        }
        match unsafe {
            quant_impl(
                coeffs.as_ptr(),
                levels.as_mut_ptr(),
                mf.as_ptr(),
                16,
                qbits,
                offset,
            )
        } {
            Some(nz) => nz,
            None => quant4_scalar(coeffs, levels, mf, qbits, offset),
        }
    }

    fn quant8(
        coeffs: &[i32; 64],
        levels: &mut [i16; 64],
        mf: &[i32; 64],
        qbits: u32,
        offset: i32,
    ) -> u32 {
        if offset < 0 || qbits > 62 {
            return quant8_scalar(coeffs, levels, mf, qbits, offset);
        }
        match unsafe {
            quant_impl(
                coeffs.as_ptr(),
                levels.as_mut_ptr(),
                mf.as_ptr(),
                64,
                qbits,
                offset,
            )
        } {
            Some(nz) => nz,
            None => quant8_scalar(coeffs, levels, mf, qbits, offset),
        }
    }
}

/// Install the best kernels `cpu` can run, one rung at a time.
pub fn install(d: &mut H264EncDsp, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_all(d);
    }
    if cpu.sse41 {
        sse41::install_all(d);
    }
    if cpu.avx {
        avx::install_all(d);
    }
    if cpu.avx2 {
        avx2::install(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::h264_enc::{Quant, qbits4, qbits8, quant_offset};
    use crate::h264::sps::ScalingLists;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Every rung the host can run, installed cumulatively as in the field.
    fn rungs() -> Vec<(&'static str, H264EncDsp)> {
        let base = Cpu::SCALAR;
        let sse2 = Cpu { sse2: true, ..base };
        let ssse3 = Cpu {
            ssse3: true,
            ..sse2
        };
        let sse41 = Cpu {
            sse41: true,
            ..ssse3
        };
        let avx = Cpu { avx: true, ..sse41 };
        let avx2 = Cpu { avx2: true, ..avx };
        [
            ("sse2", sse2, std::is_x86_feature_detected!("sse2")),
            ("ssse3", ssse3, std::is_x86_feature_detected!("ssse3")),
            ("sse4.1", sse41, std::is_x86_feature_detected!("sse4.1")),
            ("avx", avx, std::is_x86_feature_detected!("avx")),
            ("avx2", avx2, std::is_x86_feature_detected!("avx2")),
        ]
        .into_iter()
        .filter(|&(_, _, have)| have)
        .map(|(n, c, _)| {
            let mut d = H264EncDsp::SCALAR;
            install(&mut d, c);
            (n, d)
        })
        .collect()
    }

    /// A residual block: rows in the range of a `bit_depth` residual, or at
    /// an i16 extreme, so the widest intermediate of each pass is reached.
    fn residual<const N: usize>(seed: &mut u64, bit_depth: u32) -> [i16; N] {
        let span = 1i32 << bit_depth;
        let row = if N == 16 { 4 } else { 8 };
        let mut r = [0i16; N];
        let mut mode = 0;
        for (i, v) in r.iter_mut().enumerate() {
            if i % row == 0 {
                mode = lcg(seed) % 8;
            }
            *v = match mode {
                0 => 32767,
                1 => -32768,
                2 => lcg(seed) as i16,
                3 => (span - 1) as i16,
                4 => (1 - span) as i16,
                _ => ((lcg(seed) as i32 % (2 * span - 1)) - (span - 1)) as i16,
            };
        }
        r
    }

    #[test]
    fn transforms_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let mut seed = 0x4dc7_u64;
        let tables = rungs();
        assert!(!tables.is_empty(), "no x86 rung to test");
        for bit_depth in [8u32, 9, 10, 12, 14] {
            for round in 0..300 {
                let r4: [i16; 16] = residual(&mut seed, bit_depth);
                let r8: [i16; 64] = residual(&mut seed, bit_depth);
                let (mut w4, mut w8) = ([0i32; 16], [0i32; 64]);
                (s.fdct4)(&r4, &mut w4);
                (s.fdct8)(&r8, &mut w8);
                // The Hadamard's input is sixteen 4x4 DCs; scaled up now and
                // then toward the largest sum that stays in i32.
                let dc: [i32; 16] =
                    std::array::from_fn(|i| w4[i] * if round % 2 == 0 { 1 } else { 16 });
                let mut wh = dc;
                (s.hadamard4)(&mut wh);
                for (name, d) in &tables {
                    let (mut g4, mut g8) = ([0i32; 16], [0i32; 64]);
                    (d.fdct4)(&r4, &mut g4);
                    (d.fdct8)(&r8, &mut g8);
                    let mut gh = dc;
                    (d.hadamard4)(&mut gh);
                    assert_eq!(
                        g4, w4,
                        "{name}: fdct4, {bit_depth} bits, round {round}, {r4:?}"
                    );
                    assert_eq!(
                        g8, w8,
                        "{name}: fdct8, {bit_depth} bits, round {round}, {r8:?}"
                    );
                    assert_eq!(gh, wh, "{name}: hadamard4, round {round}, {dc:?}");
                }
            }
        }
    }

    /// Scaling lists: flat, the smallest weight everywhere (the largest
    /// multipliers a stream can make), and random ones.
    fn lists(seed: &mut u64) -> Vec<ScalingLists> {
        let flat = ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        };
        let small = ScalingLists {
            list4x4: [[4; 16]; 6],
            list8x8: [[4; 64]; 6],
        };
        let mut random = flat.clone();
        for l in 0..6 {
            random.list4x4[l] = std::array::from_fn(|_| 4 + (lcg(seed) % 252) as u8);
            random.list8x8[l] = std::array::from_fn(|_| 4 + (lcg(seed) % 252) as u8);
        }
        vec![flat, small, random]
    }

    /// A coefficient: mostly what a transform of a residual gives, with the
    /// i32 extremes and zero mixed in.
    fn coeff(seed: &mut u64, span: i32) -> i32 {
        match lcg(seed) % 16 {
            0 => 0,
            1 => i32::MAX,
            2 => i32::MIN + 1,
            3 => -span,
            4 => span,
            5 => lcg(seed) as i32,
            _ => (lcg(seed) as i32 % (2 * span)) - span,
        }
    }

    #[test]
    fn quantisers_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let mut seed = 0x9a41_u64;
        let tables = rungs();
        assert!(!tables.is_empty(), "no x86 rung to test");
        for lists in lists(&mut seed) {
            let q = Quant::new(&lists);
            // QP'Y up to 51 plus the 14-bit QpBdOffset of 36.
            for qp in 0..=87 {
                for intra in [true, false] {
                    let (qb4, qb8) = (qbits4(qp), qbits8(qp));
                    let (o4, o8) = (quant_offset(qb4, intra), quant_offset(qb8, intra));
                    let list = qp as usize % 6;
                    let (mf4, mf8) = (
                        &q.mf4[list][(qp % 6) as usize],
                        &q.mf8[list][(qp % 6) as usize],
                    );
                    for span in [255 * 36, 1023 * 64, 16383 * 256] {
                        let c4: [i32; 16] = std::array::from_fn(|_| coeff(&mut seed, span));
                        let c8: [i32; 64] = std::array::from_fn(|_| coeff(&mut seed, span));
                        let (mut w4, mut w8) = ([0i16; 16], [0i16; 64]);
                        let nw4 = (s.quant4)(&c4, &mut w4, mf4, qb4, o4);
                        let nw8 = (s.quant8)(&c8, &mut w8, mf8, qb8, o8);
                        for (name, d) in &tables {
                            let (mut g4, mut g8) = ([0i16; 16], [0i16; 64]);
                            let ng4 = (d.quant4)(&c4, &mut g4, mf4, qb4, o4);
                            let ng8 = (d.quant8)(&c8, &mut g8, mf8, qb8, o8);
                            assert_eq!(
                                (g4, ng4),
                                (w4, nw4),
                                "{name}: quant4, qp {qp}, intra {intra}, span {span}, {c4:?}"
                            );
                            assert_eq!(
                                (g8, ng8),
                                (w8, nw8),
                                "{name}: quant8, qp {qp}, intra {intra}, span {span}"
                            );
                        }
                    }
                }
            }
        }
        // The calls the kernels hand back to the reference: a negative
        // multiplier, a negative offset, a shift past 62.
        let c: [i32; 16] = std::array::from_fn(|i| i as i32 * 977 - 7000);
        let mut neg = [13107i32; 16];
        neg[5] = -3;
        for (mf, qb, off) in [(neg, 15, 100), ([13107; 16], 15, -5), ([13107; 16], 63, 0)] {
            let mut want = [0i16; 16];
            let nw = (s.quant4)(&c, &mut want, &mf, qb, off);
            for (name, d) in &tables {
                let mut got = [0i16; 16];
                let ng = (d.quant4)(&c, &mut got, &mf, qb, off);
                assert_eq!(
                    (got, ng),
                    (want, nw),
                    "{name}: mf[5] {}, qbits {qb}, offset {off}",
                    mf[5]
                );
            }
        }
    }

    /// ns per call, scalar against every rung, median of paired rounds; the
    /// two scalar rows are the same-table control.
    /// `cargo test --release --lib h264_enc_x86 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_bench() {
        use std::time::Instant;
        let mut seed = 0xb264_u64;
        let r4: [i16; 16] = residual(&mut seed, 8);
        let r8: [i16; 64] = residual(&mut seed, 8);
        let q = Quant::new(&ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        });
        let c4: [i32; 16] = std::array::from_fn(|_| (lcg(&mut seed) % 2001) as i32 - 1000);
        let c8: [i32; 64] = std::array::from_fn(|_| (lcg(&mut seed) % 4001) as i32 - 2000);
        let s = H264EncDsp::SCALAR;
        let mut tabs = vec![("scalar", s.clone()), ("scalar-again", s)];
        tabs.extend(rungs());
        type Call<'a> = &'a dyn Fn(&H264EncDsp) -> u64;
        let families: [(&str, Call); 5] = [
            ("fdct4", &|d| {
                let mut o = [0i32; 16];
                (d.fdct4)(&r4, &mut o);
                o[5] as u64
            }),
            ("fdct8", &|d| {
                let mut o = [0i32; 64];
                (d.fdct8)(&r8, &mut o);
                o[9] as u64
            }),
            ("hadamard4", &|d| {
                let mut o = c4;
                (d.hadamard4)(&mut o);
                o[3] as u64
            }),
            ("quant4", &|d| {
                let mut l = [0i16; 16];
                (d.quant4)(
                    &c4,
                    &mut l,
                    &q.mf4[0][2],
                    qbits4(26),
                    quant_offset(qbits4(26), false),
                ) as u64
                    + l[3] as u64
            }),
            ("quant8", &|d| {
                let mut l = [0i16; 64];
                (d.quant8)(
                    &c8,
                    &mut l,
                    &q.mf8[0][2],
                    qbits8(26),
                    quant_offset(qbits8(26), false),
                ) as u64
                    + l[3] as u64
            }),
        ];
        const ROUNDS: usize = 7;
        const PER: usize = 200_000;
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        for (label, f) in &families {
            // `ns[round][table]`: every table back to back within a round.
            let mut ns = vec![vec![0f64; tabs.len()]; ROUNDS];
            let mut sink = 0u64;
            for round in ns.iter_mut() {
                for (slot, (_, d)) in round.iter_mut().zip(&tabs) {
                    let start = Instant::now();
                    for _ in 0..PER {
                        sink = sink.wrapping_add(f(std::hint::black_box(d)));
                    }
                    *slot = start.elapsed().as_nanos() as f64 / PER as f64;
                }
            }
            for (t, (name, _)) in tabs.iter().enumerate() {
                let own = median(ns.iter().map(|r| r[t]).collect());
                let ratio = median(ns.iter().map(|r| r[0] / r[t]).collect());
                println!(
                    "{label:10} {name:13} {own:7.1} ns/call  {ratio:5.2}x scalar [{}]",
                    sink & 1
                );
            }
        }
    }
}
