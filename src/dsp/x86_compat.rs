//! The handful of 128-bit primitives whose best instruction depends on which
//! SSE level the kernel is being compiled for.
//!
//! The x86 kernels are written once and compiled once per rung of the ladder
//! (SSE2, SSSE3, SSE4.1, AVX). Almost every intrinsic they use is SSE2, which
//! is baseline on x86-64; what the later levels add is a better instruction
//! for a small number of recurring operations — a select, a zero- or
//! sign-extension, an absolute value, a 32-bit min/max. [`compat_core!`]
//! expands the best available implementation of each for the requested level,
//! so a kernel body reads the same at every rung and the ladder cannot drift
//! apart between them.
//!
//! Every implementation here is *exact*, not approximate: `sel` is only ever
//! given all-ones / all-zeros lane masks (they come from compares), for which
//! the SSE2 and-or form and `pblendvb` agree bit for bit, and `abs16` is only
//! ever given differences of samples, which cannot be `i16::MIN`.
//!
//! The level is named by a bare token — `sse2`, `ssse3` or `sse41` — and the
//! feature string by a literal, because they are not the same thing: the AVX
//! instantiation compiles the SSE4.1 primitive set with `enable = "avx"` to
//! get VEX encoding, so it passes `("avx", sse41)`.

/// Expand the level-dependent primitives for `$feat` / `$lvl`.
///
/// Callers must have `std::arch::x86_64::*` in scope.
macro_rules! compat_core {
    ($feat:literal, sse2) => {
        /// Zero-extend the low eight bytes of `v` to eight i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8(v: __m128i) -> __m128i {
            unsafe { _mm_unpacklo_epi8(v, _mm_setzero_si128()) }
        }

        /// Zero-extend the high eight bytes of `v` to eight i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8h(v: __m128i) -> __m128i {
            unsafe { _mm_unpackhi_epi8(v, _mm_setzero_si128()) }
        }

        /// Zero-extend the low four u16 lanes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16(v: __m128i) -> __m128i {
            unsafe { _mm_unpacklo_epi16(v, _mm_setzero_si128()) }
        }

        /// Zero-extend u16 lanes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16h(v: __m128i) -> __m128i {
            unsafe { _mm_unpackhi_epi16(v, _mm_setzero_si128()) }
        }

        /// Sign-extend the low four i16 lanes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16(v: __m128i) -> __m128i {
            unsafe { _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16) }
        }

        /// Sign-extend i16 lanes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16h(v: __m128i) -> __m128i {
            unsafe { _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16) }
        }


        /// Zero-extend the low four bytes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8d(v: __m128i) -> __m128i {
            unsafe { zx16(zx8(v)) }
        }

        /// Zero-extend bytes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8dh(v: __m128i) -> __m128i {
            unsafe { zx16h(zx8(v)) }
        }
        /// `b` where `m`'s lanes are all-ones, `a` where they are all-zeros.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sel(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
            unsafe { _mm_or_si128(_mm_andnot_si128(m, a), _mm_and_si128(m, b)) }
        }

        /// `|v|` per i16 lane (never given `i16::MIN`).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs16(v: __m128i) -> __m128i {
            unsafe { _mm_max_epi16(v, _mm_sub_epi16(_mm_setzero_si128(), v)) }
        }

        /// `|v|` per i32 lane.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs32(v: __m128i) -> __m128i {
            unsafe {
                let m = _mm_srai_epi32(v, 31);
                _mm_sub_epi32(_mm_xor_si128(v, m), m)
            }
        }

        /// Signed 32-bit minimum.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn min32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { sel(a, b, _mm_cmpgt_epi32(a, b)) }
        }

        /// Signed 32-bit maximum.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn max32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { sel(b, a, _mm_cmpgt_epi32(a, b)) }
        }

        /// `max(v, 0)` per signed byte.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn maxb0(v: __m128i) -> __m128i {
            unsafe { _mm_and_si128(v, _mm_cmpgt_epi8(v, _mm_setzero_si128())) }
        }

        /// Whether every bit of `v` is zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn is_zero(v: __m128i) -> bool {
            unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_setzero_si128())) == 0xFFFF }
        }
    };

    ($feat:literal, ssse3) => {
        // SSSE3 adds `pabsw` / `pabsd`; the extensions, select and 32-bit
        // min/max are still SSE2 sequences.
        crate::dsp::x86_compat::compat_core!($feat, sse2_but_abs);

        /// `|v|` per i16 lane.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs16(v: __m128i) -> __m128i {
            unsafe { _mm_abs_epi16(v) }
        }

        /// `|v|` per i32 lane.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs32(v: __m128i) -> __m128i {
            unsafe { _mm_abs_epi32(v) }
        }
    };

    // The SSE2 set minus the two absolute values, so the SSSE3 arm can
    // replace them without redefining the rest.
    ($feat:literal, sse2_but_abs) => {
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8(v: __m128i) -> __m128i {
            unsafe { _mm_unpacklo_epi8(v, _mm_setzero_si128()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8h(v: __m128i) -> __m128i {
            unsafe { _mm_unpackhi_epi8(v, _mm_setzero_si128()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16(v: __m128i) -> __m128i {
            unsafe { _mm_unpacklo_epi16(v, _mm_setzero_si128()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16h(v: __m128i) -> __m128i {
            unsafe { _mm_unpackhi_epi16(v, _mm_setzero_si128()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16(v: __m128i) -> __m128i {
            unsafe { _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16h(v: __m128i) -> __m128i {
            unsafe { _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16) }
        }


        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8d(v: __m128i) -> __m128i {
            unsafe { zx16(zx8(v)) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8dh(v: __m128i) -> __m128i {
            unsafe { zx16h(zx8(v)) }
        }
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sel(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
            unsafe { _mm_or_si128(_mm_andnot_si128(m, a), _mm_and_si128(m, b)) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn min32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { sel(a, b, _mm_cmpgt_epi32(a, b)) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn max32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { sel(b, a, _mm_cmpgt_epi32(a, b)) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn maxb0(v: __m128i) -> __m128i {
            unsafe { _mm_and_si128(v, _mm_cmpgt_epi8(v, _mm_setzero_si128())) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn is_zero(v: __m128i) -> bool {
            unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_setzero_si128())) == 0xFFFF }
        }
    };

    ($feat:literal, sse41) => {
        /// Zero-extend the low eight bytes of `v` to eight i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu8_epi16(v) }
        }

        /// Zero-extend the high eight bytes of `v` to eight i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8h(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu8_epi16(_mm_srli_si128(v, 8)) }
        }

        /// Zero-extend the low four u16 lanes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu16_epi32(v) }
        }

        /// Zero-extend u16 lanes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx16h(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu16_epi32(_mm_srli_si128(v, 8)) }
        }

        /// Sign-extend the low four i16 lanes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepi16_epi32(v) }
        }

        /// Sign-extend i16 lanes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sx16h(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepi16_epi32(_mm_srli_si128(v, 8)) }
        }


        /// Zero-extend the low four bytes of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8d(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu8_epi32(v) }
        }

        /// Zero-extend bytes 4..8 of `v` to four i32.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn zx8dh(v: __m128i) -> __m128i {
            unsafe { _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)) }
        }
        /// `b` where `m`'s lanes are all-ones, `a` where they are all-zeros.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sel(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
            unsafe { _mm_blendv_epi8(a, b, m) }
        }

        /// `|v|` per i16 lane.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs16(v: __m128i) -> __m128i {
            unsafe { _mm_abs_epi16(v) }
        }

        /// `|v|` per i32 lane.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn abs32(v: __m128i) -> __m128i {
            unsafe { _mm_abs_epi32(v) }
        }

        /// Signed 32-bit minimum.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn min32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_min_epi32(a, b) }
        }

        /// Signed 32-bit maximum.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn max32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_max_epi32(a, b) }
        }

        /// `max(v, 0)` per signed byte.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn maxb0(v: __m128i) -> __m128i {
            unsafe { _mm_max_epi8(v, _mm_setzero_si128()) }
        }

        /// Whether every bit of `v` is zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn is_zero(v: __m128i) -> bool {
            unsafe { _mm_testz_si128(v, v) != 0 }
        }
    };
}

pub(crate) use compat_core;
