//! Pixel kernels: interpolation filters, inverse transforms, deblocking and
//! SAO edges — the loops a decoder spends its time in.
//!
//! Every kernel exists as a scalar reference implementation, and the hot ones
//! also as SIMD (x86-64 AVX2, AArch64 NEON). Which runs is decided once per process
//! by [`Cpu::detect`] and threaded through as function pointers, the way
//! libavcodec's `*dsp_init` tables work, so the decoders never branch on the
//! CPU themselves and the scalar path stays the executable specification the
//! SIMD path is tested against.

pub mod h264;
pub mod hevc;
// The SIMD modules wrap their intrinsics in `unsafe {}` blocks: required on
// the crate's MSRV, redundant (and warned about) on toolchains where
// target-feature intrinsics became safe to call inside `#[target_feature]`
// functions.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub mod h264_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub mod h264_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub mod hevc_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub mod hevc_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub mod hevc_avx2_u8;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub mod hevc_neon_u8;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub mod neon_dotprod;

/// What the running CPU can do, detected once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu {
    /// x86-64 with AVX2 (implies SSE4.1 / SSSE3).
    pub avx2: bool,
    /// x86-64 with SSE4.1 (a superset of the SSSE3 kernels need).
    pub sse41: bool,
    /// AArch64 NEON (baseline on every AArch64 CPU).
    pub neon: bool,
    /// AArch64 with the ARMv8.2-A dot product extension (`sdot` / `udot`):
    /// four 8-bit products summed into a 32-bit lane, which is the shape of
    /// the byte-tap interpolation filters. Cortex-A75 and later, Apple A11
    /// and later — not an exotic target.
    pub dotprod: bool,
    /// AArch64 with the ARMv8.6-A 8-bit matrix multiply extension (`usdot` /
    /// `usmmla`). Detected, but no kernel uses it yet: `neon_dotprod`'s
    /// module documentation says why it does not pay on these filters.
    pub i8mm: bool,
}

impl Cpu {
    /// Detect the running CPU's SIMD extensions.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::is_x86_feature_detected!("avx2");
            let sse41 = std::is_x86_feature_detected!("sse4.1");
            return Self { avx2, sse41, ..Self::SCALAR };
        }
        #[cfg(target_arch = "aarch64")]
        {
            let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            let i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
            return Self { neon: true, dotprod, i8mm, ..Self::SCALAR };
        }
        #[allow(unreachable_code)]
        Self::SCALAR
    }

    /// Scalar only — the reference paths. What the SIMD kernels are checked
    /// against, and what a `H26X_NO_SIMD=1` environment asks for.
    pub const SCALAR: Self = Self { avx2: false, sse41: false, neon: false, dotprod: false, i8mm: false };

    /// [`Self::detect`], unless the environment turns SIMD off.
    pub fn detect_honouring_env() -> Self {
        if std::env::var_os("H26X_NO_SIMD").is_some_and(|v| v == "1" || v == "true") {
            Self::SCALAR
        } else {
            Self::detect()
        }
    }
}
