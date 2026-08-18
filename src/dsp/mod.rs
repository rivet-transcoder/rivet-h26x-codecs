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
pub mod h264_avx;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub mod h264_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub mod h264_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub mod hevc_avx;
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

/// What the running CPU can do, detected once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu {
    /// x86-64 with AVX2 (implies AVX / SSE4.1 / SSSE3).
    pub avx2: bool,
    /// x86-64 with AVX: the 128-bit kernels, VEX-encoded.
    pub avx: bool,
    /// x86-64 with SSE4.1 (a superset of the SSSE3 kernels need).
    pub sse41: bool,
    /// AArch64 NEON (baseline on every AArch64 CPU).
    pub neon: bool,
}

impl Cpu {
    /// Detect the running CPU's SIMD extensions.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::is_x86_feature_detected!("avx2");
            let avx = std::is_x86_feature_detected!("avx");
            let sse41 = std::is_x86_feature_detected!("sse4.1");
            return Self { avx2, avx, sse41, neon: false };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self { avx2: false, avx: false, sse41: false, neon: true };
        }
        #[allow(unreachable_code)]
        Self { avx2: false, avx: false, sse41: false, neon: false }
    }

    /// Scalar only — the reference paths. What the SIMD kernels are checked
    /// against, and what a `H26X_NO_SIMD=1` environment asks for.
    pub const SCALAR: Self = Self { avx2: false, avx: false, sse41: false, neon: false };

    /// [`Self::detect`], with what it found capped by the environment.
    ///
    /// `H26X_NO_SIMD=1` asks for the scalar reference paths. `H26X_MAX_SIMD`
    /// caps the x86 install level — `avx2` (no cap), `avx`, `sse41` or `none`
    /// — so the 128-bit and scalar paths can be exercised for bit-exactness
    /// on a machine whose CPU would otherwise always take the widest one.
    /// An unrecognised value is ignored rather than silently downgrading.
    pub fn detect_honouring_env() -> Self {
        if std::env::var_os("H26X_NO_SIMD").is_some_and(|v| v == "1" || v == "true") {
            return Self::SCALAR;
        }
        let mut cpu = Self::detect();
        match std::env::var("H26X_MAX_SIMD").as_deref() {
            Ok("none") | Ok("scalar") => cpu = Self::SCALAR,
            Ok("sse41") | Ok("sse4.1") => {
                cpu.avx2 = false;
                cpu.avx = false;
            }
            Ok("avx") => cpu.avx2 = false,
            _ => {}
        }
        cpu
    }
}
