//! Pixel kernels: interpolation filters, inverse transforms, deblocking and
//! SAO edges — the loops a decoder spends its time in.
//!
//! Every kernel exists as a scalar reference implementation, and the hot ones
//! also as SIMD: on x86-64 a ladder from SSE2 up through SSSE3, SSE4.1, AVX
//! and AVX2, and on AArch64 NEON with the dot product extension above it.
//! Which runs is decided once per process
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
pub(crate) mod x86_compat;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub(crate) mod h264_x86_128;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub(crate) mod h264_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub(crate) mod h264_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub(crate) mod hevc_x86_128;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub(crate) mod hevc_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub(crate) mod hevc_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
pub(crate) mod hevc_avx2_u8;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub(crate) mod hevc_neon_u8;
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
pub(crate) mod neon_dotprod;

/// What the running CPU can do, detected once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu {
    /// x86-64 with AVX2 (implies AVX / SSE4.1 / SSSE3).
    pub avx2: bool,
    /// x86-64 with AVX: the 128-bit kernels, VEX-encoded.
    pub avx: bool,
    /// x86-64 with SSE4.1 (`pblendvb`, `pmovzx`, `pminsd` / `pmaxsd`, `ptest`).
    pub sse41: bool,
    /// x86-64 with SSSE3 (`pmaddubsw`, `pshufb`, `pabsw` / `pabsd`).
    pub ssse3: bool,
    /// x86-64 with SSE2 — baseline on every x86-64 CPU, so on this
    /// architecture the scalar kernels are a reference, never a fallback.
    pub sse2: bool,
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
            let avx = std::is_x86_feature_detected!("avx");
            let sse41 = std::is_x86_feature_detected!("sse4.1");
            let ssse3 = std::is_x86_feature_detected!("ssse3");
            // Guaranteed by the target, but ask anyway rather than assert it.
            let sse2 = std::is_x86_feature_detected!("sse2");
            return Self { avx2, avx, sse41, ssse3, sse2, ..Self::SCALAR };
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
    pub const SCALAR: Self = Self {
        avx2: false,
        avx: false,
        sse41: false,
        ssse3: false,
        sse2: false,
        neon: false,
        dotprod: false,
        i8mm: false,
    };

    /// The name of the widest rung this `Cpu` selects, for reporting.
    ///
    /// Which rung a machine takes decides its speed by more than a factor of
    /// two, so a user who cannot ask which one they got cannot make sense of
    /// a measurement. `h26xdec --rung` prints this.
    pub fn rung(&self) -> &'static str {
        if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX (VEX-128)"
        } else if self.sse41 {
            "SSE4.1"
        } else if self.ssse3 {
            "SSSE3"
        } else if self.sse2 {
            "SSE2"
        } else if self.dotprod {
            "NEON + DotProd"
        } else if self.neon {
            "NEON"
        } else {
            "scalar"
        }
    }

    /// [`Self::detect`], with what it found capped by the environment.
    ///
    /// `H26X_NO_SIMD=1` asks for the scalar reference paths. `H26X_MAX_SIMD`
    /// caps the x86 install level — `avx2` (no cap), `avx`, `sse41`, `ssse3`,
    /// `sse2` or `none` — so every rung of the ladder can be exercised for
    /// bit-exactness on a machine whose CPU would otherwise always take the
    /// top one. An unrecognised value is ignored rather than silently
    /// downgrading.
    pub fn detect_honouring_env() -> Self {
        if std::env::var_os("H26X_NO_SIMD").is_some_and(|v| v == "1" || v == "true") {
            return Self::SCALAR;
        }
        let mut cpu = Self::detect();
        // Each arm falls through the ones above it: capping at `ssse3` also
        // clears sse41, avx and avx2.
        match std::env::var("H26X_MAX_SIMD").as_deref() {
            Ok("none") | Ok("scalar") => cpu = Self::SCALAR,
            Ok("sse2") => {
                cpu.avx2 = false;
                cpu.avx = false;
                cpu.sse41 = false;
                cpu.ssse3 = false;
            }
            Ok("ssse3") => {
                cpu.avx2 = false;
                cpu.avx = false;
                cpu.sse41 = false;
            }
            Ok("sse41") | Ok("sse4.1") => {
                cpu.avx2 = false;
                cpu.avx = false;
            }
            Ok("avx") => cpu.avx2 = false,
            // AArch64: cap at baseline NEON.
            Ok("neon") => {
                cpu.dotprod = false;
                cpu.i8mm = false;
            }
            _ => {}
        }
        cpu
    }
}
