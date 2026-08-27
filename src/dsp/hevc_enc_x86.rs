//! x86-64 SIMD versions of the H.265 forward transforms and quantiser.
//!
//! The forward DCT is the decoder's inverse ([`super::hevc_avx2`]'s
//! `idct_impl`) read the other way, and it is built from the same two
//! `pmaddwd` shapes:
//!
//! - **stage 1 (rows)** vectorises across *outputs*: for one input row,
//!   the pair `(x[2q], x[2q+1])` is broadcast as one i32 and multiplied
//!   against a table row holding `(M[j][2q], M[j][2q+1])` for eight
//!   consecutive `j`, so one `pmaddwd` yields the partial sums of eight
//!   coefficients at once;
//! - **stage 2 (columns)** vectorises across *columns*: for one output
//!   row `j`, the pair `(M[j][2q], M[j][2q+1])` is broadcast and multiplied
//!   against `(t[2q][x], t[2q+1][x])` interleaved for eight consecutive
//!   `x`.
//!
//! What removes the transpose between them is that stage 1 processes two
//! input rows at a time and stores its results already interleaved in
//! pairs, `(t[2q][j], t[2q+1][j])`, which is exactly the operand stage 2
//! reads. One table serves both stages: `FP[q][2j..2j+2]` is the pair
//! stage 1 loads eight of and stage 2 broadcasts one of. It is built at
//! compile time from the decoder's `TRANSFORM32`, as the inverse's table
//! is, so the two directions cannot disagree about a coefficient.
//!
//! Exactness: every product is `i16 x i16` summed in i32 — at most 32 terms
//! of `90 x 32767`, under 2^27 — and both roundings are the scalar
//! reference's `(sum + (1 << (s - 1))) >> s` as an arithmetic shift,
//! followed by `packs`, whose saturation is the reference's clamp to i16.
//!
//! The quantiser multiplies the u16 magnitude by the u16 scale with
//! `pmullw` / `pmulhuw`, which together are the exact 32-bit product; the
//! magnitude of `-32768` is `0x8000`, which is 32768 as u16, so the one
//! value an i16 absolute cannot represent is still right. The shift is a
//! logical one on a value that is nonnegative by construction, and the
//! sign is put back with `(m ^ s) - s`.

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::hevc_enc::{DST4, HevcEncDsp};
use crate::hevc::tables::TRANSFORM32;

/// `FP[q * 2N + 2j + t] = M[j][2q + t]` for the `N`-point matrix `M`
/// (`TRANSFORM32` every `32 / N`th row), flattened so a generic kernel can
/// index it. `L` is `N * N`.
const fn build_fp<const L: usize>(n: usize) -> [i16; L] {
    let mut t = [0i16; L];
    let step = 32 / n;
    let mut q = 0;
    while q < n / 2 {
        let mut j = 0;
        while j < n {
            t[q * 2 * n + 2 * j] = TRANSFORM32[j * step][2 * q] as i16;
            t[q * 2 * n + 2 * j + 1] = TRANSFORM32[j * step][2 * q + 1] as i16;
            j += 1;
        }
        q += 1;
    }
    t
}

/// The DST's table, from the same matrix the scalar kernel reads.
const fn build_fdst() -> [i16; 16] {
    let mut t = [0i16; 16];
    let mut q = 0;
    while q < 2 {
        let mut j = 0;
        while j < 4 {
            t[q * 8 + 2 * j] = DST4[j][2 * q] as i16;
            t[q * 8 + 2 * j + 1] = DST4[j][2 * q + 1] as i16;
            j += 1;
        }
        q += 1;
    }
    t
}

static FP4: [i16; 16] = build_fp::<16>(4);
static FP8: [i16; 64] = build_fp::<64>(8);
static FP16: [i16; 256] = build_fp::<256>(16);
static FP32: [i16; 1024] = build_fp::<1024>(32);
static FDST: [i16; 16] = build_fdst();

#[inline(always)]
fn fp<const N: usize>() -> &'static [i16] {
    match N {
        32 => &FP32,
        16 => &FP16,
        8 => &FP8,
        _ => &FP4,
    }
}

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::hevc_enc::{HevcEncDsp, fdct_scalar, fdst4_scalar, quant_scalar};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        /// Every kernel this rung carries.
        pub(crate) fn install_all(d: &mut HevcEncDsp) {
            d.fdct = [fdct::<4>, fdct::<8>, fdct::<16>, fdct::<32>];
            d.fdst4 = fdst4;
            d.quant = quant;
        }

        /// The quantiser alone — what a rung whose only change is a better
        /// absolute value re-installs.
        pub(crate) fn install_quant(d: &mut HevcEncDsp) {
            d.quant = quant;
        }

        /// Both stages of an `N`-point forward transform over `block`
        /// (raster, in place), with `table` the pair table of its matrix.
        /// `s1 >= 1`.
        #[target_feature(enable = $feat)]
        unsafe fn fwd_impl<const N: usize>(block: *mut i16, s1: i32, s2: i32, table: *const i16) {
            unsafe {
                // Stage 1's output, rows interleaved in pairs.
                let mut inter = [0i16; 32 * 32];
                let sh1 = _mm_cvtsi32_si128(s1);
                let r1 = _mm_set1_epi32(1 << (s1 - 1));
                let sh2 = _mm_cvtsi32_si128(s2);
                let r2 = _mm_set1_epi32(1 << (s2 - 1));
                let np = N / 2;
                for p in 0..np {
                    let re = block.add(2 * p * N);
                    let ro = block.add((2 * p + 1) * N);
                    let mut j = 0;
                    while j < N {
                        let (mut e0, mut e1, mut o0, mut o1) = (r1, r1, r1, r1);
                        for q in 0..np {
                            let te = _mm_set1_epi32((re.add(2 * q) as *const i32).read_unaligned());
                            let to = _mm_set1_epi32((ro.add(2 * q) as *const i32).read_unaligned());
                            let c0 =
                                _mm_loadu_si128(table.add(q * 2 * N + 2 * j) as *const __m128i);
                            e0 = _mm_add_epi32(e0, _mm_madd_epi16(c0, te));
                            o0 = _mm_add_epi32(o0, _mm_madd_epi16(c0, to));
                            if N > 4 {
                                let c1 = _mm_loadu_si128(
                                    table.add(q * 2 * N + 2 * j + 8) as *const __m128i
                                );
                                e1 = _mm_add_epi32(e1, _mm_madd_epi16(c1, te));
                                o1 = _mm_add_epi32(o1, _mm_madd_epi16(c1, to));
                            }
                        }
                        let ve = _mm_packs_epi32(_mm_sra_epi32(e0, sh1), _mm_sra_epi32(e1, sh1));
                        let vo = _mm_packs_epi32(_mm_sra_epi32(o0, sh1), _mm_sra_epi32(o1, sh1));
                        let dst = inter.as_mut_ptr().add(p * 2 * N + 2 * j);
                        _mm_storeu_si128(dst as *mut __m128i, _mm_unpacklo_epi16(ve, vo));
                        if N > 4 {
                            _mm_storeu_si128(
                                dst.add(8) as *mut __m128i,
                                _mm_unpackhi_epi16(ve, vo),
                            );
                        }
                        j += 8;
                    }
                }
                for j in 0..N {
                    let mut x = 0;
                    while x < N {
                        let (mut a0, mut a1) = (r2, r2);
                        for q in 0..np {
                            let c = _mm_set1_epi32(
                                (table.add(q * 2 * N + 2 * j) as *const i32).read_unaligned(),
                            );
                            let src = inter.as_ptr().add(q * 2 * N + 2 * x);
                            a0 = _mm_add_epi32(
                                a0,
                                _mm_madd_epi16(_mm_loadu_si128(src as *const __m128i), c),
                            );
                            if N > 4 {
                                a1 = _mm_add_epi32(
                                    a1,
                                    _mm_madd_epi16(
                                        _mm_loadu_si128(src.add(8) as *const __m128i),
                                        c,
                                    ),
                                );
                            }
                        }
                        let v = _mm_packs_epi32(_mm_sra_epi32(a0, sh2), _mm_sra_epi32(a1, sh2));
                        let dst = block.add(j * N + x);
                        if N > 4 {
                            _mm_storeu_si128(dst as *mut __m128i, v);
                        } else {
                            _mm_storel_epi64(dst as *mut __m128i, v);
                        }
                        x += 8;
                    }
                }
            }
        }

        fn fdct<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
            debug_assert_eq!(1usize << log2, N);
            let s1 = log2 as i32 + bit_depth as i32 - 9;
            let s2 = log2 as i32 + 6;
            // A first-stage shift of zero has no rounding term; only a bit
            // depth below eight gets there, and the reference handles it.
            if s1 <= 0 {
                return fdct_scalar::<N>(block, log2, bit_depth);
            }
            assert!(block.len() >= N * N, "block too small");
            unsafe { fwd_impl::<N>(block.as_mut_ptr(), s1, s2, super::fp::<N>().as_ptr()) }
        }

        fn fdst4(block: &mut [i16], bit_depth: u32) {
            let s1 = 2 + bit_depth as i32 - 9;
            if s1 <= 0 {
                return fdst4_scalar(block, bit_depth);
            }
            assert!(block.len() >= 16, "block too small");
            unsafe { fwd_impl::<4>(block.as_mut_ptr(), s1, 8, super::FDST.as_ptr()) }
        }

        #[target_feature(enable = $feat)]
        unsafe fn quant_impl(
            coeffs: *const i16,
            levels: *mut i16,
            n2: usize,
            scale: i32,
            qbits: u32,
            offset: i32,
        ) -> u32 {
            unsafe {
                let vs = _mm_set1_epi16(scale as i16);
                let vo = _mm_set1_epi32(offset);
                let sh = _mm_cvtsi32_si128(qbits as i32);
                let zero = _mm_setzero_si128();
                let mut nz = 0u32;
                let mut i = 0;
                while i < n2 {
                    let c = _mm_loadu_si128(coeffs.add(i) as *const __m128i);
                    let a = abs16(c);
                    let lo = _mm_mullo_epi16(a, vs);
                    let hi = _mm_mulhi_epu16(a, vs);
                    let m0 = _mm_srl_epi32(_mm_add_epi32(_mm_unpacklo_epi16(lo, hi), vo), sh);
                    let m1 = _mm_srl_epi32(_mm_add_epi32(_mm_unpackhi_epi16(lo, hi), vo), sh);
                    let s = _mm_srai_epi16(c, 15);
                    let s0 = _mm_unpacklo_epi16(s, s);
                    let s1 = _mm_unpackhi_epi16(s, s);
                    let v0 = _mm_sub_epi32(_mm_xor_si128(m0, s0), s0);
                    let v1 = _mm_sub_epi32(_mm_xor_si128(m1, s1), s1);
                    let v = _mm_packs_epi32(v0, v1);
                    _mm_storeu_si128(levels.add(i) as *mut __m128i, v);
                    let z = _mm_movemask_epi8(_mm_cmpeq_epi16(v, zero)) as u32;
                    nz += 8 - z.count_ones() / 2;
                    i += 8;
                }
                nz
            }
        }

        fn quant(
            coeffs: &[i16],
            levels: &mut [i16],
            n: usize,
            scale: i32,
            qbits: u32,
            offset: i32,
        ) -> u32 {
            let n2 = n * n;
            // The product needs `scale` in u16 and the sum in i32, which
            // `quant_scale` (at most 26214) and a shift of at least 14
            // guarantee; anything else is the reference's.
            if n2 % 8 != 0 || !(0..=32767).contains(&scale) || offset < 0 || qbits > 31 {
                return quant_scalar(coeffs, levels, n, scale, qbits, offset);
            }
            assert!(coeffs.len() >= n2 && levels.len() >= n2, "block too small");
            unsafe {
                quant_impl(
                    coeffs.as_ptr(),
                    levels.as_mut_ptr(),
                    n2,
                    scale,
                    qbits,
                    offset,
                )
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

/// SSSE3: `pabsw` in the quantiser.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// AVX: the SSE4.1 primitive set, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// AVX2: the 16- and 32-point transforms with eight outputs (stage 1) or
/// eight columns (stage 2) per vector, and the quantiser sixteen
/// coefficients at a time. The 4- and 8-point transforms keep the [`avx`]
/// kernels: their rows are one 128-bit vector already.
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    use crate::dsp::hevc_enc::{HevcEncDsp, fdct_scalar, quant_scalar};

    pub(crate) fn install(d: &mut HevcEncDsp) {
        d.fdct[2] = fdct::<16>;
        d.fdct[3] = fdct::<32>;
        d.quant = quant;
    }

    /// Stage 1 for a pair of rows: both packed to i16, then interleaved
    /// so `inter` reads in order. `packs` interleaves the two 128-bit
    /// lanes, and so does `unpack`; two `permute2x128` put the four
    /// eight-lane groups back in column order.
    #[target_feature(enable = "avx2")]
    unsafe fn fwd_impl<const N: usize>(block: *mut i16, s1: i32, s2: i32, table: *const i16) {
        unsafe {
            let mut inter = [0i16; 32 * 32];
            let sh1 = _mm_cvtsi32_si128(s1);
            let r1 = _mm256_set1_epi32(1 << (s1 - 1));
            let sh2 = _mm_cvtsi32_si128(s2);
            let r2 = _mm256_set1_epi32(1 << (s2 - 1));
            let np = N / 2;
            for p in 0..np {
                let re = block.add(2 * p * N);
                let ro = block.add((2 * p + 1) * N);
                let mut j = 0;
                while j < N {
                    let (mut e0, mut e1, mut o0, mut o1) = (r1, r1, r1, r1);
                    for q in 0..np {
                        let te = _mm256_set1_epi32((re.add(2 * q) as *const i32).read_unaligned());
                        let to = _mm256_set1_epi32((ro.add(2 * q) as *const i32).read_unaligned());
                        let c0 = _mm256_loadu_si256(table.add(q * 2 * N + 2 * j) as *const __m256i);
                        let c1 =
                            _mm256_loadu_si256(table.add(q * 2 * N + 2 * j + 16) as *const __m256i);
                        e0 = _mm256_add_epi32(e0, _mm256_madd_epi16(c0, te));
                        e1 = _mm256_add_epi32(e1, _mm256_madd_epi16(c1, te));
                        o0 = _mm256_add_epi32(o0, _mm256_madd_epi16(c0, to));
                        o1 = _mm256_add_epi32(o1, _mm256_madd_epi16(c1, to));
                    }
                    // Outputs j..j+16 of each row, lanes in order after the permute.
                    let ve = _mm256_permute4x64_epi64(
                        _mm256_packs_epi32(_mm256_sra_epi32(e0, sh1), _mm256_sra_epi32(e1, sh1)),
                        0b11_01_10_00,
                    );
                    let vo = _mm256_permute4x64_epi64(
                        _mm256_packs_epi32(_mm256_sra_epi32(o0, sh1), _mm256_sra_epi32(o1, sh1)),
                        0b11_01_10_00,
                    );
                    let lo = _mm256_unpacklo_epi16(ve, vo); // pairs j..j+4 | j+8..j+12
                    let hi = _mm256_unpackhi_epi16(ve, vo); // pairs j+4..j+8 | j+12..j+16
                    let dst = inter.as_mut_ptr().add(p * 2 * N + 2 * j);
                    _mm256_storeu_si256(
                        dst as *mut __m256i,
                        _mm256_permute2x128_si256(lo, hi, 0x20),
                    );
                    _mm256_storeu_si256(
                        dst.add(16) as *mut __m256i,
                        _mm256_permute2x128_si256(lo, hi, 0x31),
                    );
                    j += 16;
                }
            }
            for j in 0..N {
                let mut x = 0;
                while x < N {
                    let (mut a0, mut a1) = (r2, r2);
                    for q in 0..np {
                        let c = _mm256_set1_epi32(
                            (table.add(q * 2 * N + 2 * j) as *const i32).read_unaligned(),
                        );
                        let src = inter.as_ptr().add(q * 2 * N + 2 * x);
                        a0 = _mm256_add_epi32(
                            a0,
                            _mm256_madd_epi16(_mm256_loadu_si256(src as *const __m256i), c),
                        );
                        a1 = _mm256_add_epi32(
                            a1,
                            _mm256_madd_epi16(_mm256_loadu_si256(src.add(16) as *const __m256i), c),
                        );
                    }
                    let v = _mm256_permute4x64_epi64(
                        _mm256_packs_epi32(_mm256_sra_epi32(a0, sh2), _mm256_sra_epi32(a1, sh2)),
                        0b11_01_10_00,
                    );
                    _mm256_storeu_si256(block.add(j * N + x) as *mut __m256i, v);
                    x += 16;
                }
            }
        }
    }

    fn fdct<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
        debug_assert_eq!(1usize << log2, N);
        let s1 = log2 as i32 + bit_depth as i32 - 9;
        let s2 = log2 as i32 + 6;
        if s1 <= 0 {
            return fdct_scalar::<N>(block, log2, bit_depth);
        }
        assert!(block.len() >= N * N, "block too small");
        unsafe { fwd_impl::<N>(block.as_mut_ptr(), s1, s2, super::fp::<N>().as_ptr()) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quant_impl(
        coeffs: *const i16,
        levels: *mut i16,
        n2: usize,
        scale: i32,
        qbits: u32,
        offset: i32,
    ) -> u32 {
        unsafe {
            let vs = _mm256_set1_epi16(scale as i16);
            let vo = _mm256_set1_epi32(offset);
            let sh = _mm_cvtsi32_si128(qbits as i32);
            let zero = _mm256_setzero_si256();
            let mut nz = 0u32;
            let mut i = 0;
            while i < n2 {
                let c = _mm256_loadu_si256(coeffs.add(i) as *const __m256i);
                let a = _mm256_abs_epi16(c);
                let lo = _mm256_mullo_epi16(a, vs);
                let hi = _mm256_mulhi_epu16(a, vs);
                let m0 = _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpacklo_epi16(lo, hi), vo), sh);
                let m1 = _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpackhi_epi16(lo, hi), vo), sh);
                let s = _mm256_srai_epi16(c, 15);
                let s0 = _mm256_unpacklo_epi16(s, s);
                let s1 = _mm256_unpackhi_epi16(s, s);
                let v0 = _mm256_sub_epi32(_mm256_xor_si256(m0, s0), s0);
                let v1 = _mm256_sub_epi32(_mm256_xor_si256(m1, s1), s1);
                // unpack and packs both work per lane, so this comes back
                // in the input order without a permute.
                let v = _mm256_packs_epi32(v0, v1);
                _mm256_storeu_si256(levels.add(i) as *mut __m256i, v);
                let z = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v, zero)) as u32;
                nz += 16 - z.count_ones() / 2;
                i += 16;
            }
            nz
        }
    }

    fn quant(
        coeffs: &[i16],
        levels: &mut [i16],
        n: usize,
        scale: i32,
        qbits: u32,
        offset: i32,
    ) -> u32 {
        let n2 = n * n;
        if n2 % 16 != 0 || !(0..=32767).contains(&scale) || offset < 0 || qbits > 31 {
            return quant_scalar(coeffs, levels, n, scale, qbits, offset);
        }
        assert!(coeffs.len() >= n2 && levels.len() >= n2, "block too small");
        unsafe {
            quant_impl(
                coeffs.as_ptr(),
                levels.as_mut_ptr(),
                n2,
                scale,
                qbits,
                offset,
            )
        }
    }
}

/// Install the best kernels `cpu` can run, one rung at a time.
pub fn install(d: &mut HevcEncDsp, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_quant(d);
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
    use crate::dsp::hevc_enc::{qbits, quant_offset, quant_scale};

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn rungs() -> Vec<(&'static str, HevcEncDsp)> {
        let base = Cpu::SCALAR;
        [
            (
                "sse2",
                Cpu { sse2: true, ..base },
                std::is_x86_feature_detected!("sse2"),
            ),
            (
                "ssse3",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    ..base
                },
                std::is_x86_feature_detected!("ssse3"),
            ),
            (
                "avx",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    sse41: true,
                    avx: true,
                    ..base
                },
                std::is_x86_feature_detected!("avx"),
            ),
            (
                "avx2",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    sse41: true,
                    avx: true,
                    avx2: true,
                    ..base
                },
                std::is_x86_feature_detected!("avx2"),
            ),
        ]
        .into_iter()
        .filter(|&(_, _, have)| have)
        .map(|(n, c, _)| {
            let mut d = HevcEncDsp::scalar();
            install(&mut d, c);
            (n, d)
        })
        .collect()
    }

    /// A residual block: mostly the range a real residual has, with whole
    /// rows at the i16 extremes now and then so the clamp after each stage
    /// and the widest products are reached.
    fn block(seed: &mut u64, n: usize, bit_depth: u32) -> Vec<i16> {
        let span = 1i32 << bit_depth;
        let mut b = vec![0i16; n * n];
        for y in 0..n {
            let mode = lcg(seed) % 6;
            for x in 0..n {
                b[y * n + x] = match mode {
                    0 => 32767,
                    1 => -32768,
                    2 => (lcg(seed) as i32 & 0xffff) as i16,
                    _ => ((lcg(seed) as i32 % (2 * span)) - span) as i16,
                };
            }
        }
        b
    }

    #[test]
    fn forward_transforms_match_scalar() {
        let s = HevcEncDsp::scalar();
        let mut seed = 0xfdc7_u64;
        for (name, d) in rungs() {
            for log2 in 2..6u32 {
                let n = 1usize << log2;
                for bit_depth in [8u32, 10, 12] {
                    for round in 0..40 {
                        let src = block(&mut seed, n, bit_depth);
                        let mut want = src.clone();
                        let mut got = src.clone();
                        (s.fdct[(log2 - 2) as usize])(&mut want, log2, bit_depth);
                        (d.fdct[(log2 - 2) as usize])(&mut got, log2, bit_depth);
                        assert_eq!(
                            got, want,
                            "{name} fdct {n}x{n} bd={bit_depth} round {round}"
                        );
                        if log2 == 2 {
                            let mut want = src.clone();
                            let mut got = src.clone();
                            (s.fdst4)(&mut want, bit_depth);
                            (d.fdst4)(&mut got, bit_depth);
                            assert_eq!(got, want, "{name} fdst4 bd={bit_depth} round {round}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantiser_matches_scalar() {
        let s = HevcEncDsp::scalar();
        let mut seed = 0x9a47_u64;
        for (name, d) in rungs() {
            for log2 in 2..6u32 {
                let n = 1usize << log2;
                for bit_depth in [8u32, 10] {
                    for qp in (0..52).step_by(3) {
                        for intra in [true, false] {
                            let qb = qbits(qp, log2, bit_depth);
                            let off = quant_offset(qb, intra);
                            let scale = quant_scale((qp % 6) as usize);
                            let coeffs = block(&mut seed, n, 12);
                            let mut want = vec![0i16; n * n];
                            let mut got = vec![0i16; n * n];
                            let nw = (s.quant)(&coeffs, &mut want, n, scale, qb, off);
                            let ng = (d.quant)(&coeffs, &mut got, n, scale, qb, off);
                            assert_eq!(
                                got, want,
                                "{name} quant {n}x{n} qp={qp} bd={bit_depth} intra={intra}"
                            );
                            assert_eq!(ng, nw, "{name} quant nz {n}x{n} qp={qp}");
                        }
                    }
                }
            }
        }
    }

    /// The pair table really is the matrix: every entry against
    /// `TRANSFORM32` directly, so a transposed build cannot pass by being
    /// consistently wrong in both stages.
    #[test]
    fn pair_table_is_the_matrix() {
        for &(n, t) in &[
            (4usize, &FP4[..]),
            (8, &FP8[..]),
            (16, &FP16[..]),
            (32, &FP32[..]),
        ] {
            let step = 32 / n;
            for q in 0..n / 2 {
                for j in 0..n {
                    assert_eq!(
                        t[q * 2 * n + 2 * j],
                        TRANSFORM32[j * step][2 * q] as i16,
                        "n={n} q={q} j={j}"
                    );
                    assert_eq!(
                        t[q * 2 * n + 2 * j + 1],
                        TRANSFORM32[j * step][2 * q + 1] as i16,
                        "n={n} q={q} j={j}"
                    );
                }
            }
        }
        for q in 0..2 {
            for j in 0..4 {
                assert_eq!(FDST[q * 8 + 2 * j], DST4[j][2 * q] as i16);
                assert_eq!(FDST[q * 8 + 2 * j + 1], DST4[j][2 * q + 1] as i16);
            }
        }
    }

    /// `cargo test --release hevc_enc_x86 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_bench() {
        use std::time::Instant;
        let s = HevcEncDsp::scalar();
        let mut seed = 0xb3c4_u64;
        let mut tables = vec![("scalar", s.clone()), ("scalar-again", s)];
        tables.extend(rungs());
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            let src = block(&mut seed, n, 8);
            let iters = 4_000_000 / (n * n);
            for (name, d) in &tables {
                let mut work = src.clone();
                let mut levels = vec![0i16; n * n];
                let mut sink = 0u32;
                let t = Instant::now();
                for _ in 0..iters {
                    work.copy_from_slice(&src);
                    (d.fdct[(log2 - 2) as usize])(&mut work, log2, 8);
                    sink = sink.wrapping_add((d.quant)(
                        &work,
                        &mut levels,
                        n,
                        20560,
                        qbits(26, log2, 8),
                        1 << 10,
                    ));
                }
                let ns = t.elapsed().as_nanos() as f64 / iters as f64;
                println!(
                    "{n}x{n} {name:13} {ns:8.1} ns per (fdct+quant) [{}]",
                    sink & 1
                );
            }
        }
    }
}
