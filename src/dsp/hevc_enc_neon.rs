//! AArch64 NEON versions of the H.265 forward transforms and quantiser —
//! the same kernels as [`super::hevc_enc_x86`], with the same arithmetic
//! and roundings, on the instructions this architecture has for them.
//!
//! NEON has no `pmaddwd`, so the transform is not the pair-table shape.
//! What it has instead is a widening multiply-accumulate *by lane*
//! (`smlal` / `smlal2` with a lane index), which turns the matrix
//! multiply into its most direct form: for one input row, each sample
//! `x[k]` is a lane of the row vector and is multiplied against the eight
//! consecutive outputs' weights `M[j..j+8][k]`, a contiguous vector when
//! the matrix is stored transposed (`hevc_enc::ct`). The second stage
//! swaps the roles — the weights of one output row `M[j][k..k+8]` are the
//! lane vector (the matrix stored as it is, `hevc_enc::mt`) and the data
//! rows of the intermediate are what gets strided through — so neither
//! stage transposes anything. Every product is `i16 x i16` accumulated in
//! i32 (at most 32 terms of `90 x 32767`, under 2^27); the rounding term
//! is the accumulator's initial value, the shift an arithmetic `sshl` by a
//! negative count, and the clamp to i16 is `sqxtn`'s saturation — the
//! reference's `(sum + (1 << (s - 1))) >> s` and `clamp` exactly.
//!
//! The quantiser multiplies the u16 magnitude (`abs` of `-32768` wraps to
//! `0x8000`, which is 32768 as u16 — the one value an i16 absolute cannot
//! represent, still right) by the u16 scale with `umull`, an exact 32-bit
//! product; adds the offset; shifts logically (`ushl` by a negative
//! count) on a value nonnegative by construction; and puts the sign back
//! with `(m ^ s) - s`. Nonzero levels are counted by subtracting the
//! all-ones `cmeq`-with-zero masks into a lane counter (adding one per
//! zero) and reading it once at the end.
//!
//! Written on x86 and checked for compilation against
//! `aarch64-unknown-linux-gnu`; the bit-exactness tests below are the x86
//! module's and execute on the CI arm64 runners.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::hevc_enc::layouts::{CDST, MDST, ct, mt};
use super::hevc_enc::{HevcEncDsp, fdct_scalar, fdst4_scalar, quant_scalar};

/// Install the NEON kernels.
pub fn install(d: &mut HevcEncDsp, cpu: Cpu) {
    if cpu.neon {
        d.fdct = [fdct::<4>, fdct::<8>, fdct::<16>, fdct::<32>];
        d.fdst4 = fdst4;
        d.quant = quant;
    }
}

/// `acc += lanes[l] * v[l]` for `l` in `0..8`, where `v[l]` is the eight
/// i16 at `base + l * stride`: the low four outputs in `lo`, the high
/// four in `hi`.
#[inline(always)]
unsafe fn mla8(
    mut lo: int32x4_t,
    mut hi: int32x4_t,
    base: *const i16,
    stride: usize,
    lanes: int16x8_t,
) -> (int32x4_t, int32x4_t) {
    unsafe {
        macro_rules! lane {
            ($l:literal) => {{
                let v = vld1q_s16(base.add($l * stride));
                lo = vmlal_laneq_s16::<$l>(lo, vget_low_s16(v), lanes);
                hi = vmlal_high_laneq_s16::<$l>(hi, v, lanes);
            }};
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
        lane!(4);
        lane!(5);
        lane!(6);
        lane!(7);
        (lo, hi)
    }
}

/// The four-lane form: `acc += lanes[l] * v[l]` for `l` in `0..4`, `v[l]`
/// the four i16 at `base + l * stride`.
#[inline(always)]
unsafe fn mla4(mut acc: int32x4_t, base: *const i16, stride: usize, lanes: int16x4_t) -> int32x4_t {
    unsafe {
        macro_rules! lane {
            ($l:literal) => {{
                acc = vmlal_lane_s16::<$l>(acc, vld1_s16(base.add($l * stride)), lanes);
            }};
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
        acc
    }
}

/// `(acc >> s)` arithmetic, saturated to i16 — the rounding term is
/// already in `acc`.
#[inline(always)]
unsafe fn narrow(acc: int32x4_t, neg_shift: int32x4_t) -> int16x4_t {
    unsafe { vqmovn_s32(vshlq_s32(acc, neg_shift)) }
}

/// Both stages of an `N`-point forward transform over `block` (raster,
/// in place), `N >= 8`. `col` is the transposed matrix (`CT[k][j]`), `row`
/// the matrix itself (`MT[j][k]`). `s1 >= 1`.
#[target_feature(enable = "neon")]
unsafe fn fwd_impl<const N: usize>(
    block: *mut i16,
    s1: i32,
    s2: i32,
    col: *const i16,
    row: *const i16,
) {
    unsafe {
        let mut tmp = [0i16; 32 * 32];
        let r1 = vdupq_n_s32(1 << (s1 - 1));
        let r2 = vdupq_n_s32(1 << (s2 - 1));
        let n1 = vdupq_n_s32(-s1);
        let n2 = vdupq_n_s32(-s2);
        // Stage 1 (rows): tmp[y][j] = sum_k M[j][k] x[y][k]; eight outputs
        // j..j+8 at a time from the transposed matrix's rows k.
        for y in 0..N {
            let x = block.add(y * N);
            let mut j = 0;
            while j < N {
                let (mut lo, mut hi) = (r1, r1);
                for m in 0..N / 8 {
                    let lanes = vld1q_s16(x.add(8 * m));
                    (lo, hi) = mla8(lo, hi, col.add(8 * m * N + j), N, lanes);
                }
                vst1q_s16(
                    tmp.as_mut_ptr().add(y * N + j),
                    vcombine_s16(narrow(lo, n1), narrow(hi, n1)),
                );
                j += 8;
            }
        }
        // Stage 2 (columns): out[j][x] = sum_k M[j][k] tmp[k][x]; eight
        // columns x..x+8 at a time, the weights of output row j the lanes.
        for j in 0..N {
            let mut x = 0;
            while x < N {
                let (mut lo, mut hi) = (r2, r2);
                for m in 0..N / 8 {
                    let lanes = vld1q_s16(row.add(j * N + 8 * m));
                    (lo, hi) = mla8(lo, hi, tmp.as_ptr().add(8 * m * N + x), N, lanes);
                }
                vst1q_s16(
                    block.add(j * N + x),
                    vcombine_s16(narrow(lo, n2), narrow(hi, n2)),
                );
                x += 8;
            }
        }
    }
}

/// The 4-point form, for the 4x4 DCT and the DST: one vector a row.
#[target_feature(enable = "neon")]
unsafe fn fwd4_impl(block: *mut i16, s1: i32, s2: i32, col: *const i16, row: *const i16) {
    unsafe {
        let mut tmp = [0i16; 16];
        let r1 = vdupq_n_s32(1 << (s1 - 1));
        let r2 = vdupq_n_s32(1 << (s2 - 1));
        let n1 = vdupq_n_s32(-s1);
        let n2 = vdupq_n_s32(-s2);
        for y in 0..4 {
            let lanes = vld1_s16(block.add(y * 4));
            let acc = mla4(r1, col, 4, lanes);
            vst1_s16(tmp.as_mut_ptr().add(y * 4), narrow(acc, n1));
        }
        for j in 0..4 {
            let lanes = vld1_s16(row.add(j * 4));
            let acc = mla4(r2, tmp.as_ptr(), 4, lanes);
            vst1_s16(block.add(j * 4), narrow(acc, n2));
        }
    }
}

fn fdct<const N: usize>(block: &mut [i16], log2: u32, bit_depth: u32) {
    debug_assert_eq!(1usize << log2, N);
    let s1 = log2 as i32 + bit_depth as i32 - 9;
    let s2 = log2 as i32 + 6;
    // A first-stage shift of zero has no rounding term; only a bit depth
    // below eight gets there, and the reference handles it.
    if s1 <= 0 {
        return fdct_scalar::<N>(block, log2, bit_depth);
    }
    assert!(block.len() >= N * N, "block too small");
    unsafe {
        if N == 4 {
            fwd4_impl(
                block.as_mut_ptr(),
                s1,
                s2,
                ct::<4>().as_ptr(),
                mt::<4>().as_ptr(),
            )
        } else {
            fwd_impl::<N>(
                block.as_mut_ptr(),
                s1,
                s2,
                ct::<N>().as_ptr(),
                mt::<N>().as_ptr(),
            )
        }
    }
}

fn fdst4(block: &mut [i16], bit_depth: u32) {
    let s1 = 2 + bit_depth as i32 - 9;
    if s1 <= 0 {
        return fdst4_scalar(block, bit_depth);
    }
    assert!(block.len() >= 16, "block too small");
    unsafe { fwd4_impl(block.as_mut_ptr(), s1, 8, CDST.as_ptr(), MDST.as_ptr()) }
}

#[target_feature(enable = "neon")]
unsafe fn quant_impl(
    coeffs: *const i16,
    levels: *mut i16,
    n2: usize,
    scale: i32,
    qbits: u32,
    offset: i32,
) -> u32 {
    unsafe {
        let vs = vdup_n_u16(scale as u16);
        let vo = vdupq_n_u32(offset as u32);
        let sh = vdupq_n_s32(-(qbits as i32));
        // One per zero level, accumulated by subtracting the all-ones masks.
        let mut zeros = vdupq_n_u16(0);
        let mut i = 0;
        while i < n2 {
            let c = vld1q_s16(coeffs.add(i));
            let a = vreinterpretq_u16_s16(vabsq_s16(c));
            let m0 = vshlq_u32(vaddq_u32(vmull_u16(vget_low_u16(a), vs), vo), sh);
            let m1 = vshlq_u32(vaddq_u32(vmull_high_u16(a, vcombine_u16(vs, vs)), vo), sh);
            let s = vshrq_n_s16::<15>(c);
            let s0 = vmovl_s16(vget_low_s16(s));
            let s1 = vmovl_high_s16(s);
            let v0 = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(m0), s0), s0);
            let v1 = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(m1), s1), s1);
            let v = vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
            vst1q_s16(levels.add(i), v);
            zeros = vsubq_u16(zeros, vceqzq_s16(v));
            i += 8;
        }
        n2 as u32 - vaddvq_u16(zeros) as u32
    }
}

fn quant(coeffs: &[i16], levels: &mut [i16], n: usize, scale: i32, qbits: u32, offset: i32) -> u32 {
    let n2 = n * n;
    // The product needs `scale` in u16 and the sum in u32, which
    // `quant_scale` (at most 26214) and a shift of at least 14 guarantee;
    // anything else is the reference's.
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

    /// The NEON table. NEON is baseline on AArch64, so unlike the x86
    /// rungs there is nothing to skip on: this test cannot be vacuous on
    /// the architecture it compiles for.
    fn neon() -> HevcEncDsp {
        let mut d = HevcEncDsp::scalar();
        install(
            &mut d,
            Cpu {
                neon: true,
                ..Cpu::SCALAR
            },
        );
        let s = HevcEncDsp::scalar();
        assert!(
            d.quant as usize != s.quant as usize,
            "the NEON quantiser did not install"
        );
        assert!(
            d.fdct[3] as usize != s.fdct[3] as usize,
            "the NEON transform did not install"
        );
        d
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
        let d = neon();
        let mut seed = 0xfdc7_u64;
        for log2 in 2..6u32 {
            let n = 1usize << log2;
            for bit_depth in [8u32, 10, 12] {
                for round in 0..40 {
                    let src = block(&mut seed, n, bit_depth);
                    let mut want = src.clone();
                    let mut got = src.clone();
                    (s.fdct[(log2 - 2) as usize])(&mut want, log2, bit_depth);
                    (d.fdct[(log2 - 2) as usize])(&mut got, log2, bit_depth);
                    assert_eq!(got, want, "neon fdct {n}x{n} bd={bit_depth} round {round}");
                    if log2 == 2 {
                        let mut want = src.clone();
                        let mut got = src.clone();
                        (s.fdst4)(&mut want, bit_depth);
                        (d.fdst4)(&mut got, bit_depth);
                        assert_eq!(got, want, "neon fdst4 bd={bit_depth} round {round}");
                    }
                }
            }
        }
    }

    #[test]
    fn quantiser_matches_scalar() {
        let s = HevcEncDsp::scalar();
        let d = neon();
        let mut seed = 0x9a47_u64;
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
                            "neon quant {n}x{n} qp={qp} bd={bit_depth} intra={intra}"
                        );
                        assert_eq!(ng, nw, "neon quant nz {n}x{n} qp={qp}");
                    }
                }
            }
        }
    }

    /// `HevcEncDsp::new` reaches these through the detected CPU, and on
    /// this architecture that must be the NEON table.
    #[test]
    fn new_installs_neon() {
        let d = HevcEncDsp::new(Cpu::detect());
        let s = HevcEncDsp::scalar();
        assert!(d.cpu.neon);
        assert!(d.quant as usize != s.quant as usize);
    }
}
