//! Distortion metrics: how far one block of samples is from another.
//!
//! Shared by both encoders, because none of it is codec-specific — a sum of
//! absolute differences does not care which standard asked for it. These are
//! the kernels an encoder calls most: mode decision evaluates one per
//! candidate per block, and motion search evaluates one per position
//! searched, so between them they see more samples than everything else in
//! an encoder put together.
//!
//! Three metrics, and the choice between them is a real one:
//!
//! - **SAD**, the sum of absolute differences. Cheapest, and what a motion
//!   search uses for its wide passes.
//! - **SATD**, the same after a Hadamard transform. Costs a few times more
//!   and picks visibly better modes, because it measures the residual the
//!   way the transform that will actually code it does — a residual that
//!   happens to be a smooth ramp is expensive to code but cheap in SAD, and
//!   SATD is what notices.
//! - **SSD**, the sum of squared differences. What rate-distortion
//!   optimisation needs, since it is the distortion term that pairs with a
//!   bit count in a Lagrangian.
//!
//! All three take two strided views and a block size rather than a fixed
//! shape, so one entry serves every partition size in both codecs. If a
//! profile ever shows the size-generic dispatch costing more than it saves,
//! the table can grow per-size entries without any caller changing.

use super::Cpu;
use crate::sample::Sample;

/// Sum of absolute differences over a `w` by `h` block.
pub type SadFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32;
/// Sum of absolute Hadamard-transformed differences over a `w` by `h`
/// block, both multiples of four.
pub type SatdFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32;
/// Sum of squared differences over a `w` by `h` block. Wider than the
/// others because at 12 bits a 64x64 block overflows 32 bits.
pub type SsdFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u64;

/// The distortion kernels, filled at run time from what the CPU has.
#[derive(Clone)]
pub struct DistortionDsp<S: Sample = u8> {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Sum of absolute differences.
    pub sad: SadFn<S>,
    /// Sum of absolute Hadamard-transformed differences.
    pub satd: SatdFn<S>,
    /// Sum of squared differences.
    pub ssd: SsdFn<S>,
}

impl<S: Sample> DistortionDsp<S> {
    /// The scalar reference table — the executable definition every wider
    /// rung is checked against.
    pub fn scalar() -> Self {
        DistortionDsp {
            cpu: Cpu::SCALAR,
            sad: sad_scalar::<S>,
            satd: satd_scalar::<S>,
            ssd: ssd_scalar::<S>,
        }
    }

    /// The best table for `cpu`, built the way the decoders' tables are:
    /// the scalar reference first, then each rung of the ladder replacing
    /// the entries it has a kernel for.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        install_simd(&mut d, cpu);
        d
    }
}

/// The SIMD kernels exist for 8-bit samples, which is what both encoders
/// work in today; a 16-bit table keeps the scalar reference. Dispatched on
/// the sample type here rather than through a method on [`Sample`], so
/// this table's ladder does not touch the trait the decoders share.
#[allow(unused_variables)]
fn install_simd<S: Sample>(d: &mut DistortionDsp<S>, cpu: Cpu) {
    use std::any::Any;
    if super::enc_simd_disabled("distortion") {
        return;
    }
    if let Some(d) = (d as &mut dyn Any).downcast_mut::<DistortionDsp<u8>>() {
        #[cfg(target_arch = "x86_64")]
        super::distortion_x86::install(d, cpu);
    }
}

impl<S: Sample> Default for DistortionDsp<S> {
    fn default() -> Self {
        Self::scalar()
    }
}

pub(crate) fn sad_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let (ra, rb) = (&a[y * a_stride..], &b[y * b_stride..]);
        for x in 0..w {
            sum += ra[x].to_i32().abs_diff(rb[x].to_i32());
        }
    }
    sum
}

pub(crate) fn ssd_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u64 {
    let mut sum = 0u64;
    for y in 0..h {
        let (ra, rb) = (&a[y * a_stride..], &b[y * b_stride..]);
        for x in 0..w {
            let d = (ra[x].to_i32() - rb[x].to_i32()) as i64;
            sum += (d * d) as u64;
        }
    }
    sum
}

/// The 4x4 Hadamard butterfly, in place over rows then columns.
#[inline(always)]
fn hadamard4x4(d: &mut [i32; 16]) {
    for i in 0..4 {
        let (a, b, c, e) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let (s0, s1, s2, s3) = (a + e, b + c, b - c, a - e);
        d[i * 4] = s0 + s1;
        d[i * 4 + 1] = s3 + s2;
        d[i * 4 + 2] = s0 - s1;
        d[i * 4 + 3] = s3 - s2;
    }
    for j in 0..4 {
        let (a, b, c, e) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
        let (s0, s1, s2, s3) = (a + e, b + c, b - c, a - e);
        d[j] = s0 + s1;
        d[4 + j] = s3 + s2;
        d[8 + j] = s0 - s1;
        d[12 + j] = s3 - s2;
    }
}

/// SATD over 4x4 tiles. The `(sum + 1) >> 1` is the normalisation every
/// encoder since JM uses, chosen so that a SATD is on roughly the same
/// scale as the SAD of the same block and the two can share a Lagrangian
/// constant; it matters only that it is consistent, and it is stated here
/// so nobody has to infer it from a magic number later.
pub(crate) fn satd_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32 {
    debug_assert!(w % 4 == 0 && h % 4 == 0, "SATD wants a multiple of four");
    let mut total = 0u32;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for y in 0..4 {
                let (ra, rb) = (&a[(by + y) * a_stride + bx..], &b[(by + y) * b_stride + bx..]);
                for x in 0..4 {
                    d[y * 4 + x] = ra[x].to_i32() - rb[x].to_i32();
                }
            }
            hadamard4x4(&mut d);
            let sum: u32 = d.iter().map(|v| v.unsigned_abs()).sum();
            total += (sum + 1) >> 1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// A block against itself is zero distortion by every metric, which is
    /// the one value all three must agree on.
    #[test]
    fn a_block_against_itself_is_zero() {
        let mut seed = 5u64;
        let p: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        for &(w, h) in &[(4, 4), (8, 8), (16, 16), (16, 8), (8, 16), (64, 64)] {
            assert_eq!(sad_scalar(&p, 64, &p, 64, w, h), 0);
            assert_eq!(ssd_scalar(&p, 64, &p, 64, w, h), 0);
            assert_eq!(satd_scalar(&p, 64, &p, 64, w, h), 0);
        }
    }

    /// A constant offset has a closed form for each metric: SAD is the
    /// offset times the area, SSD its square times the area, and SATD sees
    /// it as pure DC — one coefficient of 16d a tile, halved.
    #[test]
    fn a_constant_offset_has_a_closed_form() {
        for d in [1i32, 7, 30] {
            let a = vec![100u8; 64 * 64];
            let b = vec![(100 + d) as u8; 64 * 64];
            for &(w, h) in &[(4, 4), (8, 8), (16, 16), (16, 4)] {
                let area = (w * h) as u32;
                assert_eq!(sad_scalar(&a, 64, &b, 64, w, h), d as u32 * area);
                assert_eq!(ssd_scalar(&a, 64, &b, 64, w, h), (d * d) as u64 * area as u64);
                let tiles = area / 16;
                assert_eq!(satd_scalar(&a, 64, &b, 64, w, h), tiles * ((16 * d as u32 + 1) >> 1));
            }
        }
    }

    /// SATD is the sum of absolute Hadamard coefficients, so it must agree
    /// with the transform computed the long way — a matrix multiply by the
    /// 4x4 Hadamard matrix, which shares no code with the butterfly.
    #[test]
    fn satd_agrees_with_a_direct_hadamard() {
        const H: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
        let mut seed = 99u64;
        for _ in 0..200 {
            let a: Vec<u8> = (0..16).map(|_| lcg(&mut seed) as u8).collect();
            let b: Vec<u8> = (0..16).map(|_| lcg(&mut seed) as u8).collect();
            let mut diff = [[0i32; 4]; 4];
            for y in 0..4 {
                for x in 0..4 {
                    diff[y][x] = a[y * 4 + x] as i32 - b[y * 4 + x] as i32;
                }
            }
            // H * diff * H^T, term by term.
            let mut want = 0u32;
            for u in 0..4 {
                for v in 0..4 {
                    let mut acc = 0i32;
                    for y in 0..4 {
                        for x in 0..4 {
                            acc += H[u][y] * diff[y][x] * H[v][x];
                        }
                    }
                    want += acc.unsigned_abs();
                }
            }
            assert_eq!(satd_scalar(&a, 4, &b, 4, 4, 4), (want + 1) >> 1);
        }
    }

    /// Strides must be honoured: the same block read out of a wider plane
    /// gives the same answer.
    #[test]
    fn strides_are_honoured() {
        let mut seed = 21u64;
        let wide: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        let other: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        let mut packed_a = Vec::new();
        let mut packed_b = Vec::new();
        for y in 0..8 {
            packed_a.extend_from_slice(&wide[(3 + y) * 64 + 5..(3 + y) * 64 + 13]);
            packed_b.extend_from_slice(&other[(3 + y) * 64 + 5..(3 + y) * 64 + 13]);
        }
        let a = &wide[3 * 64 + 5..];
        let b = &other[3 * 64 + 5..];
        assert_eq!(sad_scalar(a, 64, b, 64, 8, 8), sad_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
        assert_eq!(ssd_scalar(a, 64, b, 64, 8, 8), ssd_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
        assert_eq!(satd_scalar(a, 64, b, 64, 8, 8), satd_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
    }

    /// Ten bits per sample must not overflow, and a 64x64 SSD at full
    /// deflection is why `ssd` returns 64 bits: it is 2^38 there.
    #[test]
    fn wide_samples_do_not_overflow() {
        let a = vec![0u16; 64 * 64];
        let b = vec![1023u16; 64 * 64];
        assert_eq!(sad_scalar(&a, 64, &b, 64, 64, 64), 1023 * 4096);
        assert_eq!(ssd_scalar(&a, 64, &b, 64, 64, 64), 1023u64 * 1023 * 4096);
        assert_eq!(satd_scalar(&a, 64, &b, 64, 64, 64), 256 * ((16 * 1023 + 1) >> 1));
    }
}
