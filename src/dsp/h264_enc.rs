//! H.264 forward transforms and quantisation — the encode-side counterpart
//! of `dsp::h264`'s inverse transforms and of `h264::transform`'s
//! dequantisation.
//!
//! The forward direction is not normative: any encoder whose output a
//! conformant decoder reconstructs acceptably is legal. What *is* fixed is
//! the pairing — the levels these kernels produce are dequantised and
//! inverse-transformed by code that is already conformance-proven, so the
//! test that matters is the round trip, and the quantisation tables here
//! are derived from the decoder's own dequantisation tables rather than
//! written down a second time.
//!
//! The derivation, so nobody has to rediscover it. Writing `W` for a
//! forward-transformed coefficient and `D` for what the inverse transform
//! must be fed to reconstruct the residual, the two transforms fix the
//! ratio `D / W` per position: `4 * (4/5)^c` for the 4x4, where `c` is the
//! number of odd indices, and `s[i % 4] * s[j % 4]` for the 8x8 with
//! `s = [1, 256/289, 8/5, 256/289]`. With a level `Z = W * MF >> qbits`
//! and the standard's dequantisation `D = Z * weight * V << (qP/6 - 4)`,
//! that ratio pins the product `MF * weight * V` to a constant per
//! position, which is all [`Quant`] computes. It reproduces the tables
//! every H.264 encoder carries, entry for entry, and a test asserts so.
//!
//! One property of that pairing is worth naming, because it is easy to
//! undo by accident. A round-trip test is *structurally blind* to a fault
//! applied symmetrically to both sides of an inverse pair: mutate the
//! forward transform and the inverse the same way and the round trip
//! still closes. What protects the tests here is that the inverse is not
//! ours — it is the decoder's, held in place by the conformance suites —
//! so a mistake on this side has nothing to cancel against. Writing a
//! private inverse "for the tests" would quietly remove that protection
//! and leave a test that can only catch asymmetric mistakes. Don't.
//!
//! Watch the shifts: 8.5.13.1 dequantises an 8x8 block by `qP/6 - 6`
//! where 8.5.12.1 uses `qP/6 - 4`, so although both multiplier tables are
//! scaled the same way, the 8x8 quantises with two bits fewer. Getting
//! that wrong leaves a pure scale error that the round-trip test catches
//! at every QP at once, which is how it was caught here.

use super::Cpu;
use crate::h264::sps::ScalingLists;
use crate::h264::tables::{DEQUANT4_INIT, DEQUANT8_INIT, DEQUANT8_INIT_SCAN};

/// Forward 4x4 integer transform: residual in raster order to coefficients
/// (the transform of 8.5.12.2 run forwards, without scaling).
pub type Fdct4Fn = fn(residual: &[i16; 16], coeffs: &mut [i32; 16]);
/// Forward 8x8 integer transform (8.5.13.2 run forwards).
pub type Fdct8Fn = fn(residual: &[i16; 64], coeffs: &mut [i32; 64]);
/// Forward 4x4 Hadamard over the Intra_16x16 luma DC coefficients (8.5.10
/// run forwards), in raster order.
pub type Hadamard4Fn = fn(dc: &mut [i32; 16]);
/// Forward 2x2 Hadamard over the 4:2:0 chroma DC coefficients (8.5.11.1).
pub type Hadamard2x2Fn = fn(dc: &mut [i32; 4]);
/// Forward 2x4 Hadamard over the 4:2:2 chroma DC coefficients (8.5.11.2).
pub type Hadamard2x4Fn = fn(dc: &mut [i32; 8]);
/// Quantise a 4x4 block: `mf` is the block's multiplier table (one entry a
/// position), `qbits` and `offset` the shift and rounding offset. Returns
/// the number of nonzero levels, which mode decision wants and which costs
/// nothing to count here.
pub type Quant4Fn = fn(coeffs: &[i32; 16], levels: &mut [i16; 16], mf: &[i32; 16], qbits: u32, offset: i32) -> u32;
/// The same for an 8x8 block.
pub type Quant8Fn = fn(coeffs: &[i32; 64], levels: &mut [i16; 64], mf: &[i32; 64], qbits: u32, offset: i32) -> u32;

/// The encode-side kernel table, filled at run time from what the CPU has,
/// exactly as [`super::h264::H264Dsp`] is. Kept separate from it because a
/// decoder never calls any of this and should not carry it.
#[derive(Clone)]
pub struct H264EncDsp {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Forward 4x4 transform.
    pub fdct4: Fdct4Fn,
    /// Forward 8x8 transform.
    pub fdct8: Fdct8Fn,
    /// Intra_16x16 luma DC forward Hadamard.
    pub hadamard4: Hadamard4Fn,
    /// 4:2:0 chroma DC forward Hadamard.
    pub hadamard2x2: Hadamard2x2Fn,
    /// 4:2:2 chroma DC forward Hadamard.
    pub hadamard2x4: Hadamard2x4Fn,
    /// 4x4 quantisation.
    pub quant4: Quant4Fn,
    /// 8x8 quantisation.
    pub quant8: Quant8Fn,
}

impl H264EncDsp {
    /// The scalar reference table — the executable definition every wider
    /// rung is checked against.
    pub const SCALAR: H264EncDsp = H264EncDsp {
        cpu: Cpu::SCALAR,
        fdct4: fdct4_scalar,
        fdct8: fdct8_scalar,
        hadamard4: hadamard4_scalar,
        hadamard2x2: hadamard2x2_scalar,
        hadamard2x4: hadamard2x4_scalar,
        quant4: quant4_scalar,
        quant8: quant8_scalar,
    };

    /// The best table for `cpu`. No rung replaces anything yet: which of
    /// these is worth hand-writing is a question for a profile of a real
    /// encoder, not for a guess, and the ladder lets them arrive one at a
    /// time.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::SCALAR;
        d.cpu = cpu;
        d
    }
}

impl Default for H264EncDsp {
    fn default() -> Self {
        Self::SCALAR
    }
}

// ----------------------------------------------------------------------
// Forward transforms
// ----------------------------------------------------------------------

/// One row of the forward 4x4 core transform (`Cf`).
#[inline(always)]
fn fdct4_1d(x: [i32; 4]) -> [i32; 4] {
    let s0 = x[0] + x[3];
    let s1 = x[1] + x[2];
    let s2 = x[1] - x[2];
    let s3 = x[0] - x[3];
    [s0 + s1, 2 * s3 + s2, s0 - s1, s3 - 2 * s2]
}

fn fdct4_scalar(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let r = [
            residual[i * 4] as i32,
            residual[i * 4 + 1] as i32,
            residual[i * 4 + 2] as i32,
            residual[i * 4 + 3] as i32,
        ];
        let o = fdct4_1d(r);
        tmp[i * 4..i * 4 + 4].copy_from_slice(&o);
    }
    for j in 0..4 {
        let o = fdct4_1d([tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]]);
        for (i, v) in o.iter().enumerate() {
            coeffs[i * 4 + j] = *v;
        }
    }
}

/// One row of the forward 8x8 transform, the mirror of `idct8_1d`.
#[inline(always)]
fn fdct8_1d(x: [i32; 8]) -> [i32; 8] {
    let a0 = x[0] + x[7];
    let a1 = x[1] + x[6];
    let a2 = x[2] + x[5];
    let a3 = x[3] + x[4];
    let a4 = x[0] - x[7];
    let a5 = x[1] - x[6];
    let a6 = x[2] - x[5];
    let a7 = x[3] - x[4];

    let b0 = a0 + a3;
    let b1 = a1 + a2;
    let b2 = a0 - a3;
    let b3 = a1 - a2;

    let b4 = a5 + a6 + ((a4 >> 1) + a4);
    let b5 = a4 - a7 - ((a6 >> 1) + a6);
    let b6 = a4 + a7 - ((a5 >> 1) + a5);
    let b7 = a5 - a6 + ((a7 >> 1) + a7);

    [
        b0 + b1,
        b4 + (b7 >> 2),
        b2 + (b3 >> 1),
        b5 + (b6 >> 2),
        b0 - b1,
        b6 - (b5 >> 2),
        (b2 >> 1) - b3,
        -b7 + (b4 >> 2),
    ]
}

fn fdct8_scalar(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
    let mut tmp = [0i32; 64];
    for i in 0..8 {
        let mut r = [0i32; 8];
        for (k, v) in r.iter_mut().enumerate() {
            *v = residual[i * 8 + k] as i32;
        }
        tmp[i * 8..i * 8 + 8].copy_from_slice(&fdct8_1d(r));
    }
    for j in 0..8 {
        let mut c = [0i32; 8];
        for (i, v) in c.iter_mut().enumerate() {
            *v = tmp[i * 8 + j];
        }
        let o = fdct8_1d(c);
        for (i, v) in o.iter().enumerate() {
            coeffs[i * 8 + j] = *v;
        }
    }
}

/// The Hadamard butterfly is its own inverse up to scale, so the forward
/// pass over the Intra_16x16 DC coefficients is the same four adds the
/// decoder runs.
#[inline(always)]
fn had4_1d(x: [i32; 4]) -> [i32; 4] {
    let s0 = x[0] + x[3];
    let s1 = x[1] + x[2];
    let s2 = x[1] - x[2];
    let s3 = x[0] - x[3];
    [s0 + s1, s3 + s2, s0 - s1, s3 - s2]
}

fn hadamard4_scalar(dc: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let o = had4_1d([dc[i * 4], dc[i * 4 + 1], dc[i * 4 + 2], dc[i * 4 + 3]]);
        tmp[i * 4..i * 4 + 4].copy_from_slice(&o);
    }
    for j in 0..4 {
        let o = had4_1d([tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]]);
        for (i, v) in o.iter().enumerate() {
            dc[i * 4 + j] = *v;
        }
    }
}

fn hadamard2x2_scalar(dc: &mut [i32; 4]) {
    let (a, b, c, d) = (dc[0], dc[1], dc[2], dc[3]);
    dc[0] = a + b + c + d;
    dc[1] = a - b + c - d;
    dc[2] = a + b - c - d;
    dc[3] = a - b - c + d;
}

fn hadamard2x4_scalar(dc: &mut [i32; 8]) {
    // Two columns of four, then the two-point transform across them
    // (8.5.11.2): the same shape the decoder inverts.
    let mut tmp = [0i32; 8];
    for j in 0..2 {
        let o = had4_1d([dc[j], dc[2 + j], dc[4 + j], dc[6 + j]]);
        for (i, v) in o.iter().enumerate() {
            tmp[i * 2 + j] = *v;
        }
    }
    for i in 0..4 {
        let (a, b) = (tmp[i * 2], tmp[i * 2 + 1]);
        dc[i * 2] = a + b;
        dc[i * 2 + 1] = a - b;
    }
}

// ----------------------------------------------------------------------
// Quantisation
// ----------------------------------------------------------------------

fn quant4_scalar(coeffs: &[i32; 16], levels: &mut [i16; 16], mf: &[i32; 16], qbits: u32, offset: i32) -> u32 {
    let mut nz = 0;
    for i in 0..16 {
        let c = coeffs[i];
        let m = (c.unsigned_abs() as i64 * mf[i] as i64 + offset as i64) >> qbits;
        let v = if c < 0 { -(m as i32) } else { m as i32 };
        levels[i] = v as i16;
        nz += (v != 0) as u32;
    }
    nz
}

fn quant8_scalar(coeffs: &[i32; 64], levels: &mut [i16; 64], mf: &[i32; 64], qbits: u32, offset: i32) -> u32 {
    let mut nz = 0;
    for i in 0..64 {
        let c = coeffs[i];
        let m = (c.unsigned_abs() as i64 * mf[i] as i64 + offset as i64) >> qbits;
        let v = if c < 0 { -(m as i32) } else { m as i32 };
        levels[i] = v as i16;
        nz += (v != 0) as u32;
    }
    nz
}

/// The shift a 4x4 block's quantisation uses at `qp`: `15 + qP / 6`.
#[inline]
pub fn qbits4(qp: i32) -> u32 {
    (15 + qp / 6) as u32
}

/// The shift an 8x8 block's quantisation uses at `qp`: `16 + qP / 6`. Two
/// less than the multipliers' scale, because 8.5.13.1 dequantises an 8x8
/// block by `qP / 6 - 6` where 8.5.12.1 uses `qP / 6 - 4`.
#[inline]
pub fn qbits8(qp: i32) -> u32 {
    (16 + qp / 6) as u32
}

/// The rounding offset of a dead-zone quantiser at `qbits`: a third of a
/// step for intra, a sixth for inter, which is what every encoder since JM
/// has used and what the round-trip tests here assume.
#[inline]
pub fn quant_offset(qbits: u32, intra: bool) -> i32 {
    let denom = if intra { 3 } else { 6 };
    (1i32 << qbits) / denom
}

/// `MF4x4[list][qP % 6][raster]` and `MF8x8[list][qP % 6][raster]`, the
/// forward counterpart of `h264::transform::Dequant` and built
/// from the same scaling lists and the same `normAdjust` tables, so the
/// two cannot drift apart.
pub struct Quant {
    /// 4x4: `[list][qp % 6][pos]`.
    pub mf4: [[[i32; 16]; 6]; 6],
    /// 8x8: `[list][qp % 6][pos]`.
    pub mf8: [[[i32; 64]; 6]; 6],
}

/// `round(2^shift * num / (den * scale))` in integers.
#[inline]
fn mf_entry(shift: u32, num: i64, den: i64, scale: i64) -> i32 {
    let a = (1i64 << shift) * num;
    let b = den * scale;
    ((a + b / 2) / b) as i32
}

impl Quant {
    /// Build from the effective scaling lists.
    pub fn new(lists: &ScalingLists) -> Self {
        let mut mf4 = [[[0i32; 16]; 6]; 6];
        let mut mf8 = [[[0i32; 64]; 6]; 6];
        // `D / W` for the 4x4 is `4 * (4/5)^c`, and the extra 2^4 folds the
        // default weight of 16 out of the denominator: 2^15 * 4 * 2^4.
        for list in 0..6 {
            for m in 0..6 {
                for pos in 0..16 {
                    let (i, j) = (pos / 4, pos % 4);
                    let c = (i % 2) + (j % 2);
                    let (num, den) = (4i64.pow(c as u32), 5i64.pow(c as u32));
                    let scale = lists.list4x4[list][pos] as i64 * DEQUANT4_INIT[m][c] as i64;
                    mf4[list][m][pos] = mf_entry(21, num, den, scale);
                }
                for pos in 0..64 {
                    let (i, j) = (pos / 8, pos % 8);
                    let c = DEQUANT8_INIT_SCAN[(i % 4) * 4 + (j % 4)] as usize;
                    // `s = [1, 256/289, 8/5, 256/289]` by index % 4, and the
                    // position's ratio is the product of the two.
                    let s = |k: usize| -> (i64, i64) {
                        match k % 4 {
                            0 => (1, 1),
                            2 => (8, 5),
                            _ => (256, 289),
                        }
                    };
                    let (n0, d0) = s(i);
                    let (n1, d1) = s(j);
                    let scale = lists.list8x8[list][pos] as i64 * DEQUANT8_INIT[m][c] as i64;
                    mf8[list][m][pos] = mf_entry(22, n0 * n1, d0 * d1, scale);
                }
            }
        }
        Quant { mf4, mf8 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::transform::{Dequant, dequant4x4, idct4x4, idct8x8};

    fn flat_lists() -> ScalingLists {
        ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        }
    }

    fn lcg(s: &mut u64) -> i32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) & 0x1ff) as i32 - 255
    }

    /// The tables every H.264 encoder carries, which this module derives
    /// from the decoder's own `normAdjust` rather than repeating.
    #[test]
    fn quant_tables_match_the_canonical_ones() {
        let q = Quant::new(&flat_lists());
        #[rustfmt::skip]
        let canon4: [[i32; 3]; 6] = [
            [13107, 8066, 5243], [11916, 7490, 4660], [10082, 6554, 4194],
            [9362, 5825, 3647], [8192, 5243, 3355], [7282, 4559, 2893],
        ];
        for m in 0..6 {
            for pos in 0..16 {
                let c = (pos / 4) % 2 + (pos % 4) % 2;
                assert_eq!(q.mf4[0][m][pos], canon4[m][c], "mf4 m={m} pos={pos}");
            }
        }
        #[rustfmt::skip]
        let canon8: [[i32; 6]; 6] = [
            [13107, 11428, 20972, 12222, 16777, 15481],
            [11916, 10826, 19174, 11058, 14980, 14290],
            [10082, 8943, 15978, 9675, 12710, 11985],
            [9362, 8228, 14913, 8931, 11984, 11259],
            [8192, 7346, 13159, 7740, 10486, 9777],
            [7282, 6428, 11570, 6830, 9118, 8640],
        ];
        for m in 0..6 {
            for pos in 0..64 {
                let c = DEQUANT8_INIT_SCAN[((pos / 8) % 4) * 4 + (pos % 8) % 4] as usize;
                assert_eq!(q.mf8[0][m][pos], canon8[m][c], "mf8 m={m} pos={pos}");
            }
        }
    }

    /// The invariant that matters: what these kernels quantise, the
    /// decoder's own dequantisation and inverse transform bring back. The
    /// bound grows with the step, so it is checked against the step rather
    /// than a constant — a scale error anywhere in the pair breaks it at
    /// every QP at once, which is what makes this worth more than a
    /// benchmark.
    #[test]
    fn forward_and_inverse_4x4_round_trip() {
        let q = Quant::new(&flat_lists());
        let dq = Dequant::new(&flat_lists());
        let mut seed = 0x1234_5678u64;
        for qp in 0..52 {
            let (m, q6) = ((qp % 6) as usize, qp / 6);
            let qbits = qbits4(qp);
            let offset = quant_offset(qbits, true);
            let step = 1i32 << q6;
            let mut worst = 0;
            for _ in 0..64 {
                let mut res = [0i16; 16];
                for r in res.iter_mut() {
                    *r = lcg(&mut seed) as i16;
                }
                let mut co = [0i32; 16];
                fdct4_scalar(&res, &mut co);
                let mut lv = [0i16; 16];
                quant4_scalar(&co, &mut lv, &q.mf4[0][m], qbits, offset);
                let mut d = [0i32; 16];
                for i in 0..16 {
                    d[i] = lv[i] as i32;
                }
                dequant4x4(&mut d, &dq.scale4[0][m], qp, false);
                idct4x4(&mut d);
                for i in 0..16 {
                    worst = worst.max((d[i] - res[i] as i32).abs());
                }
            }
            // A dead-zone quantiser at this step cannot be tighter than the
            // step itself; the slack is for the transform's own rounding.
            assert!(worst <= 6 * step + 8, "qp={qp} worst={worst} step={step}");
        }
    }

    #[test]
    fn forward_and_inverse_8x8_round_trip() {
        let q = Quant::new(&flat_lists());
        let dq = Dequant::new(&flat_lists());
        let mut seed = 0x9876_5432u64;
        for qp in 0..52 {
            let (m, q6) = ((qp % 6) as usize, qp / 6);
            let qbits = qbits8(qp);
            let offset = quant_offset(qbits, true);
            let step = 1i32 << q6;
            let mut worst = 0;
            for _ in 0..32 {
                let mut res = [0i16; 64];
                for r in res.iter_mut() {
                    *r = lcg(&mut seed) as i16;
                }
                let mut co = [0i32; 64];
                fdct8_scalar(&res, &mut co);
                let mut lv = [0i16; 64];
                quant8_scalar(&co, &mut lv, &q.mf8[0][m], qbits, offset);
                let mut d = [0i32; 64];
                for i in 0..64 {
                    d[i] = lv[i] as i32 * dq.scale8[0][m][i];
                    d[i] = if qp >= 36 {
                        d[i] << (q6 - 6)
                    } else {
                        (d[i] + (1 << (5 - q6))) >> (6 - q6)
                    };
                }
                idct8x8(&mut d);
                for i in 0..64 {
                    worst = worst.max((d[i] - res[i] as i32).abs());
                }
            }
            assert!(worst <= 12 * step + 24, "qp={qp} worst={worst} step={step}");
        }
    }

    /// A flat block has all its energy in DC, whichever transform runs.
    #[test]
    fn a_flat_block_is_dc_only() {
        let mut co = [0i32; 16];
        fdct4_scalar(&[42i16; 16], &mut co);
        assert_eq!(co[0], 42 * 16);
        assert!(co[1..].iter().all(|&v| v == 0));
        let mut c8 = [0i32; 64];
        fdct8_scalar(&[7i16; 64], &mut c8);
        assert_eq!(c8[0], 7 * 64);
        assert!(c8[1..].iter().all(|&v| v == 0));
    }
}
