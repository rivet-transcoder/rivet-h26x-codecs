//! Adaptive quantisation: a quantiser offset per coding tree block from
//! the block's own luma variance.
//!
//! The first user of per-block quantisers, and deliberately the simplest
//! one that does something a picture-wide quantiser cannot: a flat block
//! and a textured one at the same quantiser do not look equally wrong.
//! Quantisation error in a flat region is visible as banding and blocking
//! at magnitudes that vanish under texture, so spending a finer quantiser
//! on the flat blocks and a coarser one on the textured ones moves error
//! to where it is masked. That is x265's `aq-mode 1` in spirit: log
//! variance in, quantiser offset out.
//!
//! # What it does to the numbers
//!
//! It **lowers global PSNR** at a given rate, on purpose. PSNR weighs every
//! sample the same, and this moves bits away from the textured samples
//! where a fixed quantiser earns most of its PSNR. A caller measuring PSNR
//! wants it off, which is why it is a switch and why the default is off.
//! It is reported here as a size/PSNR delta rather than claimed as an
//! improvement, because the property it improves — evenness of visible
//! error — is not one the gate measures.
//!
//! # The offset
//!
//! Per CTB, the luma variance in 8-bit units (deeper samples are scaled
//! down, so a strength means the same thing at every depth), then
//! `energy = log2(variance + 1)`. The offset is
//! `strength * SLOPE * (energy - mean energy)`, rounded, and clamped to
//! [`AQ_MAX_OFFSET`] either way. Two choices in that formula are worth
//! naming:
//!
//! - **Zero-mean over the picture**, against the picture's own mean
//!   energy rather than a fixed constant. The picture quantiser is what
//!   the rate controller chose; the offsets redistribute it and must not
//!   also move it, or the controller would be steering a number it does
//!   not see. The clamp breaks the zero mean slightly on pictures with a
//!   few extreme blocks, which is accepted.
//! - **A slope of one half**, so that strength 1.0 gives about ±3 across
//!   ordinary content (six stops of variance either side of the mean)
//!   and the clamp is reached only by the flattest blocks against the
//!   busiest. The number is a calibration, not a derivation.
//!
//! The offset is a *wish*: a block that ends up with no coded residual
//! cannot carry a `cu_qp_delta` and takes the predicted quantiser
//! instead. The encoder's quantiser chain (`encode::h265`) resolves that,
//! not this module, which only says what each block would like.

use crate::sample::Sample;

/// The largest quantiser offset adaptive quantisation may apply, either
/// way. Six steps is a factor of two in step size: enough to matter,
/// small enough that a block never lands more than one octave from its
/// neighbours, which keeps the per-block deltas cheap to code and the
/// picture from breaking into visibly different quality regions.
pub(crate) const AQ_MAX_OFFSET: i32 = 6;

/// Quantiser steps per stop of log-variance at strength 1.0. See the
/// module documentation.
const SLOPE: f64 = 0.5;

/// The quantiser offset each CTB of a `width` by `height` luma plane
/// (coded size; whole CTBs of `1 << log2_ctb`) would like, in raster CTB
/// order, at `strength`. All zero at strength 0, and all zero on a
/// picture whose blocks all have the same variance.
pub(crate) fn ctb_offsets<S: Sample>(
    luma: &[S],
    stride: usize,
    width: usize,
    height: usize,
    log2_ctb: u32,
    bit_depth: u32,
    strength: f32,
) -> Vec<i32> {
    let n = 1usize << log2_ctb;
    let (wc, hc) = (width.div_ceil(n), height.div_ceil(n));
    if strength <= 0.0 {
        return vec![0; wc * hc];
    }
    let scale = 1.0 / f64::from(1u32 << (bit_depth - 8));
    let mut energy = Vec::with_capacity(wc * hc);
    for cy in 0..hc {
        for cx in 0..wc {
            let (x0, y0) = (cx * n, cy * n);
            let (w, h) = ((width - x0).min(n), (height - y0).min(n));
            let mut sum = 0f64;
            let mut sq = 0f64;
            for y in y0..y0 + h {
                for &v in &luma[y * stride + x0..y * stride + x0 + w] {
                    let v = f64::from(v.to_i32()) * scale;
                    sum += v;
                    sq += v * v;
                }
            }
            let cnt = (w * h) as f64;
            let mean = sum / cnt;
            let var = (sq / cnt - mean * mean).max(0.0);
            energy.push((var + 1.0).log2());
        }
    }
    let mean_e = energy.iter().sum::<f64>() / energy.len() as f64;
    energy
        .iter()
        .map(|&e| {
            (f64::from(strength) * SLOPE * (e - mean_e))
                .round()
                .clamp(-f64::from(AQ_MAX_OFFSET), f64::from(AQ_MAX_OFFSET)) as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64x64 picture of four 32x32 CTBs: two flat, one mildly textured,
    /// one noisy.
    fn quadrants() -> Vec<u8> {
        let mut p = vec![128u8; 64 * 64];
        let mut seed = 0x1234u32;
        for y in 0..64 {
            for x in 0..64 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let r = (seed >> 24) as i32;
                let v = match (x >= 32, y >= 32) {
                    (false, false) => 128,
                    (true, false) => 100,
                    (false, true) => 128 + ((x as i32 % 4) - 2) * 3, // mild
                    (true, true) => r,                               // full-range noise
                };
                p[y * 64 + x] = v.clamp(0, 255) as u8;
            }
        }
        p
    }

    #[test]
    fn flat_blocks_are_quantised_finer_and_noisy_ones_coarser() {
        let off = ctb_offsets::<u8>(&quadrants(), 64, 64, 64, 5, 8, 1.0);
        assert_eq!(off.len(), 4);
        assert!(
            off[0] < 0 && off[1] < 0,
            "flat blocks should get a finer quantiser: {off:?}"
        );
        assert_eq!(
            off[0], off[1],
            "two equally flat blocks get the same offset: {off:?}"
        );
        assert!(
            off[3] > 0,
            "the noisy block should get a coarser quantiser: {off:?}"
        );
        assert!(
            off[2] > off[0] && off[2] < off[3],
            "mild texture lands between: {off:?}"
        );
        assert!(off.iter().all(|o| o.abs() <= AQ_MAX_OFFSET));
    }

    #[test]
    fn strength_zero_and_uniform_pictures_offset_nothing() {
        assert!(
            ctb_offsets::<u8>(&quadrants(), 64, 64, 64, 5, 8, 0.0)
                .iter()
                .all(|&o| o == 0)
        );
        let flat = vec![77u8; 64 * 64];
        assert!(
            ctb_offsets::<u8>(&flat, 64, 64, 64, 5, 8, 2.0)
                .iter()
                .all(|&o| o == 0)
        );
    }

    /// A deeper picture that is the 8-bit one shifted up must get the
    /// same offsets: strength means the same thing at every depth.
    #[test]
    fn depth_does_not_change_the_offsets() {
        let p8 = quadrants();
        let p10: Vec<u16> = p8.iter().map(|&v| u16::from(v) << 2).collect();
        assert_eq!(
            ctb_offsets::<u8>(&p8, 64, 64, 64, 5, 8, 1.0),
            ctb_offsets::<u16>(&p10, 64, 64, 64, 5, 10, 1.0)
        );
    }

    #[test]
    fn strength_scales_the_offsets_up_to_the_clamp() {
        let a = ctb_offsets::<u8>(&quadrants(), 64, 64, 64, 5, 8, 1.0);
        let b = ctb_offsets::<u8>(&quadrants(), 64, 64, 64, 5, 8, 4.0);
        assert!(b[3] >= a[3] && b[0] <= a[0], "{a:?} vs {b:?}");
        assert_eq!(
            b[3], AQ_MAX_OFFSET,
            "at strength 4 the noisy block hits the clamp: {b:?}"
        );
    }
}
