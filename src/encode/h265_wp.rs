//! Weighted prediction decision: a gain and an offset per reference,
//! from a least-squares fit of the source against the reference.
//!
//! H.265's explicit weighting (`pred_weight_table`, 8.5.3.3.4.3) scales
//! and shifts a reference before it predicts: `Clip(((ref * w) >> denom)
//! + o)`. Motion compensation cannot express a change of brightness —
//! every vector predicts a block at the reference's own level — so on a
//! fade every inter block carries the level difference as residual, and a
//! stream with weighting carries it once per slice in the table instead.
//!
//! # The fit
//!
//! Per component, over the display area of the reference's
//! reconstruction (what a decoder holds) against the source: the
//! least-squares line `cur ≈ w * ref + o`, `w = cov(ref, cur) / var(ref)`
//! and `o = mean(cur) - w * mean(ref)`. Quantised to the table's
//! precision — weights in units of `1 / 2^LOG2_DENOM`, offsets in 8-bit
//! sample units, both held to the ranges the syntax carries — and then
//! **checked, not trusted**: the fit is applied to the reference at zero
//! motion and its SAD against the source compared with the plain
//! reference's. Only a fit that lowers the SAD by more than
//! [`MIN_GAIN`] is used; otherwise the entry is the default and the
//! table says so in one flag bit. A fit that helps nothing costs the
//! table's bits and biases the motion search for no return.
//!
//! The fit is H.264's too (`encode::h264`'s explicit weighting for P
//! slices): H.264's table carries the weight itself where H.265's carries a
//! delta around the identity, so [`fit_samples`] takes the range the
//! caller's syntax can hold, and the reference as a plain sample layout
//! ([`RefSamples`]) rather than either decoder's plane type. The weighted
//! arithmetic the check applies is the same in both standards: H.264's
//! 8.4.2.3.2 uni-directional formula at `logWD >= 1` is H.265's.
//!
//! What this is not: a per-block decision. The table is per slice and
//! per reference, so a picture whose left half fades and whose right
//! half does not gets one compromise line. The fit is over the whole
//! picture and the check is over the whole picture, which is exactly
//! the granularity the syntax offers.

use crate::hevc::frame::Plane16;
use crate::hevc::slice::{PredWeightTable, WeightEntry};
use crate::sample::Sample;

/// `luma_log2_weight_denom` and `ChromaLog2WeightDenom`: weights in
/// sixty-fourths. Six is the finest the table's `-128..=127` weight
/// delta can express a gain of a quarter to nearly three at, which
/// covers a fade at every step of its ramp.
pub(crate) const LOG2_DENOM: u32 = 6;

/// The share of the plain zero-motion SAD a fit must remove to be used.
/// The table costs bits (a few dozen per slice) and a weighted reference
/// steers the motion search, so a fit that removes a percent of the
/// residual is noise, not a fade.
const MIN_GAIN: f64 = 0.02;

/// One component's fitted weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneFit {
    /// The weight, in units of `1 << LOG2_DENOM`. `1 << LOG2_DENOM` is
    /// the identity.
    pub weight: i32,
    /// The offset in 8-bit sample units (what the table carries; the
    /// reader shifts it to the sample depth). 0 is the identity.
    pub offset: i32,
    /// Zero-motion SAD of the source against the plain reference.
    pub sad_plain: u64,
    /// The same against the weighted reference.
    pub sad_weighted: u64,
}

impl PlaneFit {
    /// Whether the fit is worth carrying: it is not the identity and it
    /// removes more than [`MIN_GAIN`] of the plain SAD.
    pub fn used(&self) -> bool {
        (self.weight != 1 << LOG2_DENOM || self.offset != 0) && (self.sad_weighted as f64) < self.sad_plain as f64 * (1.0 - MIN_GAIN)
    }

    /// The identity: default weighting, never [`PlaneFit::used`].
    pub(crate) fn identity(sad_plain: u64) -> Self {
        PlaneFit { weight: 1 << LOG2_DENOM, offset: 0, sad_plain, sad_weighted: sad_plain }
    }
}

/// Fit one `w` by `h` plane: `cur` at `cur_stride` against the display
/// area of `refp`, at `bit_depth`.
pub(crate) fn fit_plane<S: Sample>(cur: &[S], cur_stride: usize, refp: &Plane16<S>, w: usize, h: usize, bit_depth: u32) -> PlaneFit {
    let refs = RefSamples { data: &refp.data, origin: refp.origin(), stride: refp.stride };
    fit_samples(cur, cur_stride, refs, w, h, bit_depth, H265_WEIGHTS)
}

/// A reference plane as the fit reads it: the samples, the index of the
/// display area's first sample, and the stride — what an H.265 `Plane16`
/// and an H.264 `PaddedPlane` both are, under different names.
#[derive(Clone, Copy)]
pub(crate) struct RefSamples<'a, S: Sample> {
    /// The plane's samples, border included.
    pub data: &'a [S],
    /// Index in `data` of the display area's top-left sample.
    pub origin: usize,
    /// Samples per row of `data`.
    pub stride: usize,
}

/// The weights H.265's table can carry, `(lowest, highest)` in units of
/// `1 << LOG2_DENOM`: `delta_luma_weight` spans -128..=127 around the
/// identity.
pub(crate) const H265_WEIGHTS: (i32, i32) = ((1 << LOG2_DENOM) - 128, (1 << LOG2_DENOM) + 127);

/// The weights H.264's table can carry: `luma_weight_l0` is the weight
/// itself, in -128..=127 — which is also what the decoder's SIMD weighting
/// kernels are built for.
pub(crate) const H264_WEIGHTS: (i32, i32) = (-128, 127);

/// [`fit_plane`] over any reference layout, with the weight held to
/// `weights` — the range the caller's table syntax carries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_samples<S: Sample>(cur: &[S], cur_stride: usize, refp: RefSamples<'_, S>, w: usize, h: usize, bit_depth: u32, weights: (i32, i32)) -> PlaneFit {
    let o = refp.origin;
    let n = (w * h) as f64;
    let (mut sr, mut sc, mut srr, mut src) = (0f64, 0f64, 0f64, 0f64);
    let mut sad_plain = 0u64;
    for y in 0..h {
        let rrow = &refp.data[o + y * refp.stride..];
        let crow = &cur[y * cur_stride..];
        for x in 0..w {
            let r = f64::from(rrow[x].to_i32());
            let c = f64::from(crow[x].to_i32());
            sr += r;
            sc += c;
            srr += r * r;
            src += r * c;
            sad_plain += u64::from(rrow[x].to_i32().abs_diff(crow[x].to_i32()));
        }
    }
    let var = srr / n - (sr / n) * (sr / n);
    if var <= 0.0 {
        // A flat reference has no gain to fit; an offset alone is the
        // mean difference, which a flat block's residual codes for
        // nothing anyway.
        return PlaneFit::identity(sad_plain);
    }
    let gain = (src / n - (sr / n) * (sc / n)) / var;
    let unit = f64::from(1u32 << LOG2_DENOM);
    // The weight is held to what the caller's syntax carries, and the
    // offset to the -128..=127 eight-bit units both standards' tables do.
    let weight = (gain * unit).round().clamp(f64::from(weights.0), f64::from(weights.1)) as i32;
    let scale = f64::from(1u32 << (bit_depth - 8));
    let offset_samples = sc / n - f64::from(weight) / unit * (sr / n);
    let offset = (offset_samples / scale).round().clamp(-128.0, 127.0) as i32;
    let fit = PlaneFit { weight, offset, sad_plain, sad_weighted: 0 };
    let sad_weighted = weighted_sad(cur, cur_stride, refp, w, h, bit_depth, weight, offset);
    PlaneFit { sad_weighted, ..fit }
}

/// Zero-motion SAD of `cur` against `refp` weighted by `(weight,
/// offset)` — the decoder's `weighted_uni` arithmetic at a whole-sample
/// vector: `Clip(((r * w + round) >> denom) + (o << (bitDepth - 8)))`.
#[allow(clippy::too_many_arguments)]
fn weighted_sad<S: Sample>(cur: &[S], cur_stride: usize, refp: RefSamples<'_, S>, w: usize, h: usize, bit_depth: u32, weight: i32, offset: i32) -> u64 {
    let o = refp.origin;
    let max = (1i32 << bit_depth) - 1;
    let round = 1i32 << (LOG2_DENOM - 1);
    let off = offset << (bit_depth - 8);
    let mut sad = 0u64;
    for y in 0..h {
        let rrow = &refp.data[o + y * refp.stride..];
        let crow = &cur[y * cur_stride..];
        for x in 0..w {
            let p = (((rrow[x].to_i32() * weight + round) >> LOG2_DENOM) + off).clamp(0, max);
            sad += u64::from(p.abs_diff(crow[x].to_i32()));
        }
    }
    sad
}

/// The `pred_weight_table` entry the three fits make: each component's
/// fit where it is [`PlaneFit::used`], the default otherwise, with the
/// offsets shifted to the sample depth as the reader holds them.
pub(crate) fn entry_for(fits: [PlaneFit; 3], bit_depth_luma: u32, bit_depth_chroma: u32) -> WeightEntry {
    let comp = |f: &PlaneFit, bd: u32| if f.used() { (f.weight, f.offset << (bd - 8)) } else { (1 << LOG2_DENOM, 0) };
    WeightEntry { luma: comp(&fits[0], bit_depth_luma), chroma: [comp(&fits[1], bit_depth_chroma), comp(&fits[2], bit_depth_chroma)] }
}

/// A P slice's table from its list-0 entries, one per reference in
/// `RefPicList0` order.
pub(crate) fn table_for(entries: Vec<WeightEntry>) -> PredWeightTable {
    PredWeightTable { luma_log2_denom: LOG2_DENOM, chroma_log2_denom: LOG2_DENOM, lists: [entries, Vec::new()] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::frame::Frame;
    use crate::picture::ChromaFormat;

    /// A 32x24 reference with structure, as the luma plane of a frame
    /// (the only way to make a plane, as the decoder makes them).
    fn reference() -> Plane16<u8> {
        let (w, h) = (32usize, 24usize);
        let mut p = Frame::<u8>::new(w, h, ChromaFormat::Monochrome, 8).y;
        let o = p.origin();
        for y in 0..h {
            for x in 0..w {
                p.data[o + y * p.stride + x] = (60 + ((x * 5 + y * 3) % 120)) as u8;
            }
        }
        p
    }

    fn scaled(refp: &Plane16<u8>, w: usize, h: usize, gain: f64, off: f64) -> Vec<u8> {
        let o = refp.origin();
        (0..h).flat_map(|y| (0..w).map(move |x| (f64::from(refp.data[o + y * refp.stride + x]) * gain + off).round().clamp(0.0, 255.0) as u8)).collect()
    }

    /// A fade is recovered: a source that is the reference at three
    /// quarters brightness fits a weight of 48/64 and an offset near 0,
    /// and the weighted SAD is a fraction of the plain one.
    #[test]
    fn a_gain_fade_is_recovered_and_worth_using() {
        let r = reference();
        let cur = scaled(&r, 32, 24, 0.75, 0.0);
        let f = fit_plane(&cur, 32, &r, 32, 24, 8);
        assert_eq!(f.weight, 48, "{f:?}");
        assert!(f.offset.abs() <= 1, "{f:?}");
        assert!(f.used(), "{f:?}");
        assert!(f.sad_weighted * 4 < f.sad_plain, "{f:?}");
    }

    /// An offset alone: the source is the reference plus twelve.
    #[test]
    fn an_offset_is_recovered() {
        let r = reference();
        let cur = scaled(&r, 32, 24, 1.0, 12.0);
        let f = fit_plane(&cur, 32, &r, 32, 24, 8);
        assert_eq!((f.weight, f.offset), (64, 12), "{f:?}");
        assert!(f.used() && f.sad_weighted == 0, "{f:?}");
    }

    /// The same picture fits the identity and is not used; a flat
    /// reference fits nothing.
    #[test]
    fn an_unchanged_picture_uses_no_weighting() {
        let r = reference();
        let cur = scaled(&r, 32, 24, 1.0, 0.0);
        let f = fit_plane(&cur, 32, &r, 32, 24, 8);
        assert_eq!((f.weight, f.offset), (64, 0), "{f:?}");
        assert!(!f.used(), "{f:?}");
        let flat = Frame::<u8>::new(32, 24, ChromaFormat::Monochrome, 8).y;
        let f = fit_plane(&cur, 32, &flat, 32, 24, 8);
        assert!(!f.used(), "{f:?}");
    }

    /// Ten-bit samples: the offset is fitted in samples and carried in
    /// 8-bit units, so a shift of 40 samples is an offset of 10.
    #[test]
    fn deep_offsets_are_carried_in_eight_bit_units() {
        let (w, h) = (32usize, 24usize);
        let mut p = Frame::<u16>::new(w, h, ChromaFormat::Monochrome, 10).y;
        let o = p.origin();
        for y in 0..h {
            for x in 0..w {
                p.data[o + y * p.stride + x] = (240 + ((x * 20 + y * 12) % 480)) as u16;
            }
        }
        let cur: Vec<u16> = (0..h).flat_map(|y| (0..w).map(move |x| (240 + ((x * 20 + y * 12) % 480) + 40) as u16)).collect();
        let f = fit_plane(&cur, w, &p, w, h, 10);
        assert_eq!((f.weight, f.offset), (64, 10), "{f:?}");
        let e = entry_for([f, PlaneFit::identity(1), PlaneFit::identity(1)], 10, 10);
        assert_eq!(e.luma, (64, 40), "the table holds the offset at the sample depth: {e:?}");
        assert_eq!(e.chroma, [(64, 0); 2]);
    }
}
