//! Randomised bit-exactness sweeps for the SIMD tiers of the 16-bit-sample
//! kernel tables, [`H264Dsp<u16>`] and [`DistortionDsp<u16>`].
//!
//! A SIMD kernel for 16-bit samples does not have one arithmetic, it has
//! several. The H.264 kernels change lane widths at 10 bits (the luma
//! six-tap, chroma bilinear) and keep the loop filter's sums in the unsigned
//! domain up to 14; the distortion kernels, which are told no depth at all,
//! choose per block from the samples they read. A sweep at one depth checks
//! one of those paths. So these run every table at depths on both sides of
//! every switch — 9, 10, 12 and 14 bits for H.264, 9 to 16 for the
//! distortion metrics — over uniform samples and over inputs built to reach
//! each kernel's widest intermediate, and compare every output with the
//! scalar reference.
//!
//! One sweep serves every architecture: the x86 rungs' unit tests, the NEON
//! tests the arm64 CI runners execute, and the wasm probe
//! (`examples/wasm_probe.rs`), which has no test harness to run a
//! `#[cfg(test)]` module in. That is why this module is public — the probe is
//! a separate crate — and why it is hidden from the documentation: it is a
//! test, not an interface.
//!
//! Each sweep returns `Err` naming the first disagreement (table, kernel,
//! shape, depth, input), or `Ok` with the number of comparisons it made, so
//! that a caller can tell a sweep that compared nothing from one that passed.

use super::distortion::DistortionDsp;
use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// The depths the H.264 sweeps run at: each side of the kernels' 10-bit
/// switch, and 12 and 14 bits for the range of the wide paths.
pub const H264_DEPTHS: [u32; 4] = [9, 10, 12, 14];

/// The depths the distortion sweep runs at: each side of SATD's 11-bit
/// switch, and on to the 16 bits the metrics accept although no encoder
/// produces them.
pub const DISTORTION_DEPTHS: [u32; 6] = [9, 10, 11, 12, 14, 16];

/// The LCG the 8-bit kernel tests use.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// A sample in `0..=max`.
    fn sample(&mut self, max: i32) -> u16 {
        self.below(max as u32 + 1) as u16
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 0
    }
}

/// Run `call` with the scalar table and then with every table in `tables`,
/// each on its own copy of `input`, and compare what `view` shows of the
/// results. Returns whether the scalar reference changed its copy, which is
/// what a vacuity count wants: a sweep whose inputs never reach a kernel's
/// arithmetic passes without testing it.
fn same<T>(
    tables: &[(&str, T)],
    scalar: &T,
    input: &[u16],
    n: &mut u64,
    call: impl Fn(&T, &mut [u16]),
    view: impl Fn(&[u16]) -> Vec<u16>,
    what: impl Fn() -> String,
) -> Result<bool, String> {
    let mut want = input.to_vec();
    call(scalar, &mut want);
    let want_view = view(&want);
    for (name, d) in tables {
        let mut got = input.to_vec();
        call(d, &mut got);
        *n += 1;
        if view(&got) != want_view {
            return Err(format!("{name}: {}", what()));
        }
    }
    Ok(want != input)
}

/// The whole of a buffer.
fn whole(v: &[u16]) -> Vec<u16> {
    v.to_vec()
}

/// The `w x h` block of a [`PRED_STRIDE`]-strided scratch buffer: the SIMD
/// kernels may write the rest of each row.
fn block(v: &[u16], w: usize, h: usize) -> Vec<u16> {
    (0..h).flat_map(|y| v[y * PRED_STRIDE..y * PRED_STRIDE + w].iter().copied()).collect()
}

/// Every H.264 sweep, at every depth.
pub fn h264(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    Ok(h264_interp(tables)? + h264_chroma(tables)? + h264_combine(tables)? + h264_deblock(tables)? + h264_transforms(tables)?)
}

const LUMA_SIZES: [(usize, usize); 7] = [(4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)];

/// A `64 x 64` plane for the luma interpolation sweep, by `mode`: uniform
/// samples (0); samples on the rails, 0 or `max` (1); and a pattern that
/// drives the six-tap to its extremes (2, and its complement 3). The pattern
/// is `max` where a sample's column and row are both on, or both off, a −5
/// tap position (index 1 or 4 of six), and 0 elsewhere: a window aligned with
/// it gives the horizontal intermediate `42 · max` on the rows the vertical
/// filter weights positively and `−10 · max` on the other two, which is the
/// largest centre value there is — and its complement the smallest. The sweep
/// reads it from six origins so that every output position meets every
/// alignment.
fn interp_plane(rng: &mut Rng, mode: u32, max: i32) -> Vec<u16> {
    let neg = |k: usize| k % 6 == 1 || k % 6 == 4;
    (0..64 * 64)
        .map(|i| {
            let (x, y) = (i % 64, i / 64);
            match mode {
                0 => rng.sample(max),
                1 => {
                    if rng.coin() {
                        0
                    } else {
                        max as u16
                    }
                }
                2 => {
                    if neg(x) == neg(y) {
                        max as u16
                    } else {
                        0
                    }
                }
                _ => {
                    if neg(x) == neg(y) {
                        0
                    } else {
                        max as u16
                    }
                }
            }
        })
        .collect()
}

/// Luma quarter-sample interpolation, every position.
pub fn h264_interp(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    let s = H264Dsp::<u16>::SCALAR;
    let mut rng = Rng(0x0e1_0bb5);
    let mut n = 0;
    for bd in H264_DEPTHS {
        let max = (1i32 << bd) - 1;
        for mode in 0..4 {
            let plane = interp_plane(&mut rng, mode, max);
            let origins: &[usize] = if mode < 2 { &[3] } else { &[0, 1, 2, 3, 4, 5] };
            for &o in origins {
                let src = &plane[o * 64 + o..];
                for &(w, h) in &LUMA_SIZES {
                    for pos in 0..16 {
                        same(
                            tables,
                            &s,
                            &[0u16; 16 * PRED_STRIDE],
                            &mut n,
                            |d, buf| (d.qpel[pos])(buf, src, 64, w, h, max),
                            |v| block(v, w, h),
                            || format!("qpel position {pos}, {w}x{h}, {bd} bits, plane {mode} read from origin {o}"),
                        )?;
                    }
                }
            }
        }
    }
    Ok(n)
}

const CHROMA_SIZES: [(usize, usize); 8] = [(2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 4), (8, 8), (8, 16)];

/// A `64 x 64` plane for the chroma sweep, by `mode`: uniform (0); the rails
/// (1); all `max`, the largest weighted sum (2); rows alternately of at most
/// ten bits and of the full depth, so that a kernel that picks its
/// arithmetic per row switches inside one block (3); and 16-bit samples,
/// which no H.264 stream has but which the kernel, taking no `max`, must
/// still get right (4).
fn chroma_plane(rng: &mut Rng, mode: u32, max: i32) -> Vec<u16> {
    (0..64 * 64)
        .map(|i| match mode {
            0 => rng.sample(max),
            1 => {
                if rng.coin() {
                    0
                } else {
                    max as u16
                }
            }
            2 => max as u16,
            3 => {
                if (i / 64) % 2 == 0 {
                    rng.sample(max.min(1023))
                } else {
                    rng.sample(max)
                }
            }
            _ => rng.next() as u16,
        })
        .collect()
}

/// Chroma bilinear, every eighth-sample position.
pub fn h264_chroma(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    let s = H264Dsp::<u16>::SCALAR;
    let mut rng = Rng(0xc4_0a1);
    let mut n = 0;
    for bd in H264_DEPTHS {
        let max = (1i32 << bd) - 1;
        let modes = if bd == 14 { 5 } else { 4 };
        for mode in 0..modes {
            let plane = chroma_plane(&mut rng, mode, max);
            let src = &plane[5 * 64 + 5..];
            for &(w, h) in &CHROMA_SIZES {
                for xf in 0..8 {
                    for yf in 0..8 {
                        same(
                            tables,
                            &s,
                            &[0u16; 16 * PRED_STRIDE],
                            &mut n,
                            |d, buf| (d.chroma)(buf, src, 64, w, h, xf, yf),
                            |v| block(v, w, h),
                            || format!("chroma ({xf}, {yf}), {w}x{h}, {bd} bits, plane {mode}"),
                        )?;
                    }
                }
            }
        }
    }
    Ok(n)
}

/// `(log_wd, w, o)` for weighted uni-prediction, `o` in 8-bit units: the
/// 8-bit tests' rows and the corners of the explicit ranges.
const UNI: [(i32, i32, i32); 8] = [(6, 64, 0), (0, 1, 3), (5, -20, -7), (7, 127, 127), (2, 33, -128), (0, 127, 127), (0, -128, -128), (7, -128, 127)];

/// `(log_wd, w0, w1, o0, o1)` for weighted bi-prediction: default, the
/// implicit extremes (−64, 128), and the corners of the explicit ranges.
const BI: [(i32, i32, i32, i32, i32); 7] =
    [(5, 32, 32, 0, 0), (5, -64, 128, 0, 0), (5, 128, -64, 0, 0), (6, 127, -128, 127, -128), (0, 127, 127, 127, 127), (2, -128, -128, -128, -128), (7, 64, 0, -128, 127)];

/// Copy, average and weighted combination.
pub fn h264_combine(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    let s = H264Dsp::<u16>::SCALAR;
    let mut rng = Rng(0xc0b1_ae);
    let mut n = 0;
    for bd in H264_DEPTHS {
        let max = (1i32 << bd) - 1;
        let scale = 1i32 << (bd - 8);
        for mode in 0..2 {
            let pick = |rng: &mut Rng| {
                if mode == 0 {
                    rng.sample(max)
                } else if rng.coin() {
                    0
                } else {
                    max as u16
                }
            };
            let a: Vec<u16> = (0..16 * PRED_STRIDE).map(|_| pick(&mut rng)).collect();
            let b: Vec<u16> = (0..16 * PRED_STRIDE).map(|_| pick(&mut rng)).collect();
            for &(w, h) in &LUMA_SIZES {
                let ds = w + 3;
                let dst = vec![0u16; ds * h];
                let what = |k: &str| format!("{k}, {w}x{h}, {bd} bits, samples {}", if mode == 0 { "uniform" } else { "on the rails" });
                same(tables, &s, &dst, &mut n, |d, buf| (d.avg)(buf, ds, &a, &b, w, h), whole, || what("avg"))?;
                same(tables, &s, &dst, &mut n, |d, buf| (d.copy)(buf, ds, &a, w, h), whole, || what("copy"))?;
                for &(lwd, wt, o) in &UNI {
                    let o = o * scale;
                    same(tables, &s, &dst, &mut n, |d, buf| (d.weighted_uni)(buf, ds, &a, w, h, lwd, wt, o, max), whole, || {
                        what(&format!("weighted_uni log_wd {lwd} w {wt} o {o}"))
                    })?;
                }
                for &(lwd, w0, w1, o0, o1) in &BI {
                    let (o0, o1) = (o0 * scale, o1 * scale);
                    same(tables, &s, &dst, &mut n, |d, buf| (d.weighted_bi)(buf, ds, &a, &b, w, h, lwd, w0, w1, o0, o1, max), whole, || {
                        what(&format!("weighted_bi log_wd {lwd} w {w0} {w1} o {o0} {o1}"))
                    })?;
                }
            }
        }
    }
    Ok(n)
}

/// A plane for the loop-filter sweep (`stride x 40`), by `kind`: smooth
/// random content, the 8-bit tests' kind scaled to the depth (0–2); content
/// hugging 0, where `p0 + delta` clips at the bottom (3); content hugging
/// `max`, where it clips at the top (4); and a cliff across both edges the
/// sweep filters, `alpha - 1` high with jitter under `beta`, which is the
/// largest step the filters still act on (5).
fn deblock_plane(rng: &mut Rng, kind: u32, max: i32, alpha: i32, beta: i32, stride: usize) -> Vec<u16> {
    let m = max as u32;
    let len = stride * 40;
    match kind {
        0..=2 => {
            let scale = (m + 1) / 256;
            let base = rng.below(m + 1);
            let spread = 1 + rng.below(64 * scale);
            (0..len).map(|_| (base + rng.below(spread)).min(m) as u16).collect()
        }
        3 => (0..len).map(|_| rng.below(2 + beta as u32).min(m) as u16).collect(),
        4 => (0..len).map(|_| (m - rng.below(2 + beta as u32).min(m)) as u16).collect(),
        _ => {
            let gap = (alpha - 1).clamp(0, max) as u32;
            let lo = rng.below(m - gap + 1);
            let up = rng.coin();
            let jitter = 1 + beta as u32 / 2;
            (0..len)
                .map(|i| {
                    let (x, y) = (i % stride, i / stride);
                    // The q side of the vertical edge at column 8 (rows below
                    // 8) and of the horizontal edge at row 8 (columns right of
                    // 8) alike.
                    let q_side = (x < 8) == (y < 8);
                    let level = if q_side == up { lo + gap } else { lo };
                    (level + rng.below(jitter)).min(m) as u16
                })
                .collect()
        }
    }
}

/// The loop filters, every entry: luma and chroma, normal and strong,
/// vertical and horizontal, and the MBAFF eight-line luma edges.
pub fn h264_deblock(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    let s = H264Dsp::<u16>::SCALAR;
    let mut rng = Rng(0xdeb1_0c);
    let mut n = 0;
    let stride = 48;
    let off = 8 * stride + 8;
    const TRIALS: u32 = 600;
    for bd in H264_DEPTHS {
        let max = (1i32 << bd) - 1;
        let scale = 1i32 << (bd - 8);
        let mut filtered = 0;
        for trial in 0..TRIALS {
            let alpha = if trial % 9 == 0 { 255 * scale } else { rng.below(256) as i32 * scale };
            let beta = if trial % 9 == 1 { 18 * scale } else { rng.below(19) as i32 * scale };
            let mut tc0 = [0i16; 4];
            for t in tc0.iter_mut() {
                // −1 is bS 0, left unscaled; tC0 itself runs to 25 (Table 8-17).
                let v = rng.below(27) as i32 - 1;
                *t = if v < 0 { -1 } else { (v * scale) as i16 };
            }
            let kind = rng.below(6);
            let plane = deblock_plane(&mut rng, kind, max, alpha, beta, stride);
            let which = trial % 10;
            let call = |d: &H264Dsp<u16>, p: &mut [u16]| match which {
                0 => (d.deblock_luma_v)(p, off, stride, alpha, beta, &tc0, max),
                1 => (d.deblock_luma_h)(p, off, stride, alpha, beta, &tc0, max),
                2 => (d.deblock_luma_v_intra)(p, off, stride, alpha, beta, max),
                3 => (d.deblock_luma_h_intra)(p, off, stride, alpha, beta, max),
                4 => (d.deblock_luma8_v)(p, off, stride, alpha, beta, &tc0, max),
                5 => (d.deblock_luma8_v_intra)(p, off, stride, alpha, beta, max),
                6 => (d.deblock_chroma_v)(p, off, stride, alpha, beta, &tc0, max),
                7 => (d.deblock_chroma_h)(p, off, stride, alpha, beta, &tc0, max),
                8 => (d.deblock_chroma_v_intra)(p, off, stride, alpha, beta, max),
                _ => (d.deblock_chroma_h_intra)(p, off, stride, alpha, beta, max),
            };
            let changed = same(tables, &s, &plane, &mut n, call, whole, || {
                format!("deblock entry {which}, {bd} bits, plane kind {kind}, alpha {alpha} beta {beta} tc0 {tc0:?} (trial {trial})")
            })?;
            filtered += changed as u32;
        }
        // A sweep whose planes never pass the alpha / beta tests compares
        // unfiltered planes and proves nothing.
        if filtered < TRIALS / 6 {
            return Err(format!("deblock sweep at {bd} bits: only {filtered} of {TRIALS} calls changed a sample"));
        }
    }
    Ok(n)
}

/// Inverse transforms, the DC-only adds and the residual paths.
pub fn h264_transforms(tables: &[(&str, H264Dsp<u16>)]) -> Result<u64, String> {
    let s = H264Dsp::<u16>::SCALAR;
    let mut rng = Rng(0x1dc7_8);
    let mut n = 0;
    let stride = 24;
    for bd in H264_DEPTHS {
        let max = (1i32 << bd) - 1;
        // The standard's bound on a dequantised coefficient and on every
        // intermediate (8.5.12.1): 2^(7 + BitDepth).
        let lim = 1i32 << (7 + bd);
        let span = |rng: &mut Rng, l: i32| (rng.next() % (2 * l as u32 - 1)) as i32 - (l - 1);
        for trial in 0..600u32 {
            let base: Vec<u16> = match rng.below(3) {
                0 => (0..stride * 8).map(|_| rng.sample(max)).collect(),
                1 => vec![0; stride * 8],
                _ => vec![max as u16; stride * 8],
            };
            let mode = rng.below(3);
            let nz = 1 + rng.below(64) as usize;
            let mut c16 = [0i16; 64];
            let mut c32 = [0i32; 64];
            for k in 0..nz {
                c16[k] = match mode {
                    0 => (rng.below(4001) as i32 - 2000) as i16,
                    1 => {
                        if rng.coin() {
                            i16::MAX
                        } else {
                            i16::MIN
                        }
                    }
                    _ => rng.next() as i16,
                };
                c32[k] = match mode {
                    0 => rng.below(4001) as i32 - 2000,
                    1 => {
                        if rng.coin() {
                            lim - 1
                        } else {
                            1 - lim
                        }
                    }
                    _ => span(&mut rng, lim),
                };
            }
            if trial % 7 == 3 {
                // DC only: the residual paths' other branch.
                c16[1..].fill(0);
                c32[1..].fill(0);
            }
            let which = trial % 6;
            let dc = match which {
                // A DC add is `(dc + 32) >> 6` of a coefficient up to the
                // bound, so sixty-four times that reaches past the i16 the
                // SIMD kernels saturate to.
                2 | 3 => span(&mut rng, lim.saturating_mul(64)),
                _ if trial % 12 == 4 => NO_DC,
                _ => span(&mut rng, lim),
            };
            let c4: [i16; 16] = c16[..16].try_into().unwrap();
            let r4: [i32; 16] = c32[..16].try_into().unwrap();
            let call = |d: &H264Dsp<u16>, p: &mut [u16]| match which {
                0 => (d.idct4_add)(p, stride, &c4, max),
                1 => (d.idct8_add)(p, stride, &c16, max),
                2 => (d.idct4_dc_add)(p, stride, dc, max),
                3 => (d.idct8_dc_add)(p, stride, dc, max),
                4 => (d.residual4)(p, stride, &r4, dc, max),
                _ => (d.residual8)(p, stride, &c32, max),
            };
            same(tables, &s, &base, &mut n, call, whole, || {
                format!("transform entry {which}, {bd} bits, coefficient mode {mode}, {nz} nonzero, dc {dc} (trial {trial})")
            })?;
        }
    }
    Ok(n)
}

const DIST_SIZES: [(usize, usize); 16] =
    [(4, 4), (4, 8), (4, 16), (8, 4), (8, 8), (8, 16), (12, 8), (12, 12), (16, 4), (16, 8), (16, 16), (20, 8), (24, 16), (32, 32), (48, 16), (64, 64)];

/// Two `96 x 96` planes, by `kind`: rows of the 8-bit tests' kinds — 0
/// against `max`, `max` against 0, uniform against `max`, uniform against
/// uniform — so the extremes are reached (0); samples of at most eleven bits
/// with one in two hundred at `max`, so that most SATD tile pairs take the
/// narrow path and a few are thrown off it (1); uniform (2).
fn dist_planes(rng: &mut Rng, kind: u32, max: i32) -> (Vec<u16>, Vec<u16>) {
    let n = 96 * 96;
    let m = max as u16;
    let mut a = vec![0u16; n];
    let mut b = vec![0u16; n];
    for y in 0..96 {
        let mode = rng.below(8);
        for x in 0..96 {
            let (va, vb) = match kind {
                0 => match mode {
                    0 => (0, m),
                    1 => (m, 0),
                    2 => (rng.sample(max), m),
                    _ => (rng.sample(max), rng.sample(max)),
                },
                1 => {
                    let small = |rng: &mut Rng| if rng.below(200) == 0 { m } else { rng.sample(max.min(2047)) };
                    (small(rng), small(rng))
                }
                _ => (rng.sample(max), rng.sample(max)),
            };
            a[y * 96 + x] = va;
            b[y * 96 + x] = vb;
        }
    }
    (a, b)
}

/// SAD, SATD and SSD over the block shapes both encoders use and a few they
/// do not, with random strides and offsets, plus the closed forms of a block
/// of 0 against a block of `max`.
pub fn distortion(tables: &[(&str, DistortionDsp<u16>)]) -> Result<u64, String> {
    let s = DistortionDsp::<u16>::scalar();
    let mut rng = Rng(0x5add_16);
    let mut n = 0;
    for bd in DISTORTION_DEPTHS {
        let max = ((1u32 << bd) - 1) as i32;
        for round in 0..12 {
            let kind = round % 3;
            let (a, b) = dist_planes(&mut rng, kind, max);
            for &(w, h) in &DIST_SIZES {
                let sa = w + rng.below(24) as usize;
                let sb = w + rng.below(24) as usize;
                let oa = rng.below(64) as usize;
                let ob = rng.below(64) as usize;
                let (pa, pb) = (&a[oa..], &b[ob..]);
                let want = ((s.sad)(pa, sa, pb, sb, w, h), (s.satd)(pa, sa, pb, sb, w, h), (s.ssd)(pa, sa, pb, sb, w, h));
                for (name, d) in tables {
                    let got = ((d.sad)(pa, sa, pb, sb, w, h), (d.satd)(pa, sa, pb, sb, w, h), (d.ssd)(pa, sa, pb, sb, w, h));
                    n += 1;
                    if got != want {
                        return Err(format!("{name}: {w}x{h} at {bd} bits, planes of kind {kind}: (sad, satd, ssd) {got:?}, scalar {want:?}"));
                    }
                }
            }
        }
        // The closed forms, so the sweep does not rest on the scalar
        // reference alone: SAD is `max` times the area, SSD its square
        // times the area, and SATD sees pure DC, 16·max a tile, halved.
        let m = max as u32;
        let zero = vec![0u16; 64 * 64];
        let full = vec![max as u16; 64 * 64];
        let want = (m * 4096, 256 * ((16 * m + 1) >> 1), m as u64 * m as u64 * 4096);
        for (name, d) in tables {
            for (x, y) in [(&zero, &full), (&full, &zero)] {
                let got = ((d.sad)(x, 64, y, 64, 64, 64), (d.satd)(x, 64, y, 64, 64, 64), (d.ssd)(x, 64, y, 64, 64, 64));
                n += 1;
                if got != want {
                    return Err(format!("{name}: 64x64 of 0 against {max}: (sad, satd, ssd) {got:?}, closed form {want:?}"));
                }
            }
        }
    }
    Ok(n)
}
