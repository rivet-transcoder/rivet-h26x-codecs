//! Intra sample prediction (H.265 clause 8.4.4.2): reference sample
//! availability and substitution, the reference smoothing filter (with
//! strong intra smoothing), and the planar, DC and angular predictors.
//!
//! The work is split in two, because an encoder asks the second half many
//! times for one run of the first. [`prepare`] gathers a block's reference
//! samples and substitutes the unavailable ones; [`predict_prepared`]
//! smooths them if the mode wants it (once per block, kept for the next
//! mode that wants it too) and runs the predictor. The predictors
//! themselves — planar, DC, and the angular interpolation that is most of
//! the arithmetic — are [`HevcDsp`] entries, scalar and SIMD. [`predict`]
//! is the two halves back to back, which is what the decoder does for each
//! transform block; the encoder's mode search prepares a block once and
//! predicts all 35 modes from it.

use super::frame::{Plane16, Sample};
use crate::dsp::hevc::HevcDsp;

/// `intraPredAngle` for modes 2..=34 (Table 8-4), indexed by `mode - 2`.
const INTRA_PRED_ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
/// `invAngle` for modes 11..=25 (Table 8-5), indexed by `mode - 11`.
const INV_ANGLE: [i32; 15] = [-4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096];

/// The angular predictor's reference array: `ref[k]` for `k` in
/// `-n..=2n` at index `n + k`, plus room past `3n + 1` for a SIMD kernel's
/// last vector to read (and not use).
pub(crate) const REF_LEN: usize = 3 * 64 + 1 + 32;

/// Reference-sample scratch, reused from block to block. Every entry a
/// block reads it writes first — the gathering pass and the substitution
/// between them define `left[..2n]` and `top[..2n]`, the smoothing filter
/// the same span of `fl` / `ft`, and the reference buffer is written as far
/// as the angular predictor uses it — so nothing here needs clearing,
/// while clearing it per transform block cost a kilobyte and a half of
/// stack traffic. Samples of any depth fit `u16`.
pub struct IntraScratch {
    /// `p[-1][y]` and `p[x][-1]` for y, x in 0..2n, after substitution.
    left: [u16; 64],
    top: [u16; 64],
    /// `p[-1][-1]`.
    corner: u16,
    /// The smoothing filter's output, valid while `filtered` says so.
    fl: [u16; 64],
    ft: [u16; 64],
    fc: u16,
    /// Whether `fl` / `ft` / `fc` hold the prepared block's smoothed
    /// references, and for which strong-filter eligibility (`strong &&
    /// c_idx == 0`, the only way the smoothing depends on the predict
    /// call); [`prepare`] clears it.
    filtered: Option<bool>,
    /// `ref[]` of the angular predictor, biased by `n` (see [`REF_LEN`]).
    ref_buf: [u16; REF_LEN],
    /// Which reference samples may be used.
    pub avail: RefAvail,
}

impl Default for IntraScratch {
    fn default() -> Self {
        IntraScratch {
            left: [0; 64],
            top: [0; 64],
            corner: 0,
            fl: [0; 64],
            ft: [0; 64],
            fc: 0,
            filtered: None,
            ref_buf: [0; REF_LEN],
            avail: RefAvail { corner: false, left: [false; 64], top: [false; 64] },
        }
    }
}

/// Availability of each reference sample of a block: `left[0..2n]` for
/// `p[-1][y]`, `top[0..2n]` for `p[x][-1]`, and the corner.
pub struct RefAvail {
    /// `p[-1][-1]`.
    pub corner: bool,
    /// `p[-1][y]`, y in 0..2n.
    pub left: [bool; 64],
    /// `p[x][-1]`, x in 0..2n.
    pub top: [bool; 64],
}

/// Predict an `n x n` block at `(x0, y0)` (plane coordinates) into the
/// plane. `mode` is the intra prediction mode (0 planar, 1 DC, 2..=34
/// angular), `c_idx` the component, `filter` whether the reference samples
/// get the smoothing filter (luma, or any component in 4:4:4, unless the
/// range extension disables it), `boundary_filter` whether DC and the pure
/// horizontal / vertical modes smooth the block edge (luma, unless a
/// lossless block with implicit RDPCM), `bit_depth` the sample depth,
/// `strong` the SPS strong intra smoothing flag; `sc.avail` says which
/// neighbouring samples may be used and the rest of `sc` is scratch.
#[allow(clippy::too_many_arguments)]
pub fn predict<S: Sample>(
    dsp: &HevcDsp<S>,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    x0: usize,
    y0: usize,
    n: usize,
    mode: u32,
    c_idx: usize,
    filter: bool,
    boundary_filter: bool,
    bit_depth: u32,
    strong: bool,
) {
    prepare(plane, sc, x0, y0, n, bit_depth);
    predict_prepared(dsp, plane, sc, x0, y0, n, mode, c_idx, filter, boundary_filter, bit_depth, strong);
}

/// Gather the reference samples of the `n x n` block at `(x0, y0)` into
/// `sc` and substitute the unavailable ones (8.4.4.2.2), as `sc.avail`
/// says. Any number of [`predict_prepared`] calls for this block may
/// follow: a prediction writes only inside the block, and its references
/// are all outside it.
pub fn prepare<S: Sample>(plane: &Plane16<S>, sc: &mut IntraScratch, x0: usize, y0: usize, n: usize, bit_depth: u32) {
    let IntraScratch { left, top, corner, filtered, avail, .. } = sc;
    *filtered = None;
    let stride = plane.stride;
    let base = plane.offset(x0 as isize, y0 as isize);
    let n2 = 2 * n;
    let d = &plane.data;
    // Gather p[-1][-1], p[-1][0..2n] and p[0..2n][-1]. The common case —
    // every sample available — is two straight copies.
    let all = avail.corner && avail.left[..n2].iter().all(|&a| a) && avail.top[..n2].iter().all(|&a| a);
    if all {
        *corner = d[base - stride - 1].to_i32() as u16;
        for (y, l) in left[..n2].iter_mut().enumerate() {
            *l = d[base + y * stride - 1].to_i32() as u16;
        }
        for (t, s) in top[..n2].iter_mut().zip(&d[base - stride..base - stride + n2]) {
            *t = s.to_i32() as u16;
        }
        return;
    }
    let mut any = false;
    if avail.corner {
        *corner = d[base - stride - 1].to_i32() as u16;
        any = true;
    }
    for y in 0..n2 {
        if avail.left[y] {
            left[y] = d[base + y * stride - 1].to_i32() as u16;
            any = true;
        }
    }
    for x in 0..n2 {
        if avail.top[x] {
            top[x] = d[base - stride + x].to_i32() as u16;
            any = true;
        }
    }
    // Substitution (8.4.4.2.2). Order: p[-1][2n-1] .. p[-1][-1] (bottom-left
    // upwards), then p[0][-1] .. p[2n-1][-1].
    if !any {
        let v = 1u16 << (bit_depth - 1);
        left[..n2].fill(v);
        top[..n2].fill(v);
        *corner = v;
        return;
    }
    // Search from bottom-left for the first available sample: left[2n-1]
    // down to left[0], corner, top[0..2n].
    if !avail.left[n2 - 1] {
        let found = (0..n2 - 1)
            .rev()
            .find(|&y| avail.left[y])
            .map(|y| left[y])
            .or(if avail.corner { Some(*corner) } else { None })
            .or_else(|| (0..n2).find(|&x| avail.top[x]).map(|x| top[x]));
        left[n2 - 1] = found.expect("some reference sample is available");
    }
    let mut cur = left[n2 - 1];
    for y in (0..n2 - 1).rev() {
        if !avail.left[y] {
            left[y] = cur;
        } else {
            cur = left[y];
        }
    }
    if !avail.corner {
        *corner = cur;
    } else {
        cur = *corner;
    }
    for (t, &a) in top[..n2].iter_mut().zip(&avail.top[..n2]) {
        if !a {
            *t = cur;
        } else {
            cur = *t;
        }
    }
}

/// The smoothing filter (8.4.4.2.3) over the prepared references, into
/// `fl` / `ft` / `fc`: the bilinear strong filter for a flat 32x32 luma
/// block under the SPS flag, the [1, 2, 1] filter otherwise.
fn smooth(sc: &mut IntraScratch, n: usize, c_idx: usize, bit_depth: u32, strong: bool) {
    let IntraScratch { left, top, corner, fl, ft, fc, filtered, .. } = sc;
    let n2 = 2 * n;
    let (c, l, t) = (*corner as i32, |i: usize| left[i] as i32, |i: usize| top[i] as i32);
    let bi = strong
        && c_idx == 0
        && n == 32
        && (c + t(n2 - 1) - 2 * t(n - 1)).abs() < (1 << (bit_depth - 5))
        && (c + l(n2 - 1) - 2 * l(n - 1)).abs() < (1 << (bit_depth - 5));
    if bi {
        *fc = *corner;
        for i in 0..n2 - 1 {
            // pF[-1][i] = ((63-i)*p[-1][-1] + (i+1)*p[-1][63] + 32) >> 6, i=0..62
            fl[i] = (((63 - i as i32) * c + (i as i32 + 1) * l(63) + 32) >> 6) as u16;
            ft[i] = (((63 - i as i32) * c + (i as i32 + 1) * t(63) + 32) >> 6) as u16;
        }
        fl[63] = left[63];
        ft[63] = top[63];
    } else {
        *fc = ((l(0) + 2 * c + t(0) + 2) >> 2) as u16;
        fl[0] = ((c + 2 * l(0) + l(1) + 2) >> 2) as u16;
        for (f, w) in fl[1..n2 - 1].iter_mut().zip(left[..n2].windows(3)) {
            *f = ((w[0] as i32 + 2 * w[1] as i32 + w[2] as i32 + 2) >> 2) as u16;
        }
        fl[n2 - 1] = left[n2 - 1];
        ft[0] = ((c + 2 * t(0) + t(1) + 2) >> 2) as u16;
        for (f, w) in ft[1..n2 - 1].iter_mut().zip(top[..n2].windows(3)) {
            *f = ((w[0] as i32 + 2 * w[1] as i32 + w[2] as i32 + 2) >> 2) as u16;
        }
        ft[n2 - 1] = top[n2 - 1];
    }
    *filtered = Some(strong && c_idx == 0);
}

/// Predict the block [`prepare`] gathered, with `mode`; the other
/// parameters are [`predict`]'s.
#[allow(clippy::too_many_arguments)]
pub fn predict_prepared<S: Sample>(
    dsp: &HevcDsp<S>,
    plane: &mut Plane16<S>,
    sc: &mut IntraScratch,
    x0: usize,
    y0: usize,
    n: usize,
    mode: u32,
    c_idx: usize,
    filter: bool,
    boundary_filter: bool,
    bit_depth: u32,
    strong: bool,
) {
    // Filtering (8.4.4.2.3): luma (or 4:4:4) only, not for DC / 4x4.
    let smoothed = filter && mode != 1 && n != 4 && {
        let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
        let thres = match n {
            8 => 7,
            16 => 1,
            32 => 0,
            _ => 10, // never filtered
        };
        min_dist > thres
    };
    if smoothed && sc.filtered != Some(strong && c_idx == 0) {
        smooth(sc, n, c_idx, bit_depth, strong);
    }
    let stride = plane.stride;
    let base = plane.offset(x0 as isize, y0 as isize);
    let IntraScratch { left, top, corner, fl, ft, fc, ref_buf, .. } = sc;
    let (left, top, corner) = if smoothed { (&*fl, &*ft, *fc) } else { (&*left, &*top, *corner) };
    let dst = &mut plane.data[base..];
    match mode {
        0 => (dsp.intra_planar)(dst, stride, left, top, n),
        1 => (dsp.intra_dc)(dst, stride, left, top, n, boundary_filter && n < 32),
        _ => {
            let angle = INTRA_PRED_ANGLE[(mode - 2) as usize];
            // ref[] of 8.4.4.2.6, biased by n so negative indices work:
            // the side the mode points at, extended by the other side's
            // projection (negative angles) or by its own second half.
            let (main, side) = if mode >= 18 { (top, left) } else { (left, top) };
            ref_buf[n] = corner;
            ref_buf[n + 1..=2 * n].copy_from_slice(&main[..n]);
            if angle < 0 {
                let last = (n as i32 * angle) >> 5;
                if last < -1 {
                    let inv = INV_ANGLE[(mode - 11) as usize];
                    for x in last..=-1 {
                        // ref[x] = p[-1][-1 + ((x*invAngle+128)>>8)] (or its transpose)
                        let idx = -1 + ((x * inv + 128) >> 8);
                        ref_buf[(x + n as i32) as usize] = if idx < 0 { corner } else { side[idx as usize] };
                    }
                }
            } else {
                ref_buf[2 * n + 1..=3 * n].copy_from_slice(&main[n..2 * n]);
            }
            // Horizontal-ish modes run along columns: the kernel predicts
            // the transposed block.
            (dsp.intra_angular)(dst, stride, &ref_buf[..], n, angle, mode < 18);
            if boundary_filter && n < 32 {
                let max = (1i32 << bit_depth) - 1;
                let c = corner as i32;
                if mode == 26 {
                    for y in 0..n {
                        let v = (top[0] as i32 + ((left[y] as i32 - c) >> 1)).clamp(0, max);
                        dst[y * stride] = S::from_i32(v);
                    }
                } else if mode == 10 {
                    for x in 0..n {
                        let v = (left[0] as i32 + ((top[x] as i32 - c) >> 1)).clamp(0, max);
                        dst[x] = S::from_i32(v);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;

    /// The predictor as it stood before the split and the kernel table —
    /// every step inline, on `i32` arrays — kept verbatim as the reference
    /// the table-driven one must reproduce (its index loops included).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn predict_reference<S: Sample>(
        plane: &mut Plane16<S>,
        avail: &RefAvail,
        x0: usize,
        y0: usize,
        n: usize,
        mode: u32,
        c_idx: usize,
        filter: bool,
        boundary_filter: bool,
        bit_depth: u32,
        strong: bool,
    ) {
        let mut left = [0i32; 64];
        let mut top = [0i32; 64];
        let mut fl = [0i32; 64];
        let mut ft = [0i32; 64];
        let mut ref_buf = [0i32; 3 * 64 + 1];
        let stride = plane.stride;
        let base = plane.offset(x0 as isize, y0 as isize);
        let n2 = 2 * n;
        let mut corner: i32 = 0;
        let mut any = false;
        if avail.corner {
            corner = plane.data[base - stride - 1].to_i32();
            any = true;
        }
        for y in 0..n2 {
            if avail.left[y] {
                left[y] = plane.data[base + y * stride - 1].to_i32();
                any = true;
            }
        }
        for x in 0..n2 {
            if avail.top[x] {
                top[x] = plane.data[base - stride + x].to_i32();
                any = true;
            }
        }
        if !any {
            let v = 1i32 << (bit_depth - 1);
            left[..n2].fill(v);
            top[..n2].fill(v);
            corner = v;
        } else {
            if !avail.left[n2 - 1] {
                let mut found = None;
                for y in (0..n2 - 1).rev() {
                    if avail.left[y] {
                        found = Some(left[y]);
                        break;
                    }
                }
                if found.is_none() && avail.corner {
                    found = Some(corner);
                }
                if found.is_none() {
                    for x in 0..n2 {
                        if avail.top[x] {
                            found = Some(top[x]);
                            break;
                        }
                    }
                }
                left[n2 - 1] = found.unwrap();
            }
            let mut cur = Some(left[n2 - 1]);
            for y in (0..n2 - 1).rev() {
                if !avail.left[y] {
                    left[y] = cur.unwrap();
                } else {
                    cur = Some(left[y]);
                }
            }
            if !avail.corner {
                corner = cur.unwrap();
            } else {
                cur = Some(corner);
            }
            for x in 0..n2 {
                if !avail.top[x] {
                    top[x] = cur.unwrap();
                } else {
                    cur = Some(top[x]);
                }
            }
        }
        if filter && mode != 1 && n != 4 {
            let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
            let thres = match n {
                8 => 7,
                16 => 1,
                32 => 0,
                _ => 10,
            };
            if min_dist > thres {
                let bi = strong
                    && c_idx == 0
                    && n == 32
                    && (corner + top[n2 - 1] - 2 * top[n - 1]).abs() < (1 << (bit_depth - 5))
                    && (corner + left[n2 - 1] - 2 * left[n - 1]).abs() < (1 << (bit_depth - 5));
                let fc;
                if bi {
                    fc = corner;
                    for i in 0..n2 - 1 {
                        fl[i] = ((63 - i as i32) * corner + (i as i32 + 1) * left[63] + 32) >> 6;
                        ft[i] = ((63 - i as i32) * corner + (i as i32 + 1) * top[63] + 32) >> 6;
                    }
                    fl[63] = left[63];
                    ft[63] = top[63];
                } else {
                    fc = (left[0] + 2 * corner + top[0] + 2) >> 2;
                    fl[0] = (corner + 2 * left[0] + left[1] + 2) >> 2;
                    for y in 1..n2 - 1 {
                        fl[y] = (left[y - 1] + 2 * left[y] + left[y + 1] + 2) >> 2;
                    }
                    fl[n2 - 1] = left[n2 - 1];
                    ft[0] = (corner + 2 * top[0] + top[1] + 2) >> 2;
                    for x in 1..n2 - 1 {
                        ft[x] = (top[x - 1] + 2 * top[x] + top[x + 1] + 2) >> 2;
                    }
                    ft[n2 - 1] = top[n2 - 1];
                }
                left[..n2].copy_from_slice(&fl[..n2]);
                top[..n2].copy_from_slice(&ft[..n2]);
                corner = fc;
            }
        }
        let max = (1i32 << bit_depth) - 1;
        let log2n = n.trailing_zeros();
        match mode {
            0 => {
                for y in 0..n {
                    let (ly, ln, tn) = (left[y], left[n], top[n]);
                    let ry = n as i32 - 1 - y as i32;
                    for x in 0..n {
                        let v = ((n as i32 - 1 - x as i32) * ly + (x as i32 + 1) * tn + ry * top[x] + (y as i32 + 1) * ln + n as i32) >> (log2n + 1);
                        plane.data[base + y * stride + x] = S::from_i32(v);
                    }
                }
            }
            1 => {
                let mut sum = n as i32;
                for i in 0..n {
                    sum += top[i] + left[i];
                }
                let dc = sum >> (log2n + 1);
                for y in 0..n {
                    plane.data[base + y * stride..base + y * stride + n].fill(S::from_i32(dc));
                }
                if boundary_filter && n < 32 {
                    plane.data[base] = S::from_i32((left[0] + 2 * dc + top[0] + 2) >> 2);
                    for x in 1..n {
                        plane.data[base + x] = S::from_i32((top[x] + 3 * dc + 2) >> 2);
                    }
                    for y in 1..n {
                        plane.data[base + y * stride] = S::from_i32((left[y] + 3 * dc + 2) >> 2);
                    }
                }
            }
            _ => {
                let angle = INTRA_PRED_ANGLE[(mode - 2) as usize];
                let off = n as i32;
                let (main, side): (&[i32; 64], &[i32; 64]) = if mode >= 18 { (&top, &left) } else { (&left, &top) };
                ref_buf[off as usize] = corner;
                for x in 1..=n {
                    ref_buf[off as usize + x] = main[x - 1];
                }
                if angle < 0 {
                    let last = (n as i32 * angle) >> 5;
                    if last < -1 {
                        let inv = INV_ANGLE[(mode - 11) as usize];
                        for x in last..=-1 {
                            let idx = -1 + ((x * inv + 128) >> 8);
                            let v = if idx < 0 { corner } else { side[idx as usize] };
                            ref_buf[(x + off) as usize] = v;
                        }
                    }
                } else {
                    for x in n + 1..=2 * n {
                        ref_buf[off as usize + x] = main[x - 1];
                    }
                }
                for y in 0..n {
                    let i_idx = ((y as i32 + 1) * angle) >> 5;
                    let i_fact = ((y as i32 + 1) * angle) & 31;
                    for x in 0..n {
                        let start = (x as i32 + i_idx + 1 + off) as usize;
                        let v = if i_fact != 0 { ((32 - i_fact) * ref_buf[start] + i_fact * ref_buf[start + 1] + 16) >> 5 } else { ref_buf[start] };
                        let (px, py) = if mode >= 18 { (x, y) } else { (y, x) };
                        plane.data[base + py * stride + px] = S::from_i32(v);
                    }
                }
                if mode == 26 && boundary_filter && n < 32 {
                    for y in 0..n {
                        let v = (top[0] + ((left[y] - corner) >> 1)).clamp(0, max);
                        plane.data[base + y * stride] = S::from_i32(v);
                    }
                }
                if mode == 10 && boundary_filter && n < 32 {
                    for x in 0..n {
                        let v = (left[0] + ((top[x] - corner) >> 1)).clamp(0, max);
                        plane.data[base + x] = S::from_i32(v);
                    }
                }
            }
        }
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Every table this host builds for `S`: the scalar reference and each
    /// x86 rung (or NEON, or the one wasm rung), cumulatively as in the
    /// field.
    fn tables<S: Sample>() -> Vec<(&'static str, HevcDsp<S>)> {
        let mut v = vec![("scalar", HevcDsp::<S>::new(Cpu::SCALAR))];
        #[cfg(target_arch = "x86_64")]
        {
            let b = Cpu::SCALAR;
            let sse2 = Cpu { sse2: true, ..b };
            let ssse3 = Cpu { ssse3: true, ..sse2 };
            let sse41 = Cpu { sse41: true, ..ssse3 };
            let avx = Cpu { avx: true, ..sse41 };
            let avx2 = Cpu { avx2: true, ..avx };
            let top = Cpu::detect();
            for (name, cpu, have) in [
                ("sse2", sse2, top.sse2),
                ("ssse3", ssse3, top.ssse3),
                ("sse4.1", sse41, top.sse41),
                ("avx", avx, top.avx),
                ("avx2", avx2, top.avx2),
                ("avx512", Cpu { avx512: true, ..avx2 }, top.avx512),
            ] {
                if have {
                    v.push((name, HevcDsp::<S>::new(cpu)));
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        v.push(("native", HevcDsp::<S>::new(Cpu::detect())));
        v
    }

    /// Availability patterns: all, none, and the partial shapes a block
    /// meets at picture and slice edges, under constrained intra
    /// prediction, and in z-scan order (the below-left half missing, the
    /// above-right half missing), plus a random per-unit pattern.
    fn avail_pattern(seed: &mut u64, n: usize, kind: u32) -> RefAvail {
        let n2 = 2 * n;
        let mut a = RefAvail { corner: true, left: [false; 64], top: [false; 64] };
        let all = |a: &mut RefAvail| {
            a.left[..n2].fill(true);
            a.top[..n2].fill(true);
        };
        match kind {
            0 => all(&mut a),
            1 => a.corner = false,
            2 => {
                all(&mut a);
                a.left[n..n2].fill(false);
            }
            3 => {
                all(&mut a);
                a.top[n..n2].fill(false);
            }
            4 => {
                // Left column only (top picture edge).
                a.corner = false;
                a.left[..n].fill(true);
            }
            5 => {
                // Top row only (left picture edge).
                a.corner = false;
                a.top[..n2].fill(true);
            }
            _ => {
                // Constrained intra: availability in units of four samples.
                a.corner = lcg(seed).is_multiple_of(2);
                for u in 0..n2.div_ceil(4) {
                    let (l, t) = (!lcg(seed).is_multiple_of(3), !lcg(seed).is_multiple_of(3));
                    for k in 0..4.min(n2 - 4 * u) {
                        a.left[4 * u + k] = l;
                        a.top[4 * u + k] = t;
                    }
                }
            }
        }
        a
    }

    fn sweep<S: Sample>(depths: &[u32]) {
        let tables = tables::<S>();
        let mut seed = 0x1a7a_u64;
        let mut n_cmp = 0u64;
        for &bd in depths {
            let max = (1u32 << bd) - 1;
            for &n in &[4usize, 8, 16, 32] {
                for kind in 0..8 {
                    for flat in [false, true] {
                        // A plane with the block at (n, n): its references
                        // lie inside the picture. Flat content (gentle
                        // ramps) passes the strong filter's flatness test;
                        // random content exercises the clips.
                        let w = 4 * n + 8;
                        let mut plane = Plane16::<S> { data: vec![S::default(); (w + 16) * (w + 16)], width: w, height: w, pad: 8, stride: w + 16 };
                        for y in 0..w {
                            for x in 0..w {
                                let v = if flat { ((x + y) as u32 * 3 / 2 + max / 4).min(max) } else { lcg(&mut seed) % (max + 1) };
                                let o = plane.offset(x as isize, y as isize);
                                plane.data[o] = S::from_i32(v as i32);
                            }
                        }
                        let avail = avail_pattern(&mut seed, n, kind);
                        for mode in 0..35u32 {
                            for &(c_idx, filter, boundary, strong) in &[(0usize, true, true, true), (0, true, true, false), (1, false, false, false), (1, true, false, true), (0, false, false, false)] {
                                let mut want = plane.clone();
                                predict_reference(&mut want, &avail, n, n, n, mode, c_idx, filter, boundary, bd, strong);
                                for (name, d) in &tables {
                                    let mut got = plane.clone();
                                    let mut sc = IntraScratch { avail: RefAvail { corner: avail.corner, left: avail.left, top: avail.top }, ..Default::default() };
                                    predict(d, &mut got, &mut sc, n, n, n, mode, c_idx, filter, boundary, bd, strong);
                                    assert!(got.data == want.data, "{name}: {bd} bits, {n}x{n}, mode {mode}, avail {kind}, flat {flat}, c_idx {c_idx}, filter {filter}, boundary {boundary}, strong {strong}");
                                    n_cmp += 1;
                                }
                            }
                        }
                        // Prepared once, then every mode: the encoder's
                        // search. Each mode against a fresh reference.
                        for (name, d) in &tables {
                            let mut got = plane.clone();
                            let mut sc = IntraScratch { avail: RefAvail { corner: avail.corner, left: avail.left, top: avail.top }, ..Default::default() };
                            prepare(&got, &mut sc, n, n, n, bd);
                            for mode in (0..35u32).rev() {
                                predict_prepared(d, &mut got, &mut sc, n, n, n, mode, 0, true, true, bd, true);
                                let mut want = plane.clone();
                                predict_reference(&mut want, &avail, n, n, n, mode, 0, true, true, bd, true);
                                let rows = |p: &Plane16<S>| (0..n).flat_map(|y| (0..n).map(move |x| (x, y))).map(|(x, y)| p.data[p.offset((n + x) as isize, (n + y) as isize)]).collect::<Vec<_>>();
                                assert!(rows(&got) == rows(&want), "{name} prepared: {bd} bits, {n}x{n}, mode {mode}, avail {kind}, flat {flat}");
                                n_cmp += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(n_cmp > 0);
    }

    /// The table-driven predictor against the one it replaced, every mode,
    /// size, availability pattern and filter configuration, on every table
    /// the host has: 8-bit samples.
    #[test]
    fn predict_matches_reference_u8() {
        sweep::<u8>(&[8]);
    }

    /// The same at 16 bits, at every depth the u16 table serves.
    #[test]
    fn predict_matches_reference_u16() {
        sweep::<u16>(&[8, 9, 10, 12, 14, 16]);
    }
}
