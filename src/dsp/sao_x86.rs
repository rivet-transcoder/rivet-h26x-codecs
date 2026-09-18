//! x86-64 SIMD version of the H.265 encoder's SAO edge statistics
//! ([`super::distortion::SaoEdgeStatsFn`]), for both sample widths.
//!
//! The SAO decision classifies every sample of a CTB into its edge
//! category for each of the four classes, and sums the error per category;
//! at 1080p that pass was most of the 10–11% of an H.265 encode the
//! decision took. Eight samples a vector, as i16 lanes (a byte sample is
//! zero-extended, a 16-bit one read as it is): `sign(r - a)` is the
//! difference of the two `pcmpgtw` masks, the category `2 + sign(r - a) +
//! sign(r - b)`, and per category a `pcmpeqw` mask subtracted into a
//! lane counter and `pmaddwd` of the masked error against ones into i32
//! lane sums. A sample near a rail (`reach` of 0 or of `max`) is also
//! reported one by one, as the scalar reference reports it: those are few
//! wherever the content is not crushed, and they need their exact values.
//!
//! Every sum is an integer sum, so the order the lanes add in cannot
//! change the result: the statistics are the scalar reference's exactly.
//! A `max` above 32767 (samples `pcmpgtw` would read as negative) is the
//! reference's.

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::distortion::DistortionDsp;
use crate::sample::Sample;

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::distortion::sao_edge_stats_scalar;
        use crate::sample::Sample;

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        /// Eight samples at `p` as eight i16.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load8<S: Sample>(p: *const S) -> __m128i {
            unsafe {
                if S::BYTES == 1 {
                    zx8(_mm_loadl_epi64(p as *const __m128i))
                } else {
                    _mm_loadu_si128(p as *const __m128i)
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn sao_edge_stats<S: Sample>(
            rec: &[S],
            origin: usize,
            stride: usize,
            src: &[S],
            src_stride: usize,
            w: usize,
            h: usize,
            na: isize,
            nb: isize,
            max: i32,
            reach: i32,
            tally: &mut [[i64; 2]; 5],
            near: &mut Vec<(u8, u16, i32)>,
        ) {
            // The furthest either neighbour reaches, before and after the
            // region, and the source rows the region reads.
            let (lo, hi) = (na.min(nb).min(0), na.max(nb).max(0));
            let fits = h > 0
                && w > 0
                && origin as isize + lo >= 0
                && (origin + (h - 1) * stride + w) as isize + hi <= rec.len() as isize
                && (h - 1) * src_stride + w <= src.len();
            if !fits || max > 32767 || w < 8 {
                return sao_edge_stats_scalar(rec, origin, stride, src, src_stride, w, h, na, nb, max, reach, tally, near);
            }
            unsafe { edge_impl(rec, origin, stride, src, src_stride, w, h, na, nb, max, reach, tally, near) }
        }

        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn edge_impl<S: Sample>(
            recs: &[S],
            origin: usize,
            stride: usize,
            srcs: &[S],
            src_stride: usize,
            w: usize,
            h: usize,
            na: isize,
            nb: isize,
            max: i32,
            reach: i32,
            tally: &mut [[i64; 2]; 5],
            near: &mut Vec<(u8, u16, i32)>,
        ) {
            unsafe {
                let (rec, src) = (recs.as_ptr(), srcs.as_ptr());
                let ones = _mm_set1_epi16(1);
                let lo_rail = _mm_set1_epi16(reach as i16);
                let hi_rail = _mm_set1_epi16((max - reach) as i16);
                let cats: [__m128i; 5] = std::array::from_fn(|c| _mm_set1_epi16(c as i16));
                let mut totals = [[0i64; 2]; 5];
                for y in 0..h {
                    let row = rec.add(origin + y * stride);
                    let srow = src.add(y * src_stride);
                    // Per-row lane counters and sums, folded into i64 at
                    // the end of the row: a lane then holds at most w / 8
                    // samples, well inside i16 and i32 at any width.
                    let mut cnt = [_mm_setzero_si128(); 5];
                    let mut sums = [_mm_setzero_si128(); 5];
                    let mut x = 0;
                    while x + 8 <= w {
                        let p = row.add(x);
                        let r = load8(p);
                        let a = load8(p.offset(na));
                        let b = load8(p.offset(nb));
                        let s = load8(srow.add(x));
                        // sign(r - a) = (r < a mask) - (r > a mask), masks being -1.
                        let sa = _mm_sub_epi16(_mm_cmpgt_epi16(a, r), _mm_cmpgt_epi16(r, a));
                        let sb = _mm_sub_epi16(_mm_cmpgt_epi16(b, r), _mm_cmpgt_epi16(r, b));
                        let e = _mm_add_epi16(_mm_add_epi16(sa, sb), cats[2]);
                        let err = _mm_sub_epi16(s, r);
                        for c in 0..5 {
                            let m = _mm_cmpeq_epi16(e, cats[c]);
                            cnt[c] = _mm_sub_epi16(cnt[c], m);
                            sums[c] = _mm_add_epi32(sums[c], _mm_madd_epi16(_mm_and_si128(err, m), ones));
                        }
                        let rail = _mm_or_si128(_mm_cmpgt_epi16(lo_rail, r), _mm_cmpgt_epi16(r, hi_rail));
                        let bits = _mm_movemask_epi8(rail);
                        if bits != 0 {
                            let (mut rv, mut ev, mut dv) = ([0i16; 8], [0i16; 8], [0i16; 8]);
                            _mm_storeu_si128(rv.as_mut_ptr() as *mut __m128i, r);
                            _mm_storeu_si128(ev.as_mut_ptr() as *mut __m128i, e);
                            _mm_storeu_si128(dv.as_mut_ptr() as *mut __m128i, err);
                            for l in 0..8 {
                                if bits & (1 << (2 * l)) != 0 {
                                    near.push((ev[l] as u8, rv[l] as u16, dv[l] as i32));
                                }
                            }
                        }
                        x += 8;
                    }
                    for c in 0..5 {
                        let fold = |q: __m128i| {
                            let q = _mm_add_epi32(q, _mm_shuffle_epi32(q, 0b01_00_11_10));
                            _mm_cvtsi128_si32(_mm_add_epi32(q, _mm_shuffle_epi32(q, 0b10_11_00_01))) as i64
                        };
                        totals[c][0] += fold(_mm_madd_epi16(cnt[c], ones));
                        totals[c][1] += fold(sums[c]);
                    }
                    if x < w {
                        // The last few samples of the row, as the reference does them.
                        sao_edge_stats_scalar(recs, origin + y * stride + x, stride, &srcs[y * src_stride + x..], src_stride, w - x, 1, na, nb, max, reach, tally, near);
                    }
                }
                for c in 0..5 {
                    tally[c][0] += totals[c][0];
                    tally[c][1] += totals[c][1];
                }
            }
        }
    };
}

/// SSE2: baseline on x86-64.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSE4.1: `pmovzxbw` for the byte samples' widening loads.
pub(crate) mod sse41 {
    #![allow(dead_code)]
    kernels!("sse4.1", sse41);
}

/// AVX: the SSE4.1 primitive set, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// Install the best edge-statistics kernel `cpu` can run.
pub fn install<S: Sample>(d: &mut DistortionDsp<S>, cpu: Cpu) {
    if cpu.sse2 {
        d.sao_edge_stats = sse2::sao_edge_stats::<S>;
    }
    if cpu.sse41 {
        d.sao_edge_stats = sse41::sao_edge_stats::<S>;
    }
    if cpu.avx {
        d.sao_edge_stats = avx::sao_edge_stats::<S>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn rungs<S: Sample>() -> Vec<(&'static str, DistortionDsp<S>)> {
        let b = Cpu::SCALAR;
        let sse2 = Cpu { sse2: true, ..b };
        let sse41 = Cpu { ssse3: true, sse41: true, ..sse2 };
        let avx = Cpu { avx: true, ..sse41 };
        let top = Cpu::detect();
        [("sse2", sse2, top.sse2), ("sse4.1", sse41, top.sse41), ("avx", avx, top.avx)]
            .into_iter()
            .filter(|&(_, _, have)| have)
            .map(|(n, c, _)| {
                let mut d = DistortionDsp::<S>::scalar();
                install(&mut d, c);
                (n, d)
            })
            .collect()
    }

    /// Every rung against the reference: the four classes' neighbour
    /// offsets, regions whose width is and is not a multiple of eight, and
    /// planes uniform over the depth, crushed towards either rail (so most
    /// samples come back as near), or flat (every category but 2 empty).
    fn sweep<S: Sample>(depths: &[u32]) {
        let tables = rungs::<S>();
        assert!(!tables.is_empty(), "no x86 rung to test");
        let s = DistortionDsp::<S>::scalar();
        let mut seed = 0x5a0e_u64;
        let stride = 80;
        for &bd in depths {
            let max = (1i32 << bd) - 1;
            for kind in 0..4 {
                let plane: Vec<S> = (0..stride * 80)
                    .map(|_| {
                        let v = match kind {
                            0 => lcg(&mut seed) as i32 % (max + 1),
                            1 => lcg(&mut seed) as i32 % 40,
                            2 => max - lcg(&mut seed) as i32 % 40,
                            _ => max / 2,
                        };
                        S::from_i32(v)
                    })
                    .collect();
                let src: Vec<S> = (0..stride * 80).map(|_| S::from_i32(lcg(&mut seed) as i32 % (max + 1))).collect();
                for &(dx, dy) in &[(1isize, 0isize), (0, 1), (1, 1), (-1, 1)] {
                    let (na, nb) = (-dy * stride as isize - dx, dy * stride as isize + dx);
                    for &(w, h) in &[(8usize, 8usize), (13, 7), (32, 32), (64, 64), (5, 3), (61, 2)] {
                        let origin = 2 * stride + 2;
                        let mut want = [[0i64; 2]; 5];
                        let mut near_want = Vec::new();
                        (s.sao_edge_stats)(&plane, origin, stride, &src[origin..], stride, w, h, na, nb, max, 31, &mut want, &mut near_want);
                        near_want.sort();
                        for (name, d) in &tables {
                            let mut got = [[0i64; 2]; 5];
                            let mut near_got = Vec::new();
                            (d.sao_edge_stats)(&plane, origin, stride, &src[origin..], stride, w, h, na, nb, max, 31, &mut got, &mut near_got);
                            near_got.sort();
                            assert_eq!(got, want, "{name}: {bd} bits, plane {kind}, class ({dx}, {dy}), {w}x{h}");
                            assert_eq!(near_got, near_want, "{name}: near, {bd} bits, plane {kind}, class ({dx}, {dy}), {w}x{h}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn edge_stats_match_scalar_u8() {
        sweep::<u8>(&[8]);
    }

    #[test]
    fn edge_stats_match_scalar_u16() {
        sweep::<u16>(&[9, 10, 12, 14, 15]);
    }
}
