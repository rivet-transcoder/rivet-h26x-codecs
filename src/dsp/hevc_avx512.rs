//! AVX-512 versions of the H.265 kernels for 16-bit sample planes, and of
//! the sample-size-independent ones for both tables (x86-64), installed over
//! [`super::hevc_avx2`].
//!
//! **The 32-point inverse transform** is the one place in either codec
//! where a 512-bit vector is the natural width rather than a way of
//! stacking rows: a row of a 32x32 transform block is exactly thirty-two
//! 16-bit coefficients. The AVX2 kernel runs both of its stages in two
//! 16-lane steps per output row; this runs one. The smaller transforms have
//! nothing to widen — a 16-point row *is* a 256-bit vector — and stay on
//! AVX2.
//!
//! **Luma interpolation at 16 bits** takes the shapes
//! [`super::hevc_avx512_u8`] found worth it at 8 bits, for the same
//! reasons: the second (vertical, 14-bit) stage of a diagonal block at any
//! width — that stage *is* sample-size independent, and the u16 table
//! installs the 8-bit table's very kernels — and the first stage of a block
//! 32 samples wide or more, here thirty-two u16 lanes of `pmaddwd` pairs
//! where AVX2 has sixteen. Fused into default-weighted prediction as the
//! AVX2 kernels are, with the depth's shift and clip in
//! [`super::hevc_avx2::Out16`]. Chroma, whole-sample blocks and first
//! stages narrower than 32 keep AVX2.
//!
//! Bit-exact against the scalar reference in the tests below.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::hevc::HevcDsp;
use super::hevc_avx2 as w16;
use super::hevc_avx2::{MODE_BI, MODE_I16, MODE_UNI, Out16};
use super::hevc_avx512_u8 as w8;
use crate::hevc::tables::{QPEL_FILTERS, TRANSFORM32};

/// The 32-point inverse DCT. Shared by both sample widths — the transform
/// works on 16-bit coefficients whatever the samples are.
pub(super) const IDCT32: super::hevc::IdctFn = idct32_avx512;

/// Replace the AVX2 entries of `d` that AVX-512 improves on (16-bit
/// samples). Called after [`super::hevc_avx2::install`].
pub fn install(d: &mut HevcDsp<u16>) {
    d.idct[3] = IDCT32;
    d.qpel_h = qpel_h16;
    d.qpel_v = qpel_v16;
    // Sample-size independent: the 8-bit table's kernels, as they are.
    d.qpel_v2 = w8::qpel_v2_avx512;
    d.epel_v2 = w8::epel_v2_avx512;
    d.qpel_uni = qpel_uni16;
    d.qpel_bi = qpel_bi16;
}

// ----------------------------------------------------------------------
// 16-bit-sample luma interpolation
// ----------------------------------------------------------------------

/// Whether a `w`-wide, `rows`-row window of a u16 plane of `stride` can be
/// read 32 lanes at a time, `extra` samples past the last output column.
#[inline(always)]
fn fits16(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    (rows - 1) * stride + (w - 1) / 32 * 32 + extra + 32 <= len
}

/// Finish 32 lanes of 14-bit output the way `Out16` asks (see
/// `super::hevc_avx2::emit16`): as they are, or default-weighted and
/// clipped to the depth. Returns the lanes to store.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn finish<const MODE: u8>(out: &Out16, v: __m512i, other: *const i16, n: usize) -> __m512i {
    unsafe {
        let sh = _mm_cvtsi32_si128(out.shift);
        let maxv = _mm512_set1_epi16(out.max as i16);
        let clip = |r: __m512i| _mm512_min_epi16(_mm512_max_epi16(r, _mm512_setzero_si512()), maxv);
        match MODE {
            MODE_UNI => clip(_mm512_sra_epi16(_mm512_adds_epi16(v, _mm512_set1_epi16(1 << (out.shift - 1))), sh)),
            MODE_BI => {
                let o = w8::load_i16(other, n);
                let ones = _mm512_set1_epi32(0x0001_0001);
                let round = _mm512_set1_epi32(1 << (out.shift - 1));
                let q = |u: __m512i| _mm512_sra_epi32(_mm512_add_epi32(_mm512_madd_epi16(u, ones), round), sh);
                clip(_mm512_packs_epi32(q(_mm512_unpacklo_epi16(o, v)), q(_mm512_unpackhi_epi16(o, v))))
            }
            _ => v,
        }
    }
}

/// Emit `n` (≤ 32) lanes of one row's output at (`row`, `x`).
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn emit_span16<const MODE: u8>(out: &Out16, row: usize, x: usize, v: __m512i, n: usize) {
    unsafe {
        if MODE == MODE_I16 {
            return w8::store_i16(out.i16.add(row * out.w + x), v, n);
        }
        let r = finish::<MODE>(out, v, out.other.add(row * out.w + x), n);
        w8::store_i16(out.dst.add(row * out.stride + x) as *mut i16, r, n);
    }
}

/// Emit a vector holding `rows` whole rows of a `w`-wide block (`w · rows`
/// ≤ 32 lanes), the first of them row `y0`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn emit_block16<const MODE: u8>(out: &Out16, y0: usize, rows: usize, w: usize, v: __m512i) {
    unsafe {
        let n = rows * w;
        if MODE == MODE_I16 {
            return w8::store_i16(out.i16.add(y0 * w), v, n);
        }
        let r = finish::<MODE>(out, v, out.other.add(y0 * w), n);
        let dst = out.dst.add(y0 * out.stride);
        match w {
            16 => {
                _mm256_storeu_si256(dst as *mut __m256i, _mm512_castsi512_si256(r));
                if rows > 1 {
                    _mm256_storeu_si256(dst.add(out.stride) as *mut __m256i, _mm512_extracti64x4_epi64::<1>(r));
                }
            }
            8 => {
                // One row a 128-bit lane, straight out of the register.
                let lanes = [_mm512_castsi512_si128(r), _mm512_extracti32x4_epi32::<1>(r), _mm512_extracti32x4_epi32::<2>(r), _mm512_extracti32x4_epi32::<3>(r)];
                for (k, l) in lanes.iter().enumerate().take(rows) {
                    _mm_storeu_si128(dst.add(k * out.stride) as *mut __m128i, *l);
                }
            }
            _ => {
                let mut t = [0u16; 32];
                _mm512_storeu_si512(t.as_mut_ptr() as *mut __m512i, r);
                for k in 0..rows {
                    std::ptr::copy_nonoverlapping(t.as_ptr().add(k * w), dst.add(k * out.stride), w);
                }
            }
        }
    }
}

/// `TAPS`-tap FIR over u16 samples, 32 outputs at a time, `step` samples
/// between taps (1 horizontal, the stride vertical). Only for `w >= 32`.
/// `pmaddwd` over `unpacklo` / `unpackhi` leaves lane j of `lo` holding
/// outputs 8j..8j+3 and lane j of `hi` 8j+4..8j+7, which `packs` — per
/// 128-bit lane — puts back in order.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn fir16<const TAPS: usize, const MODE: u8>(out: &Out16, src: *const u16, src_stride: usize, step: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm512_setzero_si512(); 4];
        for (k, ck) in c.iter_mut().enumerate().take(TAPS / 2) {
            *ck = _mm512_set1_epi32(w16::pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let s = src.add(y * src_stride);
            let mut x = 0;
            while x < w {
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for (k, &ck) in c.iter().enumerate().take(TAPS / 2) {
                    let a = _mm512_loadu_si512(s.add(x + 2 * k * step) as *const __m512i);
                    let b = _mm512_loadu_si512(s.add(x + (2 * k + 1) * step) as *const __m512i);
                    lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), ck));
                    hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), ck));
                }
                let r = _mm512_packs_epi32(_mm512_sra_epi32(lo, sh), _mm512_sra_epi32(hi, sh));
                emit_span16::<MODE>(out, y, x, r, (w - x).min(32));
                x += 32;
            }
        }
    }
}

/// The second stage of a diagonal block over the first stage's contiguous
/// `w`-strided 14-bit rows (`super::hevc_avx512_u8`'s `fir_v2`, finished
/// into u16 samples): at `w < 32` one 512-bit load is `32 / w` whole rows.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn fir_v2_16<const TAPS: usize, const MODE: u8>(out: &Out16, src: *const i16, w: usize, h: usize, taps: &[i8]) {
    unsafe {
        let mut c = [_mm512_setzero_si512(); 4];
        for (k, ck) in c.iter_mut().enumerate().take(TAPS / 2) {
            *ck = _mm512_set1_epi32(w16::pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sum = |at: usize| {
            let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for (k, &ck) in c.iter().enumerate().take(TAPS / 2) {
                let a = _mm512_loadu_si512(src.add(at + 2 * k * w) as *const __m512i);
                let b = _mm512_loadu_si512(src.add(at + (2 * k + 1) * w) as *const __m512i);
                lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), ck));
                hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), ck));
            }
            _mm512_packs_epi32(_mm512_srai_epi32::<6>(lo), _mm512_srai_epi32::<6>(hi))
        };
        if w >= 32 {
            for y in 0..h {
                let mut x = 0;
                while x < w {
                    emit_span16::<MODE>(out, y, x, sum(y * w + x), (w - x).min(32));
                    x += 32;
                }
            }
            return;
        }
        let rows_per = 32 / w;
        let mut y = 0;
        while y < h {
            emit_block16::<MODE>(out, y, (h - y).min(rows_per), w, sum(y * w));
            y += rows_per;
        }
    }
}

fn qpel_h16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if w < 32 || dst.len() < w * h || !fits16(src.len(), src_stride, h, w, 8) {
        return w16::qpel_h_avx2(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir16::<8, MODE_I16>(&Out16::i16(dst.as_mut_ptr(), w), src.as_ptr(), src_stride, 1, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if w < 32 || dst.len() < w * h || !fits16(src.len(), src_stride, h + 7, w, 0) {
        return w16::qpel_v_avx2(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir16::<8, MODE_I16>(&Out16::i16(dst.as_mut_ptr(), w), src.as_ptr(), src_stride, src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

/// The fused luma kernels, for the cases AVX-512 improves on; `false` hands
/// the call back to AVX2. As at 8 bits, a narrow diagonal block takes the
/// AVX2 first stage and this second one.
#[allow(clippy::too_many_arguments)]
fn fused16<const MODE: u8>(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) -> bool {
    const TAPS: usize = 8;
    let reach = TAPS / 2 - 1;
    let hh = h + TAPS - 1;
    let wide = w >= 32;
    let ok = h >= 1
        && match (fx, fy) {
            (0, 0) => false,
            (_, 0) => wide && src.len() > reach * src_stride && fits16(src.len() - reach * src_stride, src_stride, h, w, TAPS),
            (0, _) => wide && src.len() > reach && fits16(src.len() - reach, src_stride, hh, w, 0),
            _ => w8::fits_i16(super::hevc::MC_TMP_LEN, w, hh) && if wide { fits16(src.len(), src_stride, hh, w, TAPS) } else { w16::fits(src.len(), src_stride, hh, w, TAPS) },
        }
        && (8..=12).contains(&bit_depth)
        && w >= 2
        && (h - 1) * dst_stride + w <= dst.len()
        && (MODE != MODE_BI || other.len() >= w * h)
        && tmp.len() >= super::hevc::MC_TMP_LEN;
    if !ok {
        return false;
    }
    let bd = bit_depth as i32;
    let shift1 = bd.min(12) - 8;
    let (tx, ty) = (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]);
    let out = Out16 {
        i16: std::ptr::null_mut(),
        dst: dst.as_mut_ptr(),
        stride: dst_stride,
        other: other.as_ptr(),
        w,
        shift: if MODE == MODE_UNI { 14 - bd } else { 15 - bd },
        max: (1 << bd) - 1,
    };
    unsafe {
        match (fx, fy) {
            (_, 0) => fir16::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, 1, w, h, tx, shift1),
            (0, _) => fir16::<TAPS, MODE>(&out, src.as_ptr().add(reach), src_stride, src_stride, w, h, ty, shift1),
            _ => {
                let mid = Out16::i16(tmp.as_mut_ptr(), w);
                if wide {
                    fir16::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, 1, w, hh, tx, shift1);
                } else {
                    w16::fir_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, shift1);
                }
                fir_v2_16::<TAPS, MODE>(&out, tmp.as_ptr(), w, h, ty);
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn qpel_uni16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    if !fused16::<MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[], bit_depth) {
        w16::qpel_uni_avx2(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, bit_depth);
    }
}

#[allow(clippy::too_many_arguments)]
fn qpel_bi16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    if !fused16::<MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth) {
        w16::qpel_bi_avx2(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth);
    }
}

/// A 32x32 block's side.
const N: usize = 32;

fn idct32_avx512(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    // The DC shortcut and any short buffer stay with the AVX2 kernel.
    if coeffs.len() < N * N || (max_x == 0 && max_y == 0) {
        return w16::idct_avx2::<32>(coeffs, bd_shift, max_x, max_y);
    }
    unsafe { idct32_impl(coeffs, bd_shift, max_x, max_y) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn idct32_impl(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    unsafe {
        let mut tmp = [0i16; N * N];
        let nzy = max_y + 1;
        let npairs = nzy.div_ceil(2);
        // Stage 1 (columns): tmp[y][x] = clip((sum_j c[j][y] · coef[j][x] + 64) >> 7),
        // vectorised across x, pairs of input rows at a time. `madd` over
        // `unpacklo`/`unpackhi` leaves lane j of `lo` holding outputs
        // 8j..8j+3 and lane j of `hi` holding 8j+4..8j+7, so `packs` — also
        // per 128-bit lane — puts them back in order for free.
        if max_x >= 16 {
            for y in 0..N {
                let mut lo = _mm512_set1_epi32(64);
                let mut hi = lo;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = _mm512_loadu_si512(coeffs.as_ptr().add(j * N) as *const __m512i);
                    let b = if j + 1 < nzy { _mm512_loadu_si512(coeffs.as_ptr().add((j + 1) * N) as *const __m512i) } else { _mm512_setzero_si512() };
                    let c = _mm512_set1_epi32(w16::pair(TRANSFORM32[j][y], TRANSFORM32[j + 1][y]));
                    lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c));
                    hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c));
                }
                let r = _mm512_packs_epi32(_mm512_srai_epi32::<7>(lo), _mm512_srai_epi32::<7>(hi));
                _mm512_storeu_si512(tmp.as_mut_ptr().add(y * N) as *mut __m512i, r);
            }
        } else {
            // Nothing past column 15 to transform. The AVX2 kernel's 16-lane
            // step already skips that half, and transforming zeros twice as
            // wide would cost more than the wider vector saves.
            for y in 0..N {
                let mut lo = _mm256_set1_epi32(64);
                let mut hi = lo;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = w16::load_n(coeffs.as_ptr().add(j * N), N);
                    let b = if j + 1 < nzy { w16::load_n(coeffs.as_ptr().add((j + 1) * N), N) } else { _mm256_setzero_si256() };
                    let c = _mm256_set1_epi32(w16::pair(TRANSFORM32[j][y], TRANSFORM32[j + 1][y]));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c));
                }
                let r = _mm256_packs_epi32(_mm256_srai_epi32::<7>(lo), _mm256_srai_epi32::<7>(hi));
                w16::store_n(tmp.as_mut_ptr().add(y * N), r, 16);
            }
        }
        // Stage 2 (rows): out[y][x] = clip((sum_j c[j][x] · tmp[y][j] + round) >> shift),
        // vectorised across all thirty-two x at once against the matrix's
        // pre-interleaved pair rows. Here `lo` holds columns 0..15 and `hi`
        // columns 16..31, so the pack *does* interleave and one permute
        // undoes it.
        let nzx = max_x + 1;
        let npairs = nzx.div_ceil(2);
        let round2 = _mm512_set1_epi32(1 << (bd_shift - 1));
        let sh = _mm_cvtsi32_si128(bd_shift);
        let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        for y in 0..N {
            let row = tmp.as_ptr().add(y * N);
            let mut lo = round2;
            let mut hi = round2;
            for p in 0..npairs {
                let j = 2 * p;
                let t0 = *row.add(j) as i32;
                let t1 = if j + 1 < nzx { *row.add(j + 1) as i32 } else { 0 };
                let tv = _mm512_set1_epi32((t0 as u16 as i32) | ((t1 as u16 as i32) << 16));
                let pr = w16::pair_row(N, p);
                lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_loadu_si512(pr.as_ptr() as *const __m512i), tv));
                hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_loadu_si512(pr.as_ptr().add(32) as *const __m512i), tv));
            }
            let r = _mm512_packs_epi32(_mm512_sra_epi32(lo, sh), _mm512_sra_epi32(hi, sh));
            _mm512_storeu_si512(coeffs.as_mut_ptr().add(y * N) as *mut __m512i, _mm512_permutexvar_epi64(idx, r));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hevc::MC_TMP_LEN;

    /// The 16-bit table as the decoder builds it on this CPU, AVX-512 over
    /// AVX2, or `None` (announced) without the extensions.
    fn table16() -> Option<HevcDsp<u16>> {
        super::super::hevc_avx512_u8::tests::avx512()?;
        let mut d = HevcDsp::<u16>::SCALAR;
        w16::install(&mut d);
        let before = d;
        install(&mut d);
        assert!(d.qpel_uni as usize != before.qpel_uni as usize, "install left the AVX2 fused kernels in place");
        Some(d)
    }

    /// Every shape a luma prediction unit can have, and a few more.
    const SHAPES: &[(usize, usize)] = &[(4, 8), (8, 4), (8, 8), (8, 16), (12, 16), (16, 4), (16, 16), (16, 64), (24, 32), (32, 8), (32, 32), (48, 64), (64, 16), (64, 64)];

    #[test]
    fn interpolation16_matches_scalar() {
        let Some(d) = table16() else { return };
        let s = HevcDsp::<u16>::SCALAR;
        let mut seed = 0x5116_u64;
        let stride = 160;
        let mut t1 = vec![0i16; MC_TMP_LEN];
        let mut t2 = vec![0i16; MC_TMP_LEN];
        let ds = 96;
        for bd in [9u32, 10, 12] {
            let max = (1u32 << bd) - 1;
            for rails in [false, true] {
                let plane: Vec<u16> = (0..stride * 160)
                    .map(|_| if rails { if lcg(&mut seed).is_multiple_of(2) { 0 } else { max as u16 } } else { (lcg(&mut seed) % (max + 1)) as u16 })
                    .collect();
                let other: Vec<i16> = (0..64 * 64).map(|_| (lcg(&mut seed) % 24000) as i16 - 2500).collect();
                let shift1 = bd.min(12) as i32 - 8;
                for &(w, h) in SHAPES {
                    for frac in 1..4 {
                        let mut a = vec![0i16; w * h];
                        let mut b = vec![0i16; w * h];
                        (s.qpel_h)(&mut a, &plane, stride, w, h, frac, shift1);
                        (d.qpel_h)(&mut b, &plane, stride, w, h, frac, shift1);
                        assert_eq!(a, b, "qpel_h {w}x{h} frac {frac} {bd} bits");
                        (s.qpel_v)(&mut a, &plane, stride, w, h, frac, shift1);
                        (d.qpel_v)(&mut b, &plane, stride, w, h, frac, shift1);
                        assert_eq!(a, b, "qpel_v {w}x{h} frac {frac} {bd} bits");
                    }
                    let mut a = vec![0u16; ds * h];
                    let mut b = vec![0u16; ds * h];
                    for fy in 0..4 {
                        for fx in 0..4 {
                            (s.qpel_uni)(&mut a, ds, &plane, stride, w, h, fx, fy, &mut t1, bd);
                            (d.qpel_uni)(&mut b, ds, &plane, stride, w, h, fx, fy, &mut t2, bd);
                            assert_eq!(a, b, "qpel_uni {w}x{h} {fx},{fy} {bd} bits rails {rails}");
                            (s.qpel_bi)(&mut a, ds, &plane, stride, w, h, fx, fy, &mut t1, &other, bd);
                            (d.qpel_bi)(&mut b, ds, &plane, stride, w, h, fx, fy, &mut t2, &other, bd);
                            assert_eq!(a, b, "qpel_bi {w}x{h} {fx},{fy} {bd} bits rails {rails}");
                        }
                    }
                }
            }
        }
        // The second stage over 14-bit rows, which the u16 table shares with
        // the u8 one: the whole i16 range an interpolation can produce.
        let mid: Vec<i16> = (0..MC_TMP_LEN).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
        for &(w, h) in SHAPES {
            for frac in 1..8 {
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                if frac < 4 {
                    (s.qpel_v2)(&mut a, &mid, w, w, h, frac);
                    (d.qpel_v2)(&mut b, &mid, w, w, h, frac);
                    assert_eq!(a, b, "qpel_v2 {w}x{h} frac {frac}");
                }
                (s.epel_v2)(&mut a, &mid, w, w, h, frac);
                (d.epel_v2)(&mut b, &mid, w, w, h, frac);
                assert_eq!(a, b, "epel_v2 {w}x{h} frac {frac}");
            }
        }
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn idct32_matches_scalar() {
        // Same skip-is-not-coverage rule as the interpolation kernels.
        if super::super::hevc_avx512_u8::tests::avx512().is_none() {
            return;
        }
        let s = HevcDsp::<u16>::SCALAR;
        let mut seed = 0x1d_c7_u64;
        for trial in 0..300 {
            // The standard clips coefficients to 16 bits (8.6.2); the
            // nonzero bounds sweep both stage-1 branches and the tail where
            // the last coefficient pair is odd.
            let (max_x, max_y) = match trial % 6 {
                0 => (31, 31),
                1 => (15, 31),
                2 => (31, 15),
                3 => (0, 7),
                4 => (16, 1),
                _ => ((lcg(&mut seed) % 32) as usize, (lcg(&mut seed) % 32) as usize),
            };
            let range = if trial % 3 == 0 { 32767 } else { 900 };
            let mut a = [0i16; N * N];
            for y in 0..=max_y {
                for x in 0..=max_x {
                    a[y * N + x] = ((lcg(&mut seed) % (2 * range + 1)) as i32 - range as i32) as i16;
                }
            }
            let mut b = a;
            let bd_shift = 20 - [8, 10, 12][trial % 3];
            (s.idct[3])(&mut a, bd_shift, max_x, max_y);
            idct32_avx512(&mut b, bd_shift, max_x, max_y);
            assert_eq!(a, b, "idct32 trial {trial} max {max_x},{max_y} shift {bd_shift}");
        }
    }
}
