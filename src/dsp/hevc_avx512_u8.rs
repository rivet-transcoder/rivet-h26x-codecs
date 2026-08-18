//! AVX-512 versions of the H.265 interpolation kernels for 8-bit sample
//! planes (x86-64), installed *over* the AVX2 table so every kernel and
//! every block shape not carried here keeps its AVX2 version.
//!
//! The bar for being here is a *measured* win, and it is higher than
//! "wider is better". Most of these kernels are a small enough share of
//! decode time that even a good speed-up on the kernel is a fraction of a
//! percent overall — inside the run-to-run scatter of the machine they
//! were measured on — so what shipped is the handful with the largest
//! speed-ups on the shapes that are actually common, and anything that
//! measured neutral or worse inside a decode was left on AVX2. Those are
//! listed at the bottom; they are most of what was tried.
//!
//! What does beat it, and why. The AVX2 kernels already run `pmaddubsw`
//! over interleaved neighbour pairs, so doubling the vector only pays where
//! the *operands* are twice as wide too — otherwise the extra lanes cost a
//! load and a shuffle to fill, and the multiply was never the bottleneck.
//! Three cases qualify:
//!
//! * **The second (vertical, 14-bit) stage of a diagonal block**
//!   (`fir_v2` below). Its input is the horizontal stage's own output — a
//!   contiguous `w`-strided buffer — so a 512-bit load is 32 lanes of real
//!   work at any block width: 32 columns of one row for a wide block, and
//!   *several whole rows* for a narrow one, where the AVX2 kernel runs an
//!   eighth-full 128-bit vector per row. Three of every four luma
//!   fractional positions are diagonal, so this is the common path.
//! * **The first stage of a block 32 samples wide or more** (`fir_h` /
//!   `fir_v`), where 64 source bytes are 64 real outputs.
//! * **The 32-point inverse transform**, in [`super::hevc_avx512`].
//!
//! What did not, measured per kernel inside a real decode:
//!
//! * **Chroma interpolation** as a whole. A chroma block is 2 to 16 samples
//!   wide, so its first stage is all marshalling: four loads and three
//!   inserts per tap pair to fill a vector whose multiply was already free.
//!   Every chroma entry point retired *more* cycles than AVX2 when counted
//!   inside a real decode, so they keep it — all but `epel_v2`, which is
//!   the contiguous 14-bit stage above and wins for the same reason its
//!   luma twin does.
//! * **A first stage narrower than 32**, for the same reason, which is why
//!   a narrow diagonal block takes the AVX2 horizontal pass and the
//!   AVX-512 vertical one.
//! * **Deblocking, SAO and residual add**, which are together under a
//!   fiftieth of decode time — there is nothing there to win.
//!
//! Every kernel is checked bit-exact against the scalar reference in the
//! tests below.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::hevc::HevcDsp;
use super::hevc_avx2_u8 as w2;
use super::hevc_avx2_u8::{MODE_BI, MODE_I16, MODE_UNI, Out};
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS};

/// Replace the AVX2 entries of `d` that AVX-512 improves on. Called after
/// [`super::hevc_avx2_u8::install`], never instead of it.
pub fn install(d: &mut HevcDsp<u8>) {
    d.idct[3] = super::hevc_avx512::IDCT32;
    d.qpel_h = qpel_h_avx512;
    d.qpel_v = qpel_v_avx512;
    d.qpel_v2 = qpel_v2_avx512;
    d.epel_v2 = epel_v2_avx512;
    d.qpel_uni = qpel_uni_avx512;
    d.qpel_bi = qpel_bi_avx512;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// The low `n` (≤ 32) lanes.
#[inline(always)]
fn mask32(n: usize) -> __mmask32 {
    (((1u64 << n) - 1) as u32) as __mmask32
}

/// Thirty-two 16-bit lanes to thirty-two bytes, saturating to `0..=255` —
/// the clip the standard asks for. `packus` works within 128-bit lanes and
/// duplicates each half, so the wanted bytes are the even qwords.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn pack32(v: __m512i) -> __m256i {
    unsafe {
        let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        _mm512_castsi512_si256(_mm512_permutexvar_epi64(idx, _mm512_packus_epi16(v, v)))
    }
}

/// Store the first `n` (≤ 32) 16-bit lanes.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn store_i16(dst: *mut i16, v: __m512i, n: usize) {
    unsafe {
        if n >= 32 {
            _mm512_storeu_si512(dst as *mut __m512i, v);
        } else {
            _mm512_mask_storeu_epi16(dst as *mut i16, mask32(n), v);
        }
    }
}

/// Load the first `n` (≤ 32) 16-bit lanes, zero elsewhere.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn load_i16(src: *const i16, n: usize) -> __m512i {
    unsafe {
        if n >= 32 {
            _mm512_loadu_si512(src as *const __m512i)
        } else {
            _mm512_maskz_loadu_epi16(mask32(n), src)
        }
    }
}

/// Emit `n` (≤ 32) lanes of a stage's 14-bit output at (`row`, `x`).
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn emit_span<const MODE: u8>(out: &Out, row: usize, x: usize, v: __m512i, n: usize) {
    unsafe {
        match MODE {
            MODE_I16 => store_i16(out.i16.add(row * out.w + x), v, n),
            MODE_UNI => {
                let r = _mm512_srai_epi16::<6>(_mm512_adds_epi16(v, _mm512_set1_epi16(32)));
                _mm256_mask_storeu_epi8(out.u8.add(row * out.stride + x) as *mut i8, mask32(n), pack32(r));
            }
            _ => {
                // Saturating sums, exact after the clip (see the AVX2 kernel).
                let o = load_i16(out.other.add(row * out.w + x), n);
                let r = _mm512_srai_epi16::<7>(_mm512_adds_epi16(_mm512_adds_epi16(v, o), _mm512_set1_epi16(64)));
                _mm256_mask_storeu_epi8(out.u8.add(row * out.stride + x) as *mut i8, mask32(n), pack32(r));
            }
        }
    }
}

/// Emit a vector holding `rows` whole rows of a `w`-wide block (`w · rows`
/// ≤ 32 lanes), the first of them row `y0`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn emit_block<const MODE: u8>(out: &Out, y0: usize, rows: usize, w: usize, v: __m512i) {
    unsafe {
        let n = rows * w;
        match MODE {
            MODE_I16 => store_i16(out.i16.add(y0 * w), v, n),
            MODE_UNI => {
                let r = _mm512_srai_epi16::<6>(_mm512_adds_epi16(v, _mm512_set1_epi16(32)));
                scatter_rows(out.u8.add(y0 * out.stride), out.stride, w, pack32(r), rows);
            }
            _ => {
                let o = load_i16(out.other.add(y0 * w), n);
                let r = _mm512_srai_epi16::<7>(_mm512_adds_epi16(_mm512_adds_epi16(v, o), _mm512_set1_epi16(64)));
                scatter_rows(out.u8.add(y0 * out.stride), out.stride, w, pack32(r), rows);
            }
        }
    }
}

/// Store 32 packed bytes as `rows` rows of `w` (≤ 32) bytes, `stride` apart.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn scatter_rows(dst: *mut u8, stride: usize, w: usize, p: __m256i, rows: usize) {
    unsafe {
        match w {
            32 => _mm256_storeu_si256(dst as *mut __m256i, p),
            16 => {
                _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(p));
                if rows > 1 {
                    _mm_storeu_si128(dst.add(stride) as *mut __m128i, _mm256_extracti128_si256::<1>(p));
                }
            }
            8 => {
                // Straight out of the register: a spill and four narrow
                // reloads would stall on store-to-load forwarding, and an
                // 8-wide block is the commonest narrow one.
                let lo = _mm256_castsi256_si128(p);
                let hi = _mm256_extracti128_si256::<1>(p);
                _mm_storel_epi64(dst as *mut __m128i, lo);
                if rows > 1 {
                    _mm_storel_epi64(dst.add(stride) as *mut __m128i, _mm_unpackhi_epi64(lo, lo));
                }
                if rows > 2 {
                    _mm_storel_epi64(dst.add(2 * stride) as *mut __m128i, hi);
                }
                if rows > 3 {
                    _mm_storel_epi64(dst.add(3 * stride) as *mut __m128i, _mm_unpackhi_epi64(hi, hi));
                }
            }
            _ => {
                let mut t = [0u32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, p);
                if w == 4 {
                    for r in 0..rows {
                        std::ptr::write_unaligned(dst.add(r * stride) as *mut u32, t[r]);
                    }
                } else {
                    let b = t.as_ptr() as *const u8;
                    for r in 0..rows {
                        std::ptr::copy_nonoverlapping(b.add(r * w), dst.add(r * stride), w);
                    }
                }
            }
        }
    }
}

/// Whether a `w`-wide, `rows`-row window of a byte plane of `stride` can be
/// read 64 bytes at a time, `extra` samples past the last output column.
#[inline(always)]
fn fits_bytes(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    (rows - 1) * stride + (w - 1) / 64 * 64 + extra + 64 <= len
}

/// Whether the contiguous `w`-strided 14-bit rows can be read 32 lanes at a
/// time for `rows` rows. A block narrower than 32 reads whole rows at once,
/// so the last load reaches past the last row — into the buffer, never past
/// its end, which is what this checks.
#[inline(always)]
fn fits_i16(len: usize, w: usize, rows: usize) -> bool {
    let last_x = if w >= 32 { (w - 1) / 32 * 32 } else { 0 };
    (rows - 1) * w + last_x + 32 <= len
}

// ----------------------------------------------------------------------
// FIR stages
// ----------------------------------------------------------------------

/// Horizontal `TAPS`-tap FIR over bytes, 64 output samples at a time. Only
/// for `w >= 32`: below that the loads to fill the vector cost more than the
/// multiplies they save.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn fir_h<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm512_setzero_si512(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm512_set1_epi16(w2::pair8(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let s = src.add(y * src_stride);
            let mut x = 0;
            while x < w {
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for k in 0..TAPS / 2 {
                    let a = _mm512_loadu_si512(s.add(x + 2 * k) as *const __m512i);
                    let b = _mm512_loadu_si512(s.add(x + 2 * k + 1) as *const __m512i);
                    lo = _mm512_add_epi16(lo, _mm512_maddubs_epi16(_mm512_unpacklo_epi8(a, b), c[k]));
                    hi = _mm512_add_epi16(hi, _mm512_maddubs_epi16(_mm512_unpackhi_epi8(a, b), c[k]));
                }
                emit64::<MODE>(out, y, x, _mm512_sra_epi16(lo, sh), _mm512_sra_epi16(hi, sh), w - x);
                x += 64;
            }
        }
    }
}

/// Vertical `TAPS`-tap FIR over byte rows, 64 output samples at a time.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn fir_v<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm512_setzero_si512(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm512_set1_epi16(w2::pair8(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for k in 0..TAPS / 2 {
                    let a = _mm512_loadu_si512(src.add((y + 2 * k) * src_stride + x) as *const __m512i);
                    let b = _mm512_loadu_si512(src.add((y + 2 * k + 1) * src_stride + x) as *const __m512i);
                    lo = _mm512_add_epi16(lo, _mm512_maddubs_epi16(_mm512_unpacklo_epi8(a, b), c[k]));
                    hi = _mm512_add_epi16(hi, _mm512_maddubs_epi16(_mm512_unpackhi_epi8(a, b), c[k]));
                }
                emit64::<MODE>(out, y, x, _mm512_sra_epi16(lo, sh), _mm512_sra_epi16(hi, sh), w - x);
                x += 64;
            }
        }
    }
}

/// Emit the 64 outputs of one byte-stage step. `pmaddubsw` over
/// `unpacklo`/`unpackhi` leaves them interleaved by 128-bit lane — `lo`
/// holds outputs 16j..16j+7 in lane j, `hi` holds 16j+8..16j+15 — so one
/// two-source permute per half puts them back in order.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn emit64<const MODE: u8>(out: &Out, y: usize, x: usize, lo: __m512i, hi: __m512i, n: usize) {
    unsafe {
        let idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        emit_span::<MODE>(out, y, x, _mm512_permutex2var_epi64(lo, idx0, hi), n.min(32));
        if n > 32 {
            let idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
            emit_span::<MODE>(out, y, x + 32, _mm512_permutex2var_epi64(lo, idx1, hi), (n - 32).min(32));
        }
    }
}

/// Vertical `TAPS`-tap FIR over the horizontal stage's 14-bit rows —
/// `pmaddwd` on interleaved row pairs, 32-bit sums, `>> 6`.
///
/// The rows are contiguous (stride `w`), which is what makes this the best
/// AVX-512 case in the file: at `w < 32` one 512-bit load *is* `32 / w`
/// whole rows, and because the filter's row pairs stay adjacent under that
/// reinterpretation, the same two loads that serve output row `y` serve the
/// next `32 / w - 1` rows as well. The AVX2 kernel runs one row per vector,
/// eight lanes of sixteen used at `w = 8`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn fir_v2<const TAPS: usize, const MODE: u8>(out: &Out, src: *const i16, w: usize, h: usize, taps: &[i8]) {
    unsafe {
        let mut c = [_mm512_setzero_si512(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm512_set1_epi32(w2::pair16(taps[2 * k], taps[2 * k + 1]));
        }
        // `packs` is per 128-bit lane, and lane j of `lo` holds outputs
        // 8j..8j+3 with lane j of `hi` holding 8j+4..8j+7, so it lands them
        // back in order without a permute.
        let step = |a: __m512i, b: __m512i, lo: &mut __m512i, hi: &mut __m512i, k: usize| {
            *lo = _mm512_add_epi32(*lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c[k]));
            *hi = _mm512_add_epi32(*hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c[k]));
        };
        let pack = |lo: __m512i, hi: __m512i| _mm512_packs_epi32(_mm512_srai_epi32::<6>(lo), _mm512_srai_epi32::<6>(hi));
        if w >= 32 {
            for y in 0..h {
                let mut x = 0;
                while x < w {
                    let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                    for k in 0..TAPS / 2 {
                        let a = _mm512_loadu_si512(src.add((y + 2 * k) * w + x) as *const __m512i);
                        let b = _mm512_loadu_si512(src.add((y + 2 * k + 1) * w + x) as *const __m512i);
                        step(a, b, &mut lo, &mut hi, k);
                    }
                    emit_span::<MODE>(out, y, x, pack(lo, hi), (w - x).min(32));
                    x += 32;
                }
            }
            return;
        }
        let rows_per = 32 / w;
        let mut y = 0;
        while y < h {
            let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for k in 0..TAPS / 2 {
                let a = _mm512_loadu_si512(src.add((y + 2 * k) * w) as *const __m512i);
                let b = _mm512_loadu_si512(src.add((y + 2 * k + 1) * w) as *const __m512i);
                step(a, b, &mut lo, &mut hi, k);
            }
            emit_block::<MODE>(out, y, (h - y).min(rows_per), w, pack(lo, hi));
            y += rows_per;
        }
    }
}

// ----------------------------------------------------------------------
// The two-pass entry points
// ----------------------------------------------------------------------

/// The 14-bit output of a standalone stage.
fn out_i16(dst: &mut [i16], w: usize) -> Out {
    Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w }
}

fn qpel_h_avx512(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if w < 32 || dst.len() < w * h || !fits_bytes(src.len(), src_stride, h, w, 7) {
        return w2::qpel_h_avx2(dst, src, src_stride, w, h, frac, shift);
    }
    let out = out_i16(dst, w);
    unsafe { fir_h::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v_avx512(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if w < 32 || dst.len() < w * h || !fits_bytes(src.len(), src_stride, h + 7, w, 0) {
        return w2::qpel_v_avx2(dst, src, src_stride, w, h, frac, shift);
    }
    let out = out_i16(dst, w);
    unsafe { fir_v::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v2_avx512(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if src_stride != w || w < 2 || dst.len() < w * h || !fits_i16(src.len(), w, h + 7) {
        return super::hevc_avx2::qpel_v2_avx2(dst, src, src_stride, w, h, frac);
    }
    let out = out_i16(dst, w);
    unsafe { fir_v2::<8, MODE_I16>(&out, src.as_ptr(), w, h, &QPEL_FILTERS[frac][..8]) }
}

fn epel_v2_avx512(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if src_stride != w || w < 2 || dst.len() < w * h || !fits_i16(src.len(), w, h + 3) {
        return super::hevc_avx2::epel_v2_avx2(dst, src, src_stride, w, h, frac);
    }
    let out = out_i16(dst, w);
    unsafe { fir_v2::<4, MODE_I16>(&out, src.as_ptr(), w, h, &EPEL_FILTERS[frac]) }
}

// ----------------------------------------------------------------------
// Fused interpolation + prediction
// ----------------------------------------------------------------------

/// The fused kernels, for the cases AVX-512 improves on. A whole-sample
/// block, and the first stage of a block narrower than 32, are handed back
/// to AVX2 — including, in the diagonal case, *only* the first stage, so a
/// narrow diagonal block still gets the AVX-512 second stage.
///
/// Generic over the tap count, but only the 8-tap (luma) instantiation is
/// installed: see the module docs for what the 4-tap one measured at.
#[allow(clippy::too_many_arguments)]
fn fused<const TAPS: usize, const MODE: u8>(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16]) -> bool {
    let reach = TAPS / 2 - 1;
    let hh = h + TAPS - 1;
    let wide = w >= 32;
    // The shape test first: the commonest hand-back is a narrow block at a
    // pure-horizontal or pure-vertical position, and that costs one compare.
    let ok = h >= 1
        && match (fx, fy) {
            (0, 0) => false,
            (_, 0) => wide && src.len() > reach * src_stride && fits_bytes(src.len() - reach * src_stride, src_stride, h, w, TAPS - 1),
            (0, _) => wide && src.len() > reach && fits_bytes(src.len() - reach, src_stride, hh, w, 0),
            // The second stage always qualifies; the first needs whichever
            // of the two kernels will run it to be in bounds.
            _ => {
                fits_i16(super::hevc::MC_TMP_LEN, w, hh)
                    && if wide { fits_bytes(src.len(), src_stride, hh, w, TAPS - 1) } else { w2::fits(src.len(), src_stride, hh, w, TAPS - 1) }
            }
        }
        && w >= 2
        && (h - 1) * dst_stride + w <= dst.len()
        && (MODE != MODE_BI || other.len() >= w * h)
        && tmp.len() >= super::hevc::MC_TMP_LEN;
    if !ok {
        return false;
    }
    let (tx, ty): (&[i8], &[i8]) = if TAPS == 8 { (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]) } else { (&EPEL_FILTERS[fx], &EPEL_FILTERS[fy]) };
    let out = Out { i16: std::ptr::null_mut(), u8: dst.as_mut_ptr(), stride: dst_stride, other: other.as_ptr(), w };
    unsafe {
        match (fx, fy) {
            (_, 0) => fir_h::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, w, h, tx, 0),
            (0, _) => fir_v::<TAPS, MODE>(&out, src.as_ptr().add(reach), src_stride, w, h, ty, 0),
            _ => {
                let mid = Out { i16: tmp.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
                if wide {
                    fir_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, 0);
                } else {
                    w2::fir_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, 0);
                }
                fir_v2::<TAPS, MODE>(&out, tmp.as_ptr(), w, h, ty);
            }
        }
    }
    true
}

fn qpel_uni_avx512(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    if !fused::<8, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[]) {
        w2::qpel_uni_avx2(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, bit_depth);
    }
}

#[allow(clippy::too_many_arguments)]
fn qpel_bi_avx512(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    if !fused::<8, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other) {
        w2::qpel_bi_avx2(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::dsp::hevc::MC_TMP_LEN;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// The AVX-512 table, or `None` without the extensions — installed the
    /// way the decoder installs it, over AVX2. A skip is announced, because a
    /// green run on a core without AVX-512 is not coverage; set
    /// `H26X_REQUIRE_AVX512=1` to turn it into a failure.
    pub(in crate::dsp) fn avx512() -> Option<HevcDsp<u8>> {
        if !crate::dsp::Cpu::detect().avx512 {
            let required = std::env::var_os("H26X_REQUIRE_AVX512").is_some_and(|v| v == "1" || v == "true");
            assert!(!required, "H26X_REQUIRE_AVX512 is set but this CPU has no AVX-512 F + BW + VL");
            eprintln!("skipping: no AVX-512 on this CPU, the 512-bit kernels are not covered");
            return None;
        }
        let mut d = HevcDsp::<u8>::SCALAR;
        w2::install(&mut d);
        let before = d;
        install(&mut d);
        // A tier that installed nothing would let every comparison below pass
        // without running a single 512-bit instruction.
        assert!(d.qpel_v2 as usize != before.qpel_v2 as usize, "install left the AVX2 kernels in place");
        assert!(d.epel_h as usize == before.epel_h as usize, "chroma interpolation stays on AVX2");
        Some(d)
    }

    /// Every block shape an HEVC prediction unit can have, luma and chroma.
    const SHAPES: &[(usize, usize)] = &[
        (2, 4),
        (2, 8),
        (4, 4),
        (4, 8),
        (4, 16),
        (6, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (12, 16),
        (16, 4),
        (16, 12),
        (16, 16),
        (24, 32),
        (32, 8),
        (32, 32),
        (48, 64),
        (64, 16),
        (64, 64),
    ];

    #[test]
    fn two_pass_matches_scalar() {
        let Some(d) = avx512() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 0x5150_u64;
        let stride = 96;
        let plane: Vec<u8> = (0..stride * 96).map(|_| lcg(&mut seed) as u8).collect();
        let mid: Vec<i16> = (0..MC_TMP_LEN).map(|_| (lcg(&mut seed) % 32768) as i16 - 16384).collect();
        for &(w, h) in SHAPES {
            if h > 72 {
                continue;
            }
            for frac in 1..4 {
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                (s.qpel_h)(&mut a, &plane, stride, w, h, frac, 0);
                (d.qpel_h)(&mut b, &plane, stride, w, h, frac, 0);
                assert_eq!(a, b, "qpel_h {w}x{h} frac {frac}");
                (s.qpel_v)(&mut a, &plane, stride, w, h, frac, 0);
                (d.qpel_v)(&mut b, &plane, stride, w, h, frac, 0);
                assert_eq!(a, b, "qpel_v {w}x{h} frac {frac}");
                (s.qpel_v2)(&mut a, &mid, w, w, h, frac);
                (d.qpel_v2)(&mut b, &mid, w, w, h, frac);
                assert_eq!(a, b, "qpel_v2 {w}x{h} frac {frac}");
            }
            for frac in 1..8 {
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                (s.epel_v2)(&mut a, &mid, w, w, h, frac);
                (d.epel_v2)(&mut b, &mid, w, w, h, frac);
                assert_eq!(a, b, "epel_v2 {w}x{h} frac {frac}");
            }
        }
    }

    #[test]
    fn fused_matches_scalar() {
        let Some(d) = avx512() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 0xf00d_u64;
        let stride = 160;
        let plane: Vec<u8> = (0..stride * 160).map(|_| lcg(&mut seed) as u8).collect();
        let other: Vec<i16> = (0..64 * 64).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
        let mut t1 = vec![0i16; MC_TMP_LEN];
        let mut t2 = vec![0i16; MC_TMP_LEN];
        let ds = 96;
        for &(w, h) in SHAPES {
            let mut a = vec![0u8; ds * h];
            let mut b = vec![0u8; ds * h];
            for fy in 0..4 {
                for fx in 0..4 {
                    (s.qpel_uni)(&mut a, ds, &plane, stride, w, h, fx, fy, &mut t1, 8);
                    (d.qpel_uni)(&mut b, ds, &plane, stride, w, h, fx, fy, &mut t2, 8);
                    assert_eq!(a, b, "qpel_uni {w}x{h} {fx},{fy}");
                    (s.qpel_bi)(&mut a, ds, &plane, stride, w, h, fx, fy, &mut t1, &other, 8);
                    (d.qpel_bi)(&mut b, ds, &plane, stride, w, h, fx, fy, &mut t2, &other, 8);
                    assert_eq!(a, b, "qpel_bi {w}x{h} {fx},{fy}");
                }
            }
        }
    }
}
