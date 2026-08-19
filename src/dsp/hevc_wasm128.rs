//! 128-bit SIMD versions of the H.265 kernels for WebAssembly, 8-bit sample
//! planes.
//!
//! The porting doctrine is [`super::h264_wasm128`]'s, and its module header
//! is the one to read first. This file adds the HEVC half, and because anyone
//! diffing it against [`super::hevc_x86_128`] will find the two do not
//! correspond kernel for kernel, here is why, at full length — none of it is
//! rot.
//!
//! **This is the i16-domain port, not the u8-domain one.** The x86 file's
//! 8-bit interpolation path from SSSE3 up is built on `pmaddubsw` — an
//! unsigned-by-signed pairwise multiply-add that wasm `simd128` does not
//! have. Its faithful emulation is seven instructions on the hottest loop
//! (widen both halves, two `i32x4_dot_i16x8`, one saturating narrow); that
//! emulation is exact — it was measured and checked bit-exact when the H.264
//! tier was built — and it was rejected there for the same reason it is
//! rejected here: seven-for-one on the innermost loop is not a trade worth
//! making. The u8 path exists *solely to exploit `pmaddubsw`*; the same
//! filtering runs in the i16 domain on `pmaddwd`, which wasm has natively as
//! `i32x4_dot_i16x8`. So these kernels take u8 samples, widen once per row
//! to i16, filter in the i16 domain, and narrow on the way out.
//!
//! Concretely, each kernel's shape comes from the file that already computes
//! it in that domain:
//!
//! - The first-stage byte FIRs (qpel_h/v, epel_h/v) follow the **SSE2
//!   instantiation** of `hevc_x86_128.rs`: one broadcast tap per
//!   coefficient, multiply widened samples with `i16x8_mul` (`pmullw`), sum
//!   in 16-bit lanes. That is exact for 8-bit input: a single product is at
//!   most 255 · 58 = 14790, and no running sum of the HEVC luma or chroma
//!   taps leaves i16 (the largest positive prefix is 255 · 88 = 22440 for
//!   the half-sample filter). The SSE2 rung is the x86 file's own proof that
//!   the filter does not need `pmaddubsw` — it is the shape that computes
//!   the same results without it.
//! - The second-stage vertical filters over 14-bit intermediates (qpel_v2 /
//!   epel_v2), the inverse DCT and the weighted combiners follow the
//!   **`pmaddwd` shapes** shared by `hevc_x86_128.rs` and
//!   [`super::hevc_avx2`] (the 16-bit-sample file): interleave neighbour
//!   pairs, `i32x4_dot_i16x8`, 32-bit sums, saturating narrow. Those stages
//!   genuinely need 32-bit accumulation, and wasm has the exact instruction.
//! - SAO follows `hevc_x86_128.rs`'s byte-domain shapes, and is *simpler*
//!   here: wasm has unsigned byte compares (`u8x16_gt`), so the SSE2
//!   `max/cmpeq/andnot` dance for `sign(v − a)` collapses to two compares;
//!   and `u8x16_swizzle` is the SSSE3 `pshufb` offset lookup (its
//!   zero-on-out-of-range semantics never trigger — edgeIdx is 0..=4).
//!
//! One structural improvement over the x86-128 shapes: the second stage's
//! 14-bit input rows are stored contiguously (stride = width), so when they
//! are, `qpel_v2` / `epel_v2` run one flat loop over all `w · h` outputs —
//! output `i` is `Σ tapₖ · src[i + k·w]` for *every* flat index — with full
//! lane occupancy at every block width instead of a mostly idle vector per
//! narrow row. That is the same structural reason the contiguous v2 was the
//! biggest win of the AVX-512 work.
//!
//! Where wasm is better than the file this came from, it is taken: shift
//! counts are plain integers (no `_mm_cvtsi32_si128` dance), and at 128 bits
//! the narrows already land their lanes in order, so the `permute4x64`
//! fix-ups of the 256-bit kernels have nothing to undo.
//!
//! What stays on the scalar reference, deliberately:
//!
//! - **Deblocking** — out of scope for this pass, exactly as the H.264 tier
//!   left its deblocking scalar; it follows.
//! - **The fused MC entries** (`qpel_uni` … `epel_bi`; `fused_mc` stays
//!   `false`) — the decoder's two-pass path runs on the kernels here, so
//!   nothing is lost but the fused kernels' second-pass saving. A follow-up.
//! - **`idst4`** and the 4x4 IDCT — the scalar butterfly is 4 lines
//!   (`hevc_x86_128.rs` reaches the same verdict for the 4x4).
//! - **The `u16` table** — wasm decode is 8-bit today; the 10/12-bit
//!   kernels stay scalar until there is a stream to justify them.
//!
//! There is one rung here, not four: `simd128` is a compile-time target
//! feature, so this module is compiled only when it holds, and
//! `H26X_NO_SIMD=1` still selects the scalar reference through
//! [`super::Cpu::detect_honouring_env`].
//!
//! Verification: `wasm32-unknown-unknown` has no test harness, so the
//! bit-exactness sweep that would be a `#[cfg(test)]` module in the x86
//! files lives where it can run — `examples/wasm_probe.rs` exports
//! `h26x_hevc_dsp_check`, a randomized comparison of every entry of this
//! table against the scalar reference over all the block shapes the
//! dispatch serves, driven inside wasm by `tools/wasm_dsp_check.mjs`; and
//! `tools/wasm.sh` decodes the vendored streams and the fixture corpus at
//! both rungs.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::hevc::HevcDsp;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS, TRANSFORM32};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut HevcDsp<u8>) {
    d.idct = [idct::<4>, idct::<8>, idct::<16>, idct::<32>];
    d.add_residual = add_residual;
    d.qpel_copy = copy;
    d.qpel_h = qpel_h;
    d.qpel_v = qpel_v;
    d.qpel_v2 = qpel_v2;
    d.epel_copy = copy;
    d.epel_h = epel_h;
    d.epel_v = epel_v;
    d.epel_v2 = epel_v2;
    d.uni = uni;
    d.bi = bi;
    d.weighted_uni = weighted_uni;
    d.weighted_bi = weighted_bi;
    d.sao_band = sao_band;
    d.sao_edge = sao_edge;
    // idst4, the fused MC entries and the deblocking filters keep the scalar
    // reference — see the module header for why each one does.
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// A pair of i8 taps `(a, b)` broadcast as 32-bit lanes `a | b << 16`, for
/// `i32x4_dot_i16x8`.
#[inline]
const fn pair(a: i8, b: i8) -> i32 {
    (a as i16 as u16 as i32) | ((b as i16 as u16 as i32) << 16)
}

/// A pair of i16 values `(a, b)` as one 32-bit lane.
#[inline]
const fn pair16(a: i16, b: i16) -> i32 {
    (a as u16 as i32) | ((b as u16 as i32) << 16)
}

/// Eight bytes at `p` as eight i16.
#[inline]
unsafe fn load8(p: *const u8) -> v128 {
    unsafe { u16x8_extend_low_u8x16(v128_load64_zero(p as *const u64)) }
}

/// `unpacklo_epi16`: the low four lanes of each source, interleaved.
#[inline]
fn zip_lo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}

/// `unpackhi_epi16`.
#[inline]
fn zip_hi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}

/// Store the first `n` (≤ 8) i16 lanes of `v`.
#[inline]
unsafe fn store_i16_n(dst: *mut i16, v: v128, n: usize) {
    unsafe {
        match n {
            8 => v128_store(dst as *mut v128, v),
            4 => v128_store64_lane::<0>(v, dst as *mut u64),
            _ => {
                let mut t = [0i16; 8];
                v128_store(t.as_mut_ptr() as *mut v128, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Load 8 i16 lanes, or the first `avail` zero-padded.
#[inline]
unsafe fn load_i16_n(src: *const i16, avail: usize) -> v128 {
    unsafe {
        if avail >= 8 {
            v128_load(src as *const v128)
        } else if avail == 4 {
            v128_load64_zero(src as *const u64)
        } else {
            let mut t = [0i16; 8];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            v128_load(t.as_ptr() as *const v128)
        }
    }
}

/// Store the first `n` (≤ 8) bytes of `v`.
#[inline]
unsafe fn store_bytes(dst: *mut u8, v: v128, n: usize) {
    unsafe {
        match n {
            8 => v128_store64_lane::<0>(v, dst as *mut u64),
            4 => std::ptr::write_unaligned(dst as *mut u32, u32x4_extract_lane::<0>(v)),
            2 => std::ptr::write_unaligned(dst as *mut u16, u16x8_extract_lane::<0>(v)),
            _ => {
                let mut t = [0u8; 16];
                v128_store(t.as_mut_ptr() as *mut v128, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Store the first `n` (≤ 16) bytes of `v` (byte-lane kernels: SAO).
#[inline]
unsafe fn store_bytes16(dst: *mut u8, v: v128, n: usize) {
    unsafe {
        if n == 16 {
            v128_store(dst as *mut v128, v);
        } else {
            store_bytes(dst, v, n);
        }
    }
}

/// Load 16 bytes, or the first `avail` zero-padded.
#[inline]
unsafe fn load_bytes16(src: *const u8, avail: usize) -> v128 {
    unsafe {
        if avail >= 16 {
            v128_load(src as *const v128)
        } else if avail == 8 {
            v128_load64_zero(src as *const u64)
        } else {
            let mut t = [0u8; 16];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            v128_load(t.as_ptr() as *const v128)
        }
    }
}

/// 8 i16 lanes to 8 bytes, saturating to `0..=255` — `packus`'s saturation
/// is exactly the clip the standard asks for at 8-bit depth.
#[inline]
fn pack8(v: v128) -> v128 {
    u8x16_narrow_i16x8(v, v)
}

/// Whether a block of width `w` is handled as one contiguous run of samples
/// (the predictions are stored with stride `w`, so a 2- or 4-wide block is
/// 4 or 2 rows per 8-lane vector instead of a mostly idle vector per row).
#[inline]
fn narrow(w: usize) -> bool {
    w == 4 || w == 2
}

/// Store 8 bytes of `p` as `rows` rows of `w` (2 or 4) bytes.
#[inline]
unsafe fn scatter_rows(dst: *mut u8, stride: usize, w: usize, p: v128, rows: usize) {
    unsafe {
        if w == 4 {
            let mut t = [0u32; 4];
            v128_store(t.as_mut_ptr() as *mut v128, p);
            for r in 0..rows.min(2) {
                std::ptr::write_unaligned(dst.add(r * stride) as *mut u32, t[r]);
            }
        } else {
            let mut t = [0u16; 8];
            v128_store(t.as_mut_ptr() as *mut v128, p);
            for r in 0..rows.min(4) {
                std::ptr::write_unaligned(dst.add(r * stride) as *mut u16, t[r]);
            }
        }
    }
}

/// Whether reading `w` bytes into a row of `stride`, for `rows` rows, plus
/// `extra` bytes along, stays inside `len` for the vector width the byte
/// kernels use at that block width.
#[inline]
fn fits_b(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    let (vec, last_x) = if w <= 8 { (8, 0) } else { (16, (w - 1) / 16 * 16) };
    (rows - 1) * stride + last_x + extra + vec <= len
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

/// The taps of a byte FIR, one i16 broadcast per tap, and how many there
/// are — the SSE2 shape (see the module header).
struct Taps([v128; 8], usize);

fn taps_load(taps: &[i8], n: usize) -> Taps {
    let mut v = [i16x8_splat(0); 8];
    for k in 0..n {
        v[k] = i16x8_splat(taps[k] as i16);
    }
    Taps(v, n)
}

/// Eight consecutive FIR outputs from the u8 window at `p`, stepping `step`
/// bytes per tap, as eight i16.
///
/// `i16x8_mul` keeps the low 16 bits, which is exact here: a single product
/// is at most 255 · 58 = 14790 and no running sum of the HEVC luma or
/// chroma taps leaves i16 for 8-bit input.
#[inline]
unsafe fn fir8(p: *const u8, step: usize, t: &Taps) -> v128 {
    unsafe {
        let mut acc = i16x8_splat(0);
        for k in 0..t.1 {
            acc = i16x8_add(acc, i16x8_mul(load8(p.add(k * step)), t.0[k]));
        }
        acc
    }
}

/// Sixteen consecutive FIR outputs, as (low eight, high eight) i16.
#[inline]
unsafe fn fir16(p: *const u8, step: usize, t: &Taps) -> (v128, v128) {
    unsafe {
        let mut lo = i16x8_splat(0);
        let mut hi = i16x8_splat(0);
        for k in 0..t.1 {
            let v = v128_load(p.add(k * step) as *const v128);
            lo = i16x8_add(lo, i16x8_mul(u16x8_extend_low_u8x16(v), t.0[k]));
            hi = i16x8_add(hi, i16x8_mul(u16x8_extend_high_u8x16(v), t.0[k]));
        }
        (lo, hi)
    }
}

/// Horizontal FIR with `TAPS` taps over bytes, into 14-bit predictions of
/// stride `w`.
#[inline]
unsafe fn fir_h<const TAPS: usize>(dst: *mut i16, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = taps_load(taps, TAPS);
        let sh = shift as u32;
        if w <= 8 {
            // Narrow blocks: 8-byte loads, one vector per row.
            for y in 0..h {
                let acc = fir8(src.add(y * src_stride), 1, &t);
                store_i16_n(dst.add(y * w), i16x8_shr(acc, sh), w);
            }
            return;
        }
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let (lo, hi) = fir16(s.add(x), 1, &t);
                let n = w - x;
                store_i16_n(d.add(x), i16x8_shr(lo, sh), n.min(8));
                if n > 8 {
                    store_i16_n(d.add(x + 8), i16x8_shr(hi, sh), (n - 8).min(8));
                }
                x += 16;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over byte rows.
#[inline]
unsafe fn fir_v<const TAPS: usize>(dst: *mut i16, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = taps_load(taps, TAPS);
        let sh = shift as u32;
        let row = |r: usize| src.add(r * src_stride);
        if w <= 8 {
            for y in 0..h {
                let acc = fir8(row(y), src_stride, &t);
                store_i16_n(dst.add(y * w), i16x8_shr(acc, sh), w);
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let (lo, hi) = fir16(row(y).add(x), src_stride, &t);
                let n = w - x;
                store_i16_n(dst.add(y * w + x), i16x8_shr(lo, sh), n.min(8));
                if n > 8 {
                    store_i16_n(dst.add(y * w + x + 8), i16x8_shr(hi, sh), (n - 8).min(8));
                }
                x += 16;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over 14-bit rows (the second stage of hv):
/// `i32x4_dot_i16x8` on interleaved row pairs, 32-bit sums, `>> 6`, with
/// the same saturating narrow every other SIMD tier uses.
///
/// When the rows are contiguous (`src_stride == w`, which is how the
/// two-pass path stores them) this runs one flat loop over all `w · h`
/// outputs: output `i` is `Σ tapₖ · src[i + k·w]` for every flat index, so
/// every lane is busy at every block width. See the module header.
#[inline]
unsafe fn fir_v2<const TAPS: usize>(dst: *mut i16, src: *const i16, src_stride: usize, w: usize, h: usize, taps: &[i8]) {
    unsafe {
        let mut c = [i32x4_splat(0); 4];
        for k in 0..TAPS / 2 {
            c[k] = i32x4_splat(pair(taps[2 * k], taps[2 * k + 1]));
        }
        let dot8 = |at: *const i16, step: usize| -> v128 {
            let mut lo = i32x4_splat(0);
            let mut hi = i32x4_splat(0);
            for k in 0..TAPS / 2 {
                let a = v128_load(at.add(2 * k * step) as *const v128);
                let b = v128_load(at.add((2 * k + 1) * step) as *const v128);
                lo = i32x4_add(lo, i32x4_dot_i16x8(zip_lo16(a, b), c[k]));
                hi = i32x4_add(hi, i32x4_dot_i16x8(zip_hi16(a, b), c[k]));
            }
            i16x8_narrow_i32x4(i32x4_shr(lo, 6), i32x4_shr(hi, 6))
        };
        if src_stride == w {
            let total = w * h;
            let mut i = 0;
            while i < total {
                store_i16_n(dst.add(i), dot8(src.add(i), w), (total - i).min(8));
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                store_i16_n(dst.add(y * w + x), dot8(src.add(y * src_stride + x), src_stride), (w - x).min(8));
                x += 8;
            }
        }
    }
}

/// Whether the second stage's 14-bit rows can be read 8 lanes at a time
/// within `len`, for the loop shape [`fir_v2`] picks at that stride.
#[inline]
fn fits_v2(len: usize, stride: usize, w: usize, h: usize, taps: usize) -> bool {
    if stride == w {
        (w * h - 1) / 8 * 8 + (taps - 1) * w + 8 <= len
    } else {
        let last_x = if w <= 8 { 0 } else { (w - 1) / 8 * 8 };
        (h + taps - 2) * stride + last_x + 8 <= len
    }
}

fn copy(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
    // 8-byte loads at every 8-sample step of each row.
    if (h - 1) * src_stride + (w - 1) / 8 * 8 + 8 > src.len() || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe {
        let sh = shift as u32;
        for y in 0..h {
            let s = src.as_ptr().add(y * src_stride);
            let d = dst.as_mut_ptr().add(y * w);
            let mut x = 0;
            while x < w {
                store_i16_n(d.add(x), i16x8_shl(load8(s.add(x)), sh), (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn qpel_h(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h, w, 7) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h + 7, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits_v2(src.len(), src_stride, w, h, 8) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v2::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8]) }
}

fn epel_h(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h, w, 3) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h + 3, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits_v2(src.len(), src_stride, w, h, 4) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v2::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac]) }
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

fn uni(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    debug_assert_eq!(max, 255);
    let _ = max;
    unsafe {
        let round = i16x8_splat(if shift > 0 { 1 << (shift - 1) } else { 0 });
        let sh = shift.max(0) as u32;
        // 14-bit + round fits i16 (< 16384 + 8192); the saturating add is
        // belt and braces for out-of-contract inputs, exact after the clip.
        let scale = |s: v128| pack8(i16x8_shr(i16x8_add_sat(s, round), sh));
        if narrow(w) {
            let total = w * h;
            let mut i = 0;
            while i < total {
                let n = (total - i).min(8);
                let s = load_i16_n(src.as_ptr().add(i), total - i);
                scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, scale(s), n / w);
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_i16_n(src.as_ptr().add(y * w + x), w - x);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), scale(s), n);
                x += 8;
            }
        }
    }
}

fn bi(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    debug_assert_eq!(max, 255);
    let _ = max;
    unsafe {
        let round = i16x8_splat(1 << (shift - 1));
        let sh = shift as u32;
        // Saturating sums: a + b can exceed i16 only when both are far above
        // the 8-bit range, and then the clip to 255 gives the same answer as
        // the exact 32-bit sum.
        let comb = |va: v128, vb: v128| pack8(i16x8_shr(i16x8_add_sat(i16x8_add_sat(va, vb), round), sh));
        if narrow(w) {
            let total = w * h;
            let mut i = 0;
            while i < total {
                let n = (total - i).min(8);
                let va = load_i16_n(a.as_ptr().add(i), total - i);
                let vb = load_i16_n(b.as_ptr().add(i), total - i);
                scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, comb(va, vb), n / w);
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_i16_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_i16_n(b.as_ptr().add(y * w + x), w - x);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), comb(va, vb), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    debug_assert_eq!(max, 255);
    if i16::try_from(wt).is_err() {
        return (HevcDsp::<u8>::SCALAR.weighted_uni)(dst, stride, src, w, h, log2_wd, wt, o, max);
    }
    unsafe {
        let round = i32x4_splat(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let sh = log2_wd.max(0) as u32;
        let wv = i16x8_splat(wt as i16);
        let ov = i32x4_splat(o);
        // `extmul` is the widening multiply the x86 shape builds from
        // unpack + madd — one instruction here.
        let weigh = |s: v128| -> v128 {
            let lo = i32x4_add(i32x4_shr(i32x4_add(i32x4_extmul_low_i16x8(s, wv), round), sh), ov);
            let hi = i32x4_add(i32x4_shr(i32x4_add(i32x4_extmul_high_i16x8(s, wv), round), sh), ov);
            pack8(i16x8_narrow_i32x4(lo, hi))
        };
        if narrow(w) {
            let total = w * h;
            let mut i = 0;
            while i < total {
                let n = (total - i).min(8);
                let s = load_i16_n(src.as_ptr().add(i), total - i);
                scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, weigh(s), n / w);
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_i16_n(src.as_ptr().add(y * w + x), w - x);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), weigh(s), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    debug_assert_eq!(max, 255);
    if i16::try_from(w0).is_err() || i16::try_from(w1).is_err() {
        return (HevcDsp::<u8>::SCALAR.weighted_bi)(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1, max);
    }
    unsafe {
        let round = i32x4_splat((o0 + o1 + 1) << log2_wd);
        let sh = (log2_wd + 1) as u32;
        let wv = i32x4_splat(pair16(w0 as i16, w1 as i16));
        // `a · w0 + b · w1` over the two predictions interleaved is exactly
        // one `i32x4_dot_i16x8` per four outputs.
        let weigh = |va: v128, vb: v128| -> v128 {
            let quad = |v: v128| i32x4_shr(i32x4_add(i32x4_dot_i16x8(v, wv), round), sh);
            pack8(i16x8_narrow_i32x4(quad(zip_lo16(va, vb)), quad(zip_hi16(va, vb))))
        };
        if narrow(w) {
            let total = w * h;
            let mut i = 0;
            while i < total {
                let n = (total - i).min(8);
                let va = load_i16_n(a.as_ptr().add(i), total - i);
                let vb = load_i16_n(b.as_ptr().add(i), total - i);
                scatter_rows(dst.as_mut_ptr().add((i / w) * stride), stride, w, weigh(va, vb), n / w);
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_i16_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_i16_n(b.as_ptr().add(y * w + x), w - x);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), weigh(va, vb), n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add
// ----------------------------------------------------------------------

fn add_residual(dst: &mut [u8], stride: usize, res: &[i16], n: usize, max: i32) {
    debug_assert_eq!(max, 255);
    let _ = max;
    unsafe {
        if n == 4 {
            // 4x4: two rows per vector.
            let d = dst.as_mut_ptr();
            for y in (0..4).step_by(2) {
                let rd = |k: usize| std::ptr::read_unaligned(d.add(k * stride) as *const u32);
                let p = u16x8_extend_low_u8x16(u32x4(rd(y), rd(y + 1), 0, 0));
                let r = v128_load(res.as_ptr().add(y * 4) as *const v128);
                // Sample + residual can leave i16 only by saturating, and
                // wherever it saturates the clip to 0..=255 agrees with the
                // exact sum.
                let v = pack8(i16x8_add_sat(p, r));
                std::ptr::write_unaligned(d.add(y * stride) as *mut u32, u32x4_extract_lane::<0>(v));
                std::ptr::write_unaligned(d.add((y + 1) * stride) as *mut u32, u32x4_extract_lane::<1>(v));
            }
            return;
        }
        for y in 0..n {
            let mut x = 0;
            while x < n {
                let d = dst.as_mut_ptr().add(y * stride + x);
                let p = load8(d);
                let r = v128_load(res.as_ptr().add(y * n + x) as *const v128);
                v128_store64_lane::<0>(pack8(i16x8_add_sat(p, r)), d as *mut u64);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Inverse DCT
// ----------------------------------------------------------------------

/// The transform matrix rows for size `n` as interleaved pairs of rows
/// `(j, j+1)`: `[c[j][0], c[j+1][0], c[j][1], c[j+1][1], ...]` (n lanes × 2).
///
/// Its own copy, like [`super::hevc_avx2`] keeps one — it is data, not code,
/// and the x86 module that owns the original is not compiled here.
struct PairRows {
    rows32: [[i16; 64]; 16],
    rows16: [[i16; 32]; 8],
    rows8: [[i16; 16]; 4],
    rows4: [[i16; 8]; 2],
}

const fn build_pairs() -> PairRows {
    let mut p = PairRows { rows32: [[0; 64]; 16], rows16: [[0; 32]; 8], rows8: [[0; 16]; 4], rows4: [[0; 8]; 2] };
    let mut j = 0;
    while j < 16 {
        let mut k = 0;
        while k < 32 {
            p.rows32[j][2 * k] = TRANSFORM32[2 * j][k] as i16;
            p.rows32[j][2 * k + 1] = TRANSFORM32[2 * j + 1][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 8 {
        let mut k = 0;
        while k < 16 {
            p.rows16[j][2 * k] = TRANSFORM32[4 * j][k] as i16;
            p.rows16[j][2 * k + 1] = TRANSFORM32[4 * j + 2][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 4 {
        let mut k = 0;
        while k < 8 {
            p.rows8[j][2 * k] = TRANSFORM32[8 * j][k] as i16;
            p.rows8[j][2 * k + 1] = TRANSFORM32[8 * j + 4][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 2 {
        let mut k = 0;
        while k < 4 {
            p.rows4[j][2 * k] = TRANSFORM32[16 * j][k] as i16;
            p.rows4[j][2 * k + 1] = TRANSFORM32[16 * j + 8][k] as i16;
            k += 1;
        }
        j += 1;
    }
    p
}

static PAIRS: PairRows = build_pairs();

#[inline]
fn pair_row(n: usize, j: usize) -> &'static [i16] {
    match n {
        32 => &PAIRS.rows32[j],
        16 => &PAIRS.rows16[j],
        8 => &PAIRS.rows8[j],
        _ => &PAIRS.rows4[j],
    }
}

fn idct<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    if max_x == 0 && max_y == 0 {
        // DC only.
        let round2 = 1i32 << (bd_shift - 1);
        let v = ((coeffs[0] as i32 * 64 + 64) >> 7).clamp(-32768, 32767);
        let r = ((v * 64 + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        coeffs[..N * N].fill(r);
        return;
    }
    if N == 4 {
        // Not worth a vector: the scalar butterfly is 4 lines.
        return (HevcDsp::<u8>::SCALAR.idct[0])(coeffs, bd_shift, max_x, max_y);
    }
    unsafe { idct_impl::<N>(coeffs, bd_shift, max_x, max_y) }
}

unsafe fn idct_impl<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    unsafe {
        let mut tmp = [0i16; 32 * 32];
        // Stage 1 (columns): tmp[y][x] = clip((sum_j c[j][y] * coef[j][x] + 64) >> 7),
        // vectorised across x for each y; pairs of input rows (j, j+1).
        let nzy = max_y + 1;
        let npairs = nzy.div_ceil(2);
        let round1 = i32x4_splat(64);
        let step = 32 / N;
        for y in 0..N {
            let mut x = 0;
            while x <= max_x {
                let mut lo = round1;
                let mut hi = round1;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = load_i16_n(coeffs.as_ptr().add(j * N + x), N - x);
                    let b = if j + 1 < nzy { load_i16_n(coeffs.as_ptr().add((j + 1) * N + x), N - x) } else { i16x8_splat(0) };
                    let c = i32x4_splat(pair(TRANSFORM32[j * step][y], TRANSFORM32[(j + 1) * step][y]));
                    lo = i32x4_add(lo, i32x4_dot_i16x8(zip_lo16(a, b), c));
                    hi = i32x4_add(hi, i32x4_dot_i16x8(zip_hi16(a, b), c));
                }
                let r = i16x8_narrow_i32x4(i32x4_shr(lo, 7), i32x4_shr(hi, 7));
                store_i16_n(tmp.as_mut_ptr().add(y * N + x), r, (N - x).min(8));
                x += 8;
            }
        }
        // Stage 2 (rows): out[y][x] = clip((sum_j c[j][x] * tmp[y][j] + round) >> shift),
        // vectorised across x with the interleaved pair rows of the matrix.
        let nzx = max_x + 1;
        let npairs = nzx.div_ceil(2);
        let round2 = i32x4_splat(1 << (bd_shift - 1));
        let sh = bd_shift as u32;
        for y in 0..N {
            let row = tmp.as_ptr().add(y * N);
            let mut x = 0;
            while x < N {
                let mut lo = round2;
                let mut hi = round2;
                for p in 0..npairs {
                    let j = 2 * p;
                    let t0 = *row.add(j) as i32;
                    let t1 = if j + 1 < nzx { *row.add(j + 1) as i32 } else { 0 };
                    let tv = i32x4_splat((t0 as u16 as i32) | ((t1 as u16 as i32) << 16));
                    let pr = pair_row(N, p);
                    let cl = v128_load(pr.as_ptr().add(2 * x) as *const v128); // pairs for x..x+4
                    let ch = if N - x > 4 { v128_load(pr.as_ptr().add(2 * x + 8) as *const v128) } else { i16x8_splat(0) };
                    lo = i32x4_add(lo, i32x4_dot_i16x8(cl, tv));
                    hi = i32x4_add(hi, i32x4_dot_i16x8(ch, tv));
                }
                // At 128 bits the narrow keeps outputs x..x+7 in order.
                let r = i16x8_narrow_i32x4(i32x4_shr(lo, sh), i32x4_shr(hi, sh));
                store_i16_n(coeffs.as_mut_ptr().add(y * N + x), r, (N - x).min(8));
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

/// `v + off` on bytes, clipped to `0..=255`, with `off` in `-128..=127`.
#[inline]
fn add_offset_u8(v: v128, off: v128) -> v128 {
    let zero = i8x16_splat(0);
    let pos = i8x16_max(off, zero);
    let neg = i8x16_max(i8x16_sub(zero, off), zero);
    u8x16_sub_sat(u8x16_add_sat(v, pos), neg)
}

#[allow(clippy::too_many_arguments)]
fn sao_band(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    if shift != 3 || table.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_band)(dst, dst_stride, src, src_stride, w, h, table, shift, max);
    }
    debug_assert_eq!(max, 255);
    unsafe {
        // The four consecutive bands (mod 32) with nonzero offsets.
        let mut bands = [0u8; 4];
        let mut offs = [0i8; 4];
        let mut k = 0;
        for b in 0..32 {
            if table[b] != 0 && k < 4 {
                bands[k] = (b as u8) << shift;
                offs[k] = table[b] as i8;
                k += 1;
            }
        }
        let mask = u8x16_splat((0xFFu32 << shift) as u8);
        let bv: [v128; 4] = std::array::from_fn(|i| u8x16_splat(bands[i]));
        let ov: [v128; 4] = std::array::from_fn(|i| i8x16_splat(offs[i]));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let v = load_bytes16(src.as_ptr().add(y * src_stride + x), n);
                let band = v128_and(v, mask);
                let mut off = i8x16_splat(0);
                for i in 0..k {
                    off = v128_bitselect(ov[i], off, i8x16_eq(band, bv[i]));
                }
                store_bytes16(dst.as_mut_ptr().add(y * dst_stride + x), add_offset_u8(v, off), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    if off.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_edge)(dst, src, origin, stride, w, h, na, nb, off, max);
    }
    debug_assert_eq!(max, 255);
    unsafe {
        // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 indexes the offsets:
        // `u8x16_swizzle` is the lookup (indices stay below 16, so its
        // zero-on-out-of-range semantics never trigger). And wasm has the
        // unsigned byte compare x86-SSE2 lacks, so sign() is two compares.
        let o = |i: usize| off[i] as u8;
        let tab = u8x16(o(0), o(1), o(2), o(3), o(4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let two = i8x16_splat(2);
        let lo_reach = na.min(nb).min(0);
        let hi_reach = na.max(nb).max(0);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let i = origin + y * stride + x;
                if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 16 > src.len() || i + 16 > dst.len() {
                    // Tail near the buffer end: scalar.
                    for xx in x..w {
                        let ii = origin + y * stride + xx;
                        let v = src[ii] as i32;
                        let a = src[(ii as isize + na) as usize] as i32;
                        let b = src[(ii as isize + nb) as usize] as i32;
                        let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                        dst[ii] = (v + off[e] as i32).clamp(0, 255) as u8;
                    }
                    break;
                }
                let v = v128_load(src.as_ptr().add(i) as *const v128);
                let a = v128_load(src.as_ptr().offset(i as isize + na) as *const v128);
                let b = v128_load(src.as_ptr().offset(i as isize + nb) as *const v128);
                let gt_a = u8x16_gt(v, a);
                let lt_a = u8x16_gt(a, v);
                let gt_b = u8x16_gt(v, b);
                let lt_b = u8x16_gt(b, v);
                // e = 2 + gt - lt with masks of -1: 2 - gt + lt.
                let e = i8x16_add(i8x16_sub(i8x16_sub(two, gt_a), gt_b), i8x16_add(lt_a, lt_b));
                let o = u8x16_swizzle(tab, e);
                store_bytes16(dst.as_mut_ptr().add(i), add_offset_u8(v, o), n);
                x += 16;
            }
        }
    }
}
