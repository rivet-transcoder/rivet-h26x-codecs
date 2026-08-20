//! 128-bit SIMD versions of the H.265 kernels for WebAssembly.
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
//! - The deblocking filters follow `hevc_x86_128.rs`'s shared i32-lane
//!   shape with byte loads and stores: four lines of an edge are four i32
//!   lanes per sample position, one luma segment per filter call, two
//!   chroma segments per vector; the arithmetic is sample-size independent
//!   and ports instruction for instruction.
//! - The fused MC entries (`fused_mc = true`) follow `hevc_x86_128.rs`'s
//!   `Out`/`emit` driver: the same FIR loops serve the two-pass and the
//!   fused kernels, with the output stage — 14-bit store, default uni, or
//!   default bi — selected per instantiation, so the fused path writes
//!   samples straight out of the filter instead of taking a second pass
//!   over a 14-bit prediction.
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
//! The 16-bit-sample table (10/12-bit decode) is served too, at the bottom
//! of the file, and both widths live in one module for `hevc_x86_128.rs`'s
//! reason: the deblocking filters, the inverse transform and the 14-bit
//! second stage are sample-size independent, and `install_u16` installs
//! the very same fn items for them. The u16 kernels mirror that file's
//! `kernels_u16!` widening choices — see the section comment. One
//! extension past x86 parity: **fused MC for u16**. No x86 or NEON rung
//! installs it (their u8 fused paths bake the bit-8 shifts in and the
//! 16-bit tables run two-pass); here the output-stage constants travel in
//! `Out16` at runtime, so 10/12-bit decode gets the same
//! straight-out-of-the-filter path the 8-bit tier got.
//!
//! What stays on the scalar reference, deliberately: **`idst4`** and the
//! 4x4 IDCT — the scalar butterfly is 4 lines (`hevc_x86_128.rs` reaches
//! the same verdict for the 4x4).
//!
//! There is one rung here, not four: `simd128` is a compile-time target
//! feature, so this module is compiled only when it holds, and
//! `H26X_NO_SIMD=1` still selects the scalar reference through
//! [`super::Cpu::detect_honouring_env`].
//!
//! Verification: `wasm32-unknown-unknown` has no test harness, so the
//! bit-exactness sweep that would be a `#[cfg(test)]` module in the x86
//! files lives where it can run — `examples/wasm_probe.rs` exports
//! `h26x_hevc_dsp_check`, a randomized comparison of every entry of both
//! tables — the u16 one at bit depths 10 and 12 — against the scalar
//! reference over all the block shapes the dispatch serves, driven inside
//! wasm by `tools/wasm_dsp_check.mjs`; and `tools/wasm.sh` decodes the
//! vendored streams and the fixture corpus (which carries 10- and 12-bit
//! streams) at both rungs.

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
    d.qpel_uni = qpel_uni;
    d.epel_uni = epel_uni;
    d.qpel_bi = qpel_bi;
    d.epel_bi = epel_bi;
    d.fused_mc = true;
    d.sao_band = sao_band;
    d.sao_edge = sao_edge;
    d.deblock_luma_v = deblock_luma_v;
    d.deblock_luma_h = deblock_luma_h;
    d.deblock_chroma_v = deblock_chroma_v;
    d.deblock_chroma_h = deblock_chroma_h;
    // idst4 keeps the scalar reference — see the module header for why.
}

/// Replace the scalar entries of `d` with the simd128 kernels (16-bit
/// sample planes, 10/12-bit decode).
///
/// The inverse transform and the 14-bit second stage are sample-size
/// independent: the entries installed for them here are the very same fn
/// items the 8-bit table gets.
pub fn install_u16(d: &mut HevcDsp<u16>) {
    d.idct = [idct::<4>, idct::<8>, idct::<16>, idct::<32>];
    d.add_residual = add_residual16;
    d.qpel_copy = copy16;
    d.qpel_h = qpel_h16;
    d.qpel_v = qpel_v16;
    d.qpel_v2 = qpel_v2;
    d.epel_copy = copy16;
    d.epel_h = epel_h16;
    d.epel_v = epel_v16;
    d.epel_v2 = epel_v2;
    d.uni = uni16;
    d.bi = bi16;
    d.weighted_uni = weighted_uni16;
    d.weighted_bi = weighted_bi16;
    d.qpel_uni = qpel_uni16;
    d.epel_uni = epel_uni16;
    d.qpel_bi = qpel_bi16;
    d.epel_bi = epel_bi16;
    d.fused_mc = true;
    d.sao_band = sao_band16;
    d.sao_edge = sao_edge16;
    d.deblock_luma_v = deblock_luma_v16;
    d.deblock_luma_h = deblock_luma_h16;
    d.deblock_chroma_v = deblock_chroma_v16;
    d.deblock_chroma_h = deblock_chroma_h16;
    // idst4 keeps the scalar reference — see the module header for why.
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

/// What a FIR stage produces, per output kind (`MODE_*`) — the shape the
/// x86-128 fused path uses, so the same FIR loops serve the two-pass and
/// the fused kernels.
#[derive(Clone, Copy)]
struct Out {
    /// `MODE_I16`: 14-bit predictions, stride `w`.
    i16: *mut i16,
    /// `MODE_UNI` / `MODE_BI`: samples, stride `stride`.
    u8: *mut u8,
    /// Sample stride.
    stride: usize,
    /// `MODE_BI`: the other list's 14-bit prediction, stride `w`.
    other: *const i16,
    /// Block width (the stride of `i16` and `other`).
    w: usize,
}

/// 14-bit predictions (the two-pass path and the first stage of hv).
const MODE_I16: u8 = 0;
/// Default-weighted uni-prediction samples: `(v + 32) >> 6`.
const MODE_UNI: u8 = 1;
/// Default-weighted bi-prediction samples: `(v + other + 64) >> 7`.
const MODE_BI: u8 = 2;

/// Emit 8 lanes of a stage's output (`v`, 14-bit) at (`row`, `x`), the
/// first `n` lanes.
#[inline]
unsafe fn emit<const MODE: u8>(out: &Out, row: usize, x: usize, v: v128, n: usize) {
    unsafe {
        match MODE {
            MODE_I16 => store_i16_n(out.i16.add(row * out.w + x), v, n),
            MODE_UNI => {
                let r = i16x8_shr(i16x8_add_sat(v, i16x8_splat(32)), 6);
                store_bytes(out.u8.add(row * out.stride + x), pack8(r), n);
            }
            _ => {
                // Saturating sums, exact after the clip (see `bi`).
                let o = load_i16_n(out.other.add(row * out.w + x), n);
                let r = i16x8_shr(i16x8_add_sat(i16x8_add_sat(v, o), i16x8_splat(64)), 7);
                store_bytes(out.u8.add(row * out.stride + x), pack8(r), n);
            }
        }
    }
}

/// An [`Out`] for the two-pass kernels: 14-bit predictions only.
#[inline]
fn i16_out(dst: &mut [i16], w: usize) -> Out {
    Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w }
}

/// Horizontal FIR with `TAPS` taps over bytes.
#[inline]
unsafe fn fir_h<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = taps_load(taps, TAPS);
        let sh = shift as u32;
        if w <= 8 {
            // Narrow blocks: 8-byte loads, one vector per row.
            for y in 0..h {
                let acc = fir8(src.add(y * src_stride), 1, &t);
                emit::<MODE>(out, y, 0, i16x8_shr(acc, sh), w);
            }
            return;
        }
        for y in 0..h {
            let s = src.add(y * src_stride);
            let mut x = 0;
            while x < w {
                let (lo, hi) = fir16(s.add(x), 1, &t);
                let n = w - x;
                emit::<MODE>(out, y, x, i16x8_shr(lo, sh), n.min(8));
                if n > 8 {
                    emit::<MODE>(out, y, x + 8, i16x8_shr(hi, sh), (n - 8).min(8));
                }
                x += 16;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over byte rows.
#[inline]
unsafe fn fir_v<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = taps_load(taps, TAPS);
        let sh = shift as u32;
        let row = |r: usize| src.add(r * src_stride);
        if w <= 8 {
            for y in 0..h {
                let acc = fir8(row(y), src_stride, &t);
                emit::<MODE>(out, y, 0, i16x8_shr(acc, sh), w);
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let (lo, hi) = fir16(row(y).add(x), src_stride, &t);
                let n = w - x;
                emit::<MODE>(out, y, x, i16x8_shr(lo, sh), n.min(8));
                if n > 8 {
                    emit::<MODE>(out, y, x + 8, i16x8_shr(hi, sh), (n - 8).min(8));
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
unsafe fn fir_v2<const TAPS: usize, const MODE: u8>(out: &Out, src: *const i16, src_stride: usize, w: usize, h: usize, taps: &[i8]) {
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
        // The flat loop is for 14-bit output only: a chunk of eight may
        // cross a row boundary, which a contiguous i16 store can absorb but
        // a strided sample store cannot.
        if MODE == MODE_I16 && src_stride == w {
            let total = w * h;
            let mut i = 0;
            while i < total {
                store_i16_n(out.i16.add(i), dot8(src.add(i), w), (total - i).min(8));
                i += 8;
            }
            return;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                emit::<MODE>(out, y, x, dot8(src.add(y * src_stride + x), src_stride), (w - x).min(8));
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
    let out = i16_out(dst, w);
    unsafe { fir_h::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h + 7, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out(dst, w);
    unsafe { fir_v::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits_v2(src.len(), src_stride, w, h, 8) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_v2)(dst, src, src_stride, w, h, frac);
    }
    let out = i16_out(dst, w);
    unsafe { fir_v2::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8]) }
}

fn epel_h(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h, w, 3) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out(dst, w);
    unsafe { fir_h::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits_b(src.len(), src_stride, h + 3, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out(dst, w);
    unsafe { fir_v::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits_v2(src.len(), src_stride, w, h, 4) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_v2)(dst, src, src_stride, w, h, frac);
    }
    let out = i16_out(dst, w);
    unsafe { fir_v2::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac]) }
}

// ----------------------------------------------------------------------
// Fused interpolation + prediction
// ----------------------------------------------------------------------

/// Copy a `w x h` byte block (whole-sample uni-prediction: the prediction
/// is the reference block).
unsafe fn copy_rows_u8(dst: *mut u8, dst_stride: usize, src: *const u8, src_stride: usize, w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * dst_stride);
            let mut x = 0;
            while x < w {
                let n = w - x;
                if n >= 16 {
                    v128_store(d.add(x) as *mut v128, v128_load(s.add(x) as *const v128));
                    x += 16;
                } else if n >= 8 {
                    std::ptr::write_unaligned(d.add(x) as *mut u64, std::ptr::read_unaligned(s.add(x) as *const u64));
                    x += 8;
                } else if n >= 4 {
                    std::ptr::write_unaligned(d.add(x) as *mut u32, std::ptr::read_unaligned(s.add(x) as *const u32));
                    x += 4;
                } else {
                    std::ptr::write_unaligned(d.add(x) as *mut u16, std::ptr::read_unaligned(s.add(x) as *const u16));
                    x += 2;
                }
            }
        }
    }
}

/// Whether the second stage's `w`-stride 14-bit rows can be read 8 lanes
/// at a time for `rows` rows within `len`.
#[inline]
fn fits_i16(len: usize, w: usize, rows: usize) -> bool {
    let last_x = if w <= 8 { 0 } else { (w - 1) / 8 * 8 };
    (rows - 1) * w + last_x + 8 <= len
}

/// The fused kernels: `TAPS` (8 luma / 4 chroma), `MODE_UNI` or `MODE_BI`.
#[allow(clippy::too_many_arguments)]
fn fused<const TAPS: usize, const MODE: u8>(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16]) {
    let reach = TAPS / 2 - 1;
    let at_block = reach * src_stride + reach;
    let hh = h + TAPS - 1;
    let ok = w >= 2
        && h >= 1
        && (h - 1) * dst_stride + w <= dst.len()
        && (MODE != MODE_BI || other.len() >= w * h)
        && tmp.len() >= super::hevc::MC_TMP_LEN
        && match (fx, fy) {
            (0, 0) => (h - 1) * src_stride + w + at_block <= src.len(),
            (_, 0) => src.len() > reach * src_stride && fits_b(src.len() - reach * src_stride, src_stride, h, w, TAPS - 1),
            (0, _) => src.len() > reach && fits_b(src.len() - reach, src_stride, hh, w, 0),
            _ => fits_b(src.len(), src_stride, hh, w, TAPS - 1) && fits_i16(super::hevc::MC_TMP_LEN, w, hh),
        };
    if !ok {
        let s = HevcDsp::<u8>::SCALAR;
        return match (TAPS, MODE) {
            (8, MODE_UNI) => (s.qpel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
            (8, _) => (s.qpel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
            (_, MODE_UNI) => (s.epel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
            _ => (s.epel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
        };
    }
    let (tx, ty): (&[i8], &[i8]) = if TAPS == 8 { (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]) } else { (&EPEL_FILTERS[fx], &EPEL_FILTERS[fy]) };
    let out = Out { i16: std::ptr::null_mut(), u8: dst.as_mut_ptr(), stride: dst_stride, other: other.as_ptr(), w };
    unsafe {
        match (fx, fy) {
            (0, 0) => {
                if MODE == MODE_UNI {
                    copy_rows_u8(dst.as_mut_ptr(), dst_stride, src.as_ptr().add(at_block), src_stride, w, h);
                } else {
                    // Whole-sample bi: widen, then the usual average.
                    let (pred, _) = tmp.split_at_mut(w * h);
                    copy(pred, &src[at_block..], src_stride, w, h, 6);
                    bi(dst, dst_stride, other, pred, w, h, 7, 255);
                }
            }
            (_, 0) => fir_h::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, w, h, tx, 0),
            (0, _) => fir_v::<TAPS, MODE>(&out, src.as_ptr().add(reach), src_stride, w, h, ty, 0),
            _ => {
                let mid = i16_out(tmp, w);
                fir_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, 0);
                fir_v2::<TAPS, MODE>(&out, tmp.as_ptr(), w, w, h, ty);
            }
        }
    }
}

fn qpel_uni(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    let _ = bit_depth;
    fused::<8, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
}

fn epel_uni(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    let _ = bit_depth;
    fused::<4, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
}

#[allow(clippy::too_many_arguments)]
fn qpel_bi(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    let _ = bit_depth;
    fused::<8, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
}

#[allow(clippy::too_many_arguments)]
fn epel_bi(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    let _ = bit_depth;
    fused::<4, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
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

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------
//
// The shared i32-lane filters of `hevc_x86_128.rs` with byte loads and
// stores: four lines of an edge are four i32 lanes per sample position
// (p3..q3), and four lines is exactly one luma segment, so the filter runs
// once per segment and the per-segment lane masks of the 256-bit kernel
// collapse to scalar booleans. Chroma segments are two lines, so a vector
// holds two of them.

/// Eight consecutive bytes as two vectors of 4 x i32 (lines 0..3, 4..7).
#[inline]
unsafe fn ld8_u8_i32(p: *const u8) -> (v128, v128) {
    unsafe {
        let v = u16x8_extend_low_u8x16(v128_load64_zero(p as *const u64));
        (u32x4_extend_low_u16x8(v), u32x4_extend_high_u16x8(v))
    }
}

/// Two vectors of 4 x i32 back to eight u16.
///
/// Signed saturation: every lane the deblocking filters produce is a
/// sample in `0..=max` — the untouched positions are originals, the weak
/// filter clips explicitly, and the strong filter's `Clip3(p ± 2tc, avg)`
/// cannot leave the range because `avg` is in it — so the saturation never
/// fires and the narrow is exact.
#[inline]
fn pack8_i32_u16(lo: v128, hi: v128) -> v128 {
    i16x8_narrow_i32x4(lo, hi)
}

/// Two vectors of 4 x i32 (each within a byte) to eight bytes in the low
/// half.
#[inline]
fn pack8_i32_u8(lo: v128, hi: v128) -> v128 {
    let p = pack8_i32_u16(lo, hi);
    u8x16_narrow_i16x8(p, p)
}

/// `unpacklo_epi32`.
#[inline]
fn zip_lo32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 4, 1, 5>(a, b)
}

/// `unpackhi_epi32`.
#[inline]
fn zip_hi32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<2, 6, 3, 7>(a, b)
}

/// `unpacklo_epi64`.
#[inline]
fn zip_lo64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<0, 2>(a, b)
}

/// `unpackhi_epi64`.
#[inline]
fn zip_hi64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<1, 3>(a, b)
}

/// Transpose eight 8-lane u16 rows.
#[inline]
fn transpose8_u16(r: &mut [v128; 8]) {
    let a0 = zip_lo16(r[0], r[1]);
    let a1 = zip_hi16(r[0], r[1]);
    let a2 = zip_lo16(r[2], r[3]);
    let a3 = zip_hi16(r[2], r[3]);
    let a4 = zip_lo16(r[4], r[5]);
    let a5 = zip_hi16(r[4], r[5]);
    let a6 = zip_lo16(r[6], r[7]);
    let a7 = zip_hi16(r[6], r[7]);
    let b0 = zip_lo32(a0, a2);
    let b1 = zip_hi32(a0, a2);
    let b2 = zip_lo32(a1, a3);
    let b3 = zip_hi32(a1, a3);
    let b4 = zip_lo32(a4, a6);
    let b5 = zip_hi32(a4, a6);
    let b6 = zip_lo32(a5, a7);
    let b7 = zip_hi32(a5, a7);
    r[0] = zip_lo64(b0, b4);
    r[1] = zip_hi64(b0, b4);
    r[2] = zip_lo64(b1, b5);
    r[3] = zip_hi64(b1, b5);
    r[4] = zip_lo64(b2, b6);
    r[5] = zip_hi64(b2, b6);
    r[6] = zip_lo64(b3, b7);
    r[7] = zip_hi64(b3, b7);
}

/// `m ? b : a` per bit.
#[inline]
fn sel(a: v128, b: v128, m: v128) -> v128 {
    v128_bitselect(b, a, m)
}

/// The luma filter on one four-line segment, in place.
fn luma_filter4(v: &mut [v128; 8], beta: i32, tc: i32, no_p: bool, no_q: bool, max: i32) {
    if beta == 0 && tc == 0 {
        return;
    }
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let add = i32x4_add;
    let sub = i32x4_sub;
    let dbl = |a| i32x4_shl(a, 1);
    let absd = |a, b| i32x4_abs(i32x4_sub(a, b));
    // Lane-wise measures.
    let dpv = i32x4_abs(add(sub(p2, dbl(p1)), p0));
    let dqv = i32x4_abs(add(sub(q2, dbl(q1)), q0));
    let ev = add(absd(p3, p0), absd(q0, q3));
    let fv = absd(p0, q0);
    let mut dp = [0i32; 4];
    let mut dq = [0i32; 4];
    let mut e = [0i32; 4];
    let mut f = [0i32; 4];
    unsafe {
        v128_store(dp.as_mut_ptr() as *mut v128, dpv);
        v128_store(dq.as_mut_ptr() as *mut v128, dqv);
        v128_store(e.as_mut_ptr() as *mut v128, ev);
        v128_store(f.as_mut_ptr() as *mut v128, fv);
    }
    // The segment's decisions, from its lines 0 and 3.
    let dpq0 = dp[0] + dq[0];
    let dpq3 = dp[3] + dq[3];
    if dpq0 + dpq3 >= beta {
        return;
    }
    let dsam = |l: usize, dpq: i32| dpq < (beta >> 2) && e[l] < (beta >> 3) && f[l] < ((5 * tc + 1) >> 1);
    let strong = dsam(0, 2 * dpq0) && dsam(3, 2 * dpq3);
    let side = (beta + (beta >> 1)) >> 3;
    let dep = dp[0] + dp[3] < side;
    let deq = dq[0] + dq[3] < side;
    let zero = i32x4_splat(0);
    let m = |b: bool| i32x4_splat(-(b as i32));
    let strong_m = m(strong);
    let dep_m = m(dep);
    let deq_m = m(deq);
    let wp_m = m(!no_p);
    let wq_m = m(!no_q);
    let tcv = i32x4_splat(tc);
    let tc2 = dbl(tcv);
    let tch = i32x4_shr(tcv, 1);
    // 10·tc, 9·x and 3·x as shifts and adds, as the x86-128 file does.
    let tc10 = add(i32x4_shl(tcv, 3), i32x4_shl(tcv, 1));
    let maxv = i32x4_splat(max);
    let clamp = |x, lo, hi| i32x4_min(i32x4_max(x, lo), hi);
    let two = i32x4_splat(2);
    let four = i32x4_splat(4);
    // Strong.
    let p0q0 = add(p0, q0);
    let sp0 = clamp(i32x4_shr(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)), 3), sub(p0, tc2), add(p0, tc2));
    let sp1 = clamp(i32x4_shr(add(add(p2, p1), add(p0q0, two)), 2), sub(p1, tc2), add(p1, tc2));
    let sp2 = clamp(i32x4_shr(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)), 3), sub(p2, tc2), add(p2, tc2));
    let sq0 = clamp(i32x4_shr(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)), 3), sub(q0, tc2), add(q0, tc2));
    let sq1 = clamp(i32x4_shr(add(add(p0q0, q1), add(q2, two)), 2), sub(q1, tc2), add(q1, tc2));
    let sq2 = clamp(i32x4_shr(add(add(p0q0, q1), add(add(q2, dbl(q2)), add(dbl(q3), four))), 3), sub(q2, tc2), add(q2, tc2));
    // Weak.
    let d0 = sub(q0, p0);
    let d1 = sub(q1, p1);
    let d0x9 = add(i32x4_shl(d0, 3), d0);
    let d1x3 = add(i32x4_shl(d1, 1), d1);
    let delta = i32x4_shr(add(sub(d0x9, d1x3), i32x4_splat(8)), 4);
    let w_m = i32x4_gt(tc10, i32x4_abs(delta));
    let delta = clamp(delta, sub(zero, tcv), tcv);
    let wp0 = clamp(add(p0, delta), zero, maxv);
    let wq0 = clamp(sub(q0, delta), zero, maxv);
    let one = i32x4_splat(1);
    let dpv2 = clamp(i32x4_shr(add(sub(i32x4_shr(add(add(p2, p0), one), 1), p1), delta), 1), sub(zero, tch), tch);
    let dqv2 = clamp(i32x4_shr(sub(sub(i32x4_shr(add(add(q2, q0), one), 1), q1), delta), 1), sub(zero, tch), tch);
    let wp1 = clamp(add(p1, dpv2), zero, maxv);
    let wq1 = clamp(add(q1, dqv2), zero, maxv);
    // Combine: strong wins over weak; weak needs its per-line test.
    let np0 = sel(sel(p0, wp0, w_m), sp0, strong_m);
    let nq0 = sel(sel(q0, wq0, w_m), sq0, strong_m);
    let np1 = sel(sel(p1, wp1, v128_and(w_m, dep_m)), sp1, strong_m);
    let nq1 = sel(sel(q1, wq1, v128_and(w_m, deq_m)), sq1, strong_m);
    let np2 = sel(p2, sp2, strong_m);
    let nq2 = sel(q2, sq2, strong_m);
    v[1] = sel(p2, np2, wp_m);
    v[2] = sel(p1, np1, wp_m);
    v[3] = sel(p0, np0, wp_m);
    v[4] = sel(q0, nq0, wq_m);
    v[5] = sel(q1, nq1, wq_m);
    v[6] = sel(q2, nq2, wq_m);
}

/// The chroma filter on four lines (two segments): `[p1, p0, q0, q1]`.
fn chroma_filter4(v: &mut [v128; 4], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    let [p1, p0, q0, q1] = *v;
    let tcv = i32x4(tc[0], tc[0], tc[1], tc[1]);
    let m = |a: [bool; 2]| {
        let x = |b: bool| -(b as i32);
        i32x4(x(a[0]), x(a[0]), x(a[1]), x(a[1]))
    };
    let zero = i32x4_splat(0);
    let on = i32x4_gt(tcv, zero);
    let wp = v128_andnot(on, m(no_p));
    let wq = v128_andnot(on, m(no_q));
    let maxv = i32x4_splat(max);
    let d = i32x4_shr(i32x4_add(i32x4_add(i32x4_shl(i32x4_sub(q0, p0), 2), i32x4_sub(p1, q1)), i32x4_splat(4)), 3);
    let d = i32x4_min(i32x4_max(d, i32x4_sub(zero, tcv)), tcv);
    let np0 = i32x4_min(i32x4_max(i32x4_add(p0, d), zero), maxv);
    let nq0 = i32x4_min(i32x4_max(i32x4_sub(q0, d), zero), maxv);
    v[1] = sel(p0, np0, wp);
    v[2] = sel(q0, nq0, wq);
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_v(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut r = [i32x4_splat(0); 8];
        for (i, v) in r.iter_mut().enumerate() {
            *v = u16x8_extend_low_u8x16(v128_load64_zero(data.add(i * stride).sub(4) as *const u64));
        }
        transpose8_u16(&mut r);
        let mut v0 = [i32x4_splat(0); 8];
        let mut v1 = [i32x4_splat(0); 8];
        for k in 0..8 {
            v0[k] = u32x4_extend_low_u16x8(r[k]);
            v1[k] = u32x4_extend_high_u16x8(r[k]);
        }
        luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
        luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
        for k in 0..8 {
            r[k] = pack8_i32_u16(v0[k], v1[k]);
        }
        transpose8_u16(&mut r);
        for (i, v) in r.iter().enumerate() {
            v128_store64_lane::<0>(u8x16_narrow_i16x8(*v, *v), data.add(i * stride).sub(4) as *mut u64);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_h(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut v0 = [i32x4_splat(0); 8];
        let mut v1 = [i32x4_splat(0); 8];
        for k in 0..8 {
            let (a, b) = ld8_u8_i32(data.offset((k as isize - 4) * stride as isize));
            v0[k] = a;
            v1[k] = b;
        }
        luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
        luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
        for k in 1..7 {
            v128_store64_lane::<0>(pack8_i32_u8(v0[k], v1[k]), data.offset((k as isize - 4) * stride as isize) as *mut u64);
        }
    }
}

fn deblock_chroma_v(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut r = [i32x4_splat(0); 8];
        for (i, v) in r.iter_mut().enumerate() {
            let q = std::ptr::read_unaligned(data.add(i * stride).sub(2) as *const u32);
            *v = u16x8_extend_low_u8x16(u32x4(q, 0, 0, 0));
        }
        let a0 = zip_lo16(r[0], r[1]);
        let a1 = zip_lo16(r[2], r[3]);
        let a2 = zip_lo16(r[4], r[5]);
        let a3 = zip_lo16(r[6], r[7]);
        let b0 = zip_lo32(a0, a1); // p1 r0..3 | p0 r0..3
        let b1 = zip_hi32(a0, a1); // q0 r0..3 | q1 r0..3
        let b2 = zip_lo32(a2, a3); // rows 4..7
        let b3 = zip_hi32(a2, a3);
        let mut v0 = [
            u32x4_extend_low_u16x8(b0),
            u32x4_extend_high_u16x8(b0),
            u32x4_extend_low_u16x8(b1),
            u32x4_extend_high_u16x8(b1),
        ];
        let mut v1 = [
            u32x4_extend_low_u16x8(b2),
            u32x4_extend_high_u16x8(b2),
            u32x4_extend_low_u16x8(b3),
            u32x4_extend_high_u16x8(b3),
        ];
        chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
        chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
        // (p0, q0) byte pairs per row.
        let p0 = pack8_i32_u16(v0[1], v1[1]);
        let q0 = pack8_i32_u16(v0[2], v1[2]);
        let pairs = u8x16_narrow_i16x8(zip_lo16(p0, q0), zip_hi16(p0, q0));
        let mut t = [0u16; 8];
        v128_store(t.as_mut_ptr() as *mut v128, pairs);
        for (i, &pq) in t.iter().enumerate() {
            std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u16, pq);
        }
    }
}

fn deblock_chroma_h(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let (a0, a1) = ld8_u8_i32(data.sub(2 * stride));
        let (b0, b1) = ld8_u8_i32(data.sub(stride));
        let (c0, c1) = ld8_u8_i32(data);
        let (d0, d1) = ld8_u8_i32(data.add(stride));
        let mut v0 = [a0, b0, c0, d0];
        let mut v1 = [a1, b1, c1, d1];
        chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
        chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
        v128_store64_lane::<0>(pack8_i32_u8(v0[1], v1[1]), data.sub(stride) as *mut u64);
        v128_store64_lane::<0>(pack8_i32_u8(v0[2], v1[2]), data as *mut u64);
    }
}

// ----------------------------------------------------------------------
// 16-bit sample planes (10/12-bit decode)
// ----------------------------------------------------------------------
//
// The kernels below mirror `hevc_x86_128.rs`'s `kernels_u16!` widening
// choices, which encode the overflow analysis: a 12-bit sample times a tap
// leaves i16 (4095 · 58 = 237,510), so the first-stage FIR is
// `i32x4_dot_i16x8` on interleaved neighbour pairs with 32-bit sums — the
// u8 tier's broadcast-tap i16 multiply does not carry over. `uni` stays in
// i16 (14-bit + round < 32767) but clips with explicit min/max: there is
// no `packus` shortcut at max 1023 or 4095. `bi` needs a 32-bit sum, and
// the x86 file's trick is kept: `pmaddwd` against (1, 1) on the two
// interleaved predictions is exactly that sum in one instruction. The
// deblocking filters, the inverse transform and the 14-bit second stage
// are sample-size independent and shared with the 8-bit tier above —
// `install_u16` installs the very same fn items for those.

/// What a 16-bit-sample FIR stage produces, per output kind (`MODE_*`).
///
/// Unlike [`Out`], the output stages carry runtime constants: the fused
/// path's shifts and clip depend on the bit depth, which the u8 tier bakes
/// in as 8 and this one cannot.
#[derive(Clone, Copy)]
struct Out16 {
    /// `MODE_I16`: 14-bit predictions, stride `w`.
    i16: *mut i16,
    /// `MODE_UNI` / `MODE_BI`: samples, stride `stride`.
    u16: *mut u16,
    /// Sample stride.
    stride: usize,
    /// `MODE_BI`: the other list's 14-bit prediction, stride `w`.
    other: *const i16,
    /// Block width (the stride of `i16` and `other`).
    w: usize,
    /// `MODE_UNI`: `1 << (13 - bd)` as i16 lanes.
    uni_round: v128,
    /// `MODE_UNI`: `14 - bd`.
    uni_shift: u32,
    /// `MODE_BI`: `1 << (14 - bd)` as i32 lanes.
    bi_round: v128,
    /// `MODE_BI`: `15 - bd`.
    bi_shift: u32,
    /// `(1 << bd) - 1` as i16 lanes.
    maxv: v128,
}

/// An [`Out16`] for the two-pass kernels: 14-bit predictions only.
#[inline]
fn i16_out16(dst: &mut [i16], w: usize) -> Out16 {
    Out16 {
        i16: dst.as_mut_ptr(),
        u16: std::ptr::null_mut(),
        stride: 0,
        other: std::ptr::null(),
        w,
        uni_round: i16x8_splat(0),
        uni_shift: 0,
        bi_round: i32x4_splat(0),
        bi_shift: 0,
        maxv: i16x8_splat(0),
    }
}

/// Store the first `n` (≤ 8) u16 lanes of `v`.
#[inline]
unsafe fn store_u16_n(dst: *mut u16, v: v128, n: usize) {
    unsafe { store_i16_n(dst as *mut i16, v, n) }
}

/// Clip 8 lanes of i16 to `0..=max` (max < 32768) as u16 samples.
#[inline]
fn clip16(v: v128, maxv: v128) -> v128 {
    i16x8_min(i16x8_max(v, i16x8_splat(0)), maxv)
}

/// Whether reading 8 u16 lanes at every 8-sample step of a `w`-wide row,
/// for `rows` rows of `stride`, plus `extra` samples along, stays inside
/// `len`.
#[inline]
fn fits16(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    let last_x = if w == 0 { 0 } else { (w - 1) / 8 * 8 };
    (rows - 1) * stride + last_x + extra + 8 <= len
}

/// Emit 8 lanes of a 16-bit-sample stage's output (`v`, 14-bit) at
/// (`row`, `x`), the first `n` lanes.
#[inline]
unsafe fn emit16<const MODE: u8>(out: &Out16, row: usize, x: usize, v: v128, n: usize) {
    unsafe {
        match MODE {
            MODE_I16 => store_i16_n(out.i16.add(row * out.w + x), v, n),
            MODE_UNI => {
                // 14-bit + round fits i16; the clip needs no more.
                let r = i16x8_shr(i16x8_add_sat(v, out.uni_round), out.uni_shift);
                store_u16_n(out.u16.add(row * out.stride + x), clip16(r, out.maxv), n);
            }
            _ => {
                // v + other exceeds i16: the (1, 1) dot is the 32-bit sum.
                let o = load_i16_n(out.other.add(row * out.w + x), n);
                let ones = i32x4_splat(pair16(1, 1));
                let quad = |z: v128| i32x4_shr(i32x4_add(i32x4_dot_i16x8(z, ones), out.bi_round), out.bi_shift);
                let r = i16x8_narrow_i32x4(quad(zip_lo16(v, o)), quad(zip_hi16(v, o)));
                store_u16_n(out.u16.add(row * out.stride + x), clip16(r, out.maxv), n);
            }
        }
    }
}

/// Horizontal FIR with `TAPS` taps over u16 samples: `i32x4_dot_i16x8` on
/// interleaved neighbour pairs, 32-bit sums, saturating narrow.
#[inline]
unsafe fn fir16_h<const TAPS: usize, const MODE: u8>(out: &Out16, src: *const u16, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [i32x4_splat(0); 4];
        for k in 0..TAPS / 2 {
            c[k] = i32x4_splat(pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = shift as u32;
        for y in 0..h {
            let s = src.add(y * src_stride);
            let mut x = 0;
            while x < w {
                let mut lo = i32x4_splat(0);
                let mut hi = i32x4_splat(0);
                for k in 0..TAPS / 2 {
                    let a = v128_load(s.add(x + 2 * k) as *const v128);
                    let b = v128_load(s.add(x + 2 * k + 1) as *const v128);
                    lo = i32x4_add(lo, i32x4_dot_i16x8(zip_lo16(a, b), c[k]));
                    hi = i32x4_add(hi, i32x4_dot_i16x8(zip_hi16(a, b), c[k]));
                }
                let r = i16x8_narrow_i32x4(i32x4_shr(lo, sh), i32x4_shr(hi, sh));
                emit16::<MODE>(out, y, x, r, (w - x).min(8));
                x += 8;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over u16 or i16 rows (`T` = 2-byte
/// lanes): row pairs interleaved, the same 32-bit dot. i16 rows read as
/// u16 lanes are fine — the dot is signed and the lanes carry their bits.
#[inline]
unsafe fn fir16_v<const TAPS: usize, const MODE: u8, T>(out: &Out16, src: *const T, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [i32x4_splat(0); 4];
        for k in 0..TAPS / 2 {
            c[k] = i32x4_splat(pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = shift as u32;
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let mut lo = i32x4_splat(0);
                let mut hi = i32x4_splat(0);
                for k in 0..TAPS / 2 {
                    let a = v128_load(src.add((y + 2 * k) * src_stride + x) as *const v128);
                    let b = v128_load(src.add((y + 2 * k + 1) * src_stride + x) as *const v128);
                    lo = i32x4_add(lo, i32x4_dot_i16x8(zip_lo16(a, b), c[k]));
                    hi = i32x4_add(hi, i32x4_dot_i16x8(zip_hi16(a, b), c[k]));
                }
                let r = i16x8_narrow_i32x4(i32x4_shr(lo, sh), i32x4_shr(hi, sh));
                emit16::<MODE>(out, y, x, r, (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn copy16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
    if !fits16(src.len(), src_stride, h, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u16>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe {
        let sh = shift as u32;
        for y in 0..h {
            let s = src.as_ptr().add(y * src_stride);
            let d = dst.as_mut_ptr().add(y * w);
            let mut x = 0;
            while x < w {
                store_i16_n(d.add(x), i16x8_shl(v128_load(s.add(x) as *const v128), sh), (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn qpel_h16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits16(src.len(), src_stride, h, w, 8) || dst.len() < w * h {
        return (HevcDsp::<u16>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out16(dst, w);
    unsafe { fir16_h::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits16(src.len(), src_stride, h + 7, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u16>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out16(dst, w);
    unsafe { fir16_v::<8, MODE_I16, u16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn epel_h16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits16(src.len(), src_stride, h, w, 4) || dst.len() < w * h {
        return (HevcDsp::<u16>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out16(dst, w);
    unsafe { fir16_h::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v16(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits16(src.len(), src_stride, h + 3, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u16>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = i16_out16(dst, w);
    unsafe { fir16_v::<4, MODE_I16, u16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

// ----------------------------------------------------------------------
// Combination / weighting (16-bit samples)
// ----------------------------------------------------------------------

fn uni16(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = i16x8_splat(if shift > 0 { 1 << (shift - 1) } else { 0 });
        let sh = shift.max(0) as u32;
        let maxv = i16x8_splat(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_i16_n(src.as_ptr().add(y * w + x), w - x);
                // 14-bit + round fits i16 (< 16384 + 8192).
                let v = i16x8_shr(i16x8_add_sat(s, round), sh);
                store_u16_n(dst.as_mut_ptr().add(y * stride + x), clip16(v, maxv), n);
                x += 8;
            }
        }
    }
}

fn bi16(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = i32x4_splat(1 << (shift - 1));
        let sh = shift as u32;
        let maxv = i16x8_splat(max as i16);
        // a + b can exceed i16, so the sum has to be 32-bit — but the dot
        // against (1, 1) on the interleaved predictions is exactly that
        // sum, in one instruction and without widening.
        let ones = i32x4_splat(pair16(1, 1));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_i16_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_i16_n(b.as_ptr().add(y * w + x), w - x);
                let quad = |v: v128| i32x4_shr(i32x4_add(i32x4_dot_i16x8(v, ones), round), sh);
                let p = i16x8_narrow_i32x4(quad(zip_lo16(va, vb)), quad(zip_hi16(va, vb)));
                store_u16_n(dst.as_mut_ptr().add(y * stride + x), clip16(p, maxv), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni16(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    if i16::try_from(wt).is_err() {
        return (HevcDsp::<u16>::SCALAR.weighted_uni)(dst, stride, src, w, h, log2_wd, wt, o, max);
    }
    unsafe {
        let round = i32x4_splat(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let sh = log2_wd.max(0) as u32;
        let wv = i16x8_splat(wt as i16);
        let ov = i32x4_splat(o);
        let maxv = i16x8_splat(max as i16);
        let weigh = |s: v128| -> v128 {
            let lo = i32x4_add(i32x4_shr(i32x4_add(i32x4_extmul_low_i16x8(s, wv), round), sh), ov);
            let hi = i32x4_add(i32x4_shr(i32x4_add(i32x4_extmul_high_i16x8(s, wv), round), sh), ov);
            i16x8_narrow_i32x4(lo, hi)
        };
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_i16_n(src.as_ptr().add(y * w + x), w - x);
                store_u16_n(dst.as_mut_ptr().add(y * stride + x), clip16(weigh(s), maxv), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi16(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    if i16::try_from(w0).is_err() || i16::try_from(w1).is_err() {
        return (HevcDsp::<u16>::SCALAR.weighted_bi)(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1, max);
    }
    unsafe {
        let round = i32x4_splat((o0 + o1 + 1) << log2_wd);
        let sh = (log2_wd + 1) as u32;
        let wv = i32x4_splat(pair16(w0 as i16, w1 as i16));
        let maxv = i16x8_splat(max as i16);
        let weigh = |va: v128, vb: v128| -> v128 {
            let quad = |v: v128| i32x4_shr(i32x4_add(i32x4_dot_i16x8(v, wv), round), sh);
            i16x8_narrow_i32x4(quad(zip_lo16(va, vb)), quad(zip_hi16(va, vb)))
        };
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_i16_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_i16_n(b.as_ptr().add(y * w + x), w - x);
                store_u16_n(dst.as_mut_ptr().add(y * stride + x), clip16(weigh(va, vb), maxv), n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add (16-bit samples)
// ----------------------------------------------------------------------

fn add_residual16(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
    unsafe {
        let maxv = i16x8_splat(max as i16);
        if n >= 8 {
            for y in 0..n {
                let mut x = 0;
                while x < n {
                    let d = dst.as_mut_ptr().add(y * stride + x);
                    let p = v128_load(d as *const v128);
                    let r = v128_load(res.as_ptr().add(y * n + x) as *const v128);
                    // Samples < 4096 and residuals fit: adds saturate correctly.
                    v128_store(d as *mut v128, clip16(i16x8_add_sat(p, r), maxv));
                    x += 8;
                }
            }
        } else {
            // 4x4: two rows per vector.
            for y in (0..4).step_by(2) {
                let d0 = dst.as_mut_ptr().add(y * stride);
                let d1 = dst.as_mut_ptr().add((y + 1) * stride);
                let p = zip_lo64(v128_load64_zero(d0 as *const u64), v128_load64_zero(d1 as *const u64));
                let r = v128_load(res.as_ptr().add(y * 4) as *const v128);
                let v = clip16(i16x8_add_sat(p, r), maxv);
                v128_store64_lane::<0>(v, d0 as *mut u64);
                v128_store64_lane::<1>(v, d1 as *mut u64);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Fused interpolation + prediction (16-bit samples)
// ----------------------------------------------------------------------

/// Copy a `w x h` u16 block (whole-sample uni-prediction).
unsafe fn copy_rows_u16(dst: *mut u16, dst_stride: usize, src: *const u16, src_stride: usize, w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * dst_stride);
            let mut x = 0;
            while x < w {
                let n = w - x;
                if n >= 8 {
                    v128_store(d.add(x) as *mut v128, v128_load(s.add(x) as *const v128));
                    x += 8;
                } else if n >= 4 {
                    std::ptr::write_unaligned(d.add(x) as *mut u64, std::ptr::read_unaligned(s.add(x) as *const u64));
                    x += 4;
                } else {
                    std::ptr::write_unaligned(d.add(x) as *mut u32, std::ptr::read_unaligned(s.add(x) as *const u32));
                    x += 2;
                }
            }
        }
    }
}

/// The fused kernels for 16-bit samples: `TAPS` (8 luma / 4 chroma),
/// `MODE_UNI` or `MODE_BI`, at any bit depth 8..=12.
///
/// No x86 or NEON rung installs fused u16 kernels — their u8 fused paths
/// bake the bit-8 shifts in, and the 16-bit tables run two-pass. This one
/// is an extension past that parity line: the output-stage constants
/// (`shift1 = min(bd, 12) − 8`, uni `>> 14 − bd`, bi `>> 15 − bd`, the
/// clip) travel in [`Out16`] at runtime, which is all the generalisation
/// the driver needed.
#[allow(clippy::too_many_arguments)]
fn fused16<const TAPS: usize, const MODE: u8>(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    let reach = TAPS / 2 - 1;
    let at_block = reach * src_stride + reach;
    let hh = h + TAPS - 1;
    let bd = bit_depth as i32;
    let ok = (8..=12).contains(&bd)
        && w >= 2
        && h >= 1
        && (h - 1) * dst_stride + w <= dst.len()
        && (MODE != MODE_BI || other.len() >= w * h)
        && tmp.len() >= super::hevc::MC_TMP_LEN
        && match (fx, fy) {
            (0, 0) => (h - 1) * src_stride + w + at_block <= src.len(),
            (_, 0) => src.len() > reach * src_stride && fits16(src.len() - reach * src_stride, src_stride, h, w, TAPS),
            (0, _) => src.len() > reach && fits16(src.len() - reach, src_stride, hh, w, 0),
            _ => fits16(src.len(), src_stride, hh, w, TAPS) && fits_i16(super::hevc::MC_TMP_LEN, w, hh),
        };
    if !ok {
        let s = HevcDsp::<u16>::SCALAR;
        return match (TAPS, MODE) {
            (8, MODE_UNI) => (s.qpel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, bit_depth),
            (8, _) => (s.qpel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth),
            (_, MODE_UNI) => (s.epel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, bit_depth),
            _ => (s.epel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth),
        };
    }
    let (tx, ty): (&[i8], &[i8]) = if TAPS == 8 { (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]) } else { (&EPEL_FILTERS[fx], &EPEL_FILTERS[fy]) };
    let shift1 = bd.min(12) - 8;
    let max = (1 << bd) - 1;
    let out = Out16 {
        i16: std::ptr::null_mut(),
        u16: dst.as_mut_ptr(),
        stride: dst_stride,
        other: other.as_ptr(),
        w,
        uni_round: i16x8_splat(1 << (13 - bd)),
        uni_shift: (14 - bd) as u32,
        bi_round: i32x4_splat(1 << (14 - bd)),
        bi_shift: (15 - bd) as u32,
        maxv: i16x8_splat(max as i16),
    };
    unsafe {
        match (fx, fy) {
            (0, 0) => {
                if MODE == MODE_UNI {
                    copy_rows_u16(dst.as_mut_ptr(), dst_stride, src.as_ptr().add(at_block), src_stride, w, h);
                } else {
                    // Whole-sample bi: widen, then the usual average.
                    let (pred, _) = tmp.split_at_mut(w * h);
                    copy16(pred, &src[at_block..], src_stride, w, h, 14 - bd);
                    bi16(dst, dst_stride, other, pred, w, h, 15 - bd, max);
                }
            }
            (_, 0) => fir16_h::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, w, h, tx, shift1),
            (0, _) => fir16_v::<TAPS, MODE, u16>(&out, src.as_ptr().add(reach), src_stride, w, h, ty, shift1),
            _ => {
                let mid = i16_out16(tmp, w);
                fir16_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, shift1);
                fir16_v::<TAPS, MODE, i16>(&out, tmp.as_ptr(), w, w, h, ty, 6);
            }
        }
    }
}

fn qpel_uni16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    fused16::<8, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[], bit_depth)
}

fn epel_uni16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    fused16::<4, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[], bit_depth)
}

#[allow(clippy::too_many_arguments)]
fn qpel_bi16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    fused16::<8, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth)
}

#[allow(clippy::too_many_arguments)]
fn epel_bi16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    fused16::<4, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, bit_depth)
}

// ----------------------------------------------------------------------
// SAO (16-bit samples)
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sao_band16(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    unsafe {
        // The four consecutive bands (mod 32) with nonzero offsets.
        let mut bands = [0i16; 4];
        let mut offs = [0i16; 4];
        let mut k = 0;
        for b in 0..32 {
            if table[b] != 0 && k < 4 {
                bands[k] = b as i16;
                offs[k] = table[b];
                k += 1;
            }
        }
        let sh = shift as u32;
        let maxv = i16x8_splat(max as i16);
        let bv: [v128; 4] = std::array::from_fn(|i| i16x8_splat(bands[i]));
        let ov: [v128; 4] = std::array::from_fn(|i| i16x8_splat(offs[i]));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = src.as_ptr().add(y * src_stride + x);
                let v = if n == 8 { v128_load(s as *const v128) } else { load_i16_n(s as *const i16, n) };
                let band = u16x8_shr(v, sh);
                let mut off = i16x8_splat(0);
                for i in 0..k {
                    off = v128_bitselect(ov[i], off, i16x8_eq(band, bv[i]));
                }
                let r = clip16(i16x8_add(v, off), maxv);
                store_u16_n(dst.as_mut_ptr().add(y * dst_stride + x), r, n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge16(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    unsafe {
        let maxv = i16x8_splat(max as i16);
        let one = i16x8_splat(1);
        // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 → offsets via compares.
        let o0 = i16x8_splat(off[0]);
        let o1 = i16x8_splat(off[1]);
        let o3 = i16x8_splat(off[3]);
        let o4 = i16x8_splat(off[4]);
        let two = i16x8_splat(2);
        let three = i16x8_splat(3);
        let four = i16x8_splat(4);
        let zero = i16x8_splat(0);
        let lo_reach = na.min(nb).min(0);
        let hi_reach = na.max(nb).max(0);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let i = origin + y * stride + x;
                // All three loads and the store must stay inside.
                if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 8 > src.len() || i + 8 > dst.len() {
                    // Tail near the buffer end: scalar.
                    for xx in x..w {
                        let ii = origin + y * stride + xx;
                        let v = src[ii] as i32;
                        let a = src[(ii as isize + na) as usize] as i32;
                        let b = src[(ii as isize + nb) as usize] as i32;
                        let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                        dst[ii] = (v + off[e] as i32).clamp(0, max) as u16;
                    }
                    break;
                }
                let v = v128_load(src.as_ptr().add(i) as *const v128);
                let a = v128_load(src.as_ptr().offset(i as isize + na) as *const v128);
                let b = v128_load(src.as_ptr().offset(i as isize + nb) as *const v128);
                // sign(v - a) = (v > a) - (v < a); samples < 32768 so signed
                // compares are exact.
                let sa = i16x8_sub(v128_and(i16x8_gt(v, a), one), v128_and(i16x8_gt(a, v), one));
                let sb = i16x8_sub(v128_and(i16x8_gt(v, b), one), v128_and(i16x8_gt(b, v), one));
                let e = i16x8_add(i16x8_add(sa, sb), two);
                let mut o = zero;
                o = v128_bitselect(o0, o, i16x8_eq(e, zero));
                o = v128_bitselect(o1, o, i16x8_eq(e, one));
                o = v128_bitselect(o3, o, i16x8_eq(e, three));
                o = v128_bitselect(o4, o, i16x8_eq(e, four));
                let r = clip16(i16x8_add(v, o), maxv);
                store_u16_n(dst.as_mut_ptr().add(i), r, n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking (16-bit samples) — the shared i32-lane filters with u16
// loads and stores.
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn deblock_luma_v16(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut r = [i32x4_splat(0); 8];
        for (i, v) in r.iter_mut().enumerate() {
            *v = v128_load(data.add(i * stride).sub(4) as *const v128);
        }
        transpose8_u16(&mut r);
        let mut v0 = [i32x4_splat(0); 8];
        let mut v1 = [i32x4_splat(0); 8];
        for k in 0..8 {
            v0[k] = u32x4_extend_low_u16x8(r[k]);
            v1[k] = u32x4_extend_high_u16x8(r[k]);
        }
        luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
        luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
        for k in 0..8 {
            r[k] = pack8_i32_u16(v0[k], v1[k]);
        }
        transpose8_u16(&mut r);
        for (i, v) in r.iter().enumerate() {
            v128_store(data.add(i * stride).sub(4) as *mut v128, *v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_h16(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut v0 = [i32x4_splat(0); 8];
        let mut v1 = [i32x4_splat(0); 8];
        for k in 0..8 {
            let p = data.offset((k as isize - 4) * stride as isize);
            v0[k] = u32x4_extend_low_u16x8(v128_load64_zero(p as *const u64));
            v1[k] = u32x4_extend_low_u16x8(v128_load64_zero(p.add(4) as *const u64));
        }
        luma_filter4(&mut v0, beta[0], tc[0], no_p[0], no_q[0], max);
        luma_filter4(&mut v1, beta[1], tc[1], no_p[1], no_q[1], max);
        for k in 1..7 {
            v128_store(data.offset((k as isize - 4) * stride as isize) as *mut v128, pack8_i32_u16(v0[k], v1[k]));
        }
    }
}

fn deblock_chroma_v16(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let mut r = [i32x4_splat(0); 8];
        for (i, v) in r.iter_mut().enumerate() {
            *v = v128_load64_zero(data.add(i * stride).sub(2) as *const u64);
        }
        let a0 = zip_lo16(r[0], r[1]);
        let a1 = zip_lo16(r[2], r[3]);
        let a2 = zip_lo16(r[4], r[5]);
        let a3 = zip_lo16(r[6], r[7]);
        let b0 = zip_lo32(a0, a1); // p1 r0..3 | p0 r0..3
        let b1 = zip_hi32(a0, a1); // q0 r0..3 | q1 r0..3
        let b2 = zip_lo32(a2, a3); // rows 4..7
        let b3 = zip_hi32(a2, a3);
        let mut v0 = [
            u32x4_extend_low_u16x8(b0),
            u32x4_extend_high_u16x8(b0),
            u32x4_extend_low_u16x8(b1),
            u32x4_extend_high_u16x8(b1),
        ];
        let mut v1 = [
            u32x4_extend_low_u16x8(b2),
            u32x4_extend_high_u16x8(b2),
            u32x4_extend_low_u16x8(b3),
            u32x4_extend_high_u16x8(b3),
        ];
        chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
        chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
        // (p0, q0) pairs per row, stored as one 32-bit write each.
        let p0 = pack8_i32_u16(v0[1], v1[1]);
        let q0 = pack8_i32_u16(v0[2], v1[2]);
        let lo = zip_lo16(p0, q0); // rows 0..3
        let hi = zip_hi16(p0, q0); // rows 4..7
        let mut t = [0u32; 8];
        v128_store(t.as_mut_ptr() as *mut v128, lo);
        v128_store(t.as_mut_ptr().add(4) as *mut v128, hi);
        for (i, &pq) in t.iter().enumerate() {
            std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u32, pq);
        }
    }
}

fn deblock_chroma_h16(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let data = data.as_mut_ptr().add(off);
        let ld = |p: *const u16| -> (v128, v128) {
            let lo = v128_load64_zero(p as *const u64);
            let hi = v128_load64_zero(p.add(4) as *const u64);
            (u32x4_extend_low_u16x8(lo), u32x4_extend_low_u16x8(hi))
        };
        let (a0, a1) = ld(data.sub(2 * stride));
        let (b0, b1) = ld(data.sub(stride));
        let (c0, c1) = ld(data);
        let (d0, d1) = ld(data.add(stride));
        let mut v0 = [a0, b0, c0, d0];
        let mut v1 = [a1, b1, c1, d1];
        chroma_filter4(&mut v0, [tc[0], tc[1]], [no_p[0], no_p[1]], [no_q[0], no_q[1]], max);
        chroma_filter4(&mut v1, [tc[2], tc[3]], [no_p[2], no_p[3]], [no_q[2], no_q[3]], max);
        v128_store(data.sub(stride) as *mut v128, pack8_i32_u16(v0[1], v1[1]));
        v128_store(data as *mut v128, pack8_i32_u16(v0[2], v1[2]));
    }
}

