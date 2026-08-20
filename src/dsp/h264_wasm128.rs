//! 128-bit SIMD versions of the H.264 kernels for WebAssembly.
//!
//! Eight 16-bit lanes per vector, the same arithmetic and the same block
//! shapes as [`super::h264_x86_128`] — which is where this was ported from,
//! and which is the file to read alongside it. What follows is why the two do
//! not correspond line for line, since anyone diffing them will notice and
//! should not conclude rot.
//!
//! **wasm has no `pmaddubsw`.** There is no unsigned-by-signed pairwise
//! multiply-add in `simd128`, and the closest faithful replacement is seven
//! instructions: widen both halves, two `i32x4_dot_i16x8`, and one saturating
//! narrow. (That emulation is exact — it was checked against a scalar
//! reference over 200,000 random inputs including the saturating corners —
//! but seven-for-one on the innermost loop is not a trade worth making.) The
//! x86 file has a `pmaddubsw` path for the six-tap luma filter and for chroma
//! bilinear *because x86 has the instruction*, and its SSE2 rung computes the
//! same results without it, from widening loads and a shift-and-add chain. So
//! this port follows the SSE2 shape for those two, and the SSE4.1 shape
//! everywhere else — because `simd128` has SSE4.1's useful parts natively:
//! `u16x8_extend_low_u8x16` is `pmovzxbw`, `v128_bitselect` is `pblendvb`,
//! `i16x8_min`/`max` are there, and `i32x4_dot_i16x8` is `pmaddwd` exactly.
//!
//! The one place wasm is *better* than the file it came from: `i16x8_shr`
//! takes its count as a plain integer, so the "shift by a runtime amount"
//! dance that x86 needs a vector for disappears.
//!
//! Deblocking and the transforms follow the SSE4.1 shape throughout. The
//! three primitives the x86 loop filters swap per rung — `pblendvb`, `pabsw`,
//! `ptest` — are all native here (`v128_bitselect`, `i16x8_abs`,
//! `v128_any_true`), and no kernel in that half touches `pmaddubsw`, so there
//! was nothing to restructure. Two of x86's little dances also fall out to
//! single instructions: `_mm_srli_si128(v, 8)` before a low-half store is
//! `v128_store64_lane::<1>`, and `_mm_cvtsi32_si128` of an unaligned dword
//! read is `v128_load32_zero` (every wasm load is unaligned-tolerant).
//!
//! There is one rung here, not four. `simd128` is a compile-time feature
//! rather than something detected at run time, so `Cpu::simd128` is set from
//! `cfg!(target_feature = "simd128")` and this module is compiled only when
//! that holds — but `H26X_NO_SIMD=1` still selects the scalar reference, so
//! the equivalence `tools/wasm.sh` checks is still checkable.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut H264Dsp<u8>) {
    d.qpel = [
        qpel::<0, 0>,
        qpel::<1, 0>,
        qpel::<2, 0>,
        qpel::<3, 0>,
        qpel::<0, 1>,
        qpel::<1, 1>,
        qpel::<2, 1>,
        qpel::<3, 1>,
        qpel::<0, 2>,
        qpel::<1, 2>,
        qpel::<2, 2>,
        qpel::<3, 2>,
        qpel::<0, 3>,
        qpel::<1, 3>,
        qpel::<2, 3>,
        qpel::<3, 3>,
    ];
    d.chroma = chroma;
    d.copy = copy;
    d.avg = avg;
    d.weighted_uni = weighted_uni;
    d.weighted_bi = weighted_bi;
    d.deblock_luma_v = deblock_luma_v;
    d.deblock_luma8_v = deblock_luma8_v;
    d.deblock_luma8_v_intra = deblock_luma8_v_intra;
    d.deblock_luma_h = deblock_luma_h;
    d.deblock_luma_v_intra = deblock_luma_v_intra;
    d.deblock_luma_h_intra = deblock_luma_h_intra;
    d.deblock_chroma_v = deblock_chroma_v;
    d.deblock_chroma_h = deblock_chroma_h;
    d.deblock_chroma_v_intra = deblock_chroma_v_intra;
    d.deblock_chroma_h_intra = deblock_chroma_h_intra;
    d.idct4_add = idct4_add;
    d.idct8_add = idct8_add;
    d.idct4_dc_add = idct4_dc_add;
    d.idct8_dc_add = idct8_dc_add;
    d.residual4 = residual4;
    d.residual8 = residual8;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Eight bytes at `p` as eight i16.
#[inline]
unsafe fn load8(p: *const u8) -> v128 {
    unsafe { u16x8_extend_low_u8x16(v128_load64_zero(p as *const u64)) }
}

/// Eight bytes at `p`, unwidened, in the low half.
#[inline]
unsafe fn load8_raw(p: *const u8) -> v128 {
    unsafe { v128_load64_zero(p as *const u64) }
}

/// Store the low eight bytes of `v`.
#[inline]
unsafe fn store8(p: *mut u8, v: v128) {
    unsafe { v128_store64_lane::<0>(v, p as *mut u64) }
}

/// Store the first `n` (≤ 16) bytes of `v`.
///
/// The narrow cases go through `write_unaligned`, never a typed store: `dst`
/// is a row of a picture and is aligned to nothing in particular.
#[inline]
unsafe fn store_u8_n(dst: *mut u8, v: v128, n: usize) {
    unsafe {
        if n == 16 {
            v128_store(dst as *mut v128, v);
        } else if n == 8 {
            v128_store64_lane::<0>(v, dst as *mut u64);
        } else if n == 4 {
            std::ptr::write_unaligned(dst as *mut u32, u32x4_extract_lane::<0>(v));
        } else {
            let mut t = [0u8; 16];
            v128_store(t.as_mut_ptr() as *mut v128, v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// `clip((v + 16) >> 5)` of eight i16 lanes, packed to eight u8 in the low
/// half — `packus`'s saturation is exactly the clip the standard asks for.
#[inline]
fn round5_pack(v: v128) -> v128 {
    let r = i16x8_shr(i16x8_add(v, i16x8_splat(16)), 5);
    u8x16_narrow_i16x8(r, r)
}

/// A tap pair as one i32 lane, for `i32x4_dot_i16x8`.
#[inline]
const fn pair(a: i16, b: i16) -> i32 {
    (a as u16 as i32) | ((b as u16 as i32) << 16)
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

/// `unpacklo_epi8`: the low eight bytes of each source, interleaved.
#[inline]
fn zip_lo8(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
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

/// `_mm_srli_si128(v, 8)`: the high half moved down, zeros shifted in. No
/// whole-vector byte shift exists in `simd128`; one lane shuffle against a
/// zero vector is the same single instruction.
#[inline]
fn hi_half(v: v128) -> v128 {
    i64x2_shuffle::<1, 2>(v, i64x2_splat(0))
}

/// `b` where `m`'s lanes are all-ones, `a` where they are all-zeros —
/// `pblendvb`, which wasm has natively (and for full lane masks the bitwise
/// select is exact, as the x86 file's SSE2 and-or form is).
#[inline]
fn sel(a: v128, b: v128, m: v128) -> v128 {
    v128_bitselect(b, a, m)
}

/// Store eight i16 lanes as eight bytes, saturating to `0..=255` — what the
/// x86 file's `store8` does. This file's [`store8`] stores raw bytes, so the
/// deblocking and transform stores, which come out of the filters still
/// widened, get their own name.
#[inline]
unsafe fn store8_sat(p: *mut u8, v: v128) {
    unsafe { v128_store64_lane::<0>(u8x16_narrow_i16x8(v, v), p as *mut u64) }
}

// ----------------------------------------------------------------------
// Luma interpolation (8.4.2.2.1)
// ----------------------------------------------------------------------

/// Six-tap `a - 5b + 20c + 20d - 5e + f` over the eight u8 samples at `p`,
/// `p + step`, … `p + 5 * step`, as eight i16.
///
/// The shift-and-add form rather than a multiply: `t = c + d <= 510` so
/// `20t <= 10200`, `5u <= 2550`, and every intermediate stays inside i16 for
/// 8-bit input. This is the SSE2 shape, kept because the SSSE3 one it
/// replaces is built on `pmaddubsw`.
#[inline]
unsafe fn tap6_u8(p: *const u8, step: usize) -> v128 {
    unsafe {
        let ld = |k: usize| load8(p.add(k * step));
        let (a, b, c, d, e, f) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let t = i16x8_add(c, d);
        let u = i16x8_add(b, e);
        let v = i16x8_add(a, f);
        let t20 = i16x8_add(i16x8_shl(t, 4), i16x8_shl(t, 2));
        let u5 = i16x8_add(i16x8_shl(u, 2), u);
        i16x8_sub(i16x8_add(v, t20), u5)
    }
}

/// Horizontal half-sample intermediate for the eight output columns from `x`
/// of window row `row`.
#[inline]
unsafe fn b1_row(src: *const u8, stride: usize, row: usize, x: usize) -> v128 {
    unsafe { tap6_u8(src.add(row * stride + x), 1) }
}

/// Vertical half-sample intermediate at window column `col`, block row `y`.
#[inline]
unsafe fn h1_row(src: *const u8, stride: usize, col: usize, y: usize) -> v128 {
    unsafe { tap6_u8(src.add(y * stride + col), stride) }
}

/// Centre position, eight columns: vertical six-tap over the six horizontal
/// intermediates of window rows `y..y+5`, 32-bit accumulation,
/// `clip((v + 512) >> 10)`.
///
/// The six are passed in because consecutive output rows share five of them;
/// see [`qpel_centre_impl`].
#[inline]
fn j_combine(w: &[v128; 6]) -> v128 {
    let (r0, r1, r2, r3, r4, r5) = (w[0], w[1], w[2], w[3], w[4], w[5]);
    let c01 = i32x4_splat(pair(1, -5));
    let c23 = i32x4_splat(pair(20, 20));
    let c45 = i32x4_splat(pair(-5, 1));
    let round = i32x4_splat(512);
    let half = |a: v128, b: v128, c: v128| {
        i32x4_add(
            i32x4_add(i32x4_dot_i16x8(a, c01), i32x4_dot_i16x8(b, c23)),
            i32x4_add(i32x4_dot_i16x8(c, c45), round),
        )
    };
    let lo = half(zip_lo16(r0, r1), zip_lo16(r2, r3), zip_lo16(r4, r5));
    let hi = half(zip_hi16(r0, r1), zip_hi16(r2, r3), zip_hi16(r4, r5));
    // At 128 bits `narrow` already lands lanes 0..7 in order.
    let v = i16x8_narrow_i32x4(i32x4_shr(lo, 10), i32x4_shr(hi, 10));
    u8x16_narrow_i16x8(v, v)
}

/// Full samples of block row `y` from column `x` (window offset 2, 2).
#[inline]
unsafe fn g_row(src: *const u8, stride: usize, y: usize, dx: usize, x: usize) -> v128 {
    unsafe { load8_raw(src.add((y + 2) * stride + 2 + dx + x)) }
}

fn qpel<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, _max: i32) {
    // The window is (w + 5) x (h + 5); an 8-lane load from column x reads
    // x + 7, plus five for the taps.
    let need = (h + 5 - 1) * stride + 21;
    if src.len() < need {
        return (H264Dsp::<u8>::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h, 255);
    }
    unsafe { qpel_impl::<XF, YF>(dst, src, stride, w, h) }
}

unsafe fn qpel_impl<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    // The five positions whose vertical filter runs over the *horizontal*
    // intermediates rather than over samples slide a window instead of
    // refilling it, which needs the loops the other way round.
    if matches!((XF, YF), (2, 2) | (2, 1) | (2, 3) | (1, 2) | (3, 2)) {
        return unsafe { qpel_centre_impl::<XF, YF>(dst, src, stride, w, h) };
    }
    unsafe {
        let s = src.as_ptr();
        for y in 0..h {
            for c in 0..w.div_ceil(8) {
                let x = c * 8;
                let b = || round5_pack(b1_row(s, stride, y + 2, x));
                let b_below = || round5_pack(b1_row(s, stride, y + 3, x));
                let hh = || round5_pack(h1_row(s, stride, 2 + x, y));
                let hh_right = || round5_pack(h1_row(s, stride, 3 + x, y));
                let v: v128 = match (XF, YF) {
                    (0, 0) => g_row(s, stride, y, 0, x),
                    (1, 0) => u8x16_avgr(g_row(s, stride, y, 0, x), b()),
                    (2, 0) => b(),
                    (3, 0) => u8x16_avgr(g_row(s, stride, y, 1, x), b()),
                    (0, 1) => u8x16_avgr(g_row(s, stride, y, 0, x), hh()),
                    (0, 2) => hh(),
                    (0, 3) => u8x16_avgr(load8_raw(s.add((y + 3) * stride + 2 + x)), hh()),
                    (1, 1) => u8x16_avgr(b(), hh()),
                    (3, 1) => u8x16_avgr(b(), hh_right()),
                    (1, 3) => u8x16_avgr(hh(), b_below()),
                    (3, 3) => u8x16_avgr(hh_right(), b_below()),
                    _ => unreachable!(),
                };
                store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
            }
        }
    }
}

/// The centre positions, over a sliding window of horizontal intermediates.
unsafe fn qpel_centre_impl<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    unsafe {
        let s = src.as_ptr();
        for c in 0..w.div_ceil(8) {
            let x = c * 8;
            let mut win = [
                b1_row(s, stride, 0, x),
                b1_row(s, stride, 1, x),
                b1_row(s, stride, 2, x),
                b1_row(s, stride, 3, x),
                b1_row(s, stride, 4, x),
                b1_row(s, stride, 5, x),
            ];
            for y in 0..h {
                let j = j_combine(&win);
                let v: v128 = match (XF, YF) {
                    (2, 2) => j,
                    (2, 1) => u8x16_avgr(round5_pack(win[2]), j),
                    (2, 3) => u8x16_avgr(j, round5_pack(win[3])),
                    (1, 2) => u8x16_avgr(round5_pack(h1_row(s, stride, 2 + x, y)), j),
                    (3, 2) => u8x16_avgr(j, round5_pack(h1_row(s, stride, 3 + x, y))),
                    _ => unreachable!(),
                };
                store8(dst.as_mut_ptr().add(y * PRED_STRIDE + x), v);
                // Not on the last row: the caller's bounds check covers window
                // rows up to h + 4, and row h + 5 would read past the block.
                if y + 1 < h {
                    win = [win[1], win[2], win[3], win[4], win[5], b1_row(s, stride, y + 6, x)];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// Chroma interpolation (8.4.2.2.2)
// ----------------------------------------------------------------------

/// The four bilinear weights, splatted. The SSE2 shape — four multiplies —
/// because the SSSE3 one it replaces packs them into `pmaddubsw` pairs.
struct ChromaW([v128; 4]);

fn chroma_w(xf: i32, yf: i32) -> ChromaW {
    ChromaW([
        i16x8_splat(((8 - xf) * (8 - yf)) as i16),
        i16x8_splat((xf * (8 - yf)) as i16),
        i16x8_splat(((8 - xf) * yf) as i16),
        i16x8_splat((xf * yf) as i16),
    ])
}

/// One output row: the four weighted neighbours summed. The weights total 64
/// and the samples are bytes, so `255 * 64 = 16320` bounds the result and the
/// whole row stays in i16.
#[inline]
unsafe fn chroma_row(w: &ChromaW, r0: *const u8, r1: *const u8) -> v128 {
    unsafe {
        i16x8_add(
            i16x8_add(i16x8_mul(load8(r0), w.0[0]), i16x8_mul(load8(r0.add(1)), w.0[1])),
            i16x8_add(i16x8_mul(load8(r1), w.0[2]), i16x8_mul(load8(r1.add(1)), w.0[3])),
        )
    }
}

fn chroma(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + 9 {
        return (H264Dsp::<u8>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe { chroma_impl(dst, src, stride, w, h, xf, yf) }
}

unsafe fn chroma_impl(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    unsafe {
        // Chroma blocks are at most eight wide: one vector of eight i16.
        let _ = w;
        let cw = chroma_w(xf, yf);
        let round = i16x8_splat(32);
        let s = src.as_ptr();
        for y in 0..h {
            let v = chroma_row(&cw, s.add(y * stride), s.add((y + 1) * stride));
            let v = u16x8_shr(i16x8_add(v, round), 6);
            store8(dst.as_mut_ptr().add(y * PRED_STRIDE), u8x16_narrow_i16x8(v, v));
        }
    }
}

// ----------------------------------------------------------------------
// Sample combination and weighting (8.4.2.3)
// ----------------------------------------------------------------------

fn copy(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
    assert!((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len());
    unsafe {
        for y in 0..h {
            let s = src.as_ptr().add(y * PRED_STRIDE);
            let d = dst.as_mut_ptr().add(y * stride);
            match w {
                16 => v128_store(d as *mut v128, v128_load(s as *const v128)),
                8 => std::ptr::write_unaligned(d as *mut u64, std::ptr::read_unaligned(s as *const u64)),
                4 => std::ptr::write_unaligned(d as *mut u32, std::ptr::read_unaligned(s as *const u32)),
                _ => std::ptr::copy_nonoverlapping(s, d, w),
            }
        }
    }
}

fn avg(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let va = v128_load(a.as_ptr().add(y * PRED_STRIDE) as *const v128);
            let vb = v128_load(b.as_ptr().add(y * PRED_STRIDE) as *const v128);
            store_u8_n(dst.as_mut_ptr().add(y * stride), u8x16_avgr(va, vb), w);
        }
    }
}

fn weighted_uni(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, _max: i32) {
    unsafe {
        // src * wt + round fits i16 for |wt| <= 128, which is the spec's range.
        let wv = i16x8_splat(wt as i16);
        let ov = i16x8_splat(o as i16);
        let round = i16x8_splat(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = log_wd.max(0) as u32;
        let scale = |s: v128| i16x8_add(i16x8_shr(i16x8_add(i16x8_mul(s, wv), round), sh), ov);
        for y in 0..h {
            let p = src.as_ptr().add(y * PRED_STRIDE);
            let v0 = scale(load8(p));
            let v1 = if w > 8 { scale(load8(p.add(8))) } else { v0 };
            store_u8_n(dst.as_mut_ptr().add(y * stride), u8x16_narrow_i16x8(v0, v1), w);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, _max: i32) {
    unsafe {
        // a * w0 + b * w1 reaches 2 * 255 * 128, so it needs 32-bit lanes —
        // and `dot` over the two predictions interleaved is exactly that sum
        // in one instruction, for the spec's weight and sample ranges.
        let wv = i32x4_splat(pair(w0 as i16, w1 as i16));
        let round = i32x4_splat(1 << log_wd);
        let off = i32x4_splat((o0 + o1 + 1) >> 1);
        let sh = (log_wd + 1) as u32;
        for y in 0..h {
            let pa = a.as_ptr().add(y * PRED_STRIDE);
            let pb = b.as_ptr().add(y * PRED_STRIDE);
            let eight = |x: usize| -> v128 {
                let va = load8(pa.add(x));
                let vb = load8(pb.add(x));
                let quad = |v: v128| i32x4_add(i32x4_shr(i32x4_add(i32x4_dot_i16x8(v, wv), round), sh), off);
                i16x8_narrow_i32x4(quad(zip_lo16(va, vb)), quad(zip_hi16(va, vb)))
            };
            let v0 = eight(0);
            let v1 = if w > 8 { eight(8) } else { v0 };
            store_u8_n(dst.as_mut_ptr().add(y * stride), u8x16_narrow_i16x8(v0, v1), w);
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking (8.7)
// ----------------------------------------------------------------------
//
// Eight lines of an edge are eight i16 lanes of one vector per sample
// position; a sixteen-line luma edge is two such halves, and its tC0
// segments (four lines each) fall two per half. A horizontal edge loads a
// sample position as one row; a vertical edge transposes 8 rows x 8 bytes
// into 8 column vectors, filters, and transposes back. The shape is the
// x86 file's SSE4.1 rung exactly: its three level-dependent primitives —
// `pblendvb`, `pabsw`, `ptest` — are native here.

/// `|a - b| < t` per i16 lane, as a mask.
#[inline]
fn diff_lt(a: v128, b: v128, t: v128) -> v128 {
    i16x8_gt(t, i16x8_abs(i16x8_sub(a, b)))
}

/// The eight positions of eight luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
type LumaLines = [v128; 8];

/// bS < 4 luma filter on eight lines (8.7.2.3), in place on the vectors.
/// `tc0v` holds the line's tC0 (−1 = bS 0).
#[inline]
fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: v128) {
    let [_, p2, p1, p0, q0, q1, q2, _] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let zero = i16x8_splat(0);
    let bs_on = i16x8_gt(tc0v, i16x8_splat(-1));
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), v128_and(diff_lt(q1, q0, beta), bs_on));
    let ap = diff_lt(p2, p0, beta);
    let aq = diff_lt(q2, q0, beta);
    // tc = tc0 + (ap < beta) + (aq < beta); masks are -1.
    let tc = i16x8_sub(i16x8_sub(tc0v, ap), aq);
    // delta = clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3)
    let d = i16x8_shr(i16x8_add(i16x8_add(i16x8_shl(i16x8_sub(q0, p0), 2), i16x8_sub(p1, q1)), i16x8_splat(4)), 3);
    let d = i16x8_min(i16x8_max(d, i16x8_sub(zero, tc)), tc);
    let np0 = i16x8_add(p0, d);
    let nq0 = i16x8_sub(q0, d);
    // p1' = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - 2 p1) >> 1), when ap
    let avg = i16x8_shr(i16x8_add(i16x8_add(p0, q0), i16x8_splat(1)), 1);
    let ntc0 = i16x8_sub(zero, tc0v);
    let dp1 = i16x8_shr(i16x8_sub(i16x8_add(p2, avg), i16x8_shl(p1, 1)), 1);
    let dp1 = i16x8_min(i16x8_max(dp1, ntc0), tc0v);
    let np1 = i16x8_add(p1, v128_and(dp1, ap));
    let dq1 = i16x8_shr(i16x8_sub(i16x8_add(q2, avg), i16x8_shl(q1, 1)), 1);
    let dq1 = i16x8_min(i16x8_max(dq1, ntc0), tc0v);
    let nq1 = i16x8_add(q1, v128_and(dq1, aq));
    // Clip to 8 bits (p1'/q1' cannot leave the range; p0'/q0' can).
    let clip = |x: v128| i16x8_min(i16x8_max(x, zero), i16x8_splat(255));
    v[2] = sel(p1, np1, mask);
    v[3] = sel(p0, clip(np0), mask);
    v[4] = sel(q0, clip(nq0), mask);
    v[5] = sel(q1, nq1, mask);
}

/// bS 4 luma filter on eight lines (8.7.2.4).
#[inline]
fn luma_filter_intra(v: &mut LumaLines, alpha: i32, beta: i32) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let alphav = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let mask = v128_and(v128_and(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
    let strong = diff_lt(p0, q0, i16x8_splat(((alpha >> 2) + 2) as i16));
    let ap = v128_and(diff_lt(p2, p0, beta), strong);
    let aq = v128_and(diff_lt(q2, q0, beta), strong);
    let two = i16x8_splat(2);
    let four = i16x8_splat(4);
    let add = i16x8_add;
    let dbl = |a| i16x8_shl(a, 1);
    // Weak: p0' = (2 p1 + p0 + q1 + 2) >> 2, q0' = (2 q1 + q0 + p1 + 2) >> 2.
    let wp0 = i16x8_shr(add(add(dbl(p1), p0), add(q1, two)), 2);
    let wq0 = i16x8_shr(add(add(dbl(q1), q0), add(p1, two)), 2);
    // Strong p side.
    let p0q0 = add(p0, q0);
    let sp0 = i16x8_shr(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)), 3);
    let sp1 = i16x8_shr(add(add(p2, p1), add(p0q0, two)), 2);
    let sp2 = i16x8_shr(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)), 3);
    // Strong q side.
    let sq0 = i16x8_shr(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)), 3);
    let sq1 = i16x8_shr(add(add(p0q0, q1), add(q2, two)), 2);
    let sq2 = i16x8_shr(add(add(dbl(q3), add(q2, dbl(q2))), add(add(q1, p0q0), four)), 3);
    let np0 = sel(wp0, sp0, ap);
    let np1 = sel(p1, sp1, ap);
    let np2 = sel(p2, sp2, ap);
    let nq0 = sel(wq0, sq0, aq);
    let nq1 = sel(q1, sq1, aq);
    let nq2 = sel(q2, sq2, aq);
    v[1] = sel(p2, np2, mask);
    v[2] = sel(p1, np1, mask);
    v[3] = sel(p0, np0, mask);
    v[4] = sel(q0, nq0, mask);
    v[5] = sel(q1, nq1, mask);
    v[6] = sel(q2, nq2, mask);
}

/// tC0 per lane for the eight luma lines of half `half` (four per segment).
#[inline]
fn tc0_luma(tc0: &[i16; 4], half: usize) -> v128 {
    let (a, b) = (tc0[2 * half], tc0[2 * half + 1]);
    i16x8(a, a, a, a, b, b, b, b)
}

/// Load the eight rows x 8 bytes around a vertical edge (`q0` at `data`) as
/// eight column vectors p3..q3.
#[inline]
unsafe fn load_transposed_8x8(data: *const u8, stride: usize) -> LumaLines {
    unsafe {
        let mut r = [i64x2_splat(0); 8];
        for i in 0..8 {
            r[i] = v128_load64_zero(data.add(i * stride).sub(4) as *const u64);
        }
        // Bytes: pairs of rows.
        let a0 = zip_lo8(r[0], r[1]);
        let a1 = zip_lo8(r[2], r[3]);
        let a2 = zip_lo8(r[4], r[5]);
        let a3 = zip_lo8(r[6], r[7]);
        // Words: quads of rows; lo = columns 0..3, hi = columns 4..7.
        let b0 = zip_lo16(a0, a1); // cols 0..3, rows 0..3
        let b1 = zip_hi16(a0, a1); // cols 4..7, rows 0..3
        let b2 = zip_lo16(a2, a3); // cols 0..3, rows 4..7
        let b3 = zip_hi16(a2, a3); // cols 4..7, rows 4..7
        // Dwords: a column pair's eight rows per vector.
        let c0 = zip_lo32(b0, b2); // col0 rows 0..7 | col1
        let c1 = zip_hi32(b0, b2); // col2 | col3
        let c2 = zip_lo32(b1, b3); // col4 | col5
        let c3 = zip_hi32(b1, b3); // col6 | col7
        [
            u16x8_extend_low_u8x16(c0),
            u16x8_extend_high_u8x16(c0),
            u16x8_extend_low_u8x16(c1),
            u16x8_extend_high_u8x16(c1),
            u16x8_extend_low_u8x16(c2),
            u16x8_extend_high_u8x16(c2),
            u16x8_extend_low_u8x16(c3),
            u16x8_extend_high_u8x16(c3),
        ]
    }
}

/// Store eight column vectors back as eight rows x 8 bytes.
#[inline]
unsafe fn store_transposed_8x8(data: *mut u8, stride: usize, v: &LumaLines) {
    unsafe {
        let p = |x: v128| u8x16_narrow_i16x8(x, x);
        let (c0, c1, c2, c3) = (p(v[0]), p(v[1]), p(v[2]), p(v[3]));
        let (c4, c5, c6, c7) = (p(v[4]), p(v[5]), p(v[6]), p(v[7]));
        // Bytes: column pairs -> rows interleaved.
        let a01 = zip_lo8(c0, c1);
        let a23 = zip_lo8(c2, c3);
        let a45 = zip_lo8(c4, c5);
        let a67 = zip_lo8(c6, c7);
        // Words: rows with p3..p0 / q0..q3.
        let bp_lo = zip_lo16(a01, a23); // rows 0..3
        let bp_hi = zip_hi16(a01, a23); // rows 4..7
        let bq_lo = zip_lo16(a45, a67);
        let bq_hi = zip_hi16(a45, a67);
        // Dwords: whole rows (8 bytes), two per vector. The odd row of each
        // pair goes out with `store64_lane::<1>` — where x86 needs an
        // `_mm_srli_si128` before its low-half store, the lane index says it
        // directly.
        let rows = [
            zip_lo32(bp_lo, bq_lo), // rows 0,1
            zip_hi32(bp_lo, bq_lo), // rows 2,3
            zip_lo32(bp_hi, bq_hi), // rows 4,5
            zip_hi32(bp_hi, bq_hi), // rows 6,7
        ];
        for (k, pair) in rows.iter().enumerate() {
            v128_store64_lane::<0>(*pair, data.add(2 * k * stride).sub(4) as *mut u64);
            v128_store64_lane::<1>(*pair, data.add((2 * k + 1) * stride).sub(4) as *mut u64);
        }
    }
}

fn deblock_luma_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

unsafe fn deblock_luma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
    unsafe {
        for half in 0..2 {
            let d = data.add(half * 8 * stride);
            let mut v = load_transposed_8x8(d, stride);
            luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half));
            store_transposed_8x8(d, stride, &v);
        }
    }
}

/// tC0 per lane for an eight-line luma edge: `tc0[i / 2]`, an MBAFF mixed
/// edge's strength changing every two lines rather than every four.
#[inline]
fn tc0_luma8(tc0: &[i16; 4]) -> v128 {
    let t = |k: usize| tc0[k];
    i16x8(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
}

fn deblock_luma8_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe { deblock_luma8_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

/// Eight lines is exactly one half of the sixteen-line kernel's loop; only
/// the tC0 lanes differ.
unsafe fn deblock_luma8_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
    unsafe {
        let mut v = load_transposed_8x8(data, stride);
        luma_filter_normal(&mut v, alpha, beta, tc0_luma8(tc0));
        store_transposed_8x8(data, stride, &v);
    }
}

fn deblock_luma8_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe { deblock_luma8_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

unsafe fn deblock_luma8_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v = load_transposed_8x8(data, stride);
        luma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x8(data, stride, &v);
    }
}

fn deblock_luma_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

unsafe fn deblock_luma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        for half in 0..2 {
            let d = data.add(half * 8 * stride);
            let mut v = load_transposed_8x8(d, stride);
            luma_filter_intra(&mut v, alpha, beta);
            store_transposed_8x8(d, stride, &v);
        }
    }
}

fn deblock_luma_h(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 3 * stride && off + 2 * stride + 16 <= data.len());
    unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

unsafe fn deblock_luma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
    unsafe {
        let zero = i16x8_splat(0);
        for half in 0..2 {
            let d = data.add(half * 8);
            let ld = |k: isize| load8(d.offset(k * stride as isize));
            let mut v: LumaLines = [zero, ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), zero];
            luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0, half));
            store8_sat(d.offset(-2 * stride as isize), v[2]);
            store8_sat(d.offset(-(stride as isize)), v[3]);
            store8_sat(d, v[4]);
            store8_sat(d.add(stride), v[5]);
        }
    }
}

fn deblock_luma_h_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
    unsafe { deblock_luma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

unsafe fn deblock_luma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        for half in 0..2 {
            let d = data.add(half * 8);
            let ld = |k: isize| load8(d.offset(k * stride as isize));
            let mut v: LumaLines = [ld(-4), ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), ld(3)];
            luma_filter_intra(&mut v, alpha, beta);
            for k in 1..7 {
                store8_sat(d.offset((k as isize - 4) * stride as isize), v[k]);
            }
        }
    }
}

/// The four positions of eight chroma lines: `[p1, p0, q0, q1]` as 8 x i16.
type ChromaLines = [v128; 4];

#[inline]
fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: v128) {
    let [p1, p0, q0, q1] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let zero = i16x8_splat(0);
    let bs_on = i16x8_gt(tc0v, i16x8_splat(-1));
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), v128_and(diff_lt(q1, q0, beta), bs_on));
    let tc = i16x8_add(tc0v, i16x8_splat(1));
    let d = i16x8_shr(i16x8_add(i16x8_add(i16x8_shl(i16x8_sub(q0, p0), 2), i16x8_sub(p1, q1)), i16x8_splat(4)), 3);
    let d = i16x8_min(i16x8_max(d, i16x8_sub(zero, tc)), tc);
    let clip = |x: v128| i16x8_min(i16x8_max(x, zero), i16x8_splat(255));
    v[1] = sel(p0, clip(i16x8_add(p0, d)), mask);
    v[2] = sel(q0, clip(i16x8_sub(q0, d)), mask);
}

#[inline]
fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
    let [p1, p0, q0, q1] = *v;
    let alpha = i16x8_splat(alpha as i16);
    let beta = i16x8_splat(beta as i16);
    let mask = v128_and(v128_and(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
    let two = i16x8_splat(2);
    let np0 = i16x8_shr(i16x8_add(i16x8_add(i16x8_shl(p1, 1), p0), i16x8_add(q1, two)), 2);
    let nq0 = i16x8_shr(i16x8_add(i16x8_add(i16x8_shl(q1, 1), q0), i16x8_add(p1, two)), 2);
    v[1] = sel(p0, np0, mask);
    v[2] = sel(q0, nq0, mask);
}

/// tC0 per lane for eight chroma lines (two per segment).
#[inline]
fn tc0_chroma(tc0: &[i16; 4]) -> v128 {
    let t = |k: usize| tc0[k];
    i16x8(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
}

/// Load 8 rows x 4 bytes (p1 p0 q0 q1) around a vertical chroma edge as four
/// column vectors. `v128_load32_zero` is x86's `_mm_cvtsi32_si128` of an
/// unaligned dword read in one step: wasm loads carry no alignment
/// requirement.
#[inline]
unsafe fn load_transposed_8x4(data: *const u8, stride: usize) -> ChromaLines {
    unsafe {
        let mut r = [i64x2_splat(0); 8];
        for i in 0..8 {
            r[i] = v128_load32_zero(data.add(i * stride).sub(2) as *const u32);
        }
        let a0 = zip_lo8(r[0], r[1]); // p1r0 p1r1 p0r0 p0r1 q0r0 q0r1 q1r0 q1r1
        let a1 = zip_lo8(r[2], r[3]);
        let a2 = zip_lo8(r[4], r[5]);
        let a3 = zip_lo8(r[6], r[7]);
        let b0 = zip_lo16(a0, a1); // p1 r0..3, p0 r0..3, q0 r0..3, q1 r0..3
        let b1 = zip_lo16(a2, a3); // rows 4..7
        let c0 = zip_lo32(b0, b1); // p1 r0..7 | p0 r0..7
        let c1 = zip_hi32(b0, b1); // q0 r0..7 | q1 r0..7
        [u16x8_extend_low_u8x16(c0), u16x8_extend_high_u8x16(c0), u16x8_extend_low_u8x16(c1), u16x8_extend_high_u8x16(c1)]
    }
}

/// Store the p0 / q0 columns of eight rows back (p1, q1 are unchanged).
#[inline]
unsafe fn store_transposed_8x4(data: *mut u8, stride: usize, v: &ChromaLines) {
    unsafe {
        // Interleave p0 and q0 bytes per row and store two bytes at x-1, x.
        let p0 = u8x16_narrow_i16x8(v[1], v[1]);
        let q0 = u8x16_narrow_i16x8(v[2], v[2]);
        let pq = zip_lo8(p0, q0); // p0r0 q0r0 p0r1 q0r1 ...
        let mut t = [0u8; 16];
        v128_store(t.as_mut_ptr() as *mut v128, pq);
        for i in 0..8 {
            let d = data.add(i * stride).sub(1);
            *d = t[2 * i];
            *d.add(1) = t[2 * i + 1];
        }
    }
}

fn deblock_chroma_v(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

unsafe fn deblock_chroma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
    unsafe {
        let mut v = load_transposed_8x4(data, stride);
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        store_transposed_8x4(data, stride, &v);
    }
}

fn deblock_chroma_v_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe { deblock_chroma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

unsafe fn deblock_chroma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v = load_transposed_8x4(data, stride);
        chroma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x4(data, stride, &v);
    }
}

fn deblock_chroma_h(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

unsafe fn deblock_chroma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4]) {
    unsafe {
        let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        store8_sat(data.sub(stride), v[1]);
        store8_sat(data, v[2]);
    }
}

fn deblock_chroma_h_intra(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe { deblock_chroma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

unsafe fn deblock_chroma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
        chroma_filter_intra(&mut v, alpha, beta);
        store8_sat(data.sub(stride), v[1]);
        store8_sat(data, v[2]);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------
//
// The x86 file notes its versions are identical to the AVX2 kernels because
// a 4x4 or 8x8 block's row is four or eight i16 lanes whatever the vector
// width; the same holds here, one more time.

/// Add `(v + 32) >> 6` rows to `dst`, clipping, `n` = 4 or 8 samples per row.
#[inline]
unsafe fn add_row(dst: *mut u8, v: v128, n: usize) {
    unsafe {
        let r = i16x8_shr(i16x8_add(v, i16x8_splat(32)), 6);
        if n == 4 {
            let p = u16x8_extend_low_u8x16(v128_load32_zero(dst as *const u32));
            let s = u8x16_narrow_i16x8(i16x8_add(p, r), i16x8_splat(0));
            std::ptr::write_unaligned(dst as *mut u32, u32x4_extract_lane::<0>(s));
        } else {
            let p = load8(dst);
            let s = u8x16_narrow_i16x8(i16x8_add(p, r), i16x8_splat(0));
            store8(dst, s);
        }
    }
}

fn idct4_add(dst: &mut [u8], stride: usize, coeffs: &[i16; 16], _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { idct4_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

unsafe fn idct4_add_impl(dst: *mut u8, stride: usize, c: &[i16; 16]) {
    unsafe {
        let r0 = v128_load64_zero(c.as_ptr() as *const u64);
        let r1 = v128_load64_zero(c.as_ptr().add(4) as *const u64);
        let r2 = v128_load64_zero(c.as_ptr().add(8) as *const u64);
        let r3 = v128_load64_zero(c.as_ptr().add(12) as *const u64);
        // Columns of the block, four lanes each.
        let t0 = zip_lo16(r0, r1);
        let t1 = zip_lo16(r2, r3);
        let c01 = zip_lo32(t0, t1);
        let c23 = zip_hi32(t0, t1);
        let (c0, c1, c2, c3) = (c01, hi_half(c01), c23, hi_half(c23));
        // Row pass.
        let e0 = i16x8_add(c0, c2);
        let e1 = i16x8_sub(c0, c2);
        let e2 = i16x8_sub(i16x8_shr(c1, 1), c3);
        let e3 = i16x8_add(c1, i16x8_shr(c3, 1));
        let f0 = i16x8_add(e0, e3);
        let f1 = i16x8_add(e1, e2);
        let f2 = i16x8_sub(e1, e2);
        let f3 = i16x8_sub(e0, e3);
        // Back to rows.
        let u0 = zip_lo16(f0, f1);
        let u1 = zip_lo16(f2, f3);
        let r01 = zip_lo32(u0, u1);
        let r23 = zip_hi32(u0, u1);
        let (row0, row1, row2, row3) = (r01, hi_half(r01), r23, hi_half(r23));
        // Column pass.
        let g0 = i16x8_add(row0, row2);
        let g1 = i16x8_sub(row0, row2);
        let g2 = i16x8_sub(i16x8_shr(row1, 1), row3);
        let g3 = i16x8_add(row1, i16x8_shr(row3, 1));
        add_row(dst, i16x8_add(g0, g3), 4);
        add_row(dst.add(stride), i16x8_add(g1, g2), 4);
        add_row(dst.add(2 * stride), i16x8_sub(g1, g2), 4);
        add_row(dst.add(3 * stride), i16x8_sub(g0, g3), 4);
    }
}

/// Transpose eight 8-lane i16 rows.
#[inline]
fn transpose8(r: &mut [v128; 8]) {
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

/// One 8-point pass (8.5.13.2) across eight registers.
#[inline]
fn idct8_pass(d: &[v128; 8]) -> [v128; 8] {
    let add = i16x8_add;
    let sub = i16x8_sub;
    let sh1 = |a| i16x8_shr(a, 1);
    let sh2 = |a| i16x8_shr(a, 2);
    let a0 = add(d[0], d[4]);
    let a4 = sub(d[0], d[4]);
    let a2 = sub(sh1(d[2]), d[6]);
    let a6 = add(d[2], sh1(d[6]));
    let b0 = add(a0, a6);
    let b2 = add(a4, a2);
    let b4 = sub(a4, a2);
    let b6 = sub(a0, a6);
    // a1 = -d3 + d5 - d7 - (d7 >> 1)
    let a1 = sub(sub(sub(d[5], d[3]), d[7]), sh1(d[7]));
    // a3 = d1 + d7 - d3 - (d3 >> 1)
    let a3 = sub(sub(add(d[1], d[7]), d[3]), sh1(d[3]));
    // a5 = -d1 + d7 + d5 + (d5 >> 1)
    let a5 = add(add(sub(d[7], d[1]), d[5]), sh1(d[5]));
    // a7 = d3 + d5 + d1 + (d1 >> 1)
    let a7 = add(add(add(d[3], d[5]), d[1]), sh1(d[1]));
    let b1 = add(a1, sh2(a7));
    let b7 = sub(a7, sh2(a1));
    let b3 = add(a3, sh2(a5));
    let b5 = sub(sh2(a3), a5);
    [add(b0, b7), add(b2, b5), add(b4, b3), add(b6, b1), sub(b6, b1), sub(b4, b3), sub(b2, b5), sub(b0, b7)]
}

fn idct8_add(dst: &mut [u8], stride: usize, coeffs: &[i16; 64], _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

unsafe fn idct8_add_impl(dst: *mut u8, stride: usize, c: &[i16; 64]) {
    unsafe {
        let mut r = [i64x2_splat(0); 8];
        for i in 0..8 {
            r[i] = v128_load(c.as_ptr().add(i * 8) as *const v128);
        }
        transpose8(&mut r);
        let mut f = idct8_pass(&r);
        transpose8(&mut f);
        let h = idct8_pass(&f);
        for i in 0..8 {
            add_row(dst.add(i * stride), h[i], 8);
        }
    }
}

fn idct4_dc_add(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 4) }
}

fn idct8_dc_add(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 8) }
}

unsafe fn dc_add_impl(dst: *mut u8, stride: usize, dc: i32, n: usize) {
    unsafe {
        let v = i16x8_splat(dc as i16);
        for i in 0..n {
            add_row(dst.add(i * stride), v, n);
        }
    }
}

/// Eight dequantised coefficients (two vectors of four i32) as one vector of
/// eight i16, saturating.
#[inline]
unsafe fn coefs16(coefs: *const i32) -> v128 {
    unsafe { i16x8_narrow_i32x4(v128_load(coefs as *const v128), v128_load(coefs.add(4) as *const v128)) }
}

/// All-ones in lane 0, for masking the DC coefficient out of the AC test.
#[inline]
fn lane0_mask() -> v128 {
    i16x8(-1, 0, 0, 0, 0, 0, 0, 0)
}

fn residual4(dst: &mut [u8], stride: usize, coefs: &[i32; 16], dc: i32, _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { residual4_impl(dst.as_mut_ptr(), stride, coefs, dc) }
}

unsafe fn residual4_impl(dst: *mut u8, stride: usize, coefs: &[i32; 16], dc: i32) {
    unsafe {
        let mut c0 = coefs16(coefs.as_ptr());
        let c1 = coefs16(coefs.as_ptr().add(8));
        if dc != NO_DC {
            c0 = i16x8_replace_lane::<0>(c0, dc as i16);
        }
        // Any AC nonzero? Zero lane 0, test the rest. (`v128_andnot(a, b)`
        // is `a & !b` — the operands sit the other way round from x86's
        // `_mm_andnot_si128`, which is `!a & b`.)
        let ac = v128_or(v128_andnot(c0, lane0_mask()), c1);
        if !v128_any_true(ac) {
            let d = i16x8_extract_lane::<0>(c0) as i32;
            if d != 0 {
                dc_add_impl(dst, stride, d, 4);
            }
            return;
        }
        let mut coeffs = [0i16; 16];
        v128_store(coeffs.as_mut_ptr() as *mut v128, c0);
        v128_store(coeffs.as_mut_ptr().add(8) as *mut v128, c1);
        idct4_add_impl(dst, stride, &coeffs);
    }
}

fn residual8(dst: &mut [u8], stride: usize, coefs: &[i32; 64], _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { residual8_impl(dst.as_mut_ptr(), stride, coefs) }
}

unsafe fn residual8_impl(dst: *mut u8, stride: usize, coefs: &[i32; 64]) {
    unsafe {
        let mut coeffs = [0i16; 64];
        let mut ac = i64x2_splat(0);
        for k in 0..8 {
            let c = coefs16(coefs.as_ptr().add(8 * k));
            v128_store(coeffs.as_mut_ptr().add(8 * k) as *mut v128, c);
            let masked = if k == 0 { v128_andnot(c, lane0_mask()) } else { c };
            ac = v128_or(ac, masked);
        }
        if !v128_any_true(ac) {
            let d = coeffs[0] as i32;
            if d != 0 {
                dc_add_impl(dst, stride, d, 8);
            }
            return;
        }
        idct8_add_impl(dst, stride, &coeffs);
    }
}
