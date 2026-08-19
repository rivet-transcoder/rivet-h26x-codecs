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
//! There is one rung here, not four. `simd128` is a compile-time feature
//! rather than something detected at run time, so `Cpu::simd128` is set from
//! `cfg!(target_feature = "simd128")` and this module is compiled only when
//! that holds — but `H26X_NO_SIMD=1` still selects the scalar reference, so
//! the equivalence `tools/wasm.sh` checks is still checkable.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::h264::{H264Dsp, PRED_STRIDE};

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
