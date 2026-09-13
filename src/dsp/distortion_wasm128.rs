//! 128-bit SIMD versions of the distortion metrics for WebAssembly, 8-bit
//! samples — the same kernels as [`super::distortion_x86`], with the same
//! lane layout, on what `simd128` has for them.
//!
//! The porting doctrine is [`super::h264_wasm128`]'s. What differs from the
//! x86 file, kernel by kernel:
//!
//! - **SAD.** wasm has no `psadbw`. The absolute byte difference is two
//!   saturating subtractions or'd together (`u8x16_sub_sat(a, b) |
//!   u8x16_sub_sat(b, a)` — one of the two is zero, the other is `|a − b|`),
//!   and the fold to wider lanes is `u16x8_extadd_pairwise_u8x16`, which is
//!   NEON's `uaddlp` rather than x86's eight-bytes-to-one. A row's
//!   differences are summed in u16 lanes (at most four sixteen-byte chunks
//!   of 510 per lane for a 64-wide row) and folded into a u32 accumulator
//!   once a row, as the NEON kernel does.
//! - **SSD.** `i32x4_dot_i16x8` is `pmaddwd` exactly, so this is the SSE2
//!   kernel line for line: widen, subtract, dot the difference with itself,
//!   widen the row's sum to 64 bits once a row.
//! - **SATD.** The butterflies and the tile transpose are the x86 file's
//!   (the unpacks are `i16x8_shuffle` / `i32x4_shuffle` / `i64x2_shuffle`
//!   with the same lane orders), `i16x8_abs` is `pabsw`, and the
//!   `pmaddwd`-against-ones that widens a tile's sixteen absolute values is
//!   `i32x4_extadd_pairwise_i16x8`. Two tiles a vector, `[A, A, B, B]`
//!   after the fold, the per-tile `(sum + 1) >> 1` applied there.
//!
//! One rung, compiled only with `+simd128`; `H26X_NO_SIMD=1` still selects
//! the scalar reference, which is what `tools/wasm.sh` compares against.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::distortion::{DistortionDsp, sad_scalar, satd_scalar, ssd_scalar};

/// Replace the scalar entries of `d` with the simd128 kernels.
pub fn install(d: &mut DistortionDsp<u8>) {
    d.sad = sad;
    d.satd = satd;
    d.ssd = ssd;
}

/// Four bytes at `p` in the low lane of a vector, the rest zero.
#[inline]
unsafe fn load4(p: *const u8) -> v128 {
    unsafe { v128_load32_zero(p as *const u32) }
}

/// Eight bytes at `p` in the low half of a vector, the rest zero.
#[inline]
unsafe fn load8(p: *const u8) -> v128 {
    unsafe { v128_load64_zero(p as *const u64) }
}

/// `unpacklo_epi16` / `unpackhi_epi16`.
#[inline]
fn zip_lo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}
#[inline]
fn zip_hi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}
/// `unpacklo_epi32` / `unpackhi_epi32`.
#[inline]
fn zip_lo32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 4, 1, 5>(a, b)
}
#[inline]
fn zip_hi32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<2, 6, 3, 7>(a, b)
}
/// `unpacklo_epi64` / `unpackhi_epi64`.
#[inline]
fn zip_lo64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<0, 2>(a, b)
}
#[inline]
fn zip_hi64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<1, 3>(a, b)
}

/// `|a - b|` per byte.
#[inline]
fn absdiff8(a: v128, b: v128) -> v128 {
    v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a))
}

// ----------------------------------------------------------------------
// SAD
// ----------------------------------------------------------------------

unsafe fn sad_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = i32x4_splat(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut row = i16x8_splat(0);
            let mut x = 0;
            while x + 16 <= w {
                let d = absdiff8(
                    v128_load(ra.add(x) as *const v128),
                    v128_load(rb.add(x) as *const v128),
                );
                row = i16x8_add(row, u16x8_extadd_pairwise_u8x16(d));
                x += 16;
            }
            if x + 8 <= w {
                let d = absdiff8(load8(ra.add(x)), load8(rb.add(x)));
                row = i16x8_add(row, u16x8_extadd_pairwise_u8x16(d));
                x += 8;
            }
            if x + 4 <= w {
                let d = absdiff8(load4(ra.add(x)), load4(rb.add(x)));
                row = i16x8_add(row, u16x8_extadd_pairwise_u8x16(d));
            }
            acc = i32x4_add(acc, u32x4_extadd_pairwise_u16x8(row));
        }
        (i32x4_extract_lane::<0>(acc) as u32)
            .wrapping_add(i32x4_extract_lane::<1>(acc) as u32)
            .wrapping_add(i32x4_extract_lane::<2>(acc) as u32)
            .wrapping_add(i32x4_extract_lane::<3>(acc) as u32)
    }
}

fn sad(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h == 0 {
        return sad_scalar(a, a_stride, b, b_stride, w, h);
    }
    // The scalar reference indexes and would panic; this reads through
    // pointers, so the bound is checked once here.
    assert!(
        a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
        "block out of range"
    );
    unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

// ----------------------------------------------------------------------
// SSD
// ----------------------------------------------------------------------

/// Sum of squared differences of the eight byte pairs in the low halves
/// of `va` and `vb`, as four i32 (pairs of lanes summed).
#[inline]
fn sq8(va: v128, vb: v128) -> v128 {
    let d = i16x8_sub(u16x8_extend_low_u8x16(va), u16x8_extend_low_u8x16(vb));
    i32x4_dot_i16x8(d, d)
}

unsafe fn ssd_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u64 {
    unsafe {
        let mut acc = i64x2_splat(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            // One row's squares fit i32 lanes with room to spare (at most
            // 4 * 130050 per lane for a 64-wide row); widening once a row
            // keeps the 64-bit accumulator exact for any block.
            let mut row = i32x4_splat(0);
            let mut x = 0;
            while x + 16 <= w {
                let va = v128_load(ra.add(x) as *const v128);
                let vb = v128_load(rb.add(x) as *const v128);
                let dh = i16x8_sub(u16x8_extend_high_u8x16(va), u16x8_extend_high_u8x16(vb));
                row = i32x4_add(row, sq8(va, vb));
                row = i32x4_add(row, i32x4_dot_i16x8(dh, dh));
                x += 16;
            }
            if x + 8 <= w {
                row = i32x4_add(row, sq8(load8(ra.add(x)), load8(rb.add(x))));
                x += 8;
            }
            if x + 4 <= w {
                row = i32x4_add(row, sq8(load4(ra.add(x)), load4(rb.add(x))));
            }
            acc = i64x2_add(acc, u64x2_extend_low_u32x4(row));
            acc = i64x2_add(acc, u64x2_extend_high_u32x4(row));
        }
        (i64x2_extract_lane::<0>(acc) as u64).wrapping_add(i64x2_extract_lane::<1>(acc) as u64)
    }
}

fn ssd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u64 {
    if w % 4 != 0 || h == 0 {
        return ssd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(
        a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
        "block out of range"
    );
    unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

// ----------------------------------------------------------------------
// SATD
// ----------------------------------------------------------------------

/// The 4-point Hadamard butterfly, lane-wise across four vectors.
#[inline]
fn butterfly(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let s0 = i16x8_add(r0, r3);
    let s1 = i16x8_add(r1, r2);
    let s2 = i16x8_sub(r1, r2);
    let s3 = i16x8_sub(r0, r3);
    [
        i16x8_add(s0, s1),
        i16x8_add(s3, s2),
        i16x8_sub(s0, s1),
        i16x8_sub(s3, s2),
    ]
}

/// SATD of the two 4x4 tiles held in `r0..r3` (one row each, tile A in
/// the low half, B in the high), as `[A, A, B, B]` i32 with the per-tile
/// `(sum + 1) >> 1` applied.
#[inline]
fn satd_pair(r0: v128, r1: v128, r2: v128, r3: v128) -> v128 {
    let [t0, t1, t2, t3] = butterfly(r0, r1, r2, r3);
    // Transpose each tile: rows become columns, halves stay put.
    let u0 = zip_lo16(t0, t1);
    let u1 = zip_lo16(t2, t3);
    let u2 = zip_hi16(t0, t1);
    let u3 = zip_hi16(t2, t3);
    let v0 = zip_lo32(u0, u1);
    let v1 = zip_hi32(u0, u1);
    let v2 = zip_lo32(u2, u3);
    let v3 = zip_hi32(u2, u3);
    let c0 = zip_lo64(v0, v2);
    let c1 = zip_hi64(v0, v2);
    let c2 = zip_lo64(v1, v3);
    let c3 = zip_hi64(v1, v3);
    let [w0, w1, w2, w3] = butterfly(c0, c1, c2, c3);
    let s = i16x8_add(
        i16x8_add(i16x8_abs(w0), i16x8_abs(w1)),
        i16x8_add(i16x8_abs(w2), i16x8_abs(w3)),
    );
    // [A01, A23, B01, B23] -> [A, A, B, B], then the tile rounding.
    let p = i32x4_extadd_pairwise_i16x8(s);
    let q = i32x4_add(p, i32x4_shuffle::<1, 0, 3, 2>(p, p));
    u32x4_shr(i32x4_add(q, i32x4_splat(1)), 1)
}

/// A row of eight bytes from each side, as eight i16 differences.
#[inline]
unsafe fn diff8(a: *const u8, b: *const u8) -> v128 {
    unsafe {
        i16x8_sub(
            u16x8_extend_low_u8x16(load8(a)),
            u16x8_extend_low_u8x16(load8(b)),
        )
    }
}

/// A row of four bytes from each side in the low half, zeros above.
#[inline]
unsafe fn diff4(a: *const u8, b: *const u8) -> v128 {
    unsafe {
        i16x8_sub(
            u16x8_extend_low_u8x16(load4(a)),
            u16x8_extend_low_u8x16(load4(b)),
        )
    }
}

/// Rows `y` and `y + 4` of a four-wide block, four bytes each, in the two
/// halves — two tiles one above the other.
#[inline]
unsafe fn diff4x2(a: *const u8, sa: usize, b: *const u8, sb: usize) -> v128 {
    unsafe {
        let va = zip_lo32(load4(a), load4(a.add(4 * sa)));
        let vb = zip_lo32(load4(b), load4(b.add(4 * sb)));
        i16x8_sub(u16x8_extend_low_u8x16(va), u16x8_extend_low_u8x16(vb))
    }
}

unsafe fn satd_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = i32x4_splat(0);
        if w == 4 {
            let mut y = 0;
            while y + 8 <= h {
                let row = |r: usize| diff4x2(a.add((y + r) * sa), sa, b.add((y + r) * sb), sb);
                acc = i32x4_add(acc, satd_pair(row(0), row(1), row(2), row(3)));
                y += 8;
            }
            if y < h {
                let row = |r: usize| diff4(a.add((y + r) * sa), b.add((y + r) * sb));
                acc = i32x4_add(acc, satd_pair(row(0), row(1), row(2), row(3)));
            }
        } else {
            let mut y = 0;
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x + 8 <= w {
                    let row = |r: usize| diff8(ra.add(r * sa + x), rb.add(r * sb + x));
                    acc = i32x4_add(acc, satd_pair(row(0), row(1), row(2), row(3)));
                    x += 8;
                }
                if x < w {
                    let row = |r: usize| diff4(ra.add(r * sa + x), rb.add(r * sb + x));
                    acc = i32x4_add(acc, satd_pair(row(0), row(1), row(2), row(3)));
                }
                y += 4;
            }
        }
        // Lanes are [A, A, B, B] sums: one of each.
        (i32x4_extract_lane::<0>(acc) as u32).wrapping_add(i32x4_extract_lane::<2>(acc) as u32)
    }
}

fn satd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h % 4 != 0 || h == 0 {
        return satd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(
        a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
        "block out of range"
    );
    unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}
