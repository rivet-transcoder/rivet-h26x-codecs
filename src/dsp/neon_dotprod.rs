//! The AArch64 dot-product rung: `sdot`, and the byte FIR built on it.
//!
//! ARMv8.2-A's `sdot` sums four 8-bit products into each 32-bit lane, which
//! is exactly the shape of an interpolation filter over 8-bit samples: sixteen
//! multiply-accumulates per instruction against the eight that `umlal` /
//! `umlsl` manage. The eight-tap HEVC luma filter fits in two `sdot`s per four
//! outputs, and the permutes that feed them are shared between output groups,
//! so eight outputs cost one load, one `eor`, three `tbl`s, four `sdot`s and
//! one `uzp1` where the baseline needs eight overlapping loads, eight
//! multiplies and a runtime branch per tap. Counted under qemu, that is 64.7%
//! fewer instructions on the horizontal-only luma fractions.
//!
//! It does not follow that every byte FIR wants this. H.264's six-tap was
//! written this way too and measured a wash — 2.4% *more* instructions across
//! the sixteen positions, and the per-position spread was wider than the
//! difference between the two rungs — because that kernel's taps are powers of
//! two either side of 20, so LLVM folds its widening into `uaddl` / `umlal`
//! pairs and the baseline is already about ten instructions per eight
//! outputs. The dot product only pays where the baseline is paying per tap.
//!
//! Two things about the instruction have to be worked around.
//!
//! `sdot` multiplies *signed* bytes, and samples are unsigned. Biasing them
//! by −128 (an `eor` with 0x80, which is what two's complement makes of the
//! subtraction) brings them into `i8` and shifts the result by a constant:
//! `sum(t[k] * (s[k] - 128))` is `sum(t[k] * s[k]) - 128 * sum(t)`, and every
//! filter here has a fixed tap sum (32 for H.264 luma, 64 for HEVC), so
//! seeding the accumulator with `128 * sum(t)` undoes it exactly. ARMv8.6-A's
//! `usdot` would take the unsigned operand directly and save the `eor`, which
//! is one instruction per sixteen samples — see the note at the bottom of this
//! comment for why that, and `usmmla`, are detected but not used.
//!
//! And the intrinsics are not on stable Rust. `vdotq_s32` sits behind the
//! unstable `stdarch_neon_dotprod` feature gate (rust-lang/rust#117224) and
//! `vusdotq_s32` behind `stdarch_neon_i8mm` (#117223), as of 1.96; the crate's
//! MSRV is 1.85 and it must build on stable, so [`sdot`] wraps the instruction
//! in `asm!` instead. That is a stable core-language feature, the wrapper has
//! the same signature the intrinsic will have, and swapping it for the real
//! one when it lands is a one-line change. `is_aarch64_feature_detected!` and
//! `#[target_feature(enable = "dotprod")]` are both already stable, so the
//! runtime ladder needs no such treatment. SVE and SVE2 have no stable
//! intrinsics at all — `svint32_t` does not exist in `core::arch::aarch64` on
//! stable — so there is nothing to write against and they are left alone.
//!
//! i8mm is detected but unused. `usdot` saves the bias `eor` and nothing else,
//! about one instruction in twenty here. `usmmla` computes a 2x2 tile of
//! 8-element dot products, four times `umlal`'s multiply count, but a single
//! FIR has one tap vector, so half of each tile is the same output computed
//! twice; for the four-tap chroma filter the tap vector can be staggered
//! across the two columns to make all four results distinct, and even then the
//! permutes and the un-interleaving of the scrambled output order cost more
//! than the byte-wide `umlal` kernel it would replace. The flag is there so a
//! future kernel that does fit the tile — a two-filter or two-block shape —
//! can be selected without touching detection again.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;
use std::arch::asm;

/// `acc + dot(a, b)` per 32-bit lane, over four signed byte products each:
/// the `vdotq_s32` that stable Rust does not have yet.
///
/// # Safety
/// The CPU must have `dotprod`; that is what the `target_feature` asserts,
/// and every caller reaches it through a [`super::Cpu`] flag.
#[target_feature(enable = "dotprod")]
#[inline]
pub unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    unsafe {
        asm!(
            "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    r
}

/// A filter's taps arranged for [`fir8`].
#[derive(Clone, Copy)]
pub struct Taps {
    /// Taps 0..4, replicated four times.
    lo: int8x16_t,
    /// Taps 4..8, replicated four times.
    hi: int8x16_t,
    /// `128 * sum(taps)`, the bias correction, broadcast.
    bias: int32x4_t,
}

impl Taps {
    /// Arrange up to eight taps (zero-padded) for `sdot`.
    #[inline]
    pub fn new(taps: &[i8]) -> Self {
        debug_assert!(taps.len() <= 8);
        let mut t = [0i8; 8];
        t[..taps.len()].copy_from_slice(taps);
        let sum: i32 = t.iter().map(|&c| c as i32).sum();
        let mut lo = [0i8; 16];
        let mut hi = [0i8; 16];
        for j in 0..4 {
            lo[j * 4..j * 4 + 4].copy_from_slice(&t[..4]);
            hi[j * 4..j * 4 + 4].copy_from_slice(&t[4..]);
        }
        unsafe { Taps { lo: vld1q_s8(lo.as_ptr()), hi: vld1q_s8(hi.as_ptr()), bias: vdupq_n_s32(128 * sum) } }
    }
}

/// The `tbl` pattern that gathers four outputs' worth of a four-tap group:
/// lane group `j` is samples `j..j+4`.
const GATHER: [u8; 16] = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6];

/// Eight consecutive outputs of a byte FIR of up to eight taps, exact in
/// 16 bits: `out[j] = sum(t[k] * p[j + k])`.
///
/// Reads the sixteen bytes at `p`, of which the first fifteen are what eight
/// outputs of an eight-tap filter need; the sixteenth is slack the callers'
/// bounds checks account for. Three permutes serve both halves — the group
/// that supplies outputs 0..4 with taps 4..8 is the same one that supplies
/// outputs 4..8 with taps 0..4 — so the cost is one load, one `eor`, three
/// `tbl`s, four `sdot`s and one `uzp1`.
///
/// The i16 narrowing is exact rather than saturating: no filter here can leave
/// the range, since the widest is H.264's, at most `42 * 255` and at least
/// `-10 * 255`.
///
/// # Safety
/// `p[0..16]` must be readable, and the CPU must have `dotprod`.
#[target_feature(enable = "dotprod")]
#[inline]
pub unsafe fn fir8(p: *const u8, t: &Taps) -> int16x8_t {
    unsafe {
        let s = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(p), vdupq_n_u8(0x80)));
        let i = vld1q_u8(GATHER.as_ptr());
        let g0 = vqtbl1q_s8(s, i);
        let g1 = vqtbl1q_s8(s, vaddq_u8(i, vdupq_n_u8(4)));
        let g2 = vqtbl1q_s8(s, vaddq_u8(i, vdupq_n_u8(8)));
        let lo = sdot(sdot(t.bias, g0, t.lo), g1, t.hi);
        let hi = sdot(sdot(t.bias, g1, t.lo), g2, t.hi);
        vuzp1q_s16(vreinterpretq_s16_s32(lo), vreinterpretq_s16_s32(hi))
    }
}
