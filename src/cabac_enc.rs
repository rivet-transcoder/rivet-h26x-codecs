//! The CABAC arithmetic *encoder* (H.264 clause 9.3.4, H.265 clause 9.3.4) —
//! the inverse of [`crate::cabac::Cabac`].
//!
//! # Sharing with the decoder
//!
//! The probability model is not duplicated here. `rangeTabLps` and the two
//! transition tables live in [`crate::cabac`] and this module reads them, so
//! there is exactly one copy of those numbers in the crate and no way for the
//! two directions to drift apart.
//!
//! What it does *not* borrow is the decoder's fused `LPS` table, indexed
//! `(range & 0xc0) | pStateIdx`. That table caches, alongside each LPS range,
//! the range *already renormalised* and the shift that renormalised it —
//! which is what lets the decoder pick a branchless arm. The encoder cannot
//! use either field, because its renormalisation is not a shift: it emits a
//! bit per doubling, and which bit depends on `low` at that moment. So it
//! reads `LPS_RANGE` — the spec table the decoder's cache is itself derived
//! from. One source, two shapes, no second copy of the data.
//!
//! # Register model
//!
//! The standard's own: a 9-bit `range` and a `low` that the spec's `PutBit`
//! keeps inside ten bits by subtracting 512 or 256 as it emits. Carries are
//! handled the standard's way too, with an outstanding-bits counter rather
//! than by propagating into already-written bytes: a run of undecided bits is
//! counted, and settled by the first bit that resolves them.
//!
//! # The state must match the decoder's, bit for bit
//!
//! Both sides choose `ctxIdxInc` from *reconstructed* neighbours, so both
//! sides must hold the same context byte at the same bin. This module updates
//! state with the same tables and the same rule as
//! [`crate::cabac::Cabac::decision`], and the round-trip test checks the
//! whole context array afterwards rather than only the bins. A divergence
//! here does not produce a wrong pixel; it produces a decoder that is reading
//! a different bin than the one written, and it surfaces hundreds of
//! macroblocks later as a stream that falls apart with no obvious cause.
//!
//! Nothing in the crate calls this yet — the encoder that will is being built
//! alongside it — so the module allows dead code. Drop the allow when the
//! first slice writer lands.
#![allow(dead_code)]

use crate::bitwriter::BitWriter;
use crate::cabac::{Ctx, LPS_RANGE, NEXT_STATE_LPS, NEXT_STATE_MPS};

/// The arithmetic encoder, writing into a caller's [`BitWriter`].
///
/// It borrows the writer rather than owning one because CABAC slice data
/// continues the same RBSP the slice header was written into, and the header
/// is plain bits.
pub struct CabacEncoder<'a> {
    w: &'a mut BitWriter,
    /// `codILow`.
    low: u32,
    /// `codIRange`.
    range: u32,
    /// `bitsOutstanding`: bits whose value is not yet decided, to be emitted
    /// as the complement of whichever bit settles them.
    outstanding: u32,
    /// `firstBitFlag`: the first `PutBit` produces no output, because the
    /// leading bit of the interval is implicit.
    first: bool,
}

impl<'a> CabacEncoder<'a> {
    /// Start encoding into `w`, which must be byte aligned (9.3.4.1). For
    /// H.264 that means after `cabac_alignment_one_bit`.
    pub fn new(w: &'a mut BitWriter) -> Self {
        debug_assert!(w.byte_aligned(), "CABAC data starts on a byte boundary");
        Self { w, low: 0, range: 510, outstanding: 0, first: true }
    }

    /// `PutBit(b)` (9.3.4.3): emit `b`, then settle every outstanding bit as
    /// its complement.
    #[inline]
    fn put_bit(&mut self, b: u32) {
        if self.first {
            self.first = false;
        } else {
            self.w.bit(b);
        }
        while self.outstanding > 0 {
            self.w.bit(b ^ 1);
            self.outstanding -= 1;
        }
    }

    /// `RenormE` (9.3.4.3): double the range until it is back in
    /// `[256, 510]`, emitting or deferring one bit each time.
    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                // Straddling the midpoint: the bit is not decided yet.
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
        debug_assert!(self.low < 1024, "codILow left its ten bits");
    }

    /// Encode one context-coded bin (9.3.4.2), advancing `ctx` exactly as
    /// [`crate::cabac::Cabac::decision`] advances it on the way back.
    pub fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        debug_assert!(bin <= 1);
        let state = *ctx;
        let p = (state >> 1) as usize;
        let mps = (state & 1) as u32;
        let lps = LPS_RANGE[p][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        if bin != mps {
            // The LPS sub-interval sits above the MPS one, so the low end
            // moves up by the whole MPS range.
            self.low += self.range;
            self.range = lps;
            let flipped = if p == 0 { mps ^ 1 } else { mps };
            *ctx = (NEXT_STATE_LPS[p] << 1) | flipped as u8;
        } else {
            *ctx = (NEXT_STATE_MPS[p] << 1) | mps as u8;
        }
        self.renorm();
    }

    /// Encode one bypass bin (9.3.4.4). No context, no renormalisation: the
    /// range is untouched and `low` takes the doubling instead.
    pub fn encode_bypass(&mut self, bin: u32) {
        debug_assert!(bin <= 1);
        self.low <<= 1;
        if bin != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
    }

    /// Encode `n` bypass bins (`n <= 32`) of `v`, most significant first —
    /// the inverse of [`crate::cabac::Cabac::bypass_bits`].
    pub fn encode_bypass_bits(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.encode_bypass((v >> i) & 1);
        }
    }

    /// Encode a terminate bin (9.3.4.5). A `1` ends the arithmetic codeword
    /// and flushes; nothing may be encoded afterwards.
    pub fn encode_terminate(&mut self, bin: u32) {
        debug_assert!(bin <= 1);
        self.range -= 2;
        if bin != 0 {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// `EncodeFlush` (9.3.4.6): close the interval and leave the writer at
    /// the bit position the standard hands back to plain bit reading.
    ///
    /// The final two bits are `((low >> 7) & 3) | 1`; the low bit forced to
    /// one is the `rbsp_stop_one_bit`'s counterpart inside the codeword —
    /// it is what makes the decoder's terminate read back as 1.
    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1);
        self.w.bits(2, ((self.low >> 7) & 3) | 1);
    }

    /// Bits written into the underlying writer so far.
    pub fn position(&self) -> u64 {
        self.w.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{Cabac, init_ctx_h264};

    /// What a bin is, for the round trip.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Op {
        /// Context-coded, against context `n` of the pool.
        Decision(usize, u32),
        Bypass(u32),
        BypassBits(u32, u32),
        /// `terminate` with a zero bin — the codeword continues.
        Terminate,
    }

    const POOL: usize = 24;

    fn contexts() -> [Ctx; POOL] {
        // A spread of initial states, from the real initialisation formula so
        // the states are ones that actually occur.
        let mut c = [0u8; POOL];
        for (i, v) in c.iter_mut().enumerate() {
            *v = init_ctx_h264(20 - (i as i32 % 40), (i as i32 * 7) % 90 - 45, 26);
        }
        c
    }

    /// Encode a sequence, decode it back, and require both the bins and the
    /// final context states to match. This is the whole correctness argument
    /// for the encoder: it is exact, it has no floor, and it does not depend
    /// on the machine.
    fn round_trip(ops: &[Op]) {
        let mut enc_ctx = contexts();
        let mut w = BitWriter::new();
        {
            let mut e = CabacEncoder::new(&mut w);
            for op in ops {
                match *op {
                    Op::Decision(i, b) => e.encode_decision(&mut enc_ctx[i], b),
                    Op::Bypass(b) => e.encode_bypass(b),
                    Op::BypassBits(n, v) => e.encode_bypass_bits(n, v),
                    Op::Terminate => e.encode_terminate(0),
                }
            }
            // A terminate of 1 both ends the codeword and flushes it.
            e.encode_terminate(1);
        }
        w.align_zero();
        let data = w.into_rbsp();

        let mut dec_ctx = contexts();
        let mut d = Cabac::new(&data);
        for (k, op) in ops.iter().enumerate() {
            match *op {
                Op::Decision(i, b) => {
                    assert_eq!(d.decision(&mut dec_ctx[i]), b, "op {k}: {op:?}");
                }
                Op::Bypass(b) => assert_eq!(d.bypass(), b, "op {k}: {op:?}"),
                Op::BypassBits(n, v) => assert_eq!(d.bypass_bits(n), v, "op {k}: {op:?}"),
                Op::Terminate => assert_eq!(d.terminate(), 0, "op {k}: {op:?}"),
            }
        }
        assert_eq!(d.terminate(), 1, "the closing terminate did not read back as 1");
        assert!(!d.overrun(), "decoder ran past what the encoder wrote");
        assert_eq!(
            enc_ctx, dec_ctx,
            "context states diverged: the two sides would desync on the next bin"
        );
    }

    #[test]
    fn round_trips_a_handful_of_bins() {
        round_trip(&[Op::Decision(0, 0)]);
        round_trip(&[Op::Decision(0, 1)]);
        round_trip(&[Op::Bypass(0)]);
        round_trip(&[Op::Bypass(1)]);
        round_trip(&[Op::Terminate, Op::Decision(3, 1), Op::Terminate]);
    }

    /// The carry path: long runs of one symbol drive `bitsOutstanding` up,
    /// and a run of bypass ones straddling the midpoint is what makes it
    /// count past one.
    #[test]
    fn round_trips_long_runs() {
        for bin in 0..2 {
            round_trip(&(0..500).map(|_| Op::Decision(0, bin)).collect::<Vec<_>>());
            round_trip(&(0..500).map(|_| Op::Bypass(bin)).collect::<Vec<_>>());
        }
        // Alternating, which keeps the state machine moving rather than
        // saturating at one end.
        round_trip(&(0..500).map(|i| Op::Decision(i % POOL, (i % 2) as u32)).collect::<Vec<_>>());
    }

    #[test]
    fn round_trips_multi_bit_bypass() {
        let mut ops = Vec::new();
        for n in 1..=16u32 {
            let hi = (1u32 << n) - 1;
            ops.push(Op::BypassBits(n, 0));
            ops.push(Op::BypassBits(n, hi));
            ops.push(Op::BypassBits(n, hi / 3));
            ops.push(Op::Decision(n as usize % POOL, n & 1));
        }
        round_trip(&ops);
    }

    /// The property test proper: pseudo-random sequences of every op, over a
    /// pool of contexts, so states are shared between bins the way they are
    /// in a real slice.
    #[test]
    fn round_trips_random_sequences() {
        let mut seed = 0x9e3779b9u64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for trial in 0..300 {
            let len = 1 + lcg() % 400;
            let mut ops = Vec::with_capacity(len as usize);
            for _ in 0..len {
                match lcg() % 10 {
                    0..=6 => ops.push(Op::Decision((lcg() as usize) % POOL, lcg() & 1)),
                    7 | 8 => ops.push(Op::Bypass(lcg() & 1)),
                    _ => {
                        let n = 1 + lcg() % 16;
                        ops.push(Op::BypassBits(n, lcg() & ((1 << n) - 1)));
                    }
                }
            }
            // A zero terminate every so often, where a slice would have one.
            if trial % 3 == 0 {
                ops.push(Op::Terminate);
            }
            round_trip(&ops);
        }
    }
}
