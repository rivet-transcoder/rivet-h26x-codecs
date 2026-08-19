//! MSB-first bit writer producing an RBSP — the inverse of
//! [`crate::bitreader::BitReader`].
//!
//! What comes out is the *unescaped* RBSP, exactly what `BitReader` expects
//! to be handed. Emulation prevention is a separate step
//! ([`crate::nal::escape_rbsp`]) for the same reason the decoder unescapes as
//! a separate step: the two are byte transformations over a finished RBSP,
//! not part of writing bits, and the encoder needs the unescaped form to
//! count byte positions in — slice data offsets, HEVC entry points — before
//! anything is escaped. [`BitWriter::into_nal`] does both where the caller
//! just wants a NAL payload.
//!
//! Writes are not fallible. There is no bound to overrun: the buffer grows,
//! and a value that does not fit the width it was given is a bug in the
//! caller rather than a property of the data, so it is a `debug_assert`.
//!
//! Nothing in the crate calls this yet — the encoder that will is being built
//! alongside it — so the module allows dead code. Drop the allow when the
//! first parameter-set writer lands. It is here to keep the tree
//! warning-free while the two halves arrive separately, not to hide an
//! unused API.
#![allow(dead_code)]

/// A bit writer accumulating into a byte buffer, MSB first.
///
/// Bits land in a 64-bit staging register and leave a byte at a time, so a
/// write is a shift and an or; the buffer only sees whole bytes.
#[derive(Default)]
pub struct BitWriter {
    out: Vec<u8>,
    /// Pending bits, right-aligned in the low `bits` positions.
    cache: u64,
    /// How many bits of `cache` are pending (always below 8 after a write).
    bits: u32,
}

impl BitWriter {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty writer with room for `bytes` reserved.
    pub fn with_capacity(bytes: usize) -> Self {
        Self { out: Vec::with_capacity(bytes), cache: 0, bits: 0 }
    }

    /// Bits written so far, including any not yet flushed to a whole byte.
    /// Mirrors [`crate::bitreader::BitReader::position`].
    #[inline(always)]
    pub fn position(&self) -> u64 {
        (self.out.len() as u64) * 8 + self.bits as u64
    }

    /// Whether the next bit would start a new byte.
    pub fn byte_aligned(&self) -> bool {
        self.bits == 0
    }

    /// Write `n` bits (0..=32) of `v`, most significant first.
    #[inline(always)]
    pub fn bits(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32, "width {n} out of range");
        debug_assert!(
            n == 32 || n == 0 || v < (1u32 << n),
            "value {v} does not fit in {n} bits"
        );
        if n == 0 {
            return;
        }
        let masked = if n == 32 { v as u64 } else { (v as u64) & ((1u64 << n) - 1) };
        self.cache = (self.cache << n) | masked;
        self.bits += n;
        while self.bits >= 8 {
            self.bits -= 8;
            self.out.push((self.cache >> self.bits) as u8);
        }
        // Keep only the pending bits, so the shift above never carries stale
        // high bits back in.
        self.cache &= (1u64 << self.bits) - 1;
    }

    /// Write one bit.
    #[inline(always)]
    pub fn bit(&mut self, v: u32) {
        self.bits(1, v & 1);
    }

    /// Write one bit as a flag.
    #[inline(always)]
    pub fn flag(&mut self, v: bool) {
        self.bits(1, v as u32);
    }

    /// Write `n` zero bits.
    pub fn zeros(&mut self, mut n: u32) {
        while n > 32 {
            self.bits(32, 0);
            n -= 32;
        }
        self.bits(n, 0);
    }

    /// Write `n` one bits.
    pub fn ones(&mut self, mut n: u32) {
        while n > 32 {
            self.bits(32, u32::MAX);
            n -= 32;
        }
        if n > 0 {
            self.bits(n, if n == 32 { u32::MAX } else { (1u32 << n) - 1 });
        }
    }

    /// Exp-Golomb `ue(v)`: `floor(log2(v + 1))` zeros, then `v + 1` itself.
    pub fn ue(&mut self, v: u32) {
        let k = v as u64 + 1;
        let zeros = 63 - k.leading_zeros();
        self.zeros(zeros);
        // `k` has `zeros + 1` significant bits, which is 33 for the largest
        // `u32` and so does not fit one write.
        if zeros + 1 <= 32 {
            self.bits(zeros + 1, k as u32);
        } else {
            self.bits(zeros + 1 - 32, (k >> 32) as u32);
            self.bits(32, k as u32);
        }
    }

    /// Exp-Golomb `se(v)`, mapped to the code number the reader unmaps:
    /// positive values take the odd code numbers, zero and negative the even.
    pub fn se(&mut self, v: i32) {
        let k = if v > 0 {
            2 * (v as i64) - 1
        } else {
            -2 * (v as i64)
        };
        debug_assert!(k >= 0 && k <= u32::MAX as i64, "se({v}) out of range");
        self.ue(k as u32);
    }

    /// Truncated Exp-Golomb `te(v)` with range `max`.
    pub fn te(&mut self, v: u32, max: u32) {
        if max > 1 {
            self.ue(v);
        } else {
            self.bit(1 - v);
        }
    }

    /// Pad with zero bits to the next byte boundary (no-op if aligned).
    pub fn align_zero(&mut self) {
        if self.bits != 0 {
            self.zeros(8 - self.bits);
        }
    }

    /// Pad with one bits to the next byte boundary (no-op if aligned) —
    /// `cabac_alignment_one_bit`, which precedes CABAC slice data.
    pub fn align_one(&mut self) {
        if self.bits != 0 {
            self.ones(8 - self.bits);
        }
    }

    /// `rbsp_trailing_bits()`: a stop bit, then zeros to the byte boundary.
    ///
    /// This is what makes the RBSP's end findable —
    /// [`crate::bitreader::BitReader::more_rbsp_data`] locates the last set
    /// bit in the payload and treats it as this one.
    pub fn rbsp_trailing_bits(&mut self) {
        self.bit(1);
        self.align_zero();
    }

    /// The RBSP written so far. Must be byte aligned: an RBSP is a whole
    /// number of bytes, and a caller with bits pending has forgotten
    /// [`BitWriter::rbsp_trailing_bits`].
    pub fn as_rbsp(&self) -> &[u8] {
        debug_assert!(self.byte_aligned(), "RBSP is not byte aligned");
        &self.out
    }

    /// The RBSP, taking ownership.
    pub fn into_rbsp(self) -> Vec<u8> {
        debug_assert!(self.byte_aligned(), "RBSP is not byte aligned");
        self.out
    }

    /// The RBSP with emulation prevention applied, ready to follow a NAL
    /// header in an Annex-B or length-prefixed stream.
    pub fn into_nal(self) -> Vec<u8> {
        crate::nal::escape_rbsp(&self.into_rbsp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn fixed_width_and_golomb_match_the_readers_fixture() {
        // The same bits `bitreader`'s own test reads back:
        // 1 | 010 | 011 | 00100 | 00101 | 1 | 0000
        let mut w = BitWriter::new();
        for v in 0..=4 {
            w.ue(v);
        }
        w.bit(1);
        assert_eq!(w.position(), 18);
        w.align_zero();
        assert_eq!(w.as_rbsp(), &[0b1010_0110, 0b0100_0010, 0b1100_0000]);
    }

    #[test]
    fn signed_golomb_matches_the_readers_fixture() {
        let mut w = BitWriter::new();
        for v in [1, -1, 2, -2] {
            w.se(v);
        }
        w.align_zero();
        assert_eq!(w.as_rbsp()[..2], [0b0100_1100, 0b1000_0101]);
    }

    /// Every writer has a reader that must undo it, including at the widths
    /// and magnitudes where the code changes shape: `ue` past sixteen leading
    /// zeros takes the reader's slow path, and `u32::MAX` needs 65 bits.
    #[test]
    fn round_trips_through_the_reader() {
        #[derive(Clone, Copy, Debug)]
        enum Op {
            Bits(u32, u32),
            Ue(u32),
            Se(i32),
            Te(u32, u32),
        }
        let mut ops = Vec::new();
        for n in 1..=32u32 {
            let hi = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            ops.push(Op::Bits(n, 0));
            ops.push(Op::Bits(n, hi));
            ops.push(Op::Bits(n, hi / 3));
        }
        for v in [0u32, 1, 2, 3, 14, 15, 16, 17, 254, 65534, 65535, 65536, 1 << 20, u32::MAX - 1] {
            ops.push(Op::Ue(v));
        }
        for v in [0i32, 1, -1, 2, -2, 100, -100, 32767, -32768, 1 << 20, -(1 << 20)] {
            ops.push(Op::Se(v));
        }
        for (v, max) in [(0u32, 1u32), (1, 1), (0, 5), (4, 5)] {
            ops.push(Op::Te(v, max));
        }
        // Interleave, so nothing is only ever exercised byte-aligned.
        let mut seed = 0x12345678u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            seed >> 16
        };
        for _ in 0..2000 {
            let n = 1 + lcg() % 32;
            let hi: u64 = 1u64 << n;
            ops.push(Op::Bits(n, (lcg() as u64 % hi) as u32));
            ops.push(Op::Ue(lcg() % 4096));
            ops.push(Op::Se(lcg() as i32 % 2048 - 1024));
        }

        let mut w = BitWriter::new();
        for op in &ops {
            match *op {
                Op::Bits(n, v) => w.bits(n, v),
                Op::Ue(v) => w.ue(v),
                Op::Se(v) => w.se(v),
                Op::Te(v, max) => w.te(v, max),
            }
        }
        w.rbsp_trailing_bits();
        let rbsp = w.into_rbsp();

        let mut r = BitReader::new(&rbsp);
        for (i, op) in ops.iter().enumerate() {
            match *op {
                Op::Bits(n, v) => assert_eq!(r.bits(n), v, "op {i}: {op:?}"),
                Op::Ue(v) => assert_eq!(r.ue(), v, "op {i}: {op:?}"),
                Op::Se(v) => assert_eq!(r.se(), v, "op {i}: {op:?}"),
                Op::Te(v, max) => assert_eq!(r.te(max), v, "op {i}: {op:?}"),
            }
        }
        assert!(!r.overrun(), "reader overran what the writer produced");
    }

    #[test]
    fn trailing_bits_are_where_the_reader_looks_for_them() {
        let mut w = BitWriter::new();
        w.bits(3, 0b101);
        w.rbsp_trailing_bits();
        let rbsp = w.into_rbsp();
        assert_eq!(rbsp, [0b1011_0000]);
        let mut r = BitReader::new(&rbsp);
        assert!(r.more_rbsp_data());
        r.bits(3);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn alignment_pads_with_what_it_says() {
        let mut w = BitWriter::new();
        w.bits(3, 0b101);
        w.align_one();
        assert_eq!(w.as_rbsp(), &[0b1011_1111]);
        let mut w = BitWriter::new();
        w.bits(3, 0b101);
        w.align_zero();
        assert_eq!(w.as_rbsp(), &[0b1010_0000]);
        // Aligned already: neither writes anything.
        let mut w = BitWriter::new();
        w.bits(8, 0xab);
        w.align_one();
        w.align_zero();
        assert_eq!(w.as_rbsp(), &[0xab]);
    }
}
