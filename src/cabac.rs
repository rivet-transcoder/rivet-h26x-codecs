//! The CABAC arithmetic decoding engine (H.264 clause 9.3.3.2, H.265 clause
//! 9.3.4.3). The two standards share it exactly: the same 64×4 `rangeTabLps`
//! table, the same state transition tables, the same renormalisation, the same
//! bypass and terminate decisions. What differs is above this layer — the
//! context tables and initialisation values, the binarisations, and which
//! context each bin uses — and lives with each codec.
//!
//! # Register model
//!
//! This is the standard's own 9-bit register model — `range` in `[256, 510]`
//! after renormalisation, `offset` a 9-bit value — with the renormalisation
//! batched into one shift per decision and the bits pulled from a 64-bit
//! cache. Keeping the model exact means the engine's bit position is *the*
//! bit position defined by the standard, which is what the PCM sample and
//! end-of-substream paths need: at those points the caller reads raw bits
//! from where the arithmetic decoder stopped.
//!
//! Both of the per-bin routines are branchless. A bin carries something close
//! to a bit of entropy, which is the definition of unpredictable, and there
//! are millions of them in a frame; a branch on the decoded value mispredicts
//! often enough to cost more than computing both sides and masking. What the
//! tables below exist for is to keep that possible — to leave so little work
//! in either arm that the compiler has nothing worth branching over, and to
//! keep the chain from one decision's `range` to the next one's as short as
//! it can be made, since that chain is what the decoder waits on.

use crate::bitreader::BitReader;

/// `rangeTabLps[pStateIdx][qRangeIdx]` — H.264 Table 9-44 / H.265 Table 9-46.
#[rustfmt::skip]
pub static LPS_RANGE: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [ 95, 116, 137, 158], [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
    [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116], [ 66,  80,  95, 110],
    [ 62,  76,  90, 104], [ 59,  72,  86,  99], [ 56,  69,  81,  94], [ 53,  65,  77,  89],
    [ 51,  62,  73,  85], [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
    [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62], [ 35,  43,  51,  59],
    [ 33,  41,  48,  56], [ 32,  39,  46,  53], [ 30,  37,  43,  50], [ 29,  35,  41,  48],
    [ 27,  33,  39,  45], [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
    [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33], [ 19,  23,  27,  31],
    [ 18,  22,  26,  30], [ 17,  21,  25,  28], [ 16,  20,  23,  27], [ 15,  19,  22,  25],
    [ 14,  18,  21,  24], [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
    [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18], [ 10,  12,  15,  17],
    [ 10,  12,  14,  16], [  9,  11,  13,  15], [  9,  11,  12,  14], [  8,  10,  12,  14],
    [  8,   9,  11,  13], [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
    [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9], [  2,   2,   2,   2],
];

/// `transIdxLps` — H.264 Table 9-45 / H.265 Table 9-47.
#[rustfmt::skip]
pub static NEXT_STATE_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// The renormalisation shift for a range value: how many left shifts bring it
/// back into `[256, 510]`. `rangeTabLps` never exceeds 240 and the MPS range
/// never falls below 128, so one to seven covers both paths.
const fn norm_shift(v: u16) -> u16 {
    let mut r = v;
    let mut n = 0;
    while r < 256 && n < 8 {
        r <<= 1;
        n += 1;
    }
    n
}

/// `rangeTabLps`, each entry carrying everything the LPS arm needs: the LPS
/// range in the low byte, its renormalisation shift in the second, and the
/// range already renormalised above those. Indexed `(range & 0xc0) |
/// pStateIdx`.
///
/// Three things are folded in here, and each takes work off the chain from
/// one decision's range to the next one's. Masking `range` instead of
/// shifting it right, and letting `qRangeIdx` land in bits 6 and 7 where
/// `pStateIdx` does not reach, makes the index an `and` and an `or` — and the
/// `or`'s other half comes from the context byte, which was loaded long
/// before. Carrying the shift means the LPS arm never counts leading zeros.
/// Carrying the renormalised range means it does not shift by it either, so
/// what is left of that arm is three extractions from a single load, all of
/// them off to the side of the comparison.
static LPS: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut q = 0;
    while q < 4 {
        let mut p = 0;
        while p < 64 {
            let lps = LPS_RANGE[p][q] as u32;
            let shift = norm_shift(lps as u16) as u32;
            t[q * 64 + p] = lps | (shift << 8) | ((lps << shift) << 16);
            p += 1;
        }
        q += 1;
    }
    t
};

/// Both state transitions in one table: the next context byte after an MPS at
/// index `state + 128`, and the one after an LPS at index `127 - state`.
///
/// The two indices are one XOR apart. Complementing the context byte with the
/// LPS mask — all ones on the LPS path, all zeros on the MPS path — turns
/// `state + 128` into `127 - state` in wrapping `u8` arithmetic, so a single
/// lookup serves both. The same complemented byte carries the decoded bin in
/// its low bit: `valMps`, or its complement on the LPS path.
static MLPS_STATE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut s = 0;
    while s < 128 {
        let p = s >> 1;
        let mps = s as u8 & 1;
        t[s + 128] = (NEXT_STATE_MPS[p] << 1) | mps;
        let flipped = if p == 0 { 1 - mps } else { mps };
        t[127 - s] = (NEXT_STATE_LPS[p] << 1) | flipped;
        s += 1;
    }
    t
};

/// `transIdxMPS` (Table 9-45): the next state after decoding the MPS.
pub static NEXT_STATE_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// One context variable: `(pStateIdx << 1) | valMps`.
pub type Ctx = u8;

/// Initialise a context from H.264's `(m, n)` pair for slice QP `qp`
/// (H.264 9.3.1.1).
#[inline]
pub fn init_ctx_h264(m: i32, n: i32, qp: i32) -> Ctx {
    let pre = (((m * qp.clamp(0, 51)) >> 4) + n).clamp(1, 126);
    if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 }
}

/// Initialise a context from H.265's 8-bit `initValue` for slice QP `qp`
/// (H.265 9.3.2.2).
#[inline]
pub fn init_ctx_hevc(init_value: u8, qp: i32) -> Ctx {
    let slope = ((init_value >> 4) as i32) * 5 - 45;
    let offset = (((init_value & 15) as i32) << 3) - 16;
    let pre = (((slope * qp.clamp(0, 51)) >> 4) + offset).clamp(1, 126);
    if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 }
}

/// The arithmetic decoder over one slice's data.
///
/// The standard's 9-bit `codIOffset` sits at bit `bits` of `low`, with the
/// next `bits` bits of the stream prefetched below it: renormalising by `k`
/// bits is just `bits -= k`, and a comparison against `range` is a shift.
/// The exact bit position of the standard's model is
/// `consumed = fetched - bits`, which is what the PCM / end-of-substream
/// paths hand back to the plain bit reader.
pub struct Cabac<'a> {
    data: &'a [u8],
    /// Next byte to prefetch.
    pos: usize,
    /// `codIOffset << bits | prefetched bits`.
    low: u64,
    /// Prefetched bits below the offset.
    bits: u32,
    range: u32,
    /// Bits fetched into `low` so far (including the initial nine).
    fetched: u64,
    /// The plain bit reader, positioned at the engine's bit position on
    /// request ([`Cabac::reader`]).
    reader: BitReader<'a>,
}

impl<'a> Cabac<'a> {
    /// Start decoding at the beginning of `data` (which must begin at the
    /// byte-aligned first byte of `slice_data()` / a substream).
    pub fn new(data: &'a [u8]) -> Self {
        let mut c = Self { data, pos: 0, low: 0, bits: 0, range: 510, fetched: 0, reader: BitReader::new(data) };
        c.start_at(0);
        c
    }

    /// Initialise the arithmetic state at byte `byte` of the data.
    fn start_at(&mut self, byte: usize) {
        self.pos = byte;
        self.low = 0;
        self.bits = 0;
        // Absolute, so `consumed_bits` stays a position in `data` after a
        // re-initialisation mid-buffer (a second PCM block, a later substream).
        self.fetched = (byte as u64) * 8;
        self.range = 510;
        // The offset is the first nine bits: fetch 32, then treat 9 of them
        // as the offset and the rest as prefetch.
        self.refill();
        self.bits -= 9;
    }

    /// Append 32 bits (zeros past the end) to `low`.
    #[inline(always)]
    fn refill(&mut self) {
        let v = match self.data.get(self.pos..self.pos + 4) {
            Some(b) => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            None => self.refill_end(),
        };
        self.pos += 4;
        self.low = (self.low << 32) | v as u64;
        self.bits += 32;
        self.fetched += 32;
    }

    /// The same four bytes, assembled one at a time with zeros standing in
    /// for what is past the end of the slice.
    ///
    /// Out of line deliberately. It runs once at the end of a slice and
    /// inlining it planted its thirty-odd instructions and eight branches in
    /// the middle of every residual loop in both decoders, where the
    /// instruction cache has better uses for the space.
    #[cold]
    #[inline(never)]
    fn refill_end(&self) -> u32 {
        let mut v = 0u32;
        for i in 0..4 {
            v = (v << 8) | self.data.get(self.pos + i).copied().unwrap_or(0) as u32;
        }
        v
    }

    /// Bits consumed by the standard's decoding engine.
    #[inline(always)]
    fn consumed_bits(&self) -> u64 {
        self.fetched - self.bits as u64
    }

    /// Re-initialise the engine at the reader's current byte-aligned position
    /// (after PCM samples, or at the start of a new substream in the same
    /// buffer).
    pub fn reinit(&mut self) {
        self.reader.align();
        let byte = (self.reader.position() / 8) as usize;
        self.start_at(byte);
    }

    /// The bit reader underneath, positioned exactly where the arithmetic
    /// decoder has consumed to. Only meaningful right after a terminate bin
    /// decoded as 1 (PCM samples, end of substream), when the standard hands
    /// the bitstream back to plain bit reading.
    pub fn reader(&mut self) -> &mut BitReader<'a> {
        let consumed = self.consumed_bits();
        self.reader = BitReader::new(self.data);
        // Skip in chunks: `skip` takes u32.
        let mut left = consumed;
        while left > 0 {
            let n = left.min(1 << 30) as u32;
            self.reader.skip(n);
            left -= n as u64;
        }
        &mut self.reader
    }

    /// Whether the underlying data ran out (a malformed slice).
    pub fn overrun(&self) -> bool {
        self.consumed_bits() > (self.data.len() as u64) * 8
    }

    /// Bits consumed from the start of the buffer.
    pub fn position(&self) -> u64 {
        self.consumed_bits()
    }

    /// Decode one context-coded bin.
    ///
    /// Both arms are computed and then masked together rather than branched
    /// between. Written as `if`/`else` this is the same program, but the
    /// compiler reads two arms with a shift in each and decides a branch is
    /// the cheaper shape — which on a value this unpredictable it is not, by
    /// five per cent of a whole CABAC decode. Spelling the selection as
    /// arithmetic settles the question. The one branch left is the refill,
    /// taken roughly once in twenty bins and predicted.
    #[inline(always)]
    pub fn decision(&mut self, ctx: &mut Ctx) -> u32 {
        let state = *ctx as u32;
        debug_assert!(state < 128, "context byte is pStateIdx << 1 | valMps");
        let entry = LPS[((self.range & 0xc0) | (state >> 1)) as usize];
        // The MPS sub-interval, and where its top sits in the offset's frame.
        let mps_range = self.range - (entry & 0xff);
        let scaled = (mps_range as u64) << self.bits;
        // All ones when the offset lands in the LPS sub-interval.
        let take_lps = self.low >= scaled;
        let mask = 0u32.wrapping_sub(take_lps as u32);
        // The MPS range never falls below 128, so renormalising it is one
        // shift at most; the LPS arm reads its own from the table.
        let mps_shift = (mps_range < 256) as u32;
        let mps_norm = mps_range << mps_shift;
        self.low -= scaled & 0u64.wrapping_sub(take_lps as u64);
        self.range = mps_norm ^ ((mps_norm ^ (entry >> 16)) & mask);
        self.bits -= mps_shift ^ ((mps_shift ^ ((entry >> 8) & 0xff)) & mask);
        let next = (state as u8) ^ (mask as u8);
        *ctx = MLPS_STATE[next.wrapping_add(128) as usize];
        if self.bits < 8 {
            self.refill();
        }
        (next & 1) as u32
    }

    /// Decode one bypass bin — equiprobable, so as unpredictable as a bin
    /// gets, and masked rather than branched for the same reason.
    #[inline(always)]
    pub fn bypass(&mut self) -> u32 {
        self.bits -= 1;
        let scaled = (self.range as u64) << self.bits;
        let bin = (self.low >= scaled) as u64;
        self.low -= scaled & 0u64.wrapping_sub(bin);
        if self.bits < 8 {
            self.refill();
        }
        bin as u32
    }

    /// Decode `n` bypass bins (`n <= 16`) as an unsigned integer, MSB first.
    ///
    /// `n` bypass decisions in a row are one division: with the offset
    /// extended by the next `n` stream bits, the bins are the quotient by the
    /// range and the new offset is the remainder (each bypass step doubles the
    /// offset, appends a bit and subtracts the range when it fits — long
    /// division, one bit of quotient per step).
    #[inline]
    pub fn bypass_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if n < 4 {
            let mut v = 0u32;
            for _ in 0..n {
                v = (v << 1) | self.bypass();
            }
            return v;
        }
        debug_assert!(n <= 16);
        if self.bits < n + 1 {
            self.refill();
        }
        let keep = self.bits - n;
        let ext = self.low >> keep;
        let range = self.range as u64;
        let q = ext / range;
        let rem = ext - q * range;
        self.low = (rem << keep) | (self.low & ((1u64 << keep) - 1));
        self.bits = keep;
        if self.bits < 8 {
            self.refill();
        }
        q as u32
    }

    /// Decode a terminate bin. Returns 1 when the arithmetic codeword ends
    /// here (end of slice / substream, or PCM samples follow); the engine then
    /// stops and the reader is at the standard's bit position.
    #[inline]
    pub fn terminate(&mut self) -> u32 {
        self.range -= 2;
        let scaled = (self.range as u64) << self.bits;
        if self.low >= scaled {
            1
        } else {
            if self.range < 256 {
                self.range <<= 1;
                self.bits -= 1;
                if self.bits < 8 {
                    self.refill();
                }
            }
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_init_matches_the_standard_formulas() {
        // H.264: m=20, n=-15 at QP 26: pre = ((20*26)>>4) - 15 = 32 - 15 = 17
        // -> pStateIdx = 63-17 = 46, valMps = 0.
        assert_eq!(init_ctx_h264(20, -15, 26), 46 << 1);
        // pre > 63: m=0, n=100 -> pre=100 -> state = 100-64 = 36, mps 1.
        assert_eq!(init_ctx_h264(0, 100, 26), (36 << 1) | 1);
        // HEVC: initValue 154 (a common "0.5" value): slope=(9*5-45)=0,
        // offset=(10<<3)-16=64 -> pre=64 -> state 0, mps 1.
        assert_eq!(init_ctx_hevc(154, 30), 1);
    }

    #[test]
    fn engine_reads_nine_bits_at_init_and_terminates_on_a_flushed_codeword() {
        // A codeword produced by the standard's encoder for a single terminate
        // bin = 1 straight after init: EncodeFlush writes, after 7 renorm
        // shifts of an empty register, PutBit(0)+WriteBits(0b01, 2), i.e. the
        // codeword is 0000000 0 01 -> bytes 0x00 0x40 (then rbsp trailing).
        // Decoder: offset = read_bits(9) = 0; range = 510; terminate: range
        // 508, offset 0 < 508 -> 0? That is a *0* bin: with the low register
        // zero the encoder's flush cannot produce a 1. So build the case
        // where offset >= range: all-ones offset.
        let data = [0xff, 0xff, 0xff];
        let mut c = Cabac::new(&data);
        assert_eq!(c.position(), 9);
        assert_eq!(c.terminate(), 1);
        // No renormalisation on a 1: the position does not move.
        assert_eq!(c.position(), 9);
    }

    #[test]
    fn bypass_is_a_plain_binary_expansion_of_the_offset() {
        // With range 510 and offset 0, bypass bins reproduce the bits until
        // the offset catches up with the range: for a stream of zeros every
        // bypass is 0.
        let data = [0u8; 8];
        let mut c = Cabac::new(&data);
        for _ in 0..20 {
            assert_eq!(c.bypass(), 0);
        }
    }
}

#[cfg(test)]
mod engine_equivalence {
    use super::*;
    // The previous (spec-literal, one-bit-at-a-time) engine, kept as the
    // reference: the standard's flowcharts transcribed, renormalising one bit
    // at a time, with nothing folded into a table.
    pub struct OldCabac<'a> {
        reader: BitReader<'a>,
        range: u32,
        offset: u32,
    }

    impl<'a> OldCabac<'a> {
        /// Start decoding at the beginning of `data` (which must begin at the
        /// byte-aligned first byte of `slice_data()` / a substream).
        pub fn new(data: &'a [u8]) -> Self {
            let mut reader = BitReader::new(data);
            let offset = reader.bits(9);
            Self { reader, range: 510, offset }
        }

        /// Bits consumed from the start of the buffer.
        pub fn position(&self) -> u64 {
            self.reader.position()
        }

        /// Decode one context-coded bin.
        #[inline(always)]
        pub fn decision(&mut self, ctx: &mut Ctx) -> u32 {
            let state = *ctx;
            let p = (state >> 1) as usize;
            let mps = (state & 1) as u32;
            let lps = LPS_RANGE[p][((self.range >> 6) & 3) as usize] as u32;
            self.range -= lps;
            if self.offset < self.range {
                // Most probable symbol. Renormalisation is at most one shift here
                // (range is at least 128 after subtracting the LPS range).
                *ctx = (NEXT_STATE_MPS[p] << 1) | (mps as u8);
                if self.range < 256 {
                    self.range <<= 1;
                    self.offset = (self.offset << 1) | self.reader.bits(1);
                }
                mps
            } else {
                self.offset -= self.range;
                self.range = lps;
                let shift = self.range.leading_zeros() - 23;
                self.range <<= shift;
                self.offset = (self.offset << shift) | self.reader.bits(shift);
                let new_mps = if p == 0 { 1 - mps } else { mps };
                *ctx = (NEXT_STATE_LPS[p] << 1) | (new_mps as u8);
                1 - mps
            }
        }

        /// Decode one bypass bin (equiprobable).
        #[inline(always)]
        pub fn bypass(&mut self) -> u32 {
            self.offset = (self.offset << 1) | self.reader.bits(1);
            if self.offset >= self.range {
                self.offset -= self.range;
                1
            } else {
                0
            }
        }

        /// Decode `n` bypass bins as an unsigned integer, MSB first.
        #[inline]
        pub fn bypass_bits(&mut self, n: u32) -> u32 {
            let mut v = 0u32;
            for _ in 0..n {
                v = (v << 1) | self.bypass();
            }
            v
        }

        /// Decode a terminate bin. Returns 1 when the arithmetic codeword ends
        /// here (end of slice / substream, or PCM samples follow); the engine then
        /// stops and the reader is at the standard's bit position.
        #[inline]
        pub fn terminate(&mut self) -> u32 {
            self.range -= 2;
            if self.offset >= self.range {
                1
            } else {
                if self.range < 256 {
                    self.range <<= 1;
                    self.offset = (self.offset << 1) | self.reader.bits(1);
                }
                0
            }
        }
    }


    #[test]
    fn prefetching_engine_matches_reference() {
        let mut seed = 42u64;
        let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as u32 };
        for trial in 0..200 {
            let len = 4 + (lcg() % 200) as usize;
            let data: Vec<u8> = (0..len).map(|_| lcg() as u8).collect();
            let mut a = Cabac::new(&data);
            let mut b = OldCabac::new(&data);
            let mut ca = [0u8; 8];
            let mut cb = [0u8; 8];
            for i in 0..8 { ca[i] = (lcg() % 128) as u8; cb[i] = ca[i]; }
            for step in 0..(len * 8 + 40) {
                let op = lcg() % 10;
                let (x, y) = if op < 6 { let c = (lcg() % 8) as usize; (a.decision(&mut ca[c]), b.decision(&mut cb[c])) }
                    else if op < 8 { (a.bypass(), b.bypass()) }
                    else if op < 9 { let n = lcg() % 17; (a.bypass_bits(n), b.bypass_bits(n)) }
                    else { (a.terminate(), b.terminate()) };
                assert_eq!(x, y, "trial {trial} step {step} op {op}");
                assert_eq!(a.position(), b.position(), "position trial {trial} step {step}");
                assert_eq!(ca, cb);
                if op >= 9 && x == 1 { break; }
            }
        }
    }
}
