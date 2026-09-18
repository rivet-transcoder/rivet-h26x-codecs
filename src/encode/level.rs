//! The level a stream claims, derived from what the stream is.
//!
//! A level is a promise about a whole stream: this many samples per
//! picture, this many per second, a decoded picture buffer no larger than
//! this, a coded picture buffer and bit rate within these bounds. A
//! decoder sizes itself from it before it has decoded a picture — the
//! H.264 decoder in this crate takes its DPB size from `level_idc` alone
//! when the SPS says nothing more (`h264::sps::Sps::level_max_dpb_frames`),
//! and so does libavcodec — and a player negotiates capability with it. So
//! both directions of error cost something. Claiming too little is a
//! conformance violation a decoder is entitled to act on; claiming too
//! much makes a hardware decoder that could play the stream refuse it.
//! These encoders used to write constants: H.264 level 5.1, which
//! over-claimed every stream below 4K, and H.265 level 4.0, which
//! under-claimed every stream above 1080p.
//!
//! What this derives instead is the **lowest** level whose limits admit
//! everything the encoder knows about the stream before writing a byte of
//! it. The limits are the standards' own (H.264 Annex A, clauses A.3.1 to
//! A.3.3, Tables A-1, A-2 and A-4; H.265 Annex A, clauses A.4.1 and A.4.2,
//! Tables A.8 to A.10), and each check below names the item it is.
//!
//! # What the derivation reads
//!
//! - **Picture size**: the coded size, the one the SPS declares. For H.264
//!   that is whole macroblocks (`MaxFS`, and each dimension within
//!   `Sqrt(MaxFS * 8)`). For H.265 it is `PicSizeInSamplesY` (`MaxLumaPs`,
//!   the same square-root rule).
//! - **Picture rate**, `Config::fps`: macroblocks (H.264 `MaxMBPS`) or luma
//!   samples (H.265 `MaxLumaSr`) per second, and the absolute ceiling `fR`
//!   puts on pictures per second — 172 below H.264 level 6, 300 at 6 and
//!   above and throughout H.265.
//! - **The decoded picture buffer the stream needs.** H.264: the
//!   `max_num_ref_frames` the SPS writes. The B pictures are not
//!   references, and the output-order DPB (C.4.5.2) outputs one directly
//!   when it has the lowest picture order count waiting, so references
//!   alone are enough, and `MaxDpbFrames` must hold that many. A stream
//!   with a declared buffer also carries `dpb_output_delay`, and output *at
//!   those times* keeps each B picture `bframes - 1` frame intervals after
//!   it is decoded, so the timed DPB peaks at `refs + bframes - 1`. H.265:
//!   the `sps_max_dec_pic_buffering_minus1 + 1` the SPS writes, against
//!   `MaxDpbSize` (Equation A-2).
//! - **The rate**, where one is known: the declared buffer's `BitRate` and
//!   `CpbSize` when there is one, otherwise the average-rate target. Both
//!   are held to the *VCL* factor (`cpbBrVclFactor`, `CpbVclFactor` and
//!   `BrVclFactor`), the stricter of the two. The stream declares only a
//!   NAL HRD, so the VCL HRD is the one the standard *infers* at the VCL
//!   factor (H.264 E.2.2, H.265 E.3.3), and the stream has to conform to
//!   that one too.
//! - **Lossless**: the stream is as large as its samples. H.264 codes
//!   every lossless macroblock as I_PCM, so the size is exact: raw samples
//!   plus the `mb_type` and alignment around them. For H.265's
//!   transquant-bypass it is the rate of content that does not compress.
//!   So a lossless stream is held to the raw rate, to a buffer that holds
//!   one raw picture, and to the per-picture size bound `MinCR` sets
//!   (8-bit H.264 High only — A.3.3 notes 3 and 4 exempt the deeper
//!   profiles) or `MinCr` sets (every H.265 profile).
//! - **A constant quantiser gets no rate term.** The only worst case that
//!   could be justified at a fixed QP is the raw rate, and it would label a
//!   1080p30 CQP stream H.264 level 6.2. So CQP streams are labelled by
//!   what the encoder controls — size, picture rate, DPB — which is also
//!   how x264 and x265 label a stream with no VBV. Whether a CQP stream's
//!   bit rate then fits the level is the caller's choice of quantiser.
//!
//! Structural limits apply as well:
//!
//! - H.264 levels 1 to 2 and 4.2 and above require `frame_mbs_only_flag`,
//!   so an interlaced stream lives in 2.1..=4.1.
//! - H.264 A.3.3(k): below level 6, a High 10, 4:2:2 or 4:4:4 picture of
//!   more than 1620 macroblocks may put at most `MaxFS / 4` of them in one
//!   slice. This encoder codes one slice per picture, so that sets the
//!   level floor for such pictures.
//! - H.265 levels 5 and above require a 32 or 64 coding tree block.
//! - H.265 references per picture (`NumPicTotalCurr`) are at most 8 at
//!   every level.
//!
//! # Tier
//!
//! H.265 levels 4 and up have a High tier, with larger rate and buffer
//! limits and nothing else. A decoder built for the Main tier refuses a
//! High-tier stream at any level, and many consumer decoders are exactly
//! that. So the derivation takes the lowest Main-tier level that admits
//! the stream. It goes to the High tier only when no Main-tier level up to
//! 6.2 carries the rate, and then takes the lowest High-tier level that
//! does.
//!
//! # Beyond the last level
//!
//! Both tables end at 6.2. (H.265 V9 adds 6.3 and 7.x, which few decoders
//! know; they are deliberately not offered.) A stream beyond 6.2:
//!
//! - **H.264**: refused by name, before a byte is written, with the limits
//!   it exceeds. H.264 has no label for such a stream, and writing 6.2
//!   would be the false claim this module exists to stop.
//! - **H.265**: written as **level 8.5** (`general_level_idc` 255, High
//!   tier), which is the standard's own label for a stream beyond every
//!   level (A.4.1: "a suitable label for bitstreams that can exceed the
//!   limits of all other specified levels"). It is not refused.
//!
//! # What the encoder keeps for its level
//!
//! Some limits constrain the encoder's *decisions*, not its parameters,
//! and no choice of level can meet them on its behalf. The encoder derives
//! the level first and then holds its decisions to it
//! ([`MotionLimits`]):
//!
//! - H.264 `MaxVmvR` (A.3.2(g)): every luma vector's vertical component
//!   within `[-MaxVmvR, MaxVmvR - 1/4]` frame samples — 64 at level 1, 512
//!   from 3.1 — halved in a field macroblock's own rows. The motion search
//!   clamps its window to it, as x264 does (`h264_me::search_rect`).
//! - H.264 `MaxMvsPer2Mb` (A.3.2(i)): at most 32 motion vectors in any
//!   two consecutive macroblocks at level 3, 16 from 3.1 — consecutive in
//!   decoding order, across slices and pictures. Each 8x8 quarter of a
//!   macroblock is held to an eighth of that, so no macroblock takes more
//!   than half and no pair more than all, whatever its neighbours hold:
//!   level 3 loses the bi-predicted 4x4 sub-macroblock, 3.1 every 4x4 and
//!   every bi-predicted sub-macroblock below 8x8. Only `--subparts` offers
//!   those shapes; everything else spends at most eight vectors.
//! - H.264 `MinLumaBiPredSize` 8x8 (A.3.3(e), Table A-4, from 3.1): no
//!   `B_Bi_8x4`, `B_Bi_4x8` or `B_Bi_4x4`, which the budget above already
//!   excludes and the search refuses by name as well.
//!
//! # What is not checked here
//!
//! - **The rate of a constant-quantiser stream.** See above: no bound
//!   below the raw rate can be justified, so it is left to the caller's
//!   quantiser, as x264 and x265 leave it without a VBV.
//! - **The per-picture `MinCR` / `MinCr` bound** for anything but
//!   lossless. At a declared buffer each picture is held to the buffer
//!   instead, which `h26xhrd` checks.
//! - **The buffer against the level.** `h26xhrd` (examples/) walks the
//!   coded picture buffer the stream *declares* and nothing else. It does
//!   not compare that buffer with the level's `MaxBR` / `MaxCPB`. The
//!   derivation here keeps the declared `BitRate` and `CpbSize` within
//!   the VCL factor times those limits by construction, so a stream that
//!   passes `h26xhrd` also meets its level's rate limits.

use crate::Result;
use crate::encode::h264_syntax;
use crate::encode::h265_syntax::{self, Cpb};
use crate::encode::{Config, FieldCoding, RateControl};
use crate::picture::ChromaFormat;

/// A level as a stream claims it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level {
    /// `level_idc` (H.264) or `general_level_idc` (H.265), as written.
    pub idc: u8,
    /// `general_tier_flag`: the H.265 High tier. Always false for H.264.
    pub high_tier: bool,
    /// The level number as the standards print it: "4.1", "1b", "8.5".
    pub name: &'static str,
}

/// The buffer a configuration declares — the one the encoders build, by
/// the same rule, so that what the level is checked against and what the
/// stream carries are the same numbers. `None` where the encoders refuse
/// the request, which they do before any level is written.
fn declared_cpb(cfg: &Config) -> Option<Cpb> {
    match (cfg.cpb_ms, cfg.rate) {
        (ms, RateControl::Bitrate { bps }) if ms > 0 => Cpb::new(bps, ms),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// H.264
// ---------------------------------------------------------------------------

/// One row of H.264 Table A-1, as this derivation uses it.
struct H264Row {
    /// `level_idc` for the High profiles (A.3.2: 1b is 9 there).
    idc: u8,
    name: &'static str,
    /// `MaxMBPS`, macroblocks per second.
    max_mbps: u64,
    /// `MaxFS`, macroblocks.
    max_fs: u64,
    /// `MaxDpbMbs`, macroblocks.
    max_dpb_mbs: u64,
    /// `MaxBR`, in units of `cpbBrVclFactor` bits per second.
    max_br: u64,
    /// `MaxCPB`, in units of `cpbBrVclFactor` bits.
    max_cpb: u64,
    /// `MinCR`.
    min_cr: u64,
    /// `MaxVmvR`, luma frame samples.
    max_vmv_r: i32,
    /// `MaxMvsPer2Mb`, 0 where the level sets none.
    max_mvs_per_2mb: u32,
}

#[allow(clippy::too_many_arguments)]
const fn h264_row(idc: u8, name: &'static str, max_mbps: u64, max_fs: u64, max_dpb_mbs: u64, max_br: u64, max_cpb: u64, min_cr: u64, max_vmv_r: i32, max_mvs_per_2mb: u32) -> H264Row {
    H264Row { idc, name, max_mbps, max_fs, max_dpb_mbs, max_br, max_cpb, min_cr, max_vmv_r, max_mvs_per_2mb }
}

/// Table A-1 in the order the standard ranks it (A.3.1: a row nearer the
/// top is a lower level), 1b between 1 and 1.1.
const H264_LEVELS: [H264Row; 20] = [
    h264_row(10, "1", 1_485, 99, 396, 64, 175, 2, 64, 0),
    h264_row(9, "1b", 1_485, 99, 396, 128, 350, 2, 64, 0),
    h264_row(11, "1.1", 3_000, 396, 900, 192, 500, 2, 128, 0),
    h264_row(12, "1.2", 6_000, 396, 2_376, 384, 1_000, 2, 128, 0),
    h264_row(13, "1.3", 11_880, 396, 2_376, 768, 2_000, 2, 128, 0),
    h264_row(20, "2", 11_880, 396, 2_376, 2_000, 2_000, 2, 128, 0),
    h264_row(21, "2.1", 19_800, 792, 4_752, 4_000, 4_000, 2, 256, 0),
    h264_row(22, "2.2", 20_250, 1_620, 8_100, 4_000, 4_000, 2, 256, 0),
    h264_row(30, "3", 40_500, 1_620, 8_100, 10_000, 10_000, 2, 256, 32),
    h264_row(31, "3.1", 108_000, 3_600, 18_000, 14_000, 14_000, 4, 512, 16),
    h264_row(32, "3.2", 216_000, 5_120, 20_480, 20_000, 20_000, 4, 512, 16),
    h264_row(40, "4", 245_760, 8_192, 32_768, 20_000, 25_000, 4, 512, 16),
    h264_row(41, "4.1", 245_760, 8_192, 32_768, 50_000, 62_500, 2, 512, 16),
    h264_row(42, "4.2", 522_240, 8_704, 34_816, 50_000, 62_500, 2, 512, 16),
    h264_row(50, "5", 589_824, 22_080, 110_400, 135_000, 135_000, 2, 512, 16),
    h264_row(51, "5.1", 983_040, 36_864, 184_320, 240_000, 240_000, 2, 512, 16),
    h264_row(52, "5.2", 2_073_600, 36_864, 184_320, 240_000, 240_000, 2, 512, 16),
    h264_row(60, "6", 4_177_920, 139_264, 696_320, 240_000, 240_000, 2, 8192, 16),
    h264_row(61, "6.1", 8_355_840, 139_264, 696_320, 480_000, 480_000, 2, 8192, 16),
    h264_row(62, "6.2", 16_711_680, 139_264, 696_320, 800_000, 800_000, 2, 8192, 16),
];

/// `cpbBrVclFactor` (Table A-2) for the profiles this encoder writes.
fn h264_vcl_factor(profile_idc: u8) -> u64 {
    match profile_idc {
        100 => 1_250,
        110 => 3_000,
        _ => 4_000, // 122 (High 4:2:2) and 244 (High 4:4:4 Predictive)
    }
}

/// What the level has to admit, read off the configuration and geometry.
struct H264Stream {
    /// `PicWidthInMbs`.
    wide: u64,
    /// `FrameHeightInMbs`.
    high: u64,
    /// The largest coded picture, in macroblocks: the frame, or the field
    /// where every picture is one. Slice limit A.3.3(k) counts this.
    pic_mbs: u64,
    fps: u64,
    profile_idc: u8,
    interlaced: bool,
    /// Frame buffers the stream needs.
    dpb: u64,
    /// Bits per second the stream is held to, where one is known.
    rate: Option<u64>,
    /// Bits of coded picture buffer it needs.
    cpb: Option<u64>,
    /// Bytes of the largest access unit, where that is known (lossless).
    au_bytes: Option<u64>,
}

impl H264Stream {
    fn new(cfg: &Config, g: &h264_syntax::Geometry) -> Self {
        let wide = u64::from(g.mbs_wide);
        // The SPS geometry is the frame's; a field geometry is half of it.
        let high = u64::from(if g.field_pic { g.mbs_high * 2 } else { g.mbs_high });
        let frame_mbs = wide * high;
        let fps = u64::from(cfg.fps.max(1));
        let profile_idc = h264_syntax::profile_idc(g);
        let refs = u64::from(cfg.max_refs);
        let cpb = declared_cpb(cfg);
        // The timed DPB: see the module documentation.
        let dpb = if cpb.is_some() && cfg.bframes > 0 { refs + u64::from(cfg.bframes) - 1 } else { refs };
        let (mut rate, mut buffer, mut au_bytes) = match (cpb, cfg.rate) {
            (Some(c), _) => (Some(c.bit_rate), Some(c.size), None),
            (None, RateControl::Bitrate { bps }) => (Some(u64::from(bps)), None, None),
            _ => (None, None, None),
        };
        if cfg.rate == RateControl::Lossless {
            // Every macroblock I_PCM: `RawMbBits` of samples (7.4.5), and
            // at most three bytes around them — `mb_type` (nine to eleven
            // bits of CAVLC, fewer bins of CABAC and its termination) and
            // the alignment to the byte — plus the slice header, the
            // parameter sets and SEIs of an IDR access unit.
            let (cw, ch) = g.chroma_mb();
            let depth = u64::from(g.bit_depth);
            let raw_mb_bits = 256 * depth + 2 * u64::from(cw * ch) * depth;
            let bytes = frame_mbs * (raw_mb_bits / 8 + 3) + 256;
            rate = Some(bytes * 8 * fps);
            buffer = Some(bytes * 8);
            au_bytes = Some(bytes);
        }
        let pic_mbs = if g.interlaced && cfg.field_coding == FieldCoding::Field { frame_mbs / 2 } else { frame_mbs };
        H264Stream { wide, high, pic_mbs, fps, profile_idc, interlaced: g.interlaced, dpb, rate, cpb: buffer, au_bytes }
    }

    /// Every limit of `row` this stream exceeds, in words — empty when the
    /// level admits it.
    fn exceeds(&self, row: &H264Row) -> Vec<String> {
        let mut why = Vec::new();
        let frame_mbs = self.wide * self.high;
        // fR (A.3): the least interval between pictures, whatever their size.
        let max_fps = if row.idc >= 60 { 300 } else { 172 };
        if self.fps > max_fps {
            why.push(format!("{} frames/s is above the {max_fps} fR allows", self.fps));
        }
        // A.3.2(a): a picture of PicSizeInMbs every 1/fps seconds.
        if frame_mbs * self.fps > row.max_mbps {
            why.push(format!("{} macroblocks/s is above MaxMBPS {}", frame_mbs * self.fps, row.max_mbps));
        }
        // A.3.2(c) to (e).
        if frame_mbs > row.max_fs {
            why.push(format!("{frame_mbs} macroblocks a frame is above MaxFS {}", row.max_fs));
        }
        if self.wide * self.wide > 8 * row.max_fs || self.high * self.high > 8 * row.max_fs {
            why.push(format!("{}x{} macroblocks exceeds Sqrt(MaxFS * 8) on a side", self.wide, self.high));
        }
        // A.3.2(f) and 7.4.2.1.1: MaxDpbFrames holds what the stream keeps.
        let max_dpb_frames = (row.max_dpb_mbs / frame_mbs.max(1)).min(16);
        if self.dpb > max_dpb_frames {
            why.push(format!("a DPB of {} frames is above MaxDpbFrames {max_dpb_frames}", self.dpb));
        }
        // A.3.3(g), and the VCL HRD the NAL one implies (E.2.2).
        let factor = h264_vcl_factor(self.profile_idc);
        if let Some(r) = self.rate.filter(|&r| r > factor * row.max_br) {
            why.push(format!("{r} bits/s is above {} (MaxBR {} x {factor})", factor * row.max_br, row.max_br));
        }
        if let Some(c) = self.cpb.filter(|&c| c > factor * row.max_cpb) {
            why.push(format!("a {c}-bit buffer is above {} (MaxCPB {} x {factor})", factor * row.max_cpb, row.max_cpb));
        }
        // A.3.3(j), the High profile only: an access unit is at most
        // 384 * MaxMBPS * (tr(n) - tr(n - 1)) / MinCR bytes.
        let min_cr_limit = |b: &u64| self.profile_idc == 100 && b * self.fps * row.min_cr > 384 * row.max_mbps;
        if let Some(b) = self.au_bytes.filter(min_cr_limit) {
            why.push(format!("a {b}-byte picture is above 384 * MaxMBPS / MinCR {} per picture", 384 * row.max_mbps / row.min_cr / self.fps));
        }
        // A.3.3(d) and Table A-4: frame_mbs_only_flag at 1..=2 and 4.2 up.
        if self.interlaced && (row.idc <= 20 || row.idc >= 42) {
            why.push("an interlaced stream needs a level from 2.1 to 4.1".to_string());
        }
        // A.3.3(k): one slice per picture.
        if self.profile_idc != 100 && row.idc < 60 && self.pic_mbs > 1_620 && self.pic_mbs > row.max_fs / 4 {
            why.push(format!("a {}-macroblock slice is above MaxFS / 4 = {}", self.pic_mbs, row.max_fs / 4));
        }
        why
    }
}

/// The level an H.264 stream from this configuration claims: the lowest in
/// Table A-1 that admits it (see the module documentation for exactly
/// what that reads). `g` is the geometry the SPS is written from.
///
/// Refuses, naming the limits exceeded, when no level admits the stream.
/// The encoder calls this before it writes anything, so the refusal comes
/// before a header that would have claimed a level the stream breaks.
pub fn h264(cfg: &Config, g: &h264_syntax::Geometry) -> Result<Level> {
    let s = H264Stream::new(cfg, g);
    if let Some(row) = H264_LEVELS.iter().find(|row| s.exceeds(row).is_empty()) {
        return Ok(Level { idc: row.idc, high_tier: false, name: row.name });
    }
    // The highest level the stream could have used names what it breaks:
    // 4.1 for an interlaced stream, since nothing above it is interlaced.
    let top = H264_LEVELS.iter().rev().find(|row| !s.interlaced || row.idc == 41).expect("4.1 is in the table");
    Err(crate::Error::unsupported(format!(
        "H.264 encode: no level admits this stream (level {}: {})",
        top.name,
        s.exceeds(top).join("; ")
    )))
}

/// What an H.264 level asks of the encoder's motion *decisions* rather
/// than of its parameters: limits no choice of level can meet on the
/// encoder's behalf, which the motion search therefore keeps for the
/// level the stream claims. Derived from that level — the level first,
/// then the search held to it — so the claim and the vectors cannot
/// disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionLimits {
    /// `MaxVmvR` (A.3.2(g)): every luma vector's vertical component lies
    /// in `[-max_vmv_r, max_vmv_r - 1/4]`, in luma *frame* samples — so a
    /// field macroblock, whose rows are every other frame row, has half
    /// the range in its own rows.
    pub max_vmv_r: i32,
    /// The most motion vectors (`MvCnt`, 8.4.1) one 8x8 quarter of a
    /// macroblock may carry: an eighth of `MaxMvsPer2Mb` (A.3.2(i)), so
    /// that a macroblock's four quarters carry at most half of it and any
    /// two consecutive macroblocks — the pairs the limit counts, in
    /// decoding order, across slices and pictures — at most all of it.
    /// Unlimited below level 3; four at 3; two from 3.1, where it rules
    /// out every 4x4 sub-partition and every bi-predicted one below 8x8.
    /// A direct 8x8 is at most two (`subMvCnt` counts only its first
    /// sub-partition), so it is always allowed.
    pub max_mvs_per_8x8: u32,
    /// `MinLumaBiPredSize` 8x8 (A.3.3(e), Table A-4, level 3.1 and up):
    /// no `B_Bi_8x4`, `B_Bi_4x8` or `B_Bi_4x4` sub-macroblock.
    pub no_bi_below_8x8: bool,
}

impl MotionLimits {
    /// No limit at all: what a context that searches no H.264 motion (the
    /// H.265 encoder's, which shares the context type) carries.
    pub const NONE: MotionLimits = MotionLimits { max_vmv_r: i32::MAX, max_mvs_per_8x8: u32::MAX, no_bi_below_8x8: false };

    /// The limits of the H.264 level whose `level_idc` is `idc` (Table
    /// A-1). A `level_idc` outside the table — which the encoder never
    /// writes — gets none.
    pub fn h264(idc: u8) -> MotionLimits {
        match H264_LEVELS.iter().find(|row| row.idc == idc) {
            Some(row) => MotionLimits {
                max_vmv_r: row.max_vmv_r,
                max_mvs_per_8x8: if row.max_mvs_per_2mb == 0 { u32::MAX } else { row.max_mvs_per_2mb / 8 },
                // Table A-4: from level 3.1, the first with a MaxMvsPer2Mb of 16.
                no_bi_below_8x8: row.max_mvs_per_2mb == 16,
            },
            None => MotionLimits::NONE,
        }
    }

    /// Whether an 8x8 quarter may be coded as `parts` sub-partitions (1, 2
    /// or 4) predicted from `lists` lists each (1, or 2 for bi-prediction).
    pub fn allows_sub_8x8(&self, parts: usize, lists: usize) -> bool {
        (parts == 1 || lists == 1 || !self.no_bi_below_8x8) && (parts * lists) as u64 <= u64::from(self.max_mvs_per_8x8)
    }

    /// The vertical range a search in rows of this kind may use, in full
    /// samples of those rows: `±(range - 1)`, so that a quarter-sample
    /// refinement of at most three quarters either way stays inside
    /// `[-range, range - 1/4]`.
    pub fn vertical_search(&self, field: bool) -> i32 {
        let range = if field { self.max_vmv_r / 2 } else { self.max_vmv_r };
        range.saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// H.265
// ---------------------------------------------------------------------------

/// One row of H.265 Tables A.8 and A.9. The two-element arrays are
/// `[Main tier, High tier]`, the High entry 0 below level 4, where there
/// is no High tier.
struct H265Row {
    /// `general_level_idc`: thirty times the level number.
    idc: u8,
    name: &'static str,
    /// `MaxLumaPs`, samples.
    max_luma_ps: u64,
    /// `MaxCPB`, in units of `CpbVclFactor` bits.
    max_cpb: [u64; 2],
    /// `MaxLumaSr`, samples per second.
    max_luma_sr: u64,
    /// `MaxBR`, in units of `BrVclFactor` bits per second.
    max_br: [u64; 2],
    /// `MinCrBase`.
    min_cr_base: [u64; 2],
}

#[allow(clippy::too_many_arguments)]
const fn h265_row(idc: u8, name: &'static str, max_luma_ps: u64, max_cpb: [u64; 2], max_luma_sr: u64, max_br: [u64; 2], min_cr_base: [u64; 2]) -> H265Row {
    H265Row { idc, name, max_luma_ps, max_cpb, max_luma_sr, max_br, min_cr_base }
}

const H265_LEVELS: [H265Row; 13] = [
    h265_row(30, "1", 36_864, [350, 0], 552_960, [128, 0], [2, 2]),
    h265_row(60, "2", 122_880, [1_500, 0], 3_686_400, [1_500, 0], [2, 2]),
    h265_row(63, "2.1", 245_760, [3_000, 0], 7_372_800, [3_000, 0], [2, 2]),
    h265_row(90, "3", 552_960, [6_000, 0], 16_588_800, [6_000, 0], [2, 2]),
    h265_row(93, "3.1", 983_040, [10_000, 0], 33_177_600, [10_000, 0], [2, 2]),
    h265_row(120, "4", 2_228_224, [12_000, 30_000], 66_846_720, [12_000, 30_000], [4, 4]),
    h265_row(123, "4.1", 2_228_224, [20_000, 50_000], 133_693_440, [20_000, 50_000], [4, 4]),
    h265_row(150, "5", 8_912_896, [25_000, 100_000], 267_386_880, [25_000, 100_000], [6, 4]),
    h265_row(153, "5.1", 8_912_896, [40_000, 160_000], 534_773_760, [40_000, 160_000], [8, 4]),
    h265_row(156, "5.2", 8_912_896, [60_000, 240_000], 1_069_547_520, [60_000, 240_000], [8, 4]),
    h265_row(180, "6", 35_651_584, [60_000, 240_000], 1_069_547_520, [60_000, 240_000], [8, 4]),
    h265_row(183, "6.1", 35_651_584, [120_000, 480_000], 2_139_095_040, [120_000, 480_000], [8, 4]),
    h265_row(186, "6.2", 35_651_584, [240_000, 800_000], 4_278_190_080, [240_000, 800_000], [6, 4]),
];

impl H265Row {
    /// A.4.1(a) to (c): what a coded luma picture of `width` by `height`
    /// breaks at this level, if anything.
    fn picture_exceeds(&self, width: u64, height: u64) -> Option<String> {
        if width * height > self.max_luma_ps {
            Some(format!("{} luma samples a picture is above MaxLumaPs {}", width * height, self.max_luma_ps))
        } else if width * width > 8 * self.max_luma_ps || height * height > 8 * self.max_luma_ps {
            Some(format!("{width}x{height} exceeds Sqrt(MaxLumaPs * 8) on a side"))
        } else {
            None
        }
    }
}

/// `general_level_idc` of the lowest level that requires a coding tree
/// block of 32 or more (A.4.1(d)): level 5.
const H265_CTB_32_FROM: u8 = 150;

/// Whether a coded luma picture of `width` by `height` is too large for
/// every level that still admits a 16x16 coding tree block — beyond level
/// 4.1's `MaxLumaPs` or its longest side (A.4.1(a) to (c)) — so that only
/// a CTB of 32 or more can code it at any level at all (A.4.1(d)).
/// `h265_syntax::Geometry::new` asks this before it takes a 16x16 CTB.
pub(crate) fn h265_beyond_ctb16(width: u32, height: u32) -> bool {
    let top = H265_LEVELS.iter().rev().find(|row| row.idc < H265_CTB_32_FROM).expect("levels below 5 are in the table");
    top.picture_exceeds(u64::from(width), u64::from(height)).is_some()
}

/// Level 8.5 (A.4.1): the label for a stream beyond every other level,
/// which the standard requires to be High tier.
const H265_LEVEL_8_5: Level = Level { idc: 255, high_tier: true, name: "8.5" };

/// The H.265 profile a stream is coded under: what `write_ptl` claims,
/// and the Table A.10 factors its levels are measured with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265Profile {
    /// `general_profile_idc`: 1 Main, 2 Main 10, 4 the format range
    /// extensions profiles.
    pub idc: u8,
    /// The profile's name, as Table A.2 prints it.
    pub name: &'static str,
    /// For `idc` 4, the nine Table A.2 constraint flags in syntax order:
    /// `general_max_12bit`, `max_10bit`, `max_8bit`, `max_422chroma`,
    /// `max_420chroma`, `max_monochrome`, `intra`, `one_picture_only` and
    /// `lower_bit_rate_constraint_flag`. Main and Main 10 carry none.
    pub flags: [bool; 9],
    /// `CpbVclFactor` (Table A.10).
    cpb_vcl_factor: u64,
    /// `FormatCapabilityFactor`, thousandths.
    fcf_milli: u64,
    /// `MinCrScaleFactor`, tenths.
    min_cr_scale_tenths: u64,
}

impl H265Profile {
    /// `HbrFactor` (A.4.2): 1 for Main and Main 10, `2 -
    /// general_lower_bit_rate_constraint_flag` for the format range
    /// extensions profiles.
    fn hbr_factor(&self) -> u64 {
        if self.idc == 4 { 2 - u64::from(self.flags[8]) } else { 1 }
    }
}

const fn rext(name: &'static str, flags: [u8; 9], cpb_vcl_factor: u64, fcf_milli: u64, min_cr_scale_tenths: u64) -> H265Profile {
    let mut f = [false; 9];
    let mut i = 0;
    while i < 9 {
        f[i] = flags[i] == 1;
        i += 1;
    }
    H265Profile { idc: 4, name, flags: f, cpb_vcl_factor, fcf_milli, min_cr_scale_tenths }
}

const H265_MAIN: H265Profile = H265Profile { idc: 1, name: "Main", flags: [false; 9], cpb_vcl_factor: 1_000, fcf_milli: 1_500, min_cr_scale_tenths: 10 };
const H265_MAIN_10: H265Profile = H265Profile { idc: 2, name: "Main 10", flags: [false; 9], cpb_vcl_factor: 1_000, fcf_milli: 1_875, min_cr_scale_tenths: 10 };
// Table A.2's rows (max 12, 10 and 8 bit; max 4:2:2, 4:2:0, monochrome;
// intra; one picture; lower bit rate) with Table A.10's factors.
const H265_MONOCHROME: H265Profile = rext("Monochrome", [1, 1, 1, 1, 1, 1, 0, 0, 1], 667, 1_000, 10);
const H265_MONOCHROME_10: H265Profile = rext("Monochrome 10", [1, 1, 0, 1, 1, 1, 0, 0, 1], 833, 1_250, 10);
const H265_MONOCHROME_12: H265Profile = rext("Monochrome 12", [1, 0, 0, 1, 1, 1, 0, 0, 1], 1_000, 1_500, 10);
const H265_MONOCHROME_16: H265Profile = rext("Monochrome 16", [0, 0, 0, 1, 1, 1, 0, 0, 1], 1_333, 2_000, 10);
const H265_MAIN_12: H265Profile = rext("Main 12", [1, 0, 0, 1, 1, 0, 0, 0, 1], 1_500, 2_250, 10);
const H265_MAIN_422_10: H265Profile = rext("Main 4:2:2 10", [1, 1, 0, 1, 0, 0, 0, 0, 1], 1_667, 2_500, 5);
const H265_MAIN_422_12: H265Profile = rext("Main 4:2:2 12", [1, 0, 0, 1, 0, 0, 0, 0, 1], 2_000, 3_000, 5);
const H265_MAIN_444: H265Profile = rext("Main 4:4:4", [1, 1, 1, 0, 0, 0, 0, 0, 1], 2_000, 3_000, 5);
const H265_MAIN_444_10: H265Profile = rext("Main 4:4:4 10", [1, 1, 0, 0, 0, 0, 0, 0, 1], 2_500, 3_750, 5);
const H265_MAIN_444_12: H265Profile = rext("Main 4:4:4 12", [1, 0, 0, 0, 0, 0, 0, 0, 1], 3_000, 4_500, 5);
// The intra-only profile a 13- or 14-bit picture in colour has, its
// "0 or 1" lower-bit-rate flag written 1 (HbrFactor 1, the stricter).
const H265_MAIN_444_16_INTRA: H265Profile = rext("Main 4:4:4 16 Intra", [0, 0, 0, 0, 0, 0, 1, 0, 1], 4_000, 6_000, 5);

/// The profile an H.265 stream of this configuration is coded under: the
/// one Table A.2 names for its format, the narrowest — which is also the
/// one the most decoders must accept, since a decoder of a wider profile
/// decodes every stream whose flags are at least its own (a Main 12
/// decoder decodes a Monochrome 12 stream, A.3.5). 8-bit 4:2:0 is Main
/// and 9- or 10-bit 4:2:0 Main 10; everything else is a format range
/// extensions profile: Monochrome to 8, 10, 12 and 16 bits, Main 12, Main
/// 4:2:2 10 (8 bits too) and 12, Main 4:4:4 at 8, 10 and 12.
///
/// Above 12 bits in colour the only such profile is Main 4:4:4 16 Intra,
/// which an all-intra stream (`gop` 0) is. An inter stream there has no
/// profile at all — High Throughput 4:4:4 14 would need wavefront
/// parallel processing, which this encoder does not write — and is coded
/// with the flags that describe it (no bit-depth limit below 16, its
/// chroma limits, not intra, lower bit rate): a combination Table A.2
/// reserves, which no decoder is required to accept, and says so.
pub fn h265_profile(cfg: &Config, g: &h265_syntax::Geometry) -> H265Profile {
    match (g.chroma, g.bit_depth) {
        (ChromaFormat::Yuv420, 8) => H265_MAIN,
        (ChromaFormat::Yuv420, 9..=10) => H265_MAIN_10,
        (ChromaFormat::Yuv420, 11..=12) => H265_MAIN_12,
        (ChromaFormat::Monochrome, 8) => H265_MONOCHROME,
        (ChromaFormat::Monochrome, 9..=10) => H265_MONOCHROME_10,
        (ChromaFormat::Monochrome, 11..=12) => H265_MONOCHROME_12,
        (ChromaFormat::Monochrome, _) => H265_MONOCHROME_16,
        (ChromaFormat::Yuv422, 8..=10) => H265_MAIN_422_10,
        (ChromaFormat::Yuv422, 11..=12) => H265_MAIN_422_12,
        (ChromaFormat::Yuv444, 8) => H265_MAIN_444,
        (ChromaFormat::Yuv444, 9..=10) => H265_MAIN_444_10,
        (ChromaFormat::Yuv444, 11..=12) => H265_MAIN_444_12,
        _ if cfg.gop == 0 => H265_MAIN_444_16_INTRA,
        (chroma, _) => H265Profile {
            name: "none (inter coding above 12 bits in colour)",
            flags: [false, false, false, chroma != ChromaFormat::Yuv444, chroma == ChromaFormat::Yuv420, false, false, false, true],
            ..H265_MAIN_444_16_INTRA
        },
    }
}

struct H265Stream {
    /// `pic_width_in_luma_samples` and `pic_height_in_luma_samples`.
    width: u64,
    height: u64,
    fps: u64,
    /// `CtbSizeY`.
    ctb: u64,
    /// `sps_max_dec_pic_buffering_minus1 + 1`.
    dpb: u64,
    /// `NumPicTotalCurr` at its largest.
    total_curr: u64,
    rate: Option<u64>,
    cpb: Option<u64>,
    au_bytes: Option<u64>,
    /// The profile, for its Table A.10 factors.
    profile: H265Profile,
}

impl H265Stream {
    fn new(cfg: &Config, g: &h265_syntax::Geometry) -> Self {
        let (width, height) = (u64::from(g.coded_width), u64::from(g.coded_height));
        let fps = u64::from(cfg.fps.max(1));
        let (buffering_minus1, _) = h265_syntax::dpb(cfg);
        // A slice's reference set lists what the picture predicts from,
        // every entry used by it: a P picture's list 0, a B picture's two
        // anchors.
        let total_curr = u64::from(if cfg.bframes > 0 { cfg.max_refs.max(2) } else { cfg.max_refs.max(1) });
        let profile = h265_profile(cfg, g);
        let (mut rate, mut cpb, mut au_bytes) = match (declared_cpb(cfg), cfg.rate) {
            (Some(c), _) => (Some(c.bit_rate), Some(c.size), None),
            (None, RateControl::Bitrate { bps }) => (Some(u64::from(bps)), None, None),
            _ => (None, None, None),
        };
        if cfg.rate == RateControl::Lossless {
            // Transquant bypass codes every sample. Content that does not
            // compress costs its raw size, which is the figure used: the
            // coded picture's samples at the declared depth.
            let (sw, sh) = g.chroma.subsampling();
            let chroma = if g.chroma == ChromaFormat::Monochrome { 0 } else { 2 * (width / u64::from(sw)) * (height / u64::from(sh)) };
            let bits = (width * height + chroma) * u64::from(g.bit_depth);
            rate = Some(bits * fps);
            cpb = Some(bits);
            au_bytes = Some(bits / 8);
        }
        H265Stream {
            width,
            height,
            fps,
            ctb: 1 << g.log2_ctb,
            dpb: u64::from(buffering_minus1) + 1,
            total_curr,
            rate,
            cpb,
            au_bytes,
            profile,
        }
    }

    /// Every limit of `row` at the tier `high` this stream exceeds.
    fn exceeds(&self, row: &H265Row, high: bool) -> Vec<String> {
        let t = usize::from(high);
        let mut why = Vec::new();
        if high && row.max_br[1] == 0 {
            why.push(format!("level {} has no High tier", row.name));
            return why;
        }
        let pic = self.width * self.height;
        // fR (A.4.2) is 1/300 at every level offered here.
        if self.fps > 300 {
            why.push(format!("{} pictures/s is above the 300 fR allows", self.fps));
        }
        why.extend(row.picture_exceeds(self.width, self.height));
        // A.4.2(a).
        if pic * self.fps > row.max_luma_sr {
            why.push(format!("{} luma samples/s is above MaxLumaSr {}", pic * self.fps, row.max_luma_sr));
        }
        // A.4.1(d).
        if row.idc >= H265_CTB_32_FROM && self.ctb < 32 {
            why.push(format!("a {}x{0} coding tree block is below the 32 level 5 and up require", self.ctb));
        }
        // A.4.1(e).
        if self.total_curr > 8 {
            why.push(format!("{} references per picture is above the 8 NumPicTotalCurr allows", self.total_curr));
        }
        // Equation A-2, maxDpbPicBuf 6.
        let max_dpb = if 4 * pic <= row.max_luma_ps {
            16
        } else if 2 * pic <= row.max_luma_ps {
            12
        } else if 4 * pic <= 3 * row.max_luma_ps {
            8
        } else {
            6
        };
        if self.dpb > max_dpb {
            why.push(format!("a DPB of {} pictures is above MaxDpbSize {max_dpb}", self.dpb));
        }
        // A.4.1(g) and A.4.2(e), and the VCL HRD the NAL one implies: the
        // rate at BrVclFactor = CpbVclFactor * HbrFactor, the buffer at
        // CpbVclFactor.
        let p = &self.profile;
        let (fb, fc) = (p.cpb_vcl_factor * p.hbr_factor(), p.cpb_vcl_factor);
        if let Some(r) = self.rate.filter(|&r| r > fb * row.max_br[t]) {
            why.push(format!("{r} bits/s is above {} (MaxBR {} x {fb})", fb * row.max_br[t], row.max_br[t]));
        }
        if let Some(c) = self.cpb.filter(|&c| c > fc * row.max_cpb[t]) {
            why.push(format!("a {c}-bit buffer is above {} (MaxCPB {} x {fc})", fc * row.max_cpb[t], row.max_cpb[t]));
        }
        // A.4.2(h): an access unit is at most FormatCapabilityFactor *
        // MaxLumaSr * (tr(n) - tr(n - 1)) / MinCr bytes, MinCr being
        // MinCrBase * MinCrScaleFactor / HbrFactor.
        if let Some(b) = self.au_bytes {
            let lhs = u128::from(b) * u128::from(self.fps) * u128::from(row.min_cr_base[t]) * u128::from(p.min_cr_scale_tenths) * 1_000;
            let rhs = u128::from(p.fcf_milli) * u128::from(row.max_luma_sr) * 10 * u128::from(p.hbr_factor());
            if lhs > rhs {
                why.push(format!("a {b}-byte picture is above FormatCapabilityFactor * MaxLumaSr / MinCr per picture"));
            }
        }
        why
    }
}

/// The level and tier an H.265 stream from this configuration claims:
/// the lowest Main-tier level of Tables A.8 and A.9 that admits it, else
/// the lowest High-tier level, else level 8.5 (see the module
/// documentation). `g` is the geometry the SPS is written from. Never
/// refuses: H.265 has a label for a stream beyond every level.
pub fn h265(cfg: &Config, g: &h265_syntax::Geometry) -> Level {
    let s = H265Stream::new(cfg, g);
    for high in [false, true] {
        if let Some(row) = H265_LEVELS.iter().find(|row| s.exceeds(row, high).is_empty()) {
            return Level { idc: row.idc, high_tier: high, name: row.name };
        }
    }
    H265_LEVEL_8_5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Access;
    use crate::encode::h264::H264Encoder;
    use crate::encode::h265::H265Encoder;

    fn cfg(width: u32, height: u32, fps: u32) -> Config {
        Config { width, height, fps, ..Config::default() }
    }

    fn h264_of(c: &Config) -> Result<Level> {
        h264(c, &h264_syntax::Geometry::new(c))
    }

    fn h264_name(c: &Config) -> &'static str {
        h264_of(c).unwrap_or_else(|e| panic!("{}x{}@{}: {e}", c.width, c.height, c.fps)).name
    }

    /// The H.265 level with the coding tree block the encoder's geometry
    /// chooses, or with a `2^log2_ctb` one and the coded size rounded up to
    /// it — the CTB is an input to the level (A.4.1(d)), so the tests hold
    /// it rather than inherit whatever `Geometry::new`'s policy is.
    fn h265_with(c: &Config, log2_ctb: Option<u32>) -> Level {
        let mut g = h265_syntax::Geometry::new(c);
        if let Some(l) = log2_ctb {
            let n = 1 << l;
            g.log2_ctb = l;
            g.coded_width = c.width.div_ceil(n) * n;
            g.coded_height = c.height.div_ceil(n) * n;
        }
        h265(c, &g)
    }

    fn h265_name(c: &Config) -> &'static str {
        let l = h265_with(c, None);
        assert!(!l.high_tier, "{}x{}@{}: High tier at level {}", c.width, c.height, c.fps, l.name);
        l.name
    }

    /// `MaxVmvR` by level (Table A-1), and the search window it leaves: a
    /// full-sample search within `±(range - 1)` whose quarter refinement
    /// cannot leave `[-range, range - 1/4]`, half the rows for a field
    /// macroblock.
    #[test]
    fn motion_limits_follow_table_a1() {
        for (idc, vmv) in [(10, 64), (9, 64), (11, 128), (20, 128), (21, 256), (30, 256), (31, 512), (52, 512), (60, 8192), (62, 8192)] {
            assert_eq!(MotionLimits::h264(idc).max_vmv_r, vmv, "level_idc {idc}");
        }
        assert_eq!(MotionLimits::h264(99), MotionLimits::NONE);
        let l = MotionLimits::h264(31);
        assert_eq!((l.vertical_search(false), l.vertical_search(true)), (511, 255));
        assert_eq!(MotionLimits::NONE.vertical_search(true), i32::MAX / 2 - 1);
        // MaxMvsPer2Mb over eight per 8x8 quarter, MinLumaBiPredSize from 3.1.
        for (idc, per_8x8, no_bi) in [(22, u32::MAX, false), (30, 4, false), (31, 2, true), (62, 2, true)] {
            let l = MotionLimits::h264(idc);
            assert_eq!((l.max_mvs_per_8x8, l.no_bi_below_8x8), (per_8x8, no_bi), "level_idc {idc}");
        }
        // (parts, lists) for 8x8 / 8x4 / 4x4, one list and two.
        let shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2)];
        let allowed = |idc| shapes.map(|(p, l)| MotionLimits::h264(idc).allows_sub_8x8(p, l));
        assert_eq!(allowed(22), [true; 6]);
        assert_eq!(allowed(30), [true, true, true, true, true, false], "level 3: no bi 4x4 (eight vectors)");
        assert_eq!(allowed(31), [true, true, true, false, false, false], "3.1: no 4x4, no bi below 8x8");
    }

    /// Frame sizes and rates against Table A-1, each at or just past a row's
    /// `MaxMBPS` or `MaxFS`. 176x144 at 15 is exactly level 1 (99
    /// macroblocks, 1485 a second); 1280x720 at 30 exactly 3.1; 1280x720
    /// at 60 exactly 3.2; 1080p30 is 244800 macroblocks a second, under
    /// level 4's 245760; and above 172 pictures a second only level 6 and
    /// up remain, whatever the size.
    #[test]
    fn h264_levels_follow_table_a1() {
        for (w, h, fps, want) in [
            (176, 144, 15, "1"),
            (176, 144, 30, "1.1"),
            (352, 288, 30, "1.3"),
            (640, 360, 30, "3"),
            (1280, 720, 30, "3.1"),
            (1280, 720, 60, "3.2"),
            (1920, 1080, 30, "4"),
            (1920, 1080, 31, "4.2"),
            (1920, 1080, 60, "4.2"),
            (3840, 2160, 30, "5.1"),
            (3840, 2160, 60, "5.2"),
            (1920, 1080, 240, "6"),
            (7680, 4320, 30, "6"),
            (7680, 4320, 60, "6.1"),
            (7680, 4320, 120, "6.2"),
        ] {
            assert_eq!(h264_name(&cfg(w, h, fps)), want, "{w}x{h}@{fps}");
        }
    }

    /// Past level 6.2 H.264 has nothing to claim, and the encoder says why
    /// rather than writing a level the stream breaks.
    #[test]
    fn h264_refuses_what_no_level_admits() {
        for (c, needle) in [
            (cfg(1920, 1080, 400), "300 fR allows"),
            (cfg(16384, 8704, 30), "MaxFS"),
            (Config { max_refs: 17, ..cfg(64, 64, 30) }, "MaxDpbFrames"),
            (Config { interlace: Some(crate::encode::FieldOrder::TopFirst), ..cfg(1920, 1080, 60) }, "level 4.1: "),
            (Config { rate: RateControl::Bitrate { bps: 1_100_000_000 }, ..cfg(64, 64, 30) }, "MaxBR"),
        ] {
            let err = h264_of(&c).expect_err(needle).to_string();
            assert!(err.contains(needle), "{}x{}@{}: {err}", c.width, c.height, c.fps);
            let enc = H264Encoder::new(c).err().expect("the encoder refuses it too").to_string();
            assert_eq!(enc, err, "the encoder's refusal is the derivation's");
        }
    }

    /// `MaxDpbFrames` against the references the SPS declares: level 1
    /// holds four QCIF frames, level 4 four 1080p ones. With a declared
    /// buffer the pictures also wait for their output times, and a B
    /// picture waits `bframes - 1` frame intervals.
    #[test]
    fn h264_references_and_timed_output_fill_the_dpb() {
        let qcif = cfg(176, 144, 15);
        assert_eq!(h264_name(&Config { max_refs: 4, ..qcif.clone() }), "1");
        assert_eq!(h264_name(&Config { max_refs: 5, ..qcif }), "1.1");
        let hd = cfg(1920, 1080, 30);
        assert_eq!(h264_name(&Config { max_refs: 4, ..hd.clone() }), "4");
        assert_eq!(h264_name(&Config { max_refs: 5, ..hd.clone() }), "5");
        let untimed = Config { max_refs: 2, bframes: 3, ..hd };
        assert_eq!(h264_name(&untimed), "4", "B pictures are output directly, not stored");
        let timed = Config { rate: RateControl::Bitrate { bps: 10_000_000 }, cpb_ms: 1000, ..untimed };
        assert_eq!(h264_name(&timed), "4", "2 references + 3 - 1 waiting B pictures");
        assert_eq!(h264_name(&Config { bframes: 4, ..timed }), "5", "2 + 4 - 1 = 5 frames");
    }

    /// The rate and buffer, held to `cpbBrVclFactor`: 1250 for High, so
    /// level 1 carries 80 kbit/s, 1b 160, 1.1 240; 3000 for High 10.
    #[test]
    fn h264_rate_and_buffer_raise_the_level() {
        let abr = |c: Config, bps: u32| Config { rate: RateControl::Bitrate { bps }, ..c };
        let qcif = cfg(176, 144, 15);
        assert_eq!(h264_name(&abr(qcif.clone(), 80_000)), "1");
        let l = h264_of(&abr(qcif.clone(), 100_000)).unwrap();
        assert_eq!((l.name, l.idc), ("1b", 9), "1b is level_idc 9 in the High profiles (A.3.2)");
        assert_eq!(h264_name(&abr(qcif.clone(), 170_000)), "1.1");
        assert_eq!(h264_name(&abr(Config { bit_depth: 10, ..qcif }, 170_000)), "1", "High 10: 64 x 3000");
        let hd = cfg(1920, 1080, 30);
        assert_eq!(h264_name(&abr(hd.clone(), 25_000_000)), "4");
        assert_eq!(h264_name(&abr(hd.clone(), 25_000_064)), "4.1");
        // A 1.6 s buffer at 20 Mbit/s is 32 Mbit, above level 4's 31.25.
        let buffered = |ms| Config { cpb_ms: ms, ..abr(hd.clone(), 20_000_000) };
        assert_eq!(h264_name(&buffered(1500)), "4");
        assert_eq!(h264_name(&buffered(1600)), "4.1");
    }

    /// Interlaced coding exists from level 2.1 to 4.1 only.
    #[test]
    fn h264_interlaced_streams_live_between_2_1_and_4_1() {
        let i = |c: Config| Config { interlace: Some(crate::encode::FieldOrder::TopFirst), ..c };
        assert_eq!(h264_name(&i(cfg(176, 144, 15))), "2.1");
        assert_eq!(h264_name(&i(cfg(1920, 1080, 30))), "4");
    }

    /// A.3.3(k): one slice of a High 10 picture above 1620 macroblocks is at
    /// most `MaxFS / 4` of them below level 6, so a single-slice 1080p High
    /// 10 stream is level 5.1, not 4. High (8-bit) has no such rule.
    #[test]
    fn h264_deep_single_slice_pictures_need_a_quarter_of_maxfs() {
        let deep = |c: Config| Config { bit_depth: 10, ..c };
        assert_eq!(h264_name(&cfg(1920, 1080, 30)), "4");
        assert_eq!(h264_name(&deep(cfg(1920, 1080, 30))), "5.1");
        assert_eq!(h264_name(&deep(cfg(1280, 720, 30))), "5");
        assert_eq!(h264_name(&deep(cfg(640, 360, 30))), "3", "920 macroblocks: the rule starts above 1620");
        assert_eq!(h264_name(&Config { chroma: ChromaFormat::Yuv444, ..cfg(1920, 1080, 30) }), "5.1");
    }

    /// Lossless is I_PCM: 64x64 at 30 is 1.5 Mbit/s of samples, past level
    /// 1.3's 960 kbit/s. The same picture at a quantiser is level 1.
    #[test]
    fn h264_lossless_is_held_to_its_raw_rate() {
        assert_eq!(h264_name(&cfg(64, 64, 30)), "1");
        assert_eq!(h264_name(&Config { rate: RateControl::Lossless, ..cfg(64, 64, 30) }), "2");
    }

    /// Picture sizes and rates against Tables A.8 and A.9, all Main tier.
    /// 1080p codes as 1920x1088, 2088960 samples, under level 4's
    /// 2228224; at 30 it is 62.7 M samples a second against 66.8 M.
    #[test]
    fn h265_levels_follow_tables_a8_and_a9() {
        for (w, h, fps, want) in [
            (176, 144, 15, "1"),
            (640, 360, 30, "2.1"),
            (1280, 720, 30, "3.1"),
            (1920, 1080, 30, "4"),
            (1920, 1080, 60, "4.1"),
            (7680, 4320, 60, "6.1"),
        ] {
            assert_eq!(h265_name(&cfg(w, h, fps)), want, "{w}x{h}@{fps}");
        }
        for (fps, want) in [(30, "5"), (60, "5.1"), (120, "5.2")] {
            let l = h265_with(&cfg(3840, 2160, fps), Some(5));
            assert_eq!((l.name, l.high_tier), (want, false), "2160p{fps}, 32x32 CTBs");
        }
    }

    /// A.4.1(d): level 5 and up need a 32 or 64 coding tree block. A 2160p
    /// picture in 16x16 blocks would be beyond every level — labelled 8.5 —
    /// and the encoder's geometry never builds one: at every coding tree
    /// depth, whole CTBs or the quadtree's, 2160p codes in 32x32 CTBs and
    /// claims level 5 at 30 pictures a second, 5.1 at 60.
    #[test]
    fn h265_a_16x16_ctb_cannot_claim_level_5() {
        assert_eq!(h265_with(&cfg(3840, 2160, 30), Some(4)), H265_LEVEL_8_5);
        assert_eq!(h265_with(&cfg(1920, 1080, 30), Some(4)).name, "4", "below level 5 any CTB will do");
        for depth in [Some(0), Some(1), None] {
            for (fps, want) in [(30, "5"), (60, "5.1")] {
                let c = Config { max_cu_depth: depth, ..cfg(3840, 2160, fps) };
                assert_eq!(h265_syntax::Geometry::new(&c).log2_ctb, 5, "2160p cu depth {depth:?}");
                assert_eq!(h265_name(&c), want, "2160p{fps} cu depth {depth:?}");
            }
        }
    }

    /// `Geometry::new` asks the level table where a 16x16 CTB stops being
    /// possible, and the answer is where level 4.1 ends: the largest
    /// picture at level 4.1, and one macroblock row or column past it,
    /// each way the limit can be crossed — `MaxLumaPs` 2,228,224 (2048x1088
    /// is exactly that) and the longest side, Sqrt(8 * MaxLumaPs) = 4222.
    #[test]
    fn h265_ctb16_stops_where_level_4_1_does() {
        for (w, h, beyond) in [
            (2048, 1088, false),
            (2048, 1096, true),
            (4222, 8, false),
            (4224, 8, true),
            (8, 4222, false),
            (8, 4224, true),
        ] {
            assert_eq!(h265_beyond_ctb16(w, h), beyond, "{w}x{h}");
            // The same picture in 16x16 CTBs, at one picture a second so
            // that only its size decides: level 4.1 or below, or none.
            let c = Config { max_cu_depth: Some(0), ..cfg(w, h, 1) };
            let mut g = h265_syntax::Geometry::new(&c);
            (g.log2_ctb, g.coded_width, g.coded_height) = (4, w, h);
            let l = h265(&c, &g);
            assert_eq!(l == H265_LEVEL_8_5, beyond, "{w}x{h} in 16x16 CTBs claims level {}", l.name);
        }
    }

    /// `MaxDpbSize` (Equation A-2) against `sps_max_dec_pic_buffering`:
    /// references + B pictures + the current one.
    #[test]
    fn h265_the_dpb_it_declares_decides_the_level() {
        let hd = cfg(1920, 1080, 30);
        assert_eq!(h265_name(&Config { max_refs: 1, bframes: 3, ..hd.clone() }), "4", "5 buffers of 6");
        assert_eq!(h265_name(&Config { max_refs: 4, bframes: 3, ..hd }), "5", "8 buffers");
        let qcif = cfg(176, 144, 15);
        assert_eq!(h265_name(&Config { max_refs: 4, bframes: 3, ..qcif.clone() }), "1", "8 buffers, 3/4 of MaxLumaPs");
        assert_eq!(h265_name(&Config { max_refs: 5, bframes: 3, ..qcif }), "2");
    }

    /// The rate picks the Main-tier level first; the High tier only when
    /// no Main-tier level carries it; level 8.5 past that.
    #[test]
    fn h265_rate_picks_the_level_then_the_tier() {
        let abr = |bps: u32| Config { rate: RateControl::Bitrate { bps }, ..cfg(1920, 1080, 30) };
        let got = |c: &Config| {
            let l = h265_with(c, None);
            (l.name, l.high_tier)
        };
        assert_eq!(got(&abr(20_000_000)), ("4.1", false));
        assert_eq!(got(&abr(25_000_000)), ("5", false), "Main 5 rather than High 4");
        assert_eq!(got(&abr(240_000_000)), ("6.2", false));
        assert_eq!(got(&abr(300_000_000)), ("6.1", true), "past Main 6.2's 240 Mbit/s");
        assert_eq!(h265_with(&abr(1_000_000_000), None), H265_LEVEL_8_5);
        // A 2 s buffer at 12 Mbit/s: 24 Mbit, past level 4.1's 20.
        assert_eq!(got(&Config { cpb_ms: 2000, ..abr(12_000_000) }), ("5", false));
        // 4:4:4 is Main 4:4:4, CpbVclFactor 2000: level 4 carries 24 Mbit/s.
        assert_eq!(got(&Config { chroma: ChromaFormat::Yuv444, ..abr(20_000_000) }), ("4", false));
    }

    /// What no level up to 6.2 admits is level 8.5, High tier (A.4.1).
    #[test]
    fn h265_beyond_every_level_is_level_8_5() {
        assert_eq!(H265_LEVEL_8_5, Level { idc: 255, high_tier: true, name: "8.5" });
        for c in [cfg(1920, 1080, 400), Config { max_refs: 9, ..cfg(64, 64, 30) }, cfg(16384, 8704, 30)] {
            assert_eq!(h265_with(&c, None), H265_LEVEL_8_5, "{}x{}@{} refs {}", c.width, c.height, c.fps, c.max_refs);
        }
    }

    /// 64x64 at 30 in 8-bit 4:2:0 is 1.47 Mbit/s of samples: level 2's 1.5.
    #[test]
    fn h265_lossless_is_held_to_its_raw_rate() {
        assert_eq!(h265_name(&cfg(64, 64, 30)), "1");
        assert_eq!(h265_name(&Config { rate: RateControl::Lossless, ..cfg(64, 64, 30) }), "2");
    }

    /// Each format's H.265 profile is Table A.2's row for it, flag for flag,
    /// in both parameter sets: `general_profile_idc`, then — for the format
    /// range extensions profiles — the nine constraint flags that follow the
    /// four source flags (bits 44..53 of the profile_tier_level, which the
    /// VPS starts at bit 32 and the SPS at bit 8). Main and Main 10 leave
    /// the 43 bits zero. The rows are the standard's, typed out again here
    /// rather than read from the table they check.
    #[test]
    fn h265_profiles_follow_table_a2() {
        let bit = |rbsp: &[u8], at: usize| rbsp[at / 8] >> (7 - at % 8) & 1 == 1;
        for (chroma, depth, gop, idc, name, row) in [
            (ChromaFormat::Yuv420, 8, 8, 1, "Main", "000000000"),
            (ChromaFormat::Yuv420, 10, 8, 2, "Main 10", "000000000"),
            (ChromaFormat::Yuv420, 12, 8, 4, "Main 12", "100110001"),
            (ChromaFormat::Monochrome, 8, 8, 4, "Monochrome", "111111001"),
            (ChromaFormat::Monochrome, 10, 8, 4, "Monochrome 10", "110111001"),
            (ChromaFormat::Monochrome, 12, 8, 4, "Monochrome 12", "100111001"),
            (ChromaFormat::Monochrome, 14, 8, 4, "Monochrome 16", "000111001"),
            (ChromaFormat::Yuv422, 8, 8, 4, "Main 4:2:2 10", "110100001"),
            (ChromaFormat::Yuv422, 10, 8, 4, "Main 4:2:2 10", "110100001"),
            (ChromaFormat::Yuv422, 12, 8, 4, "Main 4:2:2 12", "100100001"),
            (ChromaFormat::Yuv444, 8, 8, 4, "Main 4:4:4", "111000001"),
            (ChromaFormat::Yuv444, 10, 8, 4, "Main 4:4:4 10", "110000001"),
            (ChromaFormat::Yuv444, 12, 8, 4, "Main 4:4:4 12", "100000001"),
            (ChromaFormat::Yuv444, 14, 0, 4, "Main 4:4:4 16 Intra", "000000101"),
            (ChromaFormat::Yuv420, 14, 0, 4, "Main 4:4:4 16 Intra", "000000101"),
            (ChromaFormat::Yuv422, 14, 8, 4, "none (inter coding above 12 bits in colour)", "000100001"),
        ] {
            let tag = format!("{chroma:?} {depth}-bit gop {gop}");
            let c = Config { chroma, bit_depth: depth, gop, ..cfg(64, 64, 30) };
            let g = h265_syntax::Geometry::new(&c);
            let p = h265_profile(&c, &g);
            assert_eq!((p.idc, p.name), (idc, name), "{tag}");
            for (set, rbsp, ptl) in [
                ("VPS", crate::nal::unescape_rbsp(&h265_syntax::write_vps(&c, &g)), 32),
                ("SPS", crate::nal::unescape_rbsp(&h265_syntax::write_sps(&c, &g, 8, None)), 8),
            ] {
                assert_eq!(rbsp[ptl / 8] & 0x1f, idc, "{tag} {set}: general_profile_idc");
                assert!(bit(&rbsp, ptl + 8 + usize::from(idc)), "{tag} {set}: compatibility flag {idc}");
                let flags: String = (0..9).map(|k| if bit(&rbsp, ptl + 44 + k) { '1' } else { '0' }).collect();
                assert_eq!(flags, row, "{tag} {set}: Table A.2 flags");
                assert!((ptl + 53..ptl + 88).all(|k| !bit(&rbsp, k)), "{tag} {set}: reserved bits and inbld");
            }
        }
    }

    /// The writers carry what the derivation chose — both H.265 parameter
    /// sets, which must agree — rather than any constant.
    #[test]
    fn the_parameter_sets_carry_the_derived_level() {
        // With the DPB the decoder in this crate sizes from each: Table A-1's
        // MaxDpbMbs over the frame, at most 16 — level 6's 696320 is five
        // 8K frames.
        for (c, idc, dpb) in [
            (cfg(176, 144, 15), 10, 4),
            (Config { rate: RateControl::Bitrate { bps: 100_000 }, ..cfg(176, 144, 15) }, 9, 4),
            (cfg(1920, 1080, 60), 42, 4),
            (cfg(7680, 4320, 30), 60, 5),
        ] {
            let g = h264_syntax::Geometry::new(&c);
            let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&h264_syntax::write_sps(&c, &g, 16, 16, None))).unwrap();
            assert_eq!((sps.level_idc, sps.level_max_dpb_frames()), (idc, dpb), "H.264 {}x{}@{}", c.width, c.height, c.fps);
        }
        for (c, idc, tier) in [
            (cfg(176, 144, 15), 30, false),
            (cfg(1920, 1080, 30), 120, false),
            (Config { rate: RateControl::Bitrate { bps: 300_000_000 }, ..cfg(1920, 1080, 30) }, 183, true),
        ] {
            let g = h265_syntax::Geometry::new(&c);
            let vps = crate::hevc::sps::Vps::parse(&crate::nal::unescape_rbsp(&h265_syntax::write_vps(&c, &g))).unwrap();
            let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&h265_syntax::write_sps(&c, &g, 8, None))).unwrap();
            for (set, ptl) in [("VPS", &vps.ptl), ("SPS", &sps.ptl)] {
                assert_eq!((ptl.level_idc, ptl.tier), (idc, tier), "H.265 {set} {}x{}@{}", c.width, c.height, c.fps);
            }
        }
    }

    /// A smooth grating whose every 4x4 block moves its own way from one
    /// picture to the next — up to two samples each way — so the motion
    /// search, which converges on a grating, finds each block's own vector
    /// and sub-8x8 partitions pay: the content that spends the most motion
    /// vectors per macroblock.
    fn shattered(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut roll = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 5) as i32 - 2
        };
        let grating = |x: i32, y: i32| (40 + 4 * ((x.rem_euclid(25)) - 12).abs() + 3 * ((y.rem_euclid(27)) - 13).abs()) as u8;
        let mut at: Vec<(i32, i32)> = vec![(0, 0); (w / 4) * (h / 4)];
        (0..n)
            .map(|_| {
                let mut f = vec![128u8; w * h * 3 / 2];
                for by in 0..h / 4 {
                    for bx in 0..w / 4 {
                        let o = &mut at[by * (w / 4) + bx];
                        *o = (o.0 + roll(), o.1 + roll());
                        for y in 0..4 {
                            for x in 0..4 {
                                let (px, py) = (bx * 4 + x, by * 4 + y);
                                f[py * w + px] = grating(px as i32 + o.0, py as i32 + o.1);
                            }
                        }
                    }
                }
                f
            })
            .collect()
    }

    /// `MaxMvsPer2Mb` and `MinLumaBiPredSize` (A.3.2(i), A.3.3(e)), counted
    /// on the output. The same `--subparts` content at 352x288 is level 1.3
    /// at 30 pictures a second, 3 at 80 and 3.1 at 120 (MaxMBPS alone
    /// decides it) — the three regimes of those limits for the price of a
    /// CIF picture rather than a 720p one — at QP 4, where splitting is
    /// cheap. Each stream is decoded on one thread with the decoder's census
    /// on: every macroblock's `MvCnt` and whether it holds a bi-predicted
    /// partition below 8x8. At 1.3 nothing is limited, and the content does
    /// put more than 16 vectors in a macroblock, more than 32 in two and
    /// bi-predicts below 8x8 (measured: 32, 52 and 527 macroblocks), so the
    /// checks above it are not vacuous. At 3 every two consecutive
    /// macroblocks — across pictures too — hold at most 32, and bi-predicted
    /// 8x4s and 4x8s remain (3 has no `MinLumaBiPredSize`); at 3.1 at most
    /// 16, and nothing bi-predicted is smaller than 8x8. Every picture
    /// decodes to the encoder's reconstruction throughout.
    #[test]
    fn subpartitions_keep_the_levels_vector_limits() {
        let frames = shattered(352, 288, 5);
        for (fps, idc, pair_limit) in [(30u32, 13u8, None), (80, 30, Some(32u32)), (120, 31, Some(16))] {
            let tag = format!("352x288@{fps}");
            let c = Config {
                gop: 250,
                bframes: 1,
                subparts: true,
                rate: RateControl::ConstantQp(4),
                ..cfg(352, 288, fps)
            };
            let mut e = H264Encoder::new(c).unwrap();
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap());
            }
            units.extend(e.flush().unwrap());
            let stream: Vec<u8> = units.iter().flat_map(|u| u.data.iter().copied()).collect();
            let sps_nal = crate::nal::annexb_nals(&stream).find(|n| n[0] & 0x1f == 7).expect("an SPS");
            let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps_nal[1..])).unwrap();
            assert_eq!(sps.level_idc, idc, "{tag}");
            crate::h264::recon::MV_CENSUS.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let mut dec = crate::h264::H264Decoder::with_threads(1);
            dec.push_annexb(&stream).unwrap_or_else(|err| panic!("{tag}: {err}"));
            dec.flush().unwrap();
            let census = crate::h264::recon::MV_CENSUS.with(|c| c.borrow_mut().take()).expect("census on");
            round_trip(&tag, &units, e.reconstructions(), std::iter::from_fn(|| dec.next_picture().map(|p| p.into_packed())));
            assert_eq!(census.len(), 396 * frames.len(), "{tag}: every macroblock counted");
            let most = census.iter().map(|&(n, _)| n).max().unwrap();
            let pair = census.windows(2).map(|w| w[0].0 + w[1].0).max().unwrap();
            let bi_small = census.iter().filter(|&&(_, bi)| bi).count();
            match pair_limit {
                None => assert!(
                    most > 16 && pair > 32 && bi_small > 0,
                    "{tag}: the content spent at most {most} vectors on a macroblock, {pair} on two, bi-predicted below 8x8 in {bi_small}"
                ),
                Some(limit) => assert!(pair <= limit, "{tag}: two consecutive macroblocks hold {pair} vectors, above {limit}"),
            }
            match idc {
                30 => assert!(bi_small > 0, "{tag}: level 3 has no MinLumaBiPredSize, yet nothing bi-predicted below 8x8"),
                31 => assert_eq!(bi_small, 0, "{tag}: {bi_small} macroblocks bi-predict below 8x8"),
                _ => {}
            }
        }
    }

    /// Source pictures that move, so a P picture has reason to reach back
    /// through every reference it is allowed.
    fn moving(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let mut f = vec![128u8; w * h * 3 / 2];
                for y in 0..h {
                    for x in 0..w {
                        let (u, v) = (x + 3 * i, y + 2 * i);
                        f[y * w + x] = ((u * 7 + v * 13) ^ (u / 8 * 29 + v / 8 * 17)) as u8;
                    }
                }
                f
            })
            .collect()
    }

    /// Decode `units` and hold every picture, in display order, to the
    /// reconstruction the encoder kept for it; nothing more comes out.
    fn round_trip(tag: &str, units: &[Access], recon: &[Vec<u8>], pictures: impl Iterator<Item = Vec<u8>>) {
        let mut order: Vec<&Access> = units.iter().collect();
        order.sort_by_key(|u| u.display);
        let got: Vec<Vec<u8>> = pictures.collect();
        assert_eq!(got.len(), units.len(), "{tag}: pictures out against in");
        for (u, g) in order.iter().zip(&got) {
            assert!(*g == recon[u.encode_index as usize], "{tag}: picture {} differs from its reconstruction", u.display);
        }
    }

    /// The decoders size their buffers from what the encoder writes, and a
    /// stream at the edge of its level still decodes to the encoder's
    /// reconstructions. H.264 at 176x144 with four references is level 1,
    /// whose `MaxDpbFrames` is exactly four — the size this crate's decoder
    /// takes, the SPS carrying no `max_dec_frame_buffering` — and B pictures
    /// ride on top; five references move it to level 1.1. H.265 with two
    /// references and five B pictures declares eight buffers, level 1's
    /// `MaxDpbSize` at this size; six B pictures declare nine, level 2.
    /// (Two references: with three or more beside B pictures the encoder's
    /// B-picture reference sets drop an anchor the next P picture still
    /// uses, which libavcodec refuses and this crate's decoder forgives —
    /// a defect of its own, not of the level.)
    #[test]
    fn streams_at_a_dpb_boundary_round_trip() {
        let frames = moving(176, 144, 13);
        for (refs, bframes, idc, dpb) in [(4u32, 2u32, 10u8, 4u32), (5, 2, 11, 9), (4, 0, 10, 4)] {
            let tag = format!("H.264 refs {refs} bframes {bframes}");
            let c = Config { max_refs: refs, bframes, gop: 250, rate: RateControl::ConstantQp(30), ..cfg(176, 144, 15) };
            let mut e = H264Encoder::new(c).unwrap();
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap());
            }
            units.extend(e.flush().unwrap());
            let stream: Vec<u8> = units.iter().flat_map(|u| u.data.iter().copied()).collect();
            let sps_nal = crate::nal::annexb_nals(&stream).find(|n| n[0] & 0x1f == 7).expect("an SPS");
            let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps_nal[1..])).unwrap();
            assert_eq!((sps.level_idc, sps.level_max_dpb_frames(), sps.max_num_ref_frames), (idc, dpb, refs), "{tag}");
            let mut dec = crate::h264::H264Decoder::new();
            dec.push_annexb(&stream).unwrap_or_else(|err| panic!("{tag}: {err}"));
            dec.flush().unwrap();
            round_trip(&tag, &units, e.reconstructions(), std::iter::from_fn(|| dec.next_picture().map(|p| p.into_packed())));
        }
        for (bframes, idc, buffers) in [(5u32, 30u8, 8u32), (6, 60, 9)] {
            let tag = format!("H.265 refs 2 bframes {bframes}");
            let c = Config {
                max_refs: 2,
                bframes,
                gop: 250,
                rate: RateControl::ConstantQp(30),
                max_cu_depth: Some(0),
                ..cfg(176, 144, 15)
            };
            let mut e = H265Encoder::new(c).unwrap();
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap());
            }
            units.extend(e.flush().unwrap());
            let stream: Vec<u8> = units.iter().flat_map(|u| u.data.iter().copied()).collect();
            let sps_nal = crate::nal::annexb_nals(&stream).find(|n| (n[0] >> 1) & 0x3f == 33).expect("an SPS");
            let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps_nal[2..])).unwrap();
            assert_eq!((sps.ptl.level_idc, sps.max_dec_pic_buffering), (idc, buffers), "{tag}");
            let mut dec = crate::hevc::HevcDecoder::new();
            dec.push_annexb(&stream).unwrap_or_else(|err| panic!("{tag}: {err}"));
            dec.flush().unwrap();
            round_trip(&tag, &units, e.reconstructions(), std::iter::from_fn(|| dec.next_picture().map(|p| p.into_packed())));
        }
    }
}
