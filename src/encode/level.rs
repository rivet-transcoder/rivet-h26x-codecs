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
//! # What is not checked here
//!
//! Some limits constrain the encoder's *decisions*, not its parameters,
//! and choosing a level cannot satisfy them:
//!
//! - H.264 `MaxVmvR`, the vertical motion vector range.
//! - H.264 `MaxMvsPer2Mb` and `MinLumaBiPredSize` (level 3 and above:
//!   sub-8x8 partitions, bi-predicted ones especially).
//! - The per-picture `MinCR` / `MinCr` bound for anything but lossless.
//!
//! They applied equally to the constant these encoders used to write.

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
}

#[allow(clippy::too_many_arguments)]
const fn h264_row(idc: u8, name: &'static str, max_mbps: u64, max_fs: u64, max_dpb_mbs: u64, max_br: u64, max_cpb: u64, min_cr: u64) -> H264Row {
    H264Row { idc, name, max_mbps, max_fs, max_dpb_mbs, max_br, max_cpb, min_cr }
}

/// Table A-1 in the order the standard ranks it (A.3.1: a row nearer the
/// top is a lower level), 1b between 1 and 1.1.
const H264_LEVELS: [H264Row; 20] = [
    h264_row(10, "1", 1_485, 99, 396, 64, 175, 2),
    h264_row(9, "1b", 1_485, 99, 396, 128, 350, 2),
    h264_row(11, "1.1", 3_000, 396, 900, 192, 500, 2),
    h264_row(12, "1.2", 6_000, 396, 2_376, 384, 1_000, 2),
    h264_row(13, "1.3", 11_880, 396, 2_376, 768, 2_000, 2),
    h264_row(20, "2", 11_880, 396, 2_376, 2_000, 2_000, 2),
    h264_row(21, "2.1", 19_800, 792, 4_752, 4_000, 4_000, 2),
    h264_row(22, "2.2", 20_250, 1_620, 8_100, 4_000, 4_000, 2),
    h264_row(30, "3", 40_500, 1_620, 8_100, 10_000, 10_000, 2),
    h264_row(31, "3.1", 108_000, 3_600, 18_000, 14_000, 14_000, 4),
    h264_row(32, "3.2", 216_000, 5_120, 20_480, 20_000, 20_000, 4),
    h264_row(40, "4", 245_760, 8_192, 32_768, 20_000, 25_000, 4),
    h264_row(41, "4.1", 245_760, 8_192, 32_768, 50_000, 62_500, 2),
    h264_row(42, "4.2", 522_240, 8_704, 34_816, 50_000, 62_500, 2),
    h264_row(50, "5", 589_824, 22_080, 110_400, 135_000, 135_000, 2),
    h264_row(51, "5.1", 983_040, 36_864, 184_320, 240_000, 240_000, 2),
    h264_row(52, "5.2", 2_073_600, 36_864, 184_320, 240_000, 240_000, 2),
    h264_row(60, "6", 4_177_920, 139_264, 696_320, 240_000, 240_000, 2),
    h264_row(61, "6.1", 8_355_840, 139_264, 696_320, 480_000, 480_000, 2),
    h264_row(62, "6.2", 16_711_680, 139_264, 696_320, 800_000, 800_000, 2),
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
        if let Some(b) = self.au_bytes.filter(|_| self.profile_idc == 100) {
            if b * self.fps * row.min_cr > 384 * row.max_mbps {
                why.push(format!("a {b}-byte picture is above 384 * MaxMBPS / MinCR {} per picture", 384 * row.max_mbps / row.min_cr / self.fps));
            }
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

/// Level 8.5 (A.4.1): the label for a stream beyond every other level,
/// which the standard requires to be High tier.
const H265_LEVEL_8_5: Level = Level { idc: 255, high_tier: true, name: "8.5" };

/// The Table A.10 row for a format: `CpbVclFactor`, then
/// `FormatCapabilityFactor` in thousandths and `MinCrScaleFactor` in
/// tenths, so that every comparison stays in integers.
///
/// The profile is the one the format needs, which is what `write_ptl`
/// claims: Main for 8-bit 4:2:0, Main 10 up to 10 bits, and for
/// everything else the format range extensions profile that admits it —
/// the Monochrome profiles, Main 12, Main 4:2:2 10 and 12, Main 4:4:4 up
/// to 12. Past 12 bits the only such profiles are the 16-bit ones
/// (Monochrome 16, Main 4:4:4 16 Intra). `HbrFactor` is taken as 1 —
/// `BrVclFactor` then equals `CpbVclFactor` — the value every
/// non-intra range extensions profile has
/// (`general_lower_bit_rate_constraint_flag` 1), and the smaller one.
fn h265_format(chroma: ChromaFormat, depth: u32) -> (u64, u64, u64) {
    match (chroma, depth) {
        (ChromaFormat::Yuv420, 8) => (1_000, 1_500, 10),
        (ChromaFormat::Yuv420, 9..=10) => (1_000, 1_875, 10),
        (ChromaFormat::Yuv420, 11..=12) => (1_500, 2_250, 10),
        (ChromaFormat::Monochrome, 8) => (667, 1_000, 10),
        (ChromaFormat::Monochrome, 9..=10) => (833, 1_250, 10),
        (ChromaFormat::Monochrome, 11..=12) => (1_000, 1_500, 10),
        (ChromaFormat::Monochrome, _) => (1_333, 2_000, 10),
        (ChromaFormat::Yuv422, 8..=10) => (1_667, 2_500, 5),
        (ChromaFormat::Yuv422, 11..=12) => (2_000, 3_000, 5),
        (ChromaFormat::Yuv444, 8) => (2_000, 3_000, 5),
        (ChromaFormat::Yuv444, 9..=10) => (2_500, 3_750, 5),
        (ChromaFormat::Yuv444, 11..=12) => (3_000, 4_500, 5),
        _ => (4_000, 6_000, 5),
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
    /// `h265_format`.
    vcl_factor: u64,
    fcf_milli: u64,
    min_cr_scale_tenths: u64,
}

impl H265Stream {
    fn new(cfg: &Config, g: &h265_syntax::Geometry) -> Self {
        let (width, height) = (u64::from(g.coded_width), u64::from(g.coded_height));
        let fps = u64::from(cfg.fps.max(1));
        let (buffering_minus1, _) = h265_syntax::dpb(cfg);
        // Every reference a picture keeps is in its set and used by it: a
        // P picture's list 0, a B picture's two anchors.
        let total_curr = u64::from(if cfg.bframes > 0 { cfg.max_refs.max(2) } else { cfg.max_refs.max(1) });
        let (vcl_factor, fcf_milli, min_cr_scale_tenths) = h265_format(g.chroma, g.bit_depth);
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
            vcl_factor,
            fcf_milli,
            min_cr_scale_tenths,
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
        // A.4.1(a) to (c).
        if pic > row.max_luma_ps {
            why.push(format!("{pic} luma samples a picture is above MaxLumaPs {}", row.max_luma_ps));
        }
        if self.width * self.width > 8 * row.max_luma_ps || self.height * self.height > 8 * row.max_luma_ps {
            why.push(format!("{}x{} exceeds Sqrt(MaxLumaPs * 8) on a side", self.width, self.height));
        }
        // A.4.2(a).
        if pic * self.fps > row.max_luma_sr {
            why.push(format!("{} luma samples/s is above MaxLumaSr {}", pic * self.fps, row.max_luma_sr));
        }
        // A.4.1(d).
        if row.idc >= 150 && self.ctb < 32 {
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
        // A.4.1(g) and A.4.2(e), and the VCL HRD the NAL one implies.
        let f = self.vcl_factor;
        if let Some(r) = self.rate.filter(|&r| r > f * row.max_br[t]) {
            why.push(format!("{r} bits/s is above {} (MaxBR {} x {f})", f * row.max_br[t], row.max_br[t]));
        }
        if let Some(c) = self.cpb.filter(|&c| c > f * row.max_cpb[t]) {
            why.push(format!("a {c}-bit buffer is above {} (MaxCPB {} x {f})", f * row.max_cpb[t], row.max_cpb[t]));
        }
        // A.4.2(h): an access unit is at most FormatCapabilityFactor *
        // MaxLumaSr * (tr(n) - tr(n - 1)) / MinCr bytes, MinCr being
        // MinCrBase * MinCrScaleFactor / HbrFactor.
        if let Some(b) = self.au_bytes {
            let lhs = u128::from(b) * u128::from(self.fps) * u128::from(row.min_cr_base[t]) * u128::from(self.min_cr_scale_tenths) * 1_000;
            let rhs = u128::from(self.fcf_milli) * u128::from(row.max_luma_sr) * 10;
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
    /// picture in 16x16 blocks is beyond every level, and is labelled 8.5.
    #[test]
    fn h265_a_16x16_ctb_cannot_claim_level_5() {
        assert_eq!(h265_with(&cfg(3840, 2160, 30), Some(4)), H265_LEVEL_8_5);
        assert_eq!(h265_with(&cfg(1920, 1080, 30), Some(4)).name, "4", "below level 5 any CTB will do");
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

    /// The writers carry what the derivation chose — both H.265 parameter
    /// sets, which must agree — rather than any constant.
    #[test]
    fn the_parameter_sets_carry_the_derived_level() {
        for (c, idc) in [
            (cfg(176, 144, 15), 10),
            (Config { rate: RateControl::Bitrate { bps: 100_000 }, ..cfg(176, 144, 15) }, 9),
            (cfg(1920, 1080, 60), 42),
        ] {
            let g = h264_syntax::Geometry::new(&c);
            let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&h264_syntax::write_sps(&c, &g, 16, 16, None))).unwrap();
            assert_eq!(sps.level_idc, idc, "H.264 {}x{}@{}", c.width, c.height, c.fps);
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
    /// ride on top; five references move it to level 1.1. H.265 with four
    /// references and three B pictures declares eight buffers, level 1's
    /// `MaxDpbSize` at this size; five declare nine, level 2.
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
        for (refs, idc, buffers) in [(4u32, 30u8, 8u32), (5, 60, 9)] {
            let tag = format!("H.265 refs {refs} bframes 3");
            let c = Config {
                max_refs: refs,
                bframes: 3,
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
