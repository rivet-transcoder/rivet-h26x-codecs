//! Writing H.265 parameter sets and slice segment headers.
//!
//! The same shape as [`super::h264_syntax`], and deliberately so — the two
//! encoders share the picture scheduler and the verification standard, and
//! keeping the syntax layers parallel means a bug found in one is worth
//! looking for in the other.
//!
//! Where H.265 differs in a way that matters here:
//!
//! - There is a **video parameter set** above the sequence one. It carries
//!   almost nothing this encoder varies, but it is mandatory and a decoder
//!   that does not receive it will refuse the stream.
//! - The **profile-tier-level** structure is shared between the VPS and SPS
//!   and is 12 bytes of mostly-reserved fields, which is a lot of surface to
//!   get subtly wrong; it is written once here and used by both.
//! - **There is no CAVLC.** Everything below the slice segment header is
//!   CABAC, so unlike H.264 there is no simpler entropy path to bring up
//!   first. That is why this module stops at the header: the coding tree
//!   cannot be written until the CABAC slice writer exists, and emitting a
//!   header with nothing legal behind it would be worse than refusing.
//! - The coded size is a multiple of the **minimum coding block size**, not
//!   of a fixed macroblock, so the cropping arithmetic depends on the CTU
//!   configuration rather than being fixed at 16.

use crate::bitwriter::BitWriter;
use crate::encode::gop::Kind;
use crate::encode::{ColourDescription, Config};
use crate::hevc::slice::PredWeightTable;
use crate::picture::ChromaFormat;

/// Video parameter set.
pub const NAL_VPS: u8 = 32;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 33;
/// Picture parameter set.
pub const NAL_PPS: u8 = 34;
/// Coded slice of a non-IRAP picture, trailing, not referenced — a
/// sub-layer non-reference picture, which a decoder may discard without
/// affecting anything it decodes afterwards.
pub const NAL_TRAIL_N: u8 = 0;
/// Coded slice of a non-IRAP picture, trailing, referenced.
pub const NAL_TRAIL_R: u8 = 1;
/// Coded slice of an IDR picture with no leading pictures.
pub const NAL_IDR_N_LP: u8 = 20;
/// Supplemental enhancement information that precedes the pictures it
/// describes. The buffering period message rides here.
pub const NAL_PREFIX_SEI: u8 = 39;

/// Prefix a NAL payload with the two-byte H.265 header and a start code.
///
/// The header is `forbidden_zero`, six bits of type, six of layer id, three
/// of temporal id plus one — the last stored as `temporal_id + 1`, which is
/// the field people most often write as the raw value by mistake.
pub fn annexb(nal_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 6);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.push((nal_type & 0x3f) << 1);
    out.push(1); // nuh_layer_id 0, nuh_temporal_id_plus1 1
    out.extend_from_slice(payload);
    out
}

/// Coded geometry: H.265 codes in coding tree units, and the coded picture is
/// a whole number of *minimum* coding blocks rather than of CTUs.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    /// log2 of the coding tree block size. 6 means 64x64.
    pub log2_ctb: u32,
    /// log2 of the minimum coding block size. 3 means 8x8.
    pub log2_min_cb: u32,
    /// Coded luma width, a multiple of the minimum coding block size.
    pub coded_width: u32,
    /// See `coded_width`.
    pub coded_height: u32,
    /// Coding tree units across.
    pub ctbs_wide: u32,
    /// See `ctbs_wide`.
    pub ctbs_high: u32,
    /// Displayed luma width.
    pub width: u32,
    /// See `width`.
    pub height: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Bits per sample.
    pub bit_depth: u32,
}

impl Geometry {
    /// Derive the coded geometry from a configuration.
    ///
    /// The CTB size is chosen rather than configured. Where the coding
    /// quadtree may split (`max_cu_depth` above 0, the default), the CTB is
    /// 32x32 and the coded picture is the smallest legal one, a whole number
    /// of 8x8 minimum coding blocks: the CTBs along the right and bottom
    /// edges are partial, and the tree decisions and the writer produce the
    /// splits the reader infers there (`tree_steps` in `encode::h265`).
    /// Otherwise the coded picture is a whole number of CTBs, 16 or 32,
    /// whichever pads less, the larger on a tie — the geometry every stream
    /// had before partial CTBs. That is the rule at `max_cu_depth` 0, where
    /// a whole-CTB unit cannot be partial, and, under the quadtree, for a
    /// picture narrower *and* shorter than 64.
    ///
    /// That size exception is fitted to one measured clip, not derived.
    /// The 50x34 gate clip coded as partial 32x32 CTBs (56x40) against the
    /// whole 16x16 ones (64x48), per-plane YUV BD-rate at QP 22-40 (34-43
    /// in brackets): IP +1.08% (+1.66%), IPB +1.33% (+3.74%), AQ IP +8.58%
    /// (+11.75%), AQ IPB +7.51% (+11.40%), all-intra -0.56% (-0.62%). 32x32
    /// CTBs padded to whole ones (64x64) lost as much in AQ (IP +11.1%, IPB
    /// +10.6% at QP 22-40), so the AQ loss is CTB 32 with adaptive
    /// quantisation on a picture that small, not the partial geometry. The
    /// 88x44 gate clip, partial 32x32 CTBs against whole 16x16 ones, gains
    /// in every family measured (IP -7.6%, AQ IP -6.0% YUV at QP 22-40), as
    /// do 1280x720 and 3840x2160 (the module doc of `encode::h265`). No
    /// picture between 50x34 and 88x44 was measured: 64 is where the rule
    /// was drawn, not where the loss was found to end.
    ///
    /// The whole-CTB rule never takes 16 where the picture 16x16 CTBs would
    /// code is beyond level 4.1 (`beyond_level_4_1`). Levels 5 and above
    /// require a CTB of at least 32 (A.4.1 d), so such a stream would fit
    /// no level at all. 3840x2160 at `max_cu_depth` 0 was one: CTB 16 pads
    /// nothing there, so the rule took it. It now codes 3840x2176 in 32x32
    /// CTBs. Under the quadtree the CTB is 32 anyway, and the small-picture
    /// exception never gets near the limit.
    ///
    /// The conformance window crops whatever is coded beyond the requested
    /// size. The standard's CTB floor is 16 (an 8x8 CTB is illegal — this
    /// crate's own SPS parser rejects it, which is how that constraint was
    /// rediscovered), and the decision machinery's ceiling is 32; 64x64
    /// CTBs are not produced.
    pub fn new(cfg: &Config) -> Self {
        let log2_min_cb = 3;
        let tree = cfg.max_cu_depth.unwrap_or(crate::encode::h265::DEFAULT_CU_DEPTH) > 0;
        let small = cfg.width < 64 && cfg.height < 64;
        let (log2_ctb, coded_width, coded_height) = if tree && !small {
            let m = 1u32 << log2_min_cb;
            (5, cfg.width.div_ceil(m) * m, cfg.height.div_ceil(m) * m)
        } else {
            [5u32, 4]
                .into_iter()
                .map(|v| {
                    let n = 1u32 << v;
                    (v, cfg.width.div_ceil(n) * n, cfg.height.div_ceil(n) * n)
                })
                .filter(|&(v, w, h)| v > 4 || !beyond_level_4_1(w, h))
                .min_by_key(|&(v, w, h)| (w * h, u32::MAX - v))
                .unwrap()
        };
        let ctb = 1u32 << log2_ctb;
        Self {
            log2_ctb,
            log2_min_cb,
            coded_width,
            coded_height,
            ctbs_wide: coded_width.div_ceil(ctb),
            ctbs_high: coded_height.div_ceil(ctb),
            width: cfg.width,
            height: cfg.height,
            chroma: cfg.chroma,
            bit_depth: cfg.bit_depth,
        }
    }
}

/// Whether a coded luma picture of `width` by `height` exceeds level 4.1's
/// limits (A.4.1, Table A.8): more than `MaxLumaPs` = 2,228,224 samples, or
/// a side longer than `sqrt(8 * MaxLumaPs)`, 4222. Only levels 5 and up
/// admit such a picture, and they require a CTB of 32 or more. Read off
/// the level table the parameter sets' level is chosen from
/// (`encode::level`), so the two cannot disagree about where level 4.1
/// ends.
fn beyond_level_4_1(width: u32, height: u32) -> bool {
    crate::encode::level::h265_beyond_ctb16(width, height)
}

fn chroma_idc(c: ChromaFormat) -> u32 {
    match c {
        ChromaFormat::Monochrome => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// What the decoded picture buffer must hold: every picture this encoder
/// may still reference, plus every picture held back waiting for output
/// because something coded after it displays first.
///
/// Returned as the pair the parameter sets write —
/// `max_dec_pic_buffering_minus1` and `max_num_reorder_pics` — because
/// the standard requires the second to be no larger than the first, and
/// deriving them apart is how they came to disagree: the sequence set
/// declared a reorder depth of `bframes` against a buffer sized for
/// references alone. Nothing noticed while H.265 refused B pictures, and
/// libavcodec refused the first stream that had them
/// ("sps_max_num_reorder_pics out of range"). Our own decoder was more
/// forgiving, which is exactly why CROSS exists.
pub(crate) fn dpb(cfg: &Config) -> (u32, u32) {
    let reorder = cfg.bframes;
    (cfg.max_refs.max(1) + reorder, reorder)
}

/// `profile_tier_level`, shared by the VPS and SPS.
///
/// Twelve bytes, most of them reserved and required to be zero. The
/// profile is the one Table A.2 names for the format
/// (`encode::level::h265_profile`): Main for 8-bit 4:2:0, Main 10 for 9-
/// and 10-bit, and for anything else a format range extensions profile,
/// told apart by the nine constraint flags that open the 43 bits after the
/// source flags — claiming a profile that does not admit the format is a
/// stream a decoder may refuse. The tier and level are the lowest that
/// admit the stream (`encode::level`). Both are derived from `cfg` and `g`
/// alone, so the VPS and the SPS cannot claim different ones.
fn write_ptl(w: &mut BitWriter, cfg: &Config, g: &Geometry) {
    let profile = crate::encode::level::h265_profile(cfg, g);
    let level = crate::encode::level::h265(cfg, g);
    w.bits(2, 0); // general_profile_space
    w.flag(level.high_tier); // general_tier_flag
    w.bits(5, u32::from(profile.idc)); // general_profile_idc
    // general_profile_compatibility_flag[32]
    for i in 0..32 {
        w.flag(i == u32::from(profile.idc));
    }
    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(false); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag
    if profile.idc == 4 {
        // general_max_12bit / 10bit / 8bit / 422chroma / 420chroma /
        // monochrome, intra, one_picture_only and lower_bit_rate
        // constraint flags, then general_reserved_zero_34bits.
        for f in profile.flags {
            w.flag(f);
        }
        w.zeros(34);
    } else {
        // Main's general_reserved_zero_43bits; Main 10's seven reserved
        // bits, a zero one_picture_only flag and 35 more, all zero alike.
        w.zeros(43);
    }
    w.flag(false); // general_inbld_flag
    w.bits(8, u32::from(level.idc)); // general_level_idc
}

/// The coded picture buffer this stream declares, in the exact values the
/// syntax can carry.
///
/// The rounding matters and is deliberate. `BitRate` is
/// `(bit_rate_value_minus1 + 1) << (6 + bit_rate_scale)` and `CpbSize` is
/// `(cpb_size_value_minus1 + 1) << (4 + cpb_size_scale)`, so neither is a
/// free integer. Both are rounded **down** here: a stream that declares
/// slightly less than it was asked for is held to a slightly stricter
/// standard than requested, which is the safe direction. Declaring more
/// than the caller asked for would let a stream conform to a buffer nobody
/// wanted.
///
/// The controller is then given *these* numbers rather than the caller's,
/// so what it aims at and what the stream promises are the same value. A
/// controller targeting 64000 while the stream declares 63936 would be
/// wrong by exactly the rounding, forever, in the direction of overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpb {
    /// `BitRate[0]`, bits per second, as declared.
    pub bit_rate: u64,
    /// `CpbSize[0]`, bits, as declared.
    pub size: u64,
    /// `initial_cpb_removal_delay_length_minus1 + 1`.
    pub initial_delay_length: u32,
    /// `au_cpb_removal_delay_length_minus1 + 1`.
    pub removal_delay_length: u32,
}

/// The bit-rate and buffer scales this encoder writes. Zero for both keeps
/// the coded values close to the numbers a caller recognises, at the cost
/// of a ceiling: `bit_rate_value_minus1` is `ue(v)`, so a rate above about
/// four gigabits would need a scale. Refused by name rather than silently
/// rescaled, because a stream that declares a different rate than it was
/// asked for is the one bug this whole feature exists to catch.
const BIT_RATE_SCALE: u32 = 0;
/// See [`BIT_RATE_SCALE`].
const CPB_SIZE_SCALE: u32 = 0;
/// Width of the delay fields, in bits. 24 is the usual choice and holds a
/// delay of 186 seconds at the 90 kHz clock they are counted in.
const DELAY_LENGTH: u32 = 24;

impl Cpb {
    /// The buffer for `bps` bits per second held for `cpb_ms` milliseconds,
    /// snapped down to what the syntax can carry. `None` when the request
    /// cannot be represented at the fixed scales above.
    pub fn new(bps: u32, cpb_ms: u32) -> Option<Cpb> {
        let br_unit = 1u64 << (6 + BIT_RATE_SCALE);
        let cs_unit = 1u64 << (4 + CPB_SIZE_SCALE);
        let bit_rate = (bps as u64 / br_unit) * br_unit;
        let want = (bps as u64) * (cpb_ms as u64) / 1000;
        let size = (want / cs_unit) * cs_unit;
        if bit_rate == 0 || size == 0 {
            return None;
        }
        // ue(v) in this writer is 32-bit; anything needing a scale is
        // refused by the caller rather than rescaled here.
        if bit_rate / br_unit > u32::MAX as u64 || size / cs_unit > u32::MAX as u64 {
            return None;
        }
        Some(Cpb {
            bit_rate,
            size,
            initial_delay_length: DELAY_LENGTH,
            removal_delay_length: DELAY_LENGTH,
        })
    }

    /// `initial_cpb_removal_delay`, in the 90 kHz units the syntax counts
    /// it in: the time between the first bit arriving and the first access
    /// unit being removed. This encoder starts the buffer **full**, which
    /// is the largest initial delay the declared buffer allows and the most
    /// forgiving start; a smaller one is legal and only makes conformance
    /// harder.
    pub fn initial_removal_delay_90k(&self) -> u32 {
        ((self.size * 90_000) / self.bit_rate).min(((1u64 << DELAY_LENGTH) - 1) as u64) as u32
    }
}

/// `hrd_parameters(1, 0)` — one sub-layer, NAL HRD only, one CPB, no
/// sub-picture parameters. The inverse of `hevc::sps::parse_hrd`, which
/// retains exactly the fields written here.
fn write_hrd(w: &mut BitWriter, cpb: &Cpb, fps: u32) {
    let _ = fps;
    w.flag(true); // nal_hrd_parameters_present_flag
    w.flag(false); // vcl_hrd_parameters_present_flag
    w.flag(false); // sub_pic_hrd_params_present_flag
    w.bits(4, BIT_RATE_SCALE);
    w.bits(4, CPB_SIZE_SCALE);
    w.bits(5, cpb.initial_delay_length - 1);
    w.bits(5, cpb.removal_delay_length - 1);
    w.bits(5, DELAY_LENGTH - 1); // dpb_output_delay_length_minus1
    // The single sub-layer.
    w.flag(true); // fixed_pic_rate_general_flag
    w.ue(0); // elemental_duration_in_tc_minus1 — one tick per picture
    w.ue(0); // cpb_cnt_minus1 — one buffer
    // sub_layer_hrd_parameters(0), one CPB.
    w.ue((cpb.bit_rate >> (6 + BIT_RATE_SCALE)) as u32 - 1); // bit_rate_value_minus1
    w.ue((cpb.size >> (4 + CPB_SIZE_SCALE)) as u32 - 1); // cpb_size_value_minus1
    // cbr_flag 0: variable bit rate.
    //
    // Constant bit rate means the arrival never pauses, so a stream that
    // spends less than its rate must stuff the difference with filler data
    // or the buffer overflows — that is what the flag promises. This
    // encoder's controller targets an *average* and does not stuff, and it
    // undershoots more often than not, so declaring a constant rate would
    // declare something it does not do. The same rule that kept the
    // deblocking flag off until the filter was actually applied.
    //
    // Under a variable rate the arrival simply stops at a full buffer, a
    // full buffer is not an error, and underflow — the failure that
    // actually matters to a decoder — is checked exactly as before.
    w.flag(false); // cbr_flag
}

/// `vui_parameters` (E.2.1) carrying only what was asked for: the colour
/// description and the chroma siting when the caller gave them, and the
/// frame rate the removal times are counted in plus the HRD when a buffer
/// was declared.
///
/// Everything else is absent by its own flag. A VUI is optional and this
/// encoder had none until the buffer model needed one, so the only reason
/// any of it is here is that a removal schedule without a frame rate is not
/// a schedule — and, later, that a BT.2020 PQ picture with no colour
/// description is shown as BT.709. Each part is present only under its own
/// condition, so a stream with a buffer and no colour is byte-identical to
/// one from before colour existed. The inverse of `hevc::sps::parse_vui`.
fn write_vui(
    w: &mut BitWriter,
    colour: Option<&ColourDescription>,
    chroma_loc: Option<u8>,
    cpb: Option<&Cpb>,
    (fps_num, fps_den): (u32, u32),
) {
    w.flag(false); // aspect_ratio_info_present_flag
    w.flag(false); // overscan_info_present_flag
    // E.2.1 copies E.1.1's video_signal_type and chroma_loc_info groups
    // field for field.
    crate::encode::h264_syntax::write_video_signal_type(w, colour);
    crate::encode::h264_syntax::write_chroma_loc(w, chroma_loc);
    w.flag(false); // neutral_chroma_indication_flag
    w.flag(false); // field_seq_flag
    w.flag(false); // frame_field_info_present_flag
    w.flag(false); // default_display_window_flag
    match cpb {
        Some(cpb) => {
            w.flag(true); // vui_timing_info_present_flag
            // One tick per picture, the frame rate in lowest terms: 29.97
            // is 1001 over 30000, 30 is 1 over 30 as it always was.
            w.bits(32, fps_den); // vui_num_units_in_tick
            w.bits(32, fps_num); // vui_time_scale — ticks per second
            w.flag(false); // vui_poc_proportional_to_timing_flag
            w.flag(true); // vui_hrd_parameters_present_flag
            write_hrd(w, cpb, fps_num);
        }
        None => w.flag(false), // vui_timing_info_present_flag
    }
    w.flag(false); // bitstream_restriction_flag
}

// The HDR10 static-metadata SEIs are the same bytes in both standards
// (payloadTypes 137 and 144, D.2.28 / D.2.35 here), so the H.264
// module's writers serve this one; only the NAL header differs, and
// `annexb` adds that.
pub use crate::encode::h264_syntax::{write_content_light_level_sei, write_mastering_display_sei};

/// A `buffering_period` SEI message, wrapped as a prefix SEI NAL.
///
/// It carries the one number the buffer model cannot derive: how long the
/// first access unit waits after the first bit arrives. Everything else in
/// the schedule follows from the frame rate in the VUI.
///
/// Written for every IRAP access unit, which is where a buffering period
/// may begin.
pub fn write_buffering_period_sei(cpb: &Cpb) -> Vec<u8> {
    let mut p = BitWriter::with_capacity(16);
    p.ue(0); // bp_seq_parameter_set_id
    // sub_pic_hrd_params_present_flag is 0, so this flag is present.
    p.flag(false); // irap_cpb_params_present_flag
    p.flag(false); // concatenation_flag
    p.bits(cpb.removal_delay_length, 0); // au_cpb_removal_delay_delta_minus1
    // nal_hrd_parameters_present_flag is 1, one CPB.
    p.bits(cpb.initial_delay_length, cpb.initial_removal_delay_90k());
    p.bits(cpb.initial_delay_length, 0); // initial_cpb_removal_offset
    p.rbsp_trailing_bits();
    // The raw payload: `sei_nal` sizes it as RBSP bytes and escapes the
    // whole NAL once. This wrapped `p.into_nal()` — the payload already
    // escaped, then sized and escaped again — so a buffering period whose
    // delay bits held `00 00 00` went out as `00 00 03 03`, one byte
    // longer than its payload_size said, and h26xhrd read the stray byte
    // as part of the initial delay.
    crate::encode::h264_syntax::sei_nal(0, &p.into_rbsp())
}

/// Video parameter set.
pub fn write_vps(cfg: &Config, g: &Geometry) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(32);
    w.bits(4, 0); // vps_video_parameter_set_id
    w.flag(true); // vps_base_layer_internal_flag
    w.flag(true); // vps_base_layer_available_flag
    w.bits(6, 0); // vps_max_layers_minus1
    w.bits(3, 0); // vps_max_sub_layers_minus1
    w.flag(true); // vps_temporal_id_nesting_flag
    w.bits(16, 0xffff); // vps_reserved_0xffff_16bits
    write_ptl(&mut w, cfg, g);
    w.flag(true); // vps_sub_layer_ordering_info_present_flag
    let (buffering, reorder) = dpb(cfg);
    w.ue(buffering); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(reorder); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.bits(6, 0); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.flag(false); // vps_timing_info_present_flag
    w.flag(false); // vps_extension_flag
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// Sequence parameter set.
pub fn write_sps(cfg: &Config, g: &Geometry, log2_max_poc_lsb: u32, cpb: Option<&Cpb>) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(64);
    w.bits(4, 0); // sps_video_parameter_set_id
    w.bits(3, 0); // sps_max_sub_layers_minus1
    w.flag(true); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, cfg, g);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(chroma_idc(g.chroma));
    if g.chroma == ChromaFormat::Yuv444 {
        w.flag(false); // separate_colour_plane_flag
    }
    w.ue(g.coded_width);
    w.ue(g.coded_height);
    // Conformance window, in chroma units.
    let (cw, ch) = match g.chroma {
        ChromaFormat::Monochrome | ChromaFormat::Yuv444 => (1, 1),
        ChromaFormat::Yuv420 => (2, 2),
        ChromaFormat::Yuv422 => (2, 1),
    };
    let right = (g.coded_width - g.width) / cw;
    let bottom = (g.coded_height - g.height) / ch;
    if right != 0 || bottom != 0 {
        w.flag(true);
        w.ue(0);
        w.ue(right);
        w.ue(0);
        w.ue(bottom);
    } else {
        w.flag(false);
    }
    // One depth for both components — `Config::bit_depth` is the
    // picture's, and every kernel below the NAL layer takes luma and
    // chroma at one width. Written for monochrome too: the field exists
    // regardless (7.3.2.2), the reader keeps it, and a chroma depth that
    // disagreed with luma's would be a second number for nothing to read.
    w.ue(g.bit_depth - 8); // bit_depth_luma_minus8
    w.ue(g.bit_depth - 8); // bit_depth_chroma_minus8
    w.ue(log2_max_poc_lsb - 4);
    w.flag(true); // sps_sub_layer_ordering_info_present_flag
    let (buffering, reorder) = dpb(cfg);
    w.ue(buffering); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(reorder); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(g.log2_min_cb - 3); // log2_min_luma_coding_block_size_minus3
    w.ue(g.log2_ctb - g.log2_min_cb); // log2_diff_max_min_luma_coding_block_size
    w.ue(0); // log2_min_luma_transform_block_size_minus2 -> 4x4
    // The maximum transform size equals the CTB size (the CTB is at most 32,
    // which is also the standard's largest transform), so a 2Nx2N CU can
    // carry a single CU-sized TU — the unsplit transform tree every unit may
    // take. The previous value, one below the CTB, would have forced an
    // inferred transform split under every CTB-sized CU and made that shape
    // unrepresentable.
    w.ue(g.log2_ctb - 2); // log2_diff_max_min_luma_transform_block_size
    w.ue(2); // max_transform_hierarchy_depth_inter
    w.ue(2); // max_transform_hierarchy_depth_intra
    w.flag(false); // scaling_list_enabled_flag
    w.flag(false); // amp_enabled_flag
    // Turning this on makes `slice_sao_luma_flag` (and, outside
    // monochrome, `slice_sao_chroma_flag`) appear in EVERY slice header,
    // I slices included — the reader's block sits outside the non-IDR
    // branch — so this flag's first effect is on slices that carry no SAO
    // decision at all. See `SliceHeader::sao`, which is an `Option` for
    // exactly that reason.
    //
    // Third member of the family this header keeps meeting: a parameter
    // set flag that silently changes how many bits every slice header
    // holds. The other two are recorded at their sites below.
    w.flag(cfg.sao); // sample_adaptive_offset_enabled_flag
    w.flag(false); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.flag(false); // long_term_ref_pics_present_flag
    w.flag(false); // sps_temporal_mvp_enabled_flag
    w.flag(false); // strong_intra_smoothing_enabled_flag
    if cpb.is_some() || cfg.colour.is_some() || cfg.chroma_loc.is_some() {
        w.flag(true); // vui_parameters_present_flag
        write_vui(&mut w, cfg.colour.as_ref(), cfg.chroma_loc, cpb, cfg.frame_rate());
    } else {
        w.flag(false); // vui_parameters_present_flag
    }
    w.flag(false); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// Picture parameter set.
///
/// `qp` is `init_qp`, the quantiser every slice's `slice_qp_delta` is
/// relative to, in the range the syntax gives it — `-QpBdOffsetY..=51`
/// (7.4.3.3.1), so negative at depths above 8. The encoder's
/// configuration offers `0..=51` at every depth; the writer takes the
/// whole range so a header test can reach the rest of it through the
/// production writer rather than by hand.
///
/// `bypass` writes `transquant_bypass_enabled_flag` — the lossless switch.
/// It changes nothing else here or in the slice header (the parser reads no
/// other syntax conditionally on it); what it changes is the coding tree,
/// where every CU then carries a `cu_transquant_bypass_flag`.
///
/// Everything else is [`PpsOptions::default`]: the stream every caller
/// before the per-CU quantiser and weighted prediction existed asked
/// for, byte for byte. Those go through [`write_pps_opts`].
pub fn write_pps(qp: i32, bypass: bool, deblock: bool) -> Vec<u8> {
    write_pps_opts(qp, bypass, deblock, &PpsOptions::default())
}

/// The picture-parameter-set switches that make more syntax appear in
/// the coding tree or the slice header — each `false` / `None` by default
/// so that a stream not asking for the feature is byte-identical to one
/// from an encoder that never had it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PpsOptions {
    /// `cu_qp_delta_enabled_flag`, with `diff_cu_qp_delta_depth` when
    /// set: the quantiser may then change per quantisation group, and
    /// every transform unit that carries a coded cbf and is the first in
    /// its group codes `cu_qp_delta_abs` (and a sign). `Some(0)` makes
    /// the group the CTB.
    pub cu_qp_delta_depth: Option<u32>,
    /// `weighted_pred_flag`: every P slice header carries a
    /// `pred_weight_table`.
    pub weighted_pred: bool,
    /// `weighted_bipred_flag`: every B slice header carries one.
    pub weighted_bipred: bool,
}

/// [`write_pps`] with the optional switches spelled out.
pub fn write_pps_opts(qp: i32, bypass: bool, deblock: bool, opts: &PpsOptions) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(32);
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.flag(false); // dependent_slice_segments_enabled_flag
    w.flag(false); // output_flag_present_flag
    w.bits(3, 0); // num_extra_slice_header_bits
    w.flag(false); // sign_data_hiding_enabled_flag
    w.flag(false); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(qp - 26); // init_qp_minus26
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // transform_skip_enabled_flag
    // Fifth member of the flag-that-grows-syntax family this file keeps
    // meeting: setting it makes `diff_cu_qp_delta_depth` follow here and
    // `cu_qp_delta_abs` appear in the coding tree. The reader takes the
    // depth only when the flag is set (`pps.rs`), so the writer does too.
    w.flag(opts.cu_qp_delta_depth.is_some()); // cu_qp_delta_enabled_flag
    if let Some(depth) = opts.cu_qp_delta_depth {
        w.ue(depth); // diff_cu_qp_delta_depth
    }
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
    // Each of these makes a `pred_weight_table` appear in every slice
    // header of that type, which the slice header writer must then
    // carry.
    w.flag(opts.weighted_pred); // weighted_pred_flag
    w.flag(opts.weighted_bipred); // weighted_bipred_flag
    w.flag(bypass); // transquant_bypass_enabled_flag
    w.flag(false); // tiles_enabled_flag
    w.flag(false); // entropy_coding_sync_enabled_flag
    w.flag(true); // pps_loop_filter_across_slices_enabled_flag
    // The deblocking filter, on when the encoder filters its own
    // reconstruction and off when it cannot.
    //
    // It was disabled picture-wide for a while, and the reason is worth
    // keeping: the first H.265 stream this encoder produced was
    // CROSS-identical but failed SELF on exactly 24 luma samples, every
    // one within three of the 8-sample deblocking grid — libavcodec and
    // our own decoder both filtered, and the encoder did not. Rather than
    // declare a filter it could not apply, the PPS turned it off.
    //
    // It filters now, through the decoder's own deblocker over the
    // encoder's reconstruction — intra pictures and P pictures alike, the
    // latter since intra coding units inside a P slice became spellable
    // and `deblock_inter_picture` could be called. That order mattered:
    // the flag is picture-wide rather than per-slice, so until every
    // picture kind could be filtered, declaring it would have made every
    // decoder filter pictures this encoder did not.
    w.flag(true); // deblocking_filter_control_present_flag
    w.flag(false); // deblocking_filter_override_enabled_flag
    w.flag(!deblock); // pps_deblocking_filter_disabled_flag
    if deblock {
        // Clearing that flag makes two more elements appear — the filter
        // offsets — and forgetting them truncates the PPS. The encoder's
        // own parser refused the header the first time this was written,
        // which is the fourth instance of the same shape: a flag whose
        // value decides what syntax follows it.
        w.se(0); // pps_beta_offset_div2
        w.se(0); // pps_tc_offset_div2
    }
    w.flag(false); // pps_scaling_list_data_present_flag
    w.flag(false); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.flag(false); // slice_segment_header_extension_present_flag
    w.flag(false); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// What a slice segment header needs beyond the parameter sets.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// What the slice is coded as.
    pub kind: Kind,
    /// The low bits of the picture order count.
    pub poc_lsb: u32,
    /// Quantiser for the slice, `SliceQpY`: `-QpBdOffsetY..=51`, the
    /// range the parser holds it to (negative only above 8 bits — see
    /// [`write_pps`] for why the writer takes the whole of it).
    pub qp: i32,
    /// Width of the `poc_lsb` field, from the SPS.
    pub log2_max_poc_lsb: u32,
    /// POC deltas of the reference pictures, relative to this slice's POC:
    /// negative for the past (list 0), positive for the future (list 1).
    /// Empty for an I or IDR slice. These become the slice's inline short
    /// term reference picture set, in the order the reader expects —
    /// negatives nearest-first, then positives nearest-first.
    pub ref_deltas: Vec<i32>,
    /// POC deltas of the pictures this slice does not reference but a
    /// later picture will. They go in the same set with
    /// `used_by_curr_pic` 0: the reader keeps them in the decoded picture
    /// buffer and leaves them out of this slice's lists and of
    /// `NumPicTotalCurr`. A picture the set leaves out is marked unused
    /// for reference (8.3.2), and no later set can bring it back. Disjoint
    /// from `ref_deltas`.
    pub kept_deltas: Vec<i32>,
    /// The slice's SAO switches, or `None` when the SPS leaves
    /// `sample_adaptive_offset_enabled_flag` clear and the reader takes no
    /// bit here at all.
    ///
    /// It is an `Option` rather than a pair of bools because the question
    /// the writer must answer is not "is SAO on" but "does the reader read
    /// a bin" — the distinction this header has been bitten by twice, once
    /// in each direction.
    pub sao: Option<SaoFlags>,
    /// The `pred_weight_table` this slice carries, or `None` when the
    /// reader reads none — an I slice, or a PPS whose `weighted_pred_flag`
    /// (P) / `weighted_bipred_flag` (B) is clear. The same `Option`
    /// discipline as `sao`: the question is whether the reader takes the
    /// syntax, and a P slice under a set flag must carry a table even when
    /// every weight in it is the default.
    pub pred_weights: Option<PredWeights>,
}

/// A slice's `pred_weight_table`, with what the writer needs to spell it
/// the way the reader will read it: the reader shifts every offset by
/// `WpOffsetBdShift` (the depth above 8, or nothing under high-precision
/// offsets) and takes chroma syntax only when the stream has chroma.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeights {
    /// The table as the reader will hold it — offsets already shifted to
    /// the sample depth, one entry per active reference per list.
    pub table: PredWeightTable,
    /// `ChromaArrayType != 0`: whether the chroma flags and entries exist.
    pub chroma: bool,
    /// `BitDepthY` and `BitDepthC`, for the offset shift.
    pub bit_depth_luma: u32,
    /// See `bit_depth_luma`.
    pub bit_depth_chroma: u32,
}

/// Write `pred_weight_table()`: the exact inverse of
/// `hevc::slice::parse_pred_weight_table`, for a table the reader will
/// hold as `t` — one list for a P slice, two for a B.
///
/// The reader's shape: `luma_log2_weight_denom`, a chroma delta when
/// the stream has chroma; then per list every entry's luma flag, every
/// entry's chroma flag, and then per entry the flagged deltas. A flag is
/// set exactly when the entry differs from the default `(1 << denom,
/// 0)`, since an unflagged entry is what the reader infers. Offsets are
/// spelled in 8-bit units (the reader shifts them up by the depth above
/// 8; high-precision offsets, which this encoder never declares, would
/// spell them at the sample depth), and a chroma offset is spelled as
/// the `delta_chroma_offset` the reader's derivation
/// `o = Clip(half + delta - ((half * w) >> denom))` inverts to — the
/// caller keeps `o` inside the reader's clip, or the round trip does
/// not close.
pub fn write_pred_weight_table(pw: &PredWeights, b_slice: bool, high_precision: bool, w: &mut BitWriter) {
    let t = &pw.table;
    w.ue(t.luma_log2_denom); // luma_log2_weight_denom
    if pw.chroma {
        w.se(t.chroma_log2_denom as i32 - t.luma_log2_denom as i32); // delta_chroma_log2_weight_denom
    }
    let shift_y = if high_precision { 0 } else { pw.bit_depth_luma as i32 - 8 };
    let shift_c = if high_precision { 0 } else { pw.bit_depth_chroma as i32 - 8 };
    let half_c: i32 = 1 << if high_precision { pw.bit_depth_chroma - 1 } else { 7 };
    let luma_default = (1i32 << t.luma_log2_denom, 0i32);
    let chroma_default = [(1i32 << t.chroma_log2_denom, 0i32); 2];
    for list in t.lists.iter().take(if b_slice { 2 } else { 1 }) {
        for e in list {
            w.flag(e.luma != luma_default); // luma_weight_lX_flag
        }
        if pw.chroma {
            for e in list {
                w.flag(e.chroma != chroma_default); // chroma_weight_lX_flag
            }
        }
        for e in list {
            if e.luma != luma_default {
                w.se(e.luma.0 - luma_default.0); // delta_luma_weight_lX
                debug_assert_eq!(e.luma.1 & ((1 << shift_y) - 1), 0, "a luma offset must be a multiple of the shift");
                w.se(e.luma.1 >> shift_y); // luma_offset_lX
            }
            if pw.chroma && e.chroma != chroma_default {
                for (cw, co) in e.chroma {
                    w.se(cw - chroma_default[0].0); // delta_chroma_weight_lX
                    let o = co >> shift_c;
                    debug_assert!((-half_c..half_c).contains(&o), "a chroma offset outside the reader's clip cannot round-trip");
                    w.se(o - half_c + ((half_c * cw) >> t.chroma_log2_denom)); // delta_chroma_offset_lX
                }
            }
        }
    }
}

/// `slice_sao_luma_flag` and `slice_sao_chroma_flag`.
#[derive(Debug, Clone, Copy)]
pub struct SaoFlags {
    /// `slice_sao_luma_flag`.
    pub luma: bool,
    /// `slice_sao_chroma_flag`, or `None` in monochrome, where the reader
    /// reads no second bit (its gate is `chroma_format_idc != 0`).
    pub chroma: Option<bool>,
}

/// Slice segment header, up to but not including the coded tree.
pub fn write_slice_header(h: &SliceHeader, pps_qp: i32, nal_type: u8, deblock: bool, w: &mut BitWriter) {
    w.flag(true); // first_slice_segment_in_pic_flag
    if (16..=23).contains(&nal_type) {
        w.flag(false); // no_output_of_prior_pics_flag
    }
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(match h.kind {
        Kind::B => 0,
        Kind::P => 1,
        Kind::Idr | Kind::I => 2,
    });
    // An IDR has no POC and no reference picture set: its POC is zero by
    // definition and everything before it is discarded.
    if !(16..=23).contains(&nal_type) {
        w.bits(h.log2_max_poc_lsb, h.poc_lsb);
        // The SPS declares num_short_term_ref_pic_sets = 0, so there is no
        // set to select and the slice must carry its own inline.
        // (This flag was once written as 1 with nothing behind it — a
        // placeholder that claimed an SPS set that does not exist, dead
        // only because every non-IDR path refused before reaching here. It
        // would have gone live the instant inter prediction landed; the
        // writers-beside-readers work on the coding tree is what found it.)
        w.flag(false); // short_term_ref_pic_set_sps_flag
        // st_ref_pic_set(0): with no earlier set to predict from,
        // inter_ref_pic_set_prediction_flag is not read at idx 0.
        // Every picture the decoder must keep, each with whether this
        // slice uses it.
        let entries = || h.ref_deltas.iter().map(|&d| (d, true)).chain(h.kept_deltas.iter().map(|&d| (d, false)));
        let mut negative: Vec<(i32, bool)> = entries().filter(|e| e.0 < 0).collect();
        let mut positive: Vec<(i32, bool)> = entries().filter(|e| e.0 > 0).collect();
        // Nearest first, as the deltas are coded as successive differences.
        negative.sort_by_key(|e| -e.0);
        positive.sort_by_key(|e| e.0);
        w.ue(negative.len() as u32);
        w.ue(positive.len() as u32);
        let mut prev = 0i32;
        for &(d, used) in &negative {
            w.ue((prev - d - 1) as u32); // delta_poc_s0_minus1
            w.flag(used); // used_by_curr_pic_s0_flag
            prev = d;
        }
        let mut prev = 0i32;
        for &(d, used) in &positive {
            w.ue((d - prev - 1) as u32); // delta_poc_s1_minus1
            w.flag(used); // used_by_curr_pic_s1_flag
            prev = d;
        }
        // slice_temporal_mvp_enabled_flag is absent: the SPS disables
        // temporal MVP, so the reader never reads the slice-level flag —
        // which is also what makes the spatial candidate derivation the
        // complete one rather than a subset of it.
    }
    // slice_sao_luma_flag / slice_sao_chroma_flag, present exactly when
    // the SPS enables SAO — for every slice type, this block sitting
    // outside the non-IDR branch above. (A flag was once written here
    // while the SPS disabled SAO: one spurious bit that shifted everything
    // after it, unnoticed because nothing could decode past the header
    // until the coding tree existed. `None` is that case spelled so it
    // cannot recur.)
    if let Some(sao) = h.sao {
        w.flag(sao.luma);
        if let Some(chroma) = sao.chroma {
            w.flag(chroma);
        }
    }
    if matches!(h.kind, Kind::P | Kind::B) {
        // How many references each list actually has, read off the very
        // set this header just wrote: its used negatives become
        // RefPicList0 and its used positives RefPicList1, so counting them
        // is counting the lists. The kept entries are in neither.
        // Deriving it here rather than taking it as a parameter is what
        // keeps the two from disagreeing — a header that declares more
        // entries than its own reference picture set carries is a stream
        // no decoder can build the lists for.
        let n0 = h.ref_deltas.iter().filter(|d| **d < 0).count().max(1);
        let n1 = if h.kind == Kind::B { h.ref_deltas.iter().filter(|d| **d > 0).count().max(1) } else { 1 };
        // num_ref_idx_active_override_flag. The PPS defaults are one per
        // list, so the flag is needed exactly when some list has more —
        // and when it is clear the reader resolves [1, 0] for P and
        // [1, 1] for B, which is what every stream this encoder wrote
        // before multiple references existed relied on. Verified against
        // the header parser, not inferred: it copies the PPS defaults and
        // overwrites them only when this flag is set.
        //
        // The coupling is worth naming: change those PPS defaults and
        // every header written this way silently means something else.
        // Both are written from one contract in this file, which is why
        // the derivation lives here.
        let override_counts = n0 > 1 || (h.kind == Kind::B && n1 > 1);
        w.flag(override_counts);
        if override_counts {
            w.ue(n0 as u32 - 1); // num_ref_idx_l0_active_minus1
            if h.kind == Kind::B {
                w.ue(n1 as u32 - 1); // num_ref_idx_l1_active_minus1
            }
        }
        // lists_modification_present_flag is 0 in the PPS, so no list
        // modification syntax follows; mvd_l1_zero_flag is read for B.
        if h.kind == Kind::B {
            w.flag(false); // mvd_l1_zero_flag
        }
        // cabac_init_present_flag is 0 in the PPS: no cabac_init_flag, and
        // both sides derive the P/B initialisation type from the slice
        // type alone.
        //
        // collocated_from_l0_flag / collocated_ref_idx are absent because
        // slice_temporal_mvp_enabled is off. The prediction weight table
        // is read exactly when the PPS sets `weighted_pred_flag` for a P
        // slice or `weighted_bipred_flag` for a B — the caller's `Option`
        // says which, as with SAO — and it sits here, before
        // `five_minus_max_num_merge_cand`, which is where the reader
        // takes it.
        if let Some(pw) = &h.pred_weights {
            write_pred_weight_table(pw, h.kind == Kind::B, false, w);
        }
        w.ue(0); // five_minus_max_num_merge_cand -> MaxNumMergeCand 5
    }
    w.se(h.qp - pps_qp); // slice_qp_delta
    // slice_loop_filter_across_slices_enabled_flag: the reader reads it
    // when `pps_loop_filter_across_slices_enabled_flag` is set AND either
    // SAO is on for this slice or deblocking is not disabled — a
    // three-way condition, `slice.rs`. The PPS always sets the first, so
    // the bit is present whenever *either* filter runs; and since every
    // picture is deblocked, it is present always. SAO changes which
    // disjunct is true, not whether the bit exists.
    //
    // This comment used to say something narrower and, by the end, wrong:
    // "the SPS disables SAO, so this follows the PPS's deblocking flag
    // exactly". True when written, stale the moment inter deblocking went
    // live and every picture started being filtered — and it was then
    // read, in good faith, as evidence that enabling SAO would make this
    // bit newly appear. It would not; it had been appearing for a while.
    // The superseded reasoning is kept rather than deleted because the
    // failure was reasoning from this comment instead of from the reader,
    // and a comment that records having misled is worth more than a tidy
    // one that could do it again.
    //
    // Getting the condition wrong in either direction shifts every bit
    // after it, the one-spurious-bit shape this header has been bitten by
    // twice; the production parser refusing the header is what catches it.
    let sao_on = h.sao.is_some_and(|s| s.luma || s.chroma == Some(true));
    if deblock || sao_on {
        w.flag(true); // slice_loop_filter_across_slices_enabled_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Config;

    fn geom(w: u32, h: u32, c: ChromaFormat) -> (Config, Geometry) {
        let cfg = Config { width: w, height: h, chroma: c, ..Config::default() };
        let g = Geometry::new(&cfg);
        (cfg, g)
    }

    /// The HRD the SPS declares must survive the parser that, until this
    /// change, read those fields only to stay bit-aligned and threw them
    /// away.
    ///
    /// That discard is why this round trip did not exist and could not:
    /// there was nothing to compare against. It is also why the writer
    /// needed one — a `bit_rate_scale` off by one, or the delay lengths
    /// written in the wrong order, would shift every field after it inside
    /// the VUI and nothing downstream would notice, because the VUI sits at
    /// the end of the SPS and a decoder that ignores it decodes the stream
    /// perfectly either way.
    #[test]
    fn the_declared_buffer_survives_the_parser() {
        use crate::hevc::sps::Sps;
        for (bps, ms) in [(64_000u32, 125u32), (64_000, 1000), (2_000_000, 500), (128, 1000), (7_000_000, 250)] {
            let Some(cpb) = Cpb::new(bps, ms) else { continue };
            let (cfg, g) = geom(64, 64, ChromaFormat::Yuv420);
            let cfg = Config { fps: 30, ..cfg };
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, Some(&cpb))))
                .unwrap_or_else(|e| panic!("{bps}bps/{ms}ms: the encoder's SPS must parse: {e}"));
            let vui = sps.vui.as_ref().unwrap_or_else(|| panic!("{bps}bps/{ms}ms: no VUI"));
            assert_eq!(vui.timing, Some((1, 30)), "{bps}bps/{ms}ms: frame rate");
            let hrd = vui.hrd.unwrap_or_else(|| panic!("{bps}bps/{ms}ms: no HRD"));
            assert_eq!(hrd.bit_rate, cpb.bit_rate, "{bps}bps/{ms}ms: bit rate");
            assert_eq!(hrd.cpb_size, cpb.size, "{bps}bps/{ms}ms: buffer size");
            // Variable bit rate, and the round trip pins it: declaring a
            // constant rate would promise stuffing this encoder does not
            // do, and the flag survives the parser either way, so only an
            // assertion keeps the promise honest.
            assert!(!hrd.cbr, "{bps}bps/{ms}ms: cbr flag should be clear");
            assert_eq!(hrd.initial_delay_length, cpb.initial_delay_length, "{bps}bps/{ms}ms: initial delay width");
            assert_eq!(hrd.removal_delay_length, cpb.removal_delay_length, "{bps}bps/{ms}ms: removal delay width");
        }
    }

    /// A stream that declares no buffer must carry no VUI at all — the
    /// bytes before this feature existed, unchanged.
    #[test]
    fn declaring_no_buffer_writes_no_vui() {
        use crate::hevc::sps::Sps;
        let (cfg, g) = geom(64, 64, ChromaFormat::Yuv420);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, None))).expect("SPS");
        assert!(sps.vui.is_none(), "an SPS with no buffer declared should carry no VUI");
    }

    /// The colour description round-trips through the decoder's own SPS
    /// parser, on its own and beside a buffer; a buffer alone says
    /// nothing about colour; neither writes no VUI (the test above). Each
    /// code point is asserted by name so that writing one into another's
    /// field — the mutation this exists to catch — names the field lost.
    #[test]
    fn the_colour_description_survives_the_decoders_own_sps_parser() {
        use crate::encode::ColourDescription;
        use crate::hevc::sps::Sps;
        let colours = [
            ColourDescription { primaries: 9, transfer: 16, matrix: 9, full_range: false }, // HDR10
            ColourDescription { primaries: 9, transfer: 18, matrix: 9, full_range: false }, // HLG
            ColourDescription { primaries: 1, transfer: 1, matrix: 1, full_range: true }, // BT.709 full
            ColourDescription { primaries: 12, transfer: 17, matrix: 6, full_range: false }, // P3 / SMPTE 428 / 601
        ];
        let (base, g) = geom(64, 64, ChromaFormat::Yuv420);
        for c in colours {
            let cfg = Config { colour: Some(c), ..base.clone() };
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, None)))
                .unwrap_or_else(|e| panic!("{c:?}: SPS rejected: {e}"));
            let vui = sps.vui.as_ref().unwrap_or_else(|| panic!("{c:?}: no VUI"));
            let (p, t, m) = vui.colour_description.unwrap_or_else(|| panic!("{c:?}: no colour description"));
            assert_eq!(p, c.primaries, "{c:?}: primaries");
            assert_eq!(t, c.transfer, "{c:?}: transfer");
            assert_eq!(m, c.matrix, "{c:?}: matrix");
            assert_eq!(vui.full_range, c.full_range, "{c:?}: range");
            assert_eq!(vui.chroma_loc, None, "{c:?}: no siting asked for, none written");
            assert_eq!(vui.timing, None, "{c:?}: no buffer, no clock");
            assert!(vui.hrd.is_none(), "{c:?}: no buffer, no HRD");
        }
        let cpb = Cpb::new(64_000, 125).expect("representable");
        let cfg = Config {
            colour: Some(colours[0]),
            rate: crate::encode::RateControl::Bitrate { bps: 64_000 },
            cpb_ms: 125,
            ..base.clone()
        };
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, Some(&cpb)))).expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, Some((9, 16, 9)));
        assert_eq!(vui.timing, Some((1, 30)));
        assert_eq!(vui.hrd.map(|h| h.bit_rate), Some(cpb.bit_rate));
        let cfg = Config { colour: None, ..cfg };
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, Some(&cpb)))).expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, None, "a buffer alone must not invent a colour");
        assert!(!vui.full_range);
        assert_eq!(vui.timing, Some((1, 30)));
        // The chroma siting: alone it is a VUI that says nothing about
        // colour, and every code comes back for both fields.
        for t in 0..=5u8 {
            let cfg = Config { chroma_loc: Some(t), ..base.clone() };
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, None))).expect("SPS");
            let vui = sps.vui.as_ref().expect("a siting alone is a VUI");
            assert_eq!(vui.chroma_loc, Some((t, t)), "chroma_sample_loc_type {t}");
            assert_eq!(vui.colour_description, None, "a siting alone must not invent a colour");
        }
        let cfg = Config { colour: Some(colours[0]), chroma_loc: Some(1), ..base.clone() };
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 8, None))).expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, Some((9, 16, 9)));
        assert_eq!(vui.chroma_loc, Some((1, 1)));
        let plain = write_sps(&base, &g, 8, None);
        assert_ne!(plain, write_sps(&Config { colour: Some(colours[0]), ..base.clone() }, &g, 8, None));
        assert_ne!(plain, write_sps(&Config { chroma_loc: Some(0), ..base }, &g, 8, None));
    }

    /// The buffering period SEI is escaped once and sized as RBSP: what a
    /// reader unescapes is exactly `payload_type, payload_size, payload,
    /// trailing byte`, with no emulation-prevention byte left inside the
    /// payload. Before this held the payload went out escaped, then sized
    /// and escaped again, and a delay whose bits held `00 00 00` reached
    /// the reader with a `03` inside it (`00 0b 80 00 00 03 03 …`). The
    /// last assertion keeps the test honest: this buffer's payload does
    /// contain the zero run that triggers an escape.
    #[test]
    fn the_buffering_period_sei_is_escaped_once_and_sized_as_rbsp() {
        let cpb = Cpb::new(64_000, 125).expect("representable");
        let nal = write_buffering_period_sei(&cpb);
        let rbsp = crate::nal::unescape_rbsp(&nal);
        assert_eq!(rbsp[0], 0, "payload_type buffering_period");
        let size = rbsp[1] as usize;
        assert_eq!(rbsp.len(), 2 + size + 1, "type, size, payload, one trailing byte: {rbsp:02x?}");
        assert_eq!(rbsp[2 + size], 0x80, "rbsp_trailing_bits: {rbsp:02x?}");
        let payload = &rbsp[2..2 + size];
        assert!(!payload.windows(3).any(|w| w == [0, 0, 3]), "an escape byte inside the payload: {rbsp:02x?}");
        assert!(payload.windows(3).any(|w| w == [0, 0, 0]), "the zero run that exercises the escape: {rbsp:02x?}");
    }

    /// The declared values are rounded **down** from what was asked for,
    /// never up. A buffer larger than requested, or a rate above it, would
    /// let a stream conform to something nobody asked for — and the error
    /// would be invisible, because the checker reads the declaration.
    #[test]
    fn the_declaration_never_exceeds_the_request() {
        for bps in [1u32, 63, 64, 65, 1000, 64_000, 999_999] {
            for ms in [1u32, 125, 500, 1000, 3000] {
                let Some(cpb) = Cpb::new(bps, ms) else { continue };
                assert!(cpb.bit_rate <= bps as u64, "{bps}bps: declared {} above the request", cpb.bit_rate);
                let want = (bps as u64) * (ms as u64) / 1000;
                assert!(cpb.size <= want, "{bps}bps/{ms}ms: declared buffer {} above the request {want}", cpb.size);
            }
        }
    }

    /// The crate's own HEVC parsers are proven against 178 conformance
    /// streams; anything they reject is not a legal parameter set.
    #[test]
    fn the_decoder_parses_what_the_encoder_writes() {
        for (w, h, c) in [
            (64u32, 64u32, ChromaFormat::Yuv420),
            (50, 34, ChromaFormat::Yuv420),
            (64, 64, ChromaFormat::Yuv422),
            (64, 64, ChromaFormat::Yuv444),
            (64, 64, ChromaFormat::Monochrome),
        ] {
            let (cfg, g) = geom(w, h, c);
            let vps = write_vps(&cfg, &g);
            assert!(!vps.is_empty());
            let sps = write_sps(&cfg, &g, 8, None);
            let parsed = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps))
                .unwrap_or_else(|e| panic!("{w}x{h} {c:?}: SPS rejected: {e}"));
            assert_eq!(parsed.width, g.coded_width, "{w}x{h} {c:?}");
            assert_eq!(parsed.height, g.coded_height, "{w}x{h} {c:?}");
            assert_eq!(parsed.chroma_format_idc, chroma_idc(c), "{w}x{h} {c:?}");
            assert_eq!(parsed.bit_depth_luma, cfg.bit_depth, "{w}x{h} {c:?}");
        }
    }

    /// Deep parameter sets and slice headers survive the production
    /// parsers with their depth intact, and the quantiser range the
    /// header can carry grows with it.
    ///
    /// What is held, field by field: the SPS's two `bit_depth_*_minus8`
    /// (one depth, both components, monochrome included); the profile the
    /// PTL claims — Main 10 for 4:2:0 at 9 or 10 bits, the range
    /// extensions above that or outside 4:2:0, since a Main decoder may
    /// refuse a Main 10 stream and a Main 10 one a 12-bit stream; and the
    /// slice quantiser, which the parser bounds to `-QpBdOffsetY..=51` —
    /// so at 10 bits a `slice_qp_delta` reaching QP −12 is legal and
    /// parses, where at 8 bits the same header is rejected. The last
    /// clause is the one that proves the parser is applying the depth it
    /// was handed rather than a constant.
    #[test]
    fn deep_headers_round_trip_through_the_parsers() {
        use crate::hevc::pps::Pps;
        use crate::hevc::slice::SliceHeader as ParsedHeader;
        use crate::hevc::sps::Sps;
        use crate::nal::HevcNalHeader;

        for (depth, chroma, want_profile) in [
            (10u32, ChromaFormat::Yuv420, 2u32),
            (10, ChromaFormat::Yuv422, 4),
            (10, ChromaFormat::Yuv444, 4),
            (10, ChromaFormat::Monochrome, 4),
            (12, ChromaFormat::Yuv420, 4),
            (14, ChromaFormat::Yuv420, 4),
            (8, ChromaFormat::Yuv420, 1),
        ] {
            let tag = format!("{depth}-bit {chroma:?}");
            let (cfg, g) = geom(64, 64, chroma);
            let cfg = Config { bit_depth: depth, ..cfg };
            let g = Geometry { bit_depth: depth, ..g };
            let sps_rbsp = write_sps(&cfg, &g, 16, None);
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&sps_rbsp)).unwrap_or_else(|e| panic!("{tag}: SPS rejected: {e}"));
            assert_eq!(sps.bit_depth_luma, depth, "{tag}: luma depth");
            assert_eq!(sps.bit_depth_chroma, depth, "{tag}: chroma depth");
            assert_eq!(u32::from(sps.ptl.profile_idc), want_profile, "{tag}: general_profile_idc");
            // The VPS carries the same PTL; it must at least be a legal one.
            assert!(!write_vps(&cfg, &g).is_empty());

            let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false, true))).expect("PPS");
            pps.resolve_tiles(&sps).expect("tiles");

            // The quantiser range: the parser's floor is -QpBdOffsetY,
            // which this encoder's `u8` quantiser never reaches — but the
            // header syntax does, and a deep decoder must take it.
            let qp_bd_offset = 6 * (depth as i32 - 8);
            let parse_at = |slice_qp: i32| -> Result<i32, crate::Error> {
                let mut w = BitWriter::with_capacity(64);
                w.bits(8, ((NAL_TRAIL_R as u32) & 0x3f) << 1);
                w.bits(8, 1);
                let h = SliceHeader { kind: Kind::P, poc_lsb: 2, qp: slice_qp, log2_max_poc_lsb: 16, ref_deltas: vec![-2], kept_deltas: Vec::new(), sao: None, pred_weights: None };
                write_slice_header(&h, 26, NAL_TRAIL_R, true, &mut w);
                w.flag(true);
                w.align_zero();
                let rbsp = w.into_rbsp();
                let nal = HevcNalHeader::parse(&rbsp).ok_or_else(|| crate::Error::bitstream("NAL header"))?;
                let (parsed, _, _) = ParsedHeader::parse(&rbsp, nal, &|_| Some(pps.clone()), &|_| Some(sps.clone()), None)?;
                Ok(parsed.slice_qp)
            };
            assert_eq!(parse_at(51).unwrap_or_else(|e| panic!("{tag}: QP 51: {e}")), 51, "{tag}");
            assert_eq!(parse_at(-qp_bd_offset).unwrap_or_else(|e| panic!("{tag}: QP -QpBdOffset: {e}")), -qp_bd_offset, "{tag}");
            assert!(parse_at(-qp_bd_offset - 1).is_err(), "{tag}: a quantiser below -QpBdOffsetY must be rejected");
            assert!(parse_at(52).is_err(), "{tag}: a quantiser above 51 must be rejected");

            // The PPS quantiser has the same range (`init_qp_minus26`,
            // 7.4.3.3.1: `-(26 + QpBdOffsetY)..=25`), and the parser holds
            // it to the SPS it is resolved against.
            let deep_pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(-qp_bd_offset, false, true)))
                .unwrap_or_else(|e| panic!("{tag}: PPS at init_qp -QpBdOffset rejected: {e}"));
            assert_eq!(deep_pps.init_qp, -qp_bd_offset, "{tag}: init_qp");
        }
    }

    /// Whole CTBs (50x34, below 64 both ways) and partial ones (90x34)
    /// both crop back to the requested size.
    #[test]
    fn the_conformance_window_recovers_the_requested_size() {
        for (w, h, coded) in [(50, 34, (64, 48)), (90, 34, (96, 40))] {
            let (cfg, g) = geom(w, h, ChromaFormat::Yuv420);
            assert_eq!((g.coded_width, g.coded_height), coded, "{w}x{h}");
            let sps = write_sps(&cfg, &g, 8, None);
            let parsed = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps)).unwrap();
            let (l, r, t, b) = parsed.conf_win;
            assert_eq!((g.coded_width - l - r, g.coded_height - t - b), (w, h), "{w}x{h}");
        }
    }

    /// Under the coding quadtree the CTB is 32 and the coded picture the
    /// smallest legal one, whole 8x8 minimum coding blocks, whatever the
    /// edge CTBs are left holding — unless the picture is narrower and
    /// shorter than 64. There, and at depth 0, it is whole CTBs, 16 or 32 by
    /// least padding. See `Geometry::new`.
    #[test]
    fn the_coded_picture_is_minimal_under_the_quadtree_and_whole_ctus_at_depth_0_or_below_64() {
        let at = |w: u32, h: u32, depth: Option<u32>| {
            let g = Geometry::new(&Config { max_cu_depth: depth, ..geom(w, h, ChromaFormat::Yuv420).0 });
            (g.log2_ctb, g.coded_width, g.coded_height, g.ctbs_wide, g.ctbs_high)
        };
        for depth in [None, Some(1), Some(2)] {
            assert_eq!(at(64, 64, depth), (5, 64, 64, 2, 2));
            // One side of 64 is enough to leave the exception: 16x16 CTBs
            // would code 64x48 whole, and the quadtree codes 32x32 ones.
            assert_eq!(at(64, 48, depth), (5, 64, 48, 2, 2));
            assert_eq!(at(48, 64, depth), (5, 48, 64, 2, 2));
            assert_eq!(at(88, 44, depth), (5, 88, 48, 3, 2));
            assert_eq!(at(1280, 720, depth), (5, 1280, 720, 40, 23));
            assert_eq!(at(1366, 768, depth), (5, 1368, 768, 43, 24));
            assert_eq!(at(3840, 2160, depth), (5, 3840, 2160, 120, 68));
            // Narrower and shorter than 64: the depth-0 geometry. 50x34 is
            // the clip the exception was fitted to; 63x40 and 40x63 sit on
            // its edge. 63x63 codes 64x64 either way, 32x32 CTBs on a tie.
            for (w, h) in [(50, 34), (48, 48), (24, 24), (63, 63), (63, 40), (40, 63)] {
                assert_eq!(at(w, h, depth), at(w, h, Some(0)), "{w}x{h} depth {depth:?}");
            }
            assert_eq!(at(50, 34, depth), (4, 64, 48, 4, 3));
            assert_eq!(at(63, 40, depth), (4, 64, 48, 4, 3));
            assert_eq!(at(63, 63, depth), (5, 64, 64, 2, 2));
        }
        assert_eq!(at(64, 64, Some(0)), (5, 64, 64, 2, 2));
        assert_eq!(at(64, 48, Some(0)), (4, 64, 48, 4, 3));
        assert_eq!(at(48, 48, Some(0)), (4, 48, 48, 3, 3));
        // 24x24 pads to 32x32 under either CTB size; the tie goes to 32.
        assert_eq!(at(24, 24, Some(0)), (5, 32, 32, 1, 1));
        // 50x34: CTB 16 pads to 64x48, CTB 32 to 64x64 — 16 pads less.
        assert_eq!(at(50, 34, Some(0)), (4, 64, 48, 4, 3));
        assert_eq!(at(88, 44, Some(0)), (4, 96, 48, 6, 3));
        assert_eq!(at(1280, 720, Some(0)), (4, 1280, 720, 80, 45));
    }

    /// The whole-CTB rule never codes 16x16 CTBs beyond level 4.1, where
    /// every level requires 32 or more (A.4.1 d). Each pair sits either
    /// side of one limit, and in each CTB 16 pads less: 2032x1088 is
    /// 2,210,816 samples in 16x16 CTBs and 2064x1088 is 2,245,632, against
    /// `MaxLumaPs` 2,228,224. 4208 and 4224 fall either side of the longest
    /// side, 4222, in both directions. 3840x2160 is the size that used to
    /// take CTB 16.
    #[test]
    fn whole_ctbs_are_never_16_beyond_level_4_1() {
        let at = |w: u32, h: u32| {
            let g = Geometry::new(&Config { max_cu_depth: Some(0), ..geom(w, h, ChromaFormat::Yuv420).0 });
            (g.log2_ctb, g.coded_width, g.coded_height)
        };
        assert_eq!(at(3840, 2160), (5, 3840, 2176));
        assert_eq!(at(2032, 1088), (4, 2032, 1088));
        assert_eq!(at(2064, 1088), (5, 2080, 1088));
        assert_eq!(at(4208, 16), (4, 4208, 16));
        assert_eq!(at(4224, 16), (5, 4224, 32));
        assert_eq!(at(16, 4208), (4, 16, 4208));
        assert_eq!(at(16, 4224), (5, 32, 4224));
        // The quadtree's partial CTBs are 32 at every size.
        let g = Geometry::new(&geom(3840, 2160, ChromaFormat::Yuv420).0);
        assert_eq!((g.log2_ctb, g.coded_width, g.coded_height), (5, 3840, 2160));
    }

    /// The reference picture set a P or B slice carries is written here
    /// and read by the production parser, so this test drives both: build
    /// the header, hand it to `SliceHeader::parse` with the encoder's own
    /// SPS and PPS, and check the reference pictures come back.
    ///
    /// The ordering is the trap. The set is not a list of deltas in POC
    /// order: it is *all* the negatives, nearest first, each coded as a
    /// difference from the previous one, and only then all the positives
    /// the same way. A B slice with one anchor either side is the
    /// smallest case that can tell the two arrangements apart — writing
    /// them interleaved parses as two negatives and puts the future
    /// anchor in the past. Kept pictures (`kept_deltas`) must come back
    /// in the same set, in delta order among the used ones, flagged
    /// unused, with the active counts unchanged.
    #[test]
    fn a_slice_header_carries_its_reference_pictures_where_the_parser_looks() {
        use crate::nal::HevcNalHeader;
        use crate::hevc::pps::Pps;
        use crate::hevc::slice::{SliceHeader as ParsedHeader, SliceType};
        use crate::hevc::sps::Sps;

        let (cfg, g) = geom(64, 64, ChromaFormat::Yuv420);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, None))).expect("SPS");
        let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false, true))).expect("PPS");
        pps.resolve_tiles(&sps).expect("tiles");

        // Coding order puts this picture between its anchors: POC 4, with
        // POC 2 behind it and POC 8 ahead; POC 0 and -2 are older anchors
        // a later picture still needs.
        for (kind, deltas, kept, want_neg, want_pos) in [
            (Kind::P, vec![-2i32], vec![], vec![(-2i32, true)], vec![]),
            (Kind::B, vec![-2, 4], vec![], vec![(-2, true)], vec![(4, true)]),
            (Kind::B, vec![4, -2], vec![], vec![(-2, true)], vec![(4, true)]),
            (Kind::P, vec![-2], vec![-4], vec![(-2, true), (-4, false)], vec![]),
            (Kind::B, vec![-2, 4], vec![-6, -4], vec![(-2, true), (-4, false), (-6, false)], vec![(4, true)]),
        ] {
            let h = SliceHeader {
                kind,
                poc_lsb: 4,
                qp: 30,
                log2_max_poc_lsb: 16,
                ref_deltas: deltas.clone(),
                kept_deltas: kept.clone(),
                sao: None,
                pred_weights: None,
            };
            let mut w = BitWriter::with_capacity(64);
            w.bits(8, ((NAL_TRAIL_R as u32) & 0x3f) << 1);
            w.bits(8, 1);
            write_slice_header(&h, 26, NAL_TRAIL_R, true, &mut w);
            w.flag(true); // byte_alignment()
            w.align_zero();
            let rbsp = w.into_rbsp();

            let nal = HevcNalHeader::parse(&rbsp).expect("NAL header");
            let (parsed, _, _) = ParsedHeader::parse(
                &rbsp,
                nal,
                &|_| Some(pps.clone()),
                &|_| Some(sps.clone()),
                None,
            )
            .unwrap_or_else(|e| panic!("{kind:?} header with {deltas:?} must parse: {e}"));

            assert_eq!(
                parsed.slice_type,
                match kind {
                    Kind::B => SliceType::B,
                    Kind::P => SliceType::P,
                    _ => SliceType::I,
                },
                "slice type"
            );
            assert_eq!(parsed.slice_qp, 30, "slice QP");
            assert_eq!(parsed.st_rps.neg, want_neg, "past references for {kind:?} {deltas:?} kept {kept:?}");
            assert_eq!(parsed.st_rps.pos, want_pos, "future references for {kind:?} {deltas:?} kept {kept:?}");
            // One active reference per list, taken from the PPS defaults
            // rather than overridden — and B gets a second list where P
            // does not. Kept pictures add none.
            assert_eq!(
                parsed.num_ref_idx,
                match kind {
                    Kind::B => [1, 1],
                    _ => [1, 0],
                },
                "active reference counts for {kind:?}"
            );
            assert_eq!(parsed.max_num_merge_cand, 5, "MaxNumMergeCand");
            assert!(!parsed.mvd_l1_zero, "mvd_l1_zero_flag is written false");
        }
    }

    /// `pred_weight_table` round-trips through the production header
    /// parser in every spelling regime: no weights (a table of defaults
    /// under a set PPS flag), luma only, luma and chroma, negative and
    /// large weights, offsets at the edge of their range, a B slice's
    /// second list, monochrome (no chroma syntax), and 10-bit offsets
    /// (spelled in 8-bit units, held shifted). The parsed table must
    /// equal the written one field for field, and everything after it
    /// in the header — `MaxNumMergeCand`, the slice QP — must still land.
    #[test]
    fn a_pred_weight_table_round_trips_through_the_parser() {
        use crate::hevc::pps::Pps;
        use crate::hevc::slice::{SliceHeader as ParsedHeader, WeightEntry};
        use crate::hevc::sps::Sps;
        use crate::nal::HevcNalHeader;

        let entry = |lw: i32, lo: i32, cw: [i32; 2], co: [i32; 2]| WeightEntry { luma: (lw, lo), chroma: [(cw[0], co[0]), (cw[1], co[1])] };
        // (chroma format, bit depth, kind, luma denom, chroma denom, list 0, list 1)
        let cases: Vec<(ChromaFormat, u32, Kind, u32, u32, Vec<WeightEntry>, Vec<WeightEntry>)> = vec![
            // All defaults: every flag clear, the table still present.
            (ChromaFormat::Yuv420, 8, Kind::P, 6, 6, vec![entry(64, 0, [64, 64], [0, 0])], vec![]),
            // Luma only: a fade's gain and a small offset.
            (ChromaFormat::Yuv420, 8, Kind::P, 6, 6, vec![entry(48, -3, [64, 64], [0, 0])], vec![]),
            // Luma and chroma, chroma denom differing, negative weight,
            // offsets at both edges of the 8-bit range.
            (ChromaFormat::Yuv420, 8, Kind::P, 5, 7, vec![entry(-40, 127, [200, 1], [-128, 127])], vec![]),
            // Denominator 0 (weights in whole units), 4:4:4, the luma
            // offset at the floor.
            (ChromaFormat::Yuv444, 8, Kind::P, 0, 0, vec![entry(3, -128, [2, 0], [5, -6])], vec![]),
            // Two entries in list 0 — a two-reference P, whose header
            // must declare the count for the reader to read both — with
            // mixed flags.
            (ChromaFormat::Yuv420, 8, Kind::P, 6, 6, vec![entry(64, 0, [64, 64], [0, 0]), entry(52, -4, [60, 64], [2, 0])], vec![]),
            // A B slice with both lists.
            (ChromaFormat::Yuv422, 8, Kind::B, 6, 6, vec![entry(70, 2, [64, 64], [0, 0])], vec![entry(58, -2, [60, 68], [3, -3])]),
            // Monochrome: no chroma syntax at all.
            (ChromaFormat::Monochrome, 8, Kind::P, 4, 4, vec![entry(12, 9, [16, 16], [0, 0])], vec![]),
            // Ten bits: offsets held shifted by two, spelled in 8-bit units.
            (ChromaFormat::Yuv420, 10, Kind::P, 6, 6, vec![entry(50, -12 << 2, [64, 70], [0, 8 << 2])], vec![]),
            // A B slice at ten bits, both lists weighted in every
            // component, offsets of both signs held shifted.
            (ChromaFormat::Yuv420, 10, Kind::B, 6, 6, vec![entry(56, 12 << 2, [60, 64], [6 << 2, 3 << 2])], vec![entry(72, -10 << 2, [66, 68], [-2 << 2, -6 << 2])]),
            // A B slice weighting list 0 alone: list 1's entry is the
            // default, its flags clear, while list 0's are set.
            (ChromaFormat::Yuv420, 8, Kind::B, 6, 6, vec![entry(48, -3, [64, 64], [0, 0])], vec![entry(64, 0, [64, 64], [0, 0])]),
            // A monochrome B slice: no chroma syntax in either list, and
            // list 1 weighted while list 0 is not.
            (ChromaFormat::Monochrome, 8, Kind::B, 6, 6, vec![entry(64, 0, [64, 64], [0, 0])], vec![entry(80, 5, [64, 64], [0, 0])]),
        ];
        for (chroma, bit_depth, kind, ld, cd, l0, l1) in cases {
            let tag = format!("{chroma:?} {bit_depth}-bit {kind:?} denoms {ld}/{cd} l0 {l0:?} l1 {l1:?}");
            let cfg = Config { width: 64, height: 64, chroma, bit_depth, max_refs: l0.len() as u32, bframes: 1, ..Config::default() };
            let g = Geometry::new(&cfg);
            let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, None))).expect("SPS");
            let opts = PpsOptions { weighted_pred: kind == Kind::P, weighted_bipred: kind == Kind::B, ..PpsOptions::default() };
            let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps_opts(26, false, true, &opts))).expect("PPS");
            pps.resolve_tiles(&sps).expect("tiles");
            assert_eq!(pps.weighted_pred, kind == Kind::P, "{tag}: weighted_pred_flag");
            assert_eq!(pps.weighted_bipred, kind == Kind::B, "{tag}: weighted_bipred_flag");

            let table = PredWeightTable { luma_log2_denom: ld, chroma_log2_denom: if chroma == ChromaFormat::Monochrome { ld } else { cd }, lists: [l0.clone(), l1.clone()] };
            let pw = PredWeights { table: table.clone(), chroma: chroma != ChromaFormat::Monochrome, bit_depth_luma: bit_depth, bit_depth_chroma: bit_depth };
            // One past reference per list-0 entry, one future for a B.
            let mut ref_deltas: Vec<i32> = (1..=l0.len() as i32).map(|d| -2 * d).collect();
            if kind == Kind::B {
                ref_deltas.push(2);
            }
            let h = SliceHeader { kind, poc_lsb: 8, qp: 31, log2_max_poc_lsb: 16, ref_deltas, kept_deltas: Vec::new(), sao: None, pred_weights: Some(pw) };
            let mut w = BitWriter::with_capacity(64);
            w.bits(8, ((NAL_TRAIL_R as u32) & 0x3f) << 1);
            w.bits(8, 1);
            write_slice_header(&h, 26, NAL_TRAIL_R, true, &mut w);
            w.flag(true); // byte_alignment()
            w.align_zero();
            let rbsp = w.into_rbsp();
            let nal = HevcNalHeader::parse(&rbsp).expect("NAL header");
            let (parsed, _, _) = ParsedHeader::parse(&rbsp, nal, &|_| Some(pps.clone()), &|_| Some(sps.clone()), None)
                .unwrap_or_else(|e| panic!("{tag}: the header must parse: {e}"));
            assert_eq!(parsed.num_ref_idx, [l0.len() as u32, if kind == Kind::B { 1 } else { 0 }], "{tag}: the active counts the header declares");
            assert_eq!(parsed.pred_weights.as_ref(), Some(&table), "{tag}: the parsed table differs from the written one");
            assert_eq!(parsed.max_num_merge_cand, 5, "{tag}: what follows the table did not land");
            assert_eq!(parsed.slice_qp, 31, "{tag}: the slice QP after the table");
        }
    }

    #[test]
    fn the_nal_header_is_two_bytes_and_carries_temporal_id_plus_one() {
        let n = annexb(NAL_SPS, &[0xaa]);
        assert_eq!(&n[..4], &[0, 0, 0, 1]);
        assert_eq!(n[4] >> 1, NAL_SPS);
        assert_eq!(n[5], 1, "nuh_temporal_id_plus1 must be 1, not 0");
    }
}
