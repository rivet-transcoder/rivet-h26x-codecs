//! Writing H.264 headers and the macroblock layer.
//!
//! The first thing built here is the simplest bitstream that is legal and
//! decodable: every macroblock coded as `I_PCM`, which carries its samples
//! raw. There is no prediction, no transform, no quantisation and no residual
//! coding in it, so nothing about picture quality is being decided yet — and
//! that is the point. It exercises the whole envelope in one step: sequence
//! and picture parameter sets, the slice header, the macroblock layer, the
//! CAVLC and CABAC paths through `mb_type`, NAL framing and
//! emulation prevention.
//!
//! It is also *exactly* lossless by construction, which means the gate's
//! strictest property — reconstruction equal to the source, byte for byte —
//! applies to it immediately. An encoder whose first output is exact removes
//! every quality question from the first round of debugging, leaving only the
//! question of whether the bitstream is well-formed. Everything after this is
//! a quality improvement on an envelope that is already proven against
//! libavcodec.

use crate::bitwriter::BitWriter;
use crate::cabac_enc::CabacEncoder;
use crate::encode::gop::Kind;
use crate::h264::SliceType;
use crate::h264::slice::PredWeightTable;
use crate::h264::cabac_mb::{CabacState, MB_TYPE_I_PCM, write_mb_type_i_cabac};
use crate::encode::{ColourDescription, Config, ContentLightLevel, Entropy, MasteringDisplay};
use crate::picture::ChromaFormat;
use crate::sample::Sample;

pub use crate::encode::h265_syntax::Cpb;

/// Coded slice of a non-IDR picture.
pub const NAL_SLICE: u8 = 1;
/// Coded slice of an IDR picture.
pub const NAL_IDR: u8 = 5;
/// Supplemental enhancement information.
pub const NAL_SEI: u8 = 6;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 7;
/// Picture parameter set.
pub const NAL_PPS: u8 = 8;

/// Width of `dpb_output_delay`, in bits — the same 24 the other two
/// delays have (`Cpb`'s lengths), because there is no reason for the
/// three to differ.
const OUTPUT_DELAY_LENGTH: u32 = 24;

/// The clock ticks one frame lasts. `time_scale` is written as twice the
/// frame rate with `fixed_frame_rate_flag` set, because that flag's
/// definition counts a *frame* as `DeltaTfiDivisor` ticks and the divisor
/// is 2 for a frame picture without `pic_struct` (E.2.1) — the field-rate
/// clock every H.264 encoder writes, so that `cpb_removal_delay` steps by
/// two per frame.
pub const TICKS_PER_FRAME: u32 = 2;

/// `hrd_parameters()` (E.1.2) — one CPB, NAL HRD only. The inverse of
/// `h264::sps::parse_hrd`, which retains exactly the fields written here.
///
/// `BitRate` and `CpbSize` are `(value + 1) << (6 + scale)` and
/// `(value + 1) << (4 + scale)` exactly as in H.265, so the one [`Cpb`]
/// — already snapped to what those can carry — serves both codecs.
fn write_hrd(w: &mut BitWriter, cpb: &Cpb) {
    w.ue(0); // cpb_cnt_minus1 — one buffer
    w.bits(4, 0); // bit_rate_scale
    w.bits(4, 0); // cpb_size_scale
    w.ue((cpb.bit_rate >> 6) as u32 - 1); // bit_rate_value_minus1[0]
    w.ue((cpb.size >> 4) as u32 - 1); // cpb_size_value_minus1[0]
    // cbr_flag 0: a variable rate, for the reason the H.265 side gives —
    // the controller targets an average and stuffs nothing, so a constant
    // rate would declare something it does not do.
    w.flag(false); // cbr_flag[0]
    w.bits(5, cpb.initial_delay_length - 1); // initial_cpb_removal_delay_length_minus1
    w.bits(5, cpb.removal_delay_length - 1); // cpb_removal_delay_length_minus1
    w.bits(5, OUTPUT_DELAY_LENGTH - 1); // dpb_output_delay_length_minus1
    w.bits(5, 0); // time_offset_length — no pic_struct, so no time_offset
}

/// `vui_parameters()` (E.1.1) carrying only what was asked for: the
/// colour description and the chroma siting when the caller gave them,
/// and the clock the removal delays are counted in plus the NAL HRD when
/// a buffer was declared. Everything else is absent by its own flag — a
/// VUI is optional and this encoder wrote none until a buffer needed one
/// — and each part is present only under its own condition, so a stream
/// with a buffer and no colour is byte-identical to one from before
/// colour existed. The inverse of `h264::sps::parse_vui`, which keeps
/// every field written here.
fn write_vui(
    w: &mut BitWriter,
    colour: Option<&ColourDescription>,
    chroma_loc: Option<u8>,
    cpb: Option<&Cpb>,
    fps: u32,
) {
    w.flag(false); // aspect_ratio_info_present_flag
    w.flag(false); // overscan_info_present_flag
    write_video_signal_type(w, colour);
    write_chroma_loc(w, chroma_loc);
    match cpb {
        Some(cpb) => {
            w.flag(true); // timing_info_present_flag
            w.bits(32, 1); // num_units_in_tick
            w.bits(32, TICKS_PER_FRAME * fps.max(1)); // time_scale
            w.flag(true); // fixed_frame_rate_flag
            w.flag(true); // nal_hrd_parameters_present_flag
            write_hrd(w, cpb);
            w.flag(false); // vcl_hrd_parameters_present_flag
            w.flag(false); // low_delay_hrd_flag (present: a NAL HRD is)
        }
        None => {
            w.flag(false); // timing_info_present_flag
            w.flag(false); // nal_hrd_parameters_present_flag
            w.flag(false); // vcl_hrd_parameters_present_flag
            // low_delay_hrd_flag is absent: neither HRD is present.
        }
    }
    w.flag(false); // pic_struct_present_flag
    w.flag(false); // bitstream_restriction_flag
}

/// The `video_signal_type_present_flag` group of a VUI (E.1.1): the
/// three H.273 code points and the range flag, or the one zero flag that
/// says nothing about colour. `video_format` is 5, "unspecified" — the
/// value for content that is not a broadcast standard's — which is what
/// every encoder that writes this group writes.
///
/// Identical in H.264 and H.265 (E.2.1 copies E.1.1 field for field), so
/// the H.265 writer calls this one.
pub(crate) fn write_video_signal_type(w: &mut BitWriter, colour: Option<&ColourDescription>) {
    match colour {
        Some(c) => {
            w.flag(true); // video_signal_type_present_flag
            w.bits(3, 5); // video_format: unspecified
            w.flag(c.full_range); // video_full_range_flag
            w.flag(true); // colour_description_present_flag
            w.bits(8, u32::from(c.primaries)); // colour_primaries
            w.bits(8, u32::from(c.transfer)); // transfer_characteristics
            w.bits(8, u32::from(c.matrix)); // matrix_coefficients
        }
        None => w.flag(false), // video_signal_type_present_flag
    }
}

/// The `chroma_loc_info_present_flag` group of a VUI (E.1.1): one
/// `chroma_sample_loc_type` for each field, the same value twice because
/// this encoder codes frames, or the one zero flag that says nothing
/// about siting — under which every decoder assumes type 0.
///
/// Identical in H.264 and H.265 (E.2.1), so the H.265 writer calls this
/// one.
pub(crate) fn write_chroma_loc(w: &mut BitWriter, chroma_loc: Option<u8>) {
    match chroma_loc {
        Some(t) => {
            w.flag(true); // chroma_loc_info_present_flag
            w.ue(u32::from(t)); // chroma_sample_loc_type_top_field
            w.ue(u32::from(t)); // chroma_sample_loc_type_bottom_field
        }
        None => w.flag(false), // chroma_loc_info_present_flag
    }
}

/// One SEI message wrapped as an SEI NAL payload: `payloadType`,
/// `payloadSize` in the standard's 255-at-a-time form, the payload bytes
/// (already byte-aligned by their own trailing bits), then the RBSP's.
///
/// The payload arrives as *raw* RBSP bytes and the emulation prevention
/// is applied once, here, to the whole NAL. A payload that had already
/// been escaped would be escaped again — a timing SEI is mostly zero
/// bytes, exactly the pattern the escape targets — and a reader would
/// find `0x03` where a delay's bits should be. (Which is what the H.265
/// buffering period did until it was routed through here: its payload
/// went out escaped and sized as escaped, then the NAL was escaped
/// again — `00 00 03 03` on the wire, and a checker reading a stray
/// byte inside the initial delay.)
///
/// Shared by both syntaxes: an SEI message is the same bytes in H.264
/// and H.265, only the NAL header around it differs.
pub(crate) fn sei_nal(payload_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(payload.len() + 8);
    let mut t = payload_type;
    while t >= 255 {
        w.bits(8, 255);
        t -= 255;
    }
    w.bits(8, t);
    let mut n = payload.len();
    while n >= 255 {
        w.bits(8, 255);
        n -= 255;
    }
    w.bits(8, n as u32);
    for b in payload {
        w.bits(8, *b as u32);
    }
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// The `mastering_display_colour_volume` SEI (payloadType 137; D.1.29,
/// and H.265 D.2.28 field for field): the three primaries in the SEI's
/// own order — green, blue, red, the order both standards recommend and
/// every writer follows — then the white point and the two luminances,
/// all as the caller gave them. 24 bytes, byte-aligned by its own
/// syntax, so no payload alignment bits. The test below holds it
/// byte-identical to what x265 writes for the same values.
pub fn write_mastering_display_sei(m: &MasteringDisplay) -> Vec<u8> {
    let mut p = BitWriter::with_capacity(24);
    for (x, y) in [m.green, m.blue, m.red] {
        p.bits(16, u32::from(x)); // display_primaries_x[c]
        p.bits(16, u32::from(y)); // display_primaries_y[c]
    }
    p.bits(16, u32::from(m.white_point.0)); // white_point_x
    p.bits(16, u32::from(m.white_point.1)); // white_point_y
    p.bits(32, m.max_luminance); // max_display_mastering_luminance
    p.bits(32, m.min_luminance); // min_display_mastering_luminance
    sei_nal(137, &p.into_rbsp())
}

/// The `content_light_level_info` SEI (payloadType 144; D.1.31, H.265
/// D.2.35): the two light levels, four bytes.
pub fn write_content_light_level_sei(c: &ContentLightLevel) -> Vec<u8> {
    let mut p = BitWriter::with_capacity(4);
    p.bits(16, u32::from(c.max_cll)); // max_content_light_level
    p.bits(16, u32::from(c.max_fall)); // max_pic_average_light_level
    sei_nal(144, &p.into_rbsp())
}

/// A `buffering_period` SEI (D.1.2), for every IDR access unit: the
/// initial removal delay — the one number the schedule cannot derive —
/// and its offset, at the widths the SPS declared. `cpb` is what that SPS
/// wrote.
pub fn write_buffering_period_sei(cpb: &Cpb) -> Vec<u8> {
    let mut p = BitWriter::with_capacity(16);
    p.ue(0); // seq_parameter_set_id
    // NalHrdBpPresentFlag: one SchedSelIdx.
    p.bits(cpb.initial_delay_length, cpb.initial_removal_delay_90k()); // initial_cpb_removal_delay
    p.bits(cpb.initial_delay_length, 0); // initial_cpb_removal_delay_offset
    p.rbsp_trailing_bits();
    sei_nal(0, &p.into_rbsp())
}

/// A `pic_timing` SEI (D.1.3), for every access unit of a stream with a
/// NAL HRD: `cpb_removal_delay` — clock ticks since the removal of the
/// last buffering-period access unit, which is what fixes this picture's
/// removal time (C.1.2) — and `dpb_output_delay`, ticks from removal to
/// output. No `pic_struct`: the VUI does not present one.
pub fn write_pic_timing_sei(cpb: &Cpb, cpb_removal_delay: u32, dpb_output_delay: u32) -> Vec<u8> {
    let mut p = BitWriter::with_capacity(8);
    p.bits(cpb.removal_delay_length, cpb_removal_delay);
    p.bits(OUTPUT_DELAY_LENGTH, dpb_output_delay);
    p.rbsp_trailing_bits();
    sei_nal(1, &p.into_rbsp())
}

/// Prefix a NAL payload with its header byte and an Annex B start code.
///
/// Four-byte start codes throughout. Three would be legal and marginally
/// smaller, but the saving is a byte per NAL against the risk of getting the
/// rule about which may use the short form wrong, and this encoder has no
/// bitrate pressure yet.
pub fn annexb(nal_type: u8, nal_ref_idc: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 5);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.push(((nal_ref_idc & 3) << 5) | (nal_type & 0x1f));
    out.extend_from_slice(payload);
    out
}

/// Geometry the headers and the macroblock loop both need, derived once so
/// the two cannot disagree about it.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    /// Coded size, in macroblocks.
    pub mbs_wide: u32,
    /// See `mbs_wide`.
    pub mbs_high: u32,
    /// Coded luma size, which is the macroblock grid.
    pub coded_width: u32,
    /// See `coded_width`.
    pub coded_height: u32,
    /// Displayed luma size, which is what the caller asked for.
    pub width: u32,
    /// See `width`.
    pub height: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Bits per sample, 8 to 14.
    pub bit_depth: u32,
    /// The stream is interlaced (`frame_mbs_only_flag` 0): the frame
    /// height is a whole number of macroblock *pairs*, the SPS counts map
    /// units of two macroblock rows, and cropping is in field rows.
    pub interlaced: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mbaff: bool,
    /// This is the geometry of one *field* picture ([`Geometry::field`]):
    /// half the frame's rows, its macroblocks field macroblocks — the
    /// field scans, the field residual contexts, and a loop filter that
    /// treats horizontal edges as field edges.
    pub field_pic: bool,
    /// Per reference list, the vertical chroma vector offset of 8.4.1.4
    /// (Table 8-10) between this field picture and the field its
    /// reference index 0 names: 0 for the same parity (and always for a
    /// frame picture, and outside 4:2:0), −2 for a top field predicting
    /// from a bottom one, +2 for the reverse — in the eighth-sample units
    /// the chroma vector is in.
    pub chroma_mv_dy: [i32; 2],
}

impl Geometry {
    /// Derive the coded geometry from a configuration. An interlaced one
    /// rounds the height up to whole macroblock pairs (7.4.2.1.1:
    /// `FrameHeightInMbs = (2 - frame_mbs_only_flag) * PicHeightInMapUnits`).
    pub fn new(cfg: &Config) -> Self {
        let interlaced = cfg.interlace.is_some();
        let mbs_wide = cfg.width.div_ceil(16);
        let mbs_high = if interlaced { cfg.height.div_ceil(32) * 2 } else { cfg.height.div_ceil(16) };
        Self {
            mbs_wide,
            mbs_high,
            coded_width: mbs_wide * 16,
            coded_height: mbs_high * 16,
            width: cfg.width,
            height: cfg.height,
            chroma: cfg.chroma,
            bit_depth: cfg.bit_depth,
            interlaced,
            mbaff: interlaced && cfg.field_coding == crate::encode::FieldCoding::Mbaff,
            field_pic: false,
            chroma_mv_dy: [0; 2],
        }
    }

    /// The geometry of one field of this interlaced frame: every other
    /// row of it, so half the macroblock rows, half the coded height and
    /// half the displayed height (the height is even — the encoder refuses
    /// one that is not).
    pub fn field(&self) -> Geometry {
        debug_assert!(self.interlaced && self.mbs_high.is_multiple_of(2) && self.height.is_multiple_of(2));
        Geometry {
            mbs_high: self.mbs_high / 2,
            coded_height: self.coded_height / 2,
            height: self.height / 2,
            mbaff: false,
            field_pic: true,
            ..*self
        }
    }

    /// Chroma samples per macroblock, per plane.
    pub fn chroma_mb(&self) -> (u32, u32) {
        match self.chroma {
            ChromaFormat::Monochrome => (0, 0),
            ChromaFormat::Yuv420 => (8, 8),
            ChromaFormat::Yuv422 => (8, 16),
            ChromaFormat::Yuv444 => (16, 16),
        }
    }
}

/// The profile that admits this configuration.
///
/// I_PCM is in every profile, so what decides this is the format rather than
/// the coding tools (A.2): 4:4:4 needs High 4:4:4 Predictive, 4:2:2 High
/// 4:2:2, depths of 9 and 10 bits High 10 — and anything deeper than 10
/// bits, whatever its chroma format, is High 4:4:4 Predictive again, the
/// only profile whose `bit_depth_luma_minus8` may exceed 2. Monochrome
/// needs High. Claiming a lower profile than the stream needs is the kind
/// of error a decoder is entitled to reject the stream over, so this errs
/// upwards.
fn profile_idc(g: &Geometry) -> u8 {
    match g.chroma {
        _ if g.bit_depth > 10 => 244,
        ChromaFormat::Yuv444 => 244,
        ChromaFormat::Yuv422 => 122,
        _ if g.bit_depth > 8 => 110,
        ChromaFormat::Monochrome => 100,
        ChromaFormat::Yuv420 => 100,
    }
}

/// Whether the SPS carries the chroma/depth extension fields. Everything from
/// High upwards does.
fn has_chroma_extension(profile: u8) -> bool {
    matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135)
}

/// Sequence parameter set. With a coded picture buffer declared it
/// carries a VUI — the frame clock and the NAL HRD — and with a colour
/// description the VUI carries that; with neither, no VUI at all, so a
/// stream that declares no buffer and no colour is byte-identical to one
/// from before either existed.
pub fn write_sps(
    cfg: &Config,
    g: &Geometry,
    log2_max_frame_num: u32,
    log2_max_poc_lsb: u32,
    cpb: Option<&Cpb>,
) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(64);
    let profile = profile_idc(g);
    w.bits(8, profile as u32);
    // constraint_set0..5 then two reserved zero bits.
    w.bits(8, 0);
    // Level 5.1 unconditionally. Deriving the true level from size and rate
    // is a table lookup this encoder will want later; until then, claiming a
    // level that admits everything is honest, whereas claiming one too low
    // would be a stream a conforming decoder may refuse.
    w.bits(8, 51);
    w.ue(0); // seq_parameter_set_id
    if has_chroma_extension(profile) {
        w.ue(match g.chroma {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        });
        if g.chroma == ChromaFormat::Yuv444 {
            w.flag(false); // separate_colour_plane_flag
        }
        // One depth for both: the decoder refuses a stream whose luma and
        // chroma depths differ (`check_supported`, src/h264/decoder.rs),
        // monochrome included — the chroma field is still parsed and
        // compared there even though no chroma sample exists.
        w.ue(g.bit_depth - 8); // bit_depth_luma_minus8
        w.ue(g.bit_depth - 8); // bit_depth_chroma_minus8
        w.flag(false); // qpprime_y_zero_transform_bypass_flag
        w.flag(false); // seq_scaling_matrix_present_flag
    }
    w.ue(log2_max_frame_num - 4);
    w.ue(0); // pic_order_cnt_type 0
    w.ue(log2_max_poc_lsb - 4);
    w.ue(cfg.max_refs); // max_num_ref_frames
    w.flag(false); // gaps_in_frame_num_value_allowed_flag
    w.ue(g.mbs_wide - 1);
    if g.interlaced {
        // Map units are macroblock pairs: two rows each.
        w.ue(g.mbs_high / 2 - 1); // pic_height_in_map_units_minus1
        w.flag(false); // frame_mbs_only_flag
        w.flag(g.mbaff); // mb_adaptive_frame_field_flag
    } else {
        w.ue(g.mbs_high - 1); // frame_mbs_only_flag is 1, so map units are MBs
        w.flag(true); // frame_mbs_only_flag
    }
    // Required to be 1 when frame_mbs_only_flag is 0 (7.4.2.1.1), and 1
    // for every stream this encoder writes.
    w.flag(true); // direct_8x8_inference_flag
    // Cropping, because the coded size is rounded up to whole macroblocks and
    // the displayed size is not. The units are chroma samples horizontally
    // and, for frame pictures, chroma samples vertically — doubled in an
    // interlaced stream, which crops in field rows (`CropUnitY = SubHeightC
    // * (2 - frame_mbs_only_flag)`).
    let (cw, ch) = match g.chroma {
        ChromaFormat::Monochrome => (1, 1),
        ChromaFormat::Yuv420 => (2, 2),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv444 => (1, 1),
    };
    let ch = if g.interlaced { 2 * ch } else { ch };
    debug_assert!((g.coded_height - g.height).is_multiple_of(ch), "the encoder refuses a height its crop unit cannot reach");
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
    if cpb.is_some() || cfg.colour.is_some() || cfg.chroma_loc.is_some() {
        w.flag(true); // vui_parameters_present_flag
        write_vui(&mut w, cfg.colour.as_ref(), cfg.chroma_loc, cpb, cfg.fps);
    } else {
        w.flag(false); // vui_parameters_present_flag
    }
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// Picture parameter set.
pub fn write_pps(cfg: &Config, qp: u8) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(32);
    w.ue(0); // pic_parameter_set_id
    w.ue(0); // seq_parameter_set_id
    w.flag(cfg.entropy == Entropy::Cabac);
    // An interlaced frame picture carries its bottom field's POC as a
    // delta (`delta_pic_order_cnt_bottom`), which is what puts the field
    // order into a frame picture at all; a progressive stream has no
    // fields to order and keeps the flag 0.
    w.flag(cfg.interlace.is_some()); // bottom_field_pic_order_in_frame_present_flag
    w.ue(0); // num_slice_groups_minus1
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    // Explicit weighted prediction for P slices when asked for, and then
    // every P slice carries a `pred_weight_table`. B slices keep default
    // weighting — `weighted_bipred_idc` stays 0 — because the fit is one
    // line per reference and a two-list decision would need its own.
    w.flag(cfg.weighted_pred); // weighted_pred_flag
    w.bits(2, 0); // weighted_bipred_idc
    w.se(qp as i32 - 26); // pic_init_qp_minus26
    w.se(0); // pic_init_qs_minus26
    w.se(0); // chroma_qp_index_offset
    w.flag(true); // deblocking_filter_control_present_flag
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // redundant_pic_cnt_present_flag
    // The PPS extension (7.3.2.2). The reader takes it only when
    // `more_rbsp_data()` says the RBSP has not reached its stop bit
    // (src/h264/pps.rs:119), so writing nothing at all is what leaves a
    // PPS that offers no 8x8 transform byte-identical to the one this
    // encoder wrote before the transform existed — and that identity is
    // what makes "everything not using it is unchanged" checkable.
    //
    // `transform_8x8_mode_flag` needs a High profile, which every profile
    // `profile_idc` claims already is.
    if cfg.transform_8x8 {
        w.flag(true); // transform_8x8_mode_flag
        w.flag(false); // pic_scaling_matrix_present_flag
        w.se(0); // second_chroma_qp_index_offset, matching the first
    }
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// A P slice's `pred_weight_table` as the writer spells it: the table a
/// decoder will hold, and whether the stream has chroma — the reader's
/// `ChromaArrayType != 0` gate on the table's chroma half.
#[derive(Debug, Clone)]
pub struct PredWeights {
    /// The table: one list-0 entry per active reference, offsets in the
    /// syntax's 8-bit units (the reader shifts them to the sample depth).
    pub table: PredWeightTable,
    /// Whether the chroma denominator and weights are present.
    pub chroma: bool,
}

/// What a slice header needs that is not in the parameter sets.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// What the slice is coded as.
    pub kind: Kind,
    /// Counts reference pictures, and wraps at `log2_max_frame_num`.
    pub frame_num: u32,
    /// Distinguishes consecutive IDRs so a decoder cannot merge them.
    pub idr_pic_id: u32,
    /// The low bits of the picture order count.
    pub poc_lsb: u32,
    /// Quantiser for the slice.
    pub qp: u8,
    /// Width of the `frame_num` field, from the SPS.
    pub log2_max_frame_num: u32,
    /// Width of the `poc_lsb` field, from the SPS.
    pub log2_max_poc_lsb: u32,
    /// Whether later pictures may reference this one.
    pub reference: bool,
    /// Whether the deblocking filter runs over this slice. Always true
    /// today: the transform picture writers run the decoder's own filter
    /// over their reconstruction, and on the PCM and all-skip paths the
    /// filter provably does nothing (PCM macroblocks average to a qP of
    /// zero, all-skip edges have boundary strength zero). Kept as a field
    /// because a slice that legitimately wants the filter off — offsets,
    /// rate experiments — is a header question, not a rewrite.
    pub deblock: bool,
    /// Whether the slice is entropy-coded with CABAC — a P or B header
    /// then carries `cabac_init_idc`.
    pub cabac: bool,
    /// `direct_spatial_mv_pred_flag` (B slices only): true for the
    /// transform B path, whose encoder mirrors the spatial derivation;
    /// false for the legacy all-skip path, whose reconstruction assumes
    /// temporal direct over zero colocated motion.
    pub direct_spatial: bool,
    /// `pred_weight_table()`: present exactly for a P slice under a PPS
    /// that sets `weighted_pred_flag`, and `None` for every I and B slice
    /// (`weighted_bipred_idc` is 0).
    pub pred_weights: Option<PredWeights>,
    /// The SPS writes `frame_mbs_only_flag` 0, so `field_pic_flag` is in
    /// the header — and, since this encoder's PPS sets
    /// `bottom_field_pic_order_in_frame_present_flag` for exactly those
    /// streams, a frame picture's `delta_pic_order_cnt_bottom` too.
    pub interlaced: bool,
    /// A field picture's `bottom_field_flag`; `None` for a frame picture.
    pub bottom_field: Option<bool>,
    /// `delta_pic_order_cnt_bottom` of an interlaced frame picture: the
    /// bottom field's POC less the top's, +1 top field first and −1
    /// bottom field first.
    pub delta_poc_bottom: i32,
}

/// `slice_type` for an I, P or B slice, in the "all slices of this picture
/// have this type" form (5..9), which is true here and lets a decoder know it.
fn slice_type_code(kind: Kind) -> u32 {
    match kind {
        Kind::Idr | Kind::I => 7,
        Kind::P => 5,
        Kind::B => 6,
    }
}

/// Slice header, up to but not including the macroblock data.
pub fn write_slice_header(h: &SliceHeader, pps_qp: u8, w: &mut BitWriter) {
    w.ue(0); // first_mb_in_slice
    w.ue(slice_type_code(h.kind));
    w.ue(0); // pic_parameter_set_id
    w.bits(h.log2_max_frame_num, h.frame_num);
    if h.interlaced {
        w.flag(h.bottom_field.is_some()); // field_pic_flag
        if let Some(bottom) = h.bottom_field {
            w.flag(bottom); // bottom_field_flag
        }
    } else {
        // frame_mbs_only_flag is 1, so no field_pic_flag here.
        debug_assert!(h.bottom_field.is_none(), "a progressive stream has no field pictures");
    }
    if h.kind == Kind::Idr {
        w.ue(h.idr_pic_id);
    }
    w.bits(h.log2_max_poc_lsb, h.poc_lsb);
    if h.interlaced && h.bottom_field.is_none() {
        w.se(h.delta_poc_bottom); // delta_pic_order_cnt_bottom
    }
    if h.kind == Kind::B {
        w.flag(h.direct_spatial); // direct_spatial_mv_pred_flag
    }
    if h.kind == Kind::P || h.kind == Kind::B {
        w.flag(false); // num_ref_idx_active_override_flag
        // ref_pic_list_modification
        w.flag(false);
        if h.kind == Kind::B {
            w.flag(false);
        }
    }
    // pred_weight_table(), between the list modifications and the reference
    // marking (7.3.3).
    if let Some(pw) = h.pred_weights.as_ref() {
        debug_assert_eq!(h.kind, Kind::P, "only a P slice carries a table: weighted_bipred_idc is 0");
        write_pred_weight_table(pw, w);
    }
    if h.reference {
        if h.kind == Kind::Idr {
            w.flag(false); // no_output_of_prior_pics_flag
            w.flag(false); // long_term_reference_flag
        } else {
            w.flag(false); // adaptive_ref_pic_marking_mode_flag
        }
    }
    if h.cabac && h.kind != Kind::Idr && h.kind != Kind::I {
        // `cabac_init_idc`, the missing-bit twin of the one-spurious-bit
        // class: the reader takes it on every CABAC P/B slice, before
        // `slice_qp_delta` (7.3.3), and a writer that omits it hands the
        // QP field's bits to the initialisation index. Zero, matching the
        // `CabacState::new(_, 0, _)` the slice-data writers run.
        w.ue(0);
    }
    w.se(h.qp as i32 - pps_qp as i32); // slice_qp_delta
    // deblocking_filter_control_present_flag is 1 in the PPS. The offsets
    // are only present while the filter is on (7.3.3).
    w.ue(if h.deblock { 0 } else { 1 }); // disable_deblocking_filter_idc
    if h.deblock {
        w.se(0); // slice_alpha_c0_offset_div2
        w.se(0); // slice_beta_offset_div2
    }
}

/// Write `pred_weight_table()` for a P slice (7.3.3.2): the exact inverse of
/// the reader's parse (src/h264/slice.rs) for a table it will hold as
/// `pw.table` — `luma_log2_weight_denom`, `chroma_log2_weight_denom` when
/// the stream has chroma, then per list-0 entry the luma flag with the
/// weight and offset behind it, and the chroma flag with both components'
/// weights and offsets behind that. A flag is set exactly when the entry
/// differs from the default `(1 << denom, 0)` the reader infers for an
/// unflagged one, so a table of defaults costs its flags and nothing more.
/// Weights and offsets are the syntax's own values: the weight itself (not
/// a delta, unlike H.265), the offset in 8-bit units.
pub fn write_pred_weight_table(pw: &PredWeights, w: &mut BitWriter) {
    let t = &pw.table;
    w.ue(t.luma_log2_denom); // luma_log2_weight_denom
    if pw.chroma {
        w.ue(t.chroma_log2_denom); // chroma_log2_weight_denom
    }
    let luma_default = (1i32 << t.luma_log2_denom, 0i32);
    let chroma_default = [(1i32 << t.chroma_log2_denom, 0i32); 2];
    for e in &t.lists[0] {
        let luma = e.luma != luma_default;
        w.flag(luma); // luma_weight_l0_flag
        if luma {
            w.se(e.luma.0); // luma_weight_l0
            w.se(e.luma.1); // luma_offset_l0
        }
        if pw.chroma {
            let chroma = e.chroma != chroma_default;
            w.flag(chroma); // chroma_weight_l0_flag
            if chroma {
                for (cw, co) in e.chroma {
                    w.se(cw); // chroma_weight_l0
                    w.se(co); // chroma_offset_l0
                }
            }
        }
    }
}

/// Write one macroblock as `I_PCM`, taking its samples from the source and
/// leaving the same samples in `dst`, the padded reconstruction.
///
/// Samples outside the displayed picture — the padding a non-multiple-of-16
/// size implies — are filled by edge replication. Any value would be legal,
/// since the cropping rectangle excludes them from display, but replication
/// keeps the coded picture free of edges that would cost bits once this
/// encoder predicts and transforms rather than copying.
pub fn write_pcm_macroblock<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    mb_x: u32,
    mb_y: u32,
    planes: &[Plane<'_, S>],
    dst: &mut [Recon<S>],
) {
    // `mb_type` 25 is I_PCM in an I slice, as ue(v) for CAVLC.
    w.ue(25);
    w.align_zero(); // pcm_alignment_zero_bit
    write_pcm_samples(w, g, mb_x, mb_y, planes, dst);
}

/// The CABAC slice data of an all-`I_PCM` picture: `cabac_alignment_one_bit`,
/// every macroblock, then the terminate that closes the slice.
///
/// The arithmetic engine does not run across the whole slice. `I_PCM`'s
/// `mb_type` ends in a terminate bin of 1, and that *flushes* the codeword;
/// the samples follow byte-aligned as plain bits, and a new engine
/// initialises after them (9.3.1.2). So the slice is a chain of short
/// codewords, each starting on a byte boundary and each closed by the
/// terminate that introduces the next block of samples:
///
/// ```text
/// align_one | mbtype(0) term=1 | PCM(0) | eos=0 mbtype(1) term=1 | PCM(1) | ... | eos=1
/// ```
///
/// The *context state* is carried across all of it. Re-initialising the
/// engine does not re-initialise the contexts — the decoder's
/// `Cabac::reinit` likewise leaves its `CabacState` alone — and a writer that
/// reset them per macroblock would agree with the decoder on the first
/// macroblock and diverge on the second.
pub fn write_pcm_slice_data_cabac<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    qp: u8,
    planes: &[Plane<'_, S>],
    dst: &mut [Recon<S>],
) {
    // `cabac_alignment_one_bit` until the slice data starts on a byte.
    w.align_one();
    // The same initialisation the decoder runs, from the same tables. I
    // slices have no `cabac_init_idc`, so the value passed is not read.
    let mut st = CabacState::new(SliceType::I, 0, qp as i32);
    let total = g.mbs_wide * g.mbs_high;
    for idx in 0..total {
        {
            let mut e = CabacEncoder::new(w);
            if idx > 0 {
                // `end_of_slice_flag` of the macroblock before this one. A
                // zero does not flush, so this engine carries on into the
                // `mb_type` below.
                e.encode_terminate(0);
            }
            // `mb_type`, spelled by the macroblock-layer writer (its
            // terminate bin of 1 flushes the engine, as I_PCM requires).
            // The first bin's ctxIdxInc counts available neighbours that
            // are not I_NxN (9.3.3.1.1.3); every macroblock here is I_PCM,
            // so an available neighbour always contributes one, and with a
            // single slice per picture "available" is just "inside the
            // picture".
            let inc = (idx % g.mbs_wide > 0) as usize + (idx / g.mbs_wide > 0) as usize;
            write_mb_type_i_cabac(&mut e, &mut st, inc, MB_TYPE_I_PCM);
        }
        // `pcm_alignment_zero_bit`, then the samples as plain bits.
        w.align_zero();
        write_pcm_samples(w, g, idx % g.mbs_wide, idx / g.mbs_wide, planes, dst);
    }
    // The last macroblock's `end_of_slice_flag`, in an engine of its own
    // because the one before it was flushed by that macroblock's I_PCM.
    {
        let mut e = CabacEncoder::new(w);
        e.encode_terminate(1);
    }
    // The flush already wrote a one as its last bit, and that bit *is* the
    // `rbsp_stop_one_bit` (9.3.4.6). What is left is padding to the byte —
    // writing `rbsp_trailing_bits` here instead would emit a second stop bit
    // and leave a byte of rubbish after the slice.
    w.align_zero();
}

/// The raw samples of one `I_PCM` macroblock, and the same values into the
/// reconstruction — which is what makes the coding exactly lossless.
///
/// Shared by both entropy coders: only how `mb_type` is spelled differs
/// between them, and the alignment bit and samples that follow are identical.
/// Sources narrower than a whole macroblock repeat their edge sample, which
/// is what the cropping in the SPS then hides.
///
/// Each sample is `BitDepth` bits wide (7.3.5: `pcm_sample_luma` is
/// `u(v)` with `v = BitDepth_Y`, and the chroma likewise) — a 10-bit
/// picture's PCM macroblock is 1.25 times the bytes of an 8-bit one, and
/// a writer that kept eight bits would hand the reader every sample's
/// low byte shifted into its neighbour.
fn write_pcm_samples<S: Sample>(
    w: &mut BitWriter,
    g: &Geometry,
    mb_x: u32,
    mb_y: u32,
    planes: &[Plane<'_, S>],
    dst: &mut [Recon<S>],
) {
    let bd = g.bit_depth;
    let (cw, ch) = g.chroma_mb();
    let sizes: [(u32, u32); 3] = [(16, 16), (cw, ch), (cw, ch)];
    for (p, &(bw, bh)) in sizes.iter().enumerate() {
        if bw == 0 || p >= planes.len() {
            continue;
        }
        let src = &planes[p];
        let (sx, sy) = (mb_x * bw, mb_y * bh);
        for y in 0..bh {
            let syy = (sy + y).min(src.height.saturating_sub(1));
            for x in 0..bw {
                let sxx = (sx + x).min(src.width.saturating_sub(1));
                let v = src.data[syy as usize * src.stride + sxx as usize].to_i32();
                w.bits(bd, v as u32);
                let d = &mut dst[p];
                let i = ((sy + y) as usize + d.pad) * d.stride + (sx + x) as usize + d.pad;
                d.data[i] = S::from_i32(v);
            }
        }
    }
}

/// A source plane: samples, stride, and the size actually present. The
/// sample type is the picture's — `u8` at 8 bits, `u16` deeper — already
/// unpacked from the caller's bytes by the encoder's face.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a, S: Sample> {
    /// Samples, row-major.
    pub data: &'a [S],
    /// Samples per row, which may exceed `width`.
    pub stride: usize,
    /// Samples present horizontally.
    pub width: u32,
    /// Rows present.
    pub height: u32,
}

/// The reconstruction plane, which is the *decoder's* padded plane.
///
/// Deliberately not a type of the encoder's own. `h264::intra`'s predictors
/// read their neighbours directly out of the border of this layout, so
/// sharing the type is what lets the encoder reuse them — and reusing them is
/// what makes the encoder's reconstruction identical to a decoder's by
/// construction rather than by care. A second set of predictors would be a
/// second thing to keep in step, and the drift would show up as a SELF
/// failure hundreds of macroblocks after the cause.
///
/// Generic over the sample type for the same reason the decoder's plane
/// is: a 10-bit picture is `u16` samples on both sides.
pub type Recon<S> = crate::h264::frame::PaddedPlane<S>;

/// A zeroed reconstruction plane of the given coded size.
pub fn recon_plane<S: Sample>(width: u32, height: u32, pad: usize) -> Recon<S> {
    Recon::new(width as usize, height as usize, pad)
}

/// Copy the displayed top-left rectangle out of a padded plane, packed
/// as bytes — one per sample at 8 bits, little-endian pairs deeper, the
/// layout [`crate::Picture::into_packed`] emits — which is what a decoder
/// emits and therefore what the SELF check compares against.
pub fn crop_into<S: Sample>(p: &Recon<S>, w: u32, h: u32, out: &mut Vec<u8>) {
    for y in 0..h as usize {
        let row = (y + p.pad) * p.stride + p.pad;
        crate::encode::pack_row(&p.data[row..row + w as usize], out);
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

    /// The parameter sets have to survive the crate's own parsers, which are
    /// the ones proven against 412 conformance streams. Anything they reject
    /// is not a legal parameter set.
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
            let sps = write_sps(&cfg, &g, 4, 4, None);
            let parsed = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps))
                .unwrap_or_else(|e| panic!("{w}x{h} {c:?}: SPS rejected: {e}"));
            assert_eq!(parsed.pic_width_in_mbs * 16, g.coded_width, "{w}x{h} {c:?}");
            assert_eq!(parsed.chroma_format_idc, match c {
                ChromaFormat::Monochrome => 0,
                ChromaFormat::Yuv420 => 1,
                ChromaFormat::Yuv422 => 2,
                ChromaFormat::Yuv444 => 3,
            });
        }
    }

    #[test]
    fn cropping_is_written_when_the_size_is_not_a_whole_macroblock() {
        let (cfg, g) = geom(50, 34, ChromaFormat::Yuv420);
        assert_eq!((g.coded_width, g.coded_height), (64, 48));
        let sps = write_sps(&cfg, &g, 4, 4, None);
        let parsed = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps)).unwrap();
        // The property that matters is not the field values but what a
        // decoder ends up displaying: the size the caller asked for.
        let (left, right, top, bottom) = parsed.crop;
        assert_eq!(g.coded_width - left - right, 50);
        assert_eq!(g.coded_height - top - bottom, 34);
    }

    /// The picture parameter set has to survive the crate's own parser
    /// too, and `transform_8x8_mode_flag` has to arrive as what was
    /// written — it lives behind `more_rbsp_data()`, so a writer that
    /// forgot the extension would be read back as "off" rather than
    /// rejected.
    #[test]
    fn the_decoder_parses_the_picture_parameter_set() {
        let (cfg, _) = geom(64, 64, ChromaFormat::Yuv420);
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(
            &cfg,
            &Geometry::new(&cfg),
            4,
            4,
            None,
        )))
        .expect("SPS");
        for t8x8 in [false, true] {
            let cfg = Config { transform_8x8: t8x8, ..cfg.clone() };
            let pps = write_pps(&cfg, 26);
            let look = |_id: u32| Some(sps.clone());
            let parsed = crate::h264::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps), &look)
                .unwrap_or_else(|e| panic!("t8x8={t8x8}: PPS rejected: {e}"));
            assert_eq!(parsed.transform_8x8_mode, t8x8);
            assert_eq!(parsed.pic_init_qp, 26);
            assert_eq!(parsed.second_chroma_qp_index_offset, 0);
        }
        // And the PPS of a stream that does not ask for the 8x8 transform
        // is byte-identical to one from before the field existed: the
        // extension is absent, not present-and-zero.
        let off = write_pps(&Config { transform_8x8: false, ..cfg.clone() }, 26);
        let on = write_pps(&Config { transform_8x8: true, ..cfg }, 26);
        assert_ne!(off, on, "the flag has to reach the bitstream");
        assert_eq!(off.len(), 3, "no extension means the historical three-byte PPS");
    }

    /// The HRD the SPS declares must survive the parser that, until this
    /// change, read the fields only to stay bit-aligned: the rate and
    /// size as `Cpb` snapped them, the clock as twice the frame rate, and
    /// the delay widths the SEI messages will be written at. And an SPS
    /// that declares no buffer must carry no VUI — the flag, not a VUI
    /// full of zeros — so every stream without one is byte-identical to
    /// what it was.
    #[test]
    fn the_hrd_survives_the_decoders_own_sps_parser() {
        for (bps, ms) in [(64_000u32, 125u32), (128_000, 500), (1_000_000, 1000)] {
            let cfg = Config {
                width: 64,
                height: 64,
                rate: crate::encode::RateControl::Bitrate { bps },
                cpb_ms: ms,
                fps: 30,
                ..Config::default()
            };
            let g = Geometry::new(&cfg);
            let Some(cpb) = Cpb::new(bps, ms) else { panic!("{bps}bps/{ms}ms: representable") };
            let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(
                &cfg, &g, 16, 16, Some(&cpb),
            )))
            .unwrap_or_else(|e| panic!("{bps}bps/{ms}ms: SPS rejected: {e}"));
            let vui = sps.vui.as_ref().unwrap_or_else(|| panic!("{bps}bps/{ms}ms: no VUI"));
            assert_eq!(vui.timing, Some((1, 60)), "{bps}bps/{ms}ms: clock");
            assert!(vui.fixed_frame_rate);
            let hrd = vui.nal_hrd.unwrap_or_else(|| panic!("{bps}bps/{ms}ms: no NAL HRD"));
            assert_eq!(hrd.bit_rate, cpb.bit_rate, "{bps}bps/{ms}ms: bit rate");
            assert_eq!(hrd.cpb_size, cpb.size, "{bps}bps/{ms}ms: buffer size");
            assert!(!hrd.cbr);
            assert_eq!(hrd.initial_delay_length, cpb.initial_delay_length);
            assert_eq!(hrd.removal_delay_length, cpb.removal_delay_length);
            assert_eq!(hrd.output_delay_length, OUTPUT_DELAY_LENGTH);
            assert_eq!(hrd.time_offset_length, 0);
        }
        let (cfg, g) = geom(64, 64, ChromaFormat::Yuv420);
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(
            &cfg, &g, 16, 16, None,
        )))
        .unwrap();
        assert!(sps.vui.is_none(), "no buffer, no VUI");
    }

    /// A deep SPS survives the production parser with the depth it was
    /// written at — both fields, luma and chroma, at every depth and
    /// chroma format, monochrome included (the decoder refuses unequal
    /// depths, so the chroma field has to say the same even where no
    /// chroma sample exists) — and claims a profile that admits it: High
    /// 10 for 9 and 10 bits, High 4:2:2 for 4:2:2 up to 10 bits, High
    /// 4:4:4 Predictive for 4:4:4 and for anything above 10 bits (A.2).
    /// And the 8-bit SPS is byte for byte what it was.
    #[test]
    fn a_deep_sps_carries_its_depth_through_the_decoders_parser() {
        for depth in [8u32, 9, 10, 12, 14] {
            for c in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
                let cfg = Config { width: 64, height: 64, chroma: c, bit_depth: depth, ..Config::default() };
                let g = Geometry::new(&cfg);
                let sps = write_sps(&cfg, &g, 16, 16, None);
                let parsed = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps))
                    .unwrap_or_else(|e| panic!("{depth}-bit {c:?}: SPS rejected: {e}"));
                assert_eq!(parsed.bit_depth_luma, depth, "{depth}-bit {c:?}: luma depth");
                assert_eq!(parsed.bit_depth_chroma, depth, "{depth}-bit {c:?}: chroma depth");
                let want_profile = match (c, depth) {
                    (_, d) if d > 10 => 244,
                    (ChromaFormat::Yuv444, _) => 244,
                    (ChromaFormat::Yuv422, _) => 122,
                    (_, d) if d > 8 => 110,
                    _ => 100,
                };
                assert_eq!(parsed.profile_idc, want_profile, "{depth}-bit {c:?}: profile");
                // The decoder's own admission test, which is what a stream
                // has to pass before a single slice is read.
                if depth == 8 {
                    let eight = write_sps(&Config { bit_depth: 8, ..cfg.clone() }, &Geometry::new(&cfg), 16, 16, None);
                    assert_eq!(sps, eight, "{c:?}: the 8-bit SPS moved");
                }
            }
        }
    }

    /// An interlaced SPS survives the production parser as an interlaced
    /// stream: `frame_mbs_only_flag` 0, the MBAFF flag as configured, the
    /// frame height rebuilt from map units of two macroblock rows, and a
    /// crop — in field rows — that lands on the displayed height. And the
    /// progressive SPS for the same picture is what it always was.
    #[test]
    fn an_interlaced_sps_declares_macroblock_pairs_and_crops_in_field_rows() {
        use crate::encode::{FieldCoding, FieldOrder};
        for (w, h, c) in [
            (64u32, 64u32, ChromaFormat::Yuv420),
            (64, 60, ChromaFormat::Yuv420),
            (48, 36, ChromaFormat::Yuv420),
            (64, 34, ChromaFormat::Yuv422),
            (64, 50, ChromaFormat::Yuv444),
            (64, 18, ChromaFormat::Monochrome),
        ] {
            for coding in [FieldCoding::Field, FieldCoding::Paff, FieldCoding::Mbaff] {
                let tag = format!("{w}x{h} {c:?} {coding:?}");
                let cfg = Config {
                    width: w,
                    height: h,
                    chroma: c,
                    interlace: Some(FieldOrder::TopFirst),
                    field_coding: coding,
                    ..Config::default()
                };
                let g = Geometry::new(&cfg);
                assert!(g.mbs_high.is_multiple_of(2), "{tag}: whole macroblock pairs");
                let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None)))
                    .unwrap_or_else(|e| panic!("{tag}: SPS rejected: {e}"));
                assert!(!sps.frame_mbs_only, "{tag}");
                assert_eq!(sps.mb_adaptive_frame_field, coding == FieldCoding::Mbaff, "{tag}");
                assert_eq!(sps.frame_height_in_mbs() * 16, g.coded_height, "{tag}");
                let (_, _, top, bottom) = sps.crop;
                assert_eq!(g.coded_height - top - bottom, h, "{tag}: displayed height");
                let f = g.field();
                assert_eq!((f.mbs_high * 2, f.coded_height * 2, f.height * 2), (g.mbs_high, g.coded_height, g.height), "{tag}: a field is half the frame");
                assert!(f.field_pic && !g.field_pic && !f.mbaff, "{tag}");
                let pps = write_pps(&cfg, 26);
                let look = |_id: u32| Some(sps.clone());
                let pps = crate::h264::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps), &look).unwrap();
                assert!(pps.bottom_field_pic_order_in_frame_present, "{tag}");
            }
            let progressive = Config { width: w, height: h, chroma: c, ..Config::default() };
            let pg = Geometry::new(&progressive);
            assert!(!pg.interlaced && !pg.mbaff);
            let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&progressive, &pg, 16, 16, None))).unwrap();
            assert!(sps.frame_mbs_only);
            assert_eq!(write_pps(&progressive, 26).len(), 3, "{w}x{h} {c:?}: the progressive PPS is the historical three bytes");
        }
    }

    /// Field and frame slice headers of an interlaced stream come back
    /// through the production slice parser as written: `field_pic_flag`,
    /// `bottom_field_flag`, a frame picture's `delta_pic_order_cnt_bottom`,
    /// and everything after them still aligned (the quantiser, which sits
    /// near the end, arrives intact) — for every slice type, reference and
    /// not.
    #[test]
    fn interlaced_slice_headers_round_trip() {
        use crate::encode::{FieldCoding, FieldOrder};
        for (coding, cabac) in [(FieldCoding::Paff, false), (FieldCoding::Paff, true), (FieldCoding::Mbaff, false), (FieldCoding::Mbaff, true)] {
            // The header's `cabac` has to be the PPS's entropy coder: the
            // reader takes `cabac_init_idc` from the PPS's word for it.
            let cfg = Config {
                width: 64,
                height: 64,
                bframes: 2,
                entropy: if cabac { Entropy::Cabac } else { Entropy::Cavlc },
                interlace: Some(FieldOrder::BottomFirst),
                field_coding: coding,
                ..Config::default()
            };
            let g = Geometry::new(&cfg);
            let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None))).unwrap();
            let sps_look = |_id: u32| Some(sps.clone());
            let pps = crate::h264::pps::Pps::parse(&crate::nal::unescape_rbsp(&write_pps(&cfg, 26)), &sps_look).unwrap();
            let pps_look = |_id: u32| Some(pps.clone());
            for kind in [Kind::Idr, Kind::I, Kind::P, Kind::B] {
                for (bottom_field, delta) in [(None, -1), (None, 1), (Some(false), 0), (Some(true), 0)] {
                    {
                        let tag = format!("{coding:?} {kind:?} field {bottom_field:?} delta {delta} cabac {cabac}");
                        let mut w = BitWriter::new();
                        write_slice_header(
                            &SliceHeader {
                                kind,
                                frame_num: 5,
                                idr_pic_id: 1,
                                poc_lsb: 9,
                                qp: 31,
                                log2_max_frame_num: 16,
                                log2_max_poc_lsb: 16,
                                reference: kind != Kind::B,
                                deblock: true,
                                cabac,
                                direct_spatial: true,
                                pred_weights: None,
                                interlaced: true,
                                bottom_field,
                                delta_poc_bottom: delta,
                            },
                            26,
                            &mut w,
                        );
                        w.rbsp_trailing_bits();
                        let nal_type = if kind == Kind::Idr { NAL_IDR } else { NAL_SLICE };
                        let nal = annexb(nal_type, if kind != Kind::B { 3 } else { 0 }, &w.into_nal());
                        let rbsp = crate::nal::unescape_rbsp(&nal[4..]);
                        let hdr = crate::nal::H264NalHeader::parse(&nal[4..]).unwrap();
                        let (parsed, _, _) = crate::h264::slice::SliceHeader::parse(&rbsp, hdr, &pps_look, &sps_look)
                            .unwrap_or_else(|e| panic!("{tag}: slice header rejected: {e}"));
                        assert_eq!(parsed.field_pic, bottom_field.is_some(), "{tag}");
                        assert_eq!(parsed.bottom_field, bottom_field == Some(true), "{tag}");
                        assert_eq!(parsed.delta_poc_bottom, if bottom_field.is_none() { delta } else { 0 }, "{tag}");
                        assert_eq!(parsed.mbaff(&sps), coding == FieldCoding::Mbaff && bottom_field.is_none(), "{tag}");
                        assert_eq!((parsed.frame_num, parsed.poc_lsb, parsed.slice_qp), (5, 9, 31), "{tag}");
                    }
                }
            }
        }
    }

    /// A slice header written at depth — a lossless picture's slice QP of
    /// zero against a PPS quantiser of 26, and a QP 40 one — comes back
    /// through the production slice parser with the same `SliceQP_Y`. The
    /// header carries the *unprimed* quantiser (7.4.3: `SliceQP_Y` in
    /// `-QpBdOffset_Y..=51`); the primed one is derived by the reader
    /// per macroblock, so a writer that primed it here would double the
    /// offset on the way through.
    #[test]
    fn a_deep_slice_header_round_trips_its_quantiser() {
        for depth in [8u32, 10, 12, 14] {
            for (kind, qp) in [(Kind::Idr, 0u8), (Kind::Idr, 40), (Kind::P, 23), (Kind::B, 51)] {
                let cfg = Config { width: 64, height: 64, bit_depth: depth, bframes: 2, ..Config::default() };
                let g = Geometry::new(&cfg);
                let sps_nal = write_sps(&cfg, &g, 16, 16, None);
                let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps_nal)).unwrap();
                let pps_nal = write_pps(&cfg, 26);
                let sps_look = |_id: u32| Some(sps.clone());
                let pps = crate::h264::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps_nal), &sps_look).unwrap();
                let mut w = BitWriter::new();
                write_slice_header(
                    &SliceHeader {
                        kind,
                        frame_num: 3,
                        idr_pic_id: 1,
                        poc_lsb: 6,
                        qp,
                        log2_max_frame_num: 16,
                        log2_max_poc_lsb: 16,
                        reference: kind != Kind::B,
                        deblock: true,
                        cabac: true,
                        direct_spatial: true,
                        pred_weights: None,
                        interlaced: false,
                        bottom_field: None,
                        delta_poc_bottom: 0,
                    },
                    26,
                    &mut w,
                );
                w.rbsp_trailing_bits();
                let nal_type = if kind == Kind::Idr { NAL_IDR } else { NAL_SLICE };
                // `nal_ref_idc` as the encoder writes it: the marking
                // syntax exists only in a reference picture's header.
                let nal = annexb(nal_type, if kind != Kind::B { 3 } else { 0 }, &w.into_nal());
                // The parser wants the NAL header byte in front of the RBSP.
                let rbsp = crate::nal::unescape_rbsp(&nal[4..]);
                let hdr = crate::nal::H264NalHeader::parse(&nal[4..]).unwrap();
                let pps_look = |_id: u32| Some(pps.clone());
                let (parsed, _, _) = crate::h264::slice::SliceHeader::parse(&rbsp, hdr, &pps_look, &sps_look)
                    .unwrap_or_else(|e| panic!("{depth}-bit {kind:?} qp {qp}: slice header rejected: {e}"));
                assert_eq!(parsed.slice_qp, qp as i32, "{depth}-bit {kind:?}: SliceQP_Y");
                assert_eq!(parsed.frame_num, 3);
                assert_eq!(parsed.poc_lsb, 6);
                assert_eq!(parsed.disable_deblocking_filter_idc, 0);
            }
        }
    }

    /// The two HDR10 static-metadata SEIs are byte-identical to the ones
    /// x265 writes for the same values (the NAL bytes after its two-byte
    /// HEVC header, from an x265 3.x stream with
    /// `master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)`
    /// and `max-cll=1000,400` — BT.2020 primaries, D65, 1000 nits down to
    /// 0.0001). There is no reader for these in the crate, so the second
    /// writer is the fixture: it agrees with libavcodec's reading in the
    /// gate (tools/vui_probe.py) and these bytes are what it produced.
    /// The mastering display's `00 00 00 01` tail (min luminance 1) is
    /// what puts an emulation-prevention byte in the fixture, so the
    /// single escape in `sei_nal` is exercised too.
    #[test]
    fn the_hdr10_static_metadata_seis_match_x265_byte_for_byte() {
        use crate::encode::{ContentLightLevel, MasteringDisplay};
        let m = MasteringDisplay {
            red: (34000, 16000),
            green: (13250, 34500),
            blue: (7500, 3000),
            white_point: (15635, 16450),
            max_luminance: 10_000_000,
            min_luminance: 1,
        };
        let x265_mdcv = "891833c286c41d4c0bb884d03e803d13404200989680000003000180";
        let ours: String = write_mastering_display_sei(&m).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(ours, x265_mdcv, "mastering display SEI");
        let c = ContentLightLevel { max_cll: 1000, max_fall: 400 };
        let x265_cll = "900403e8019080";
        let ours: String = write_content_light_level_sei(&c).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(ours, x265_cll, "content light level SEI");
    }

    /// The colour description round-trips through the decoder's own SPS
    /// parser, field for field, on its own and beside a buffer; with a
    /// buffer and no colour the VUI says nothing about colour, and with
    /// neither there is no VUI at all — the flag, not a VUI of zeros —
    /// so every stream from before colour existed is byte-identical.
    ///
    /// Three code points are asserted separately rather than as one
    /// tuple so that writing one of them into another's field — the
    /// mutation this test exists to catch — names the field it lost.
    #[test]
    fn the_colour_description_survives_the_decoders_own_sps_parser() {
        use crate::encode::ColourDescription;
        let colours = [
            ColourDescription { primaries: 9, transfer: 16, matrix: 9, full_range: false }, // HDR10
            ColourDescription { primaries: 9, transfer: 18, matrix: 9, full_range: false }, // HLG
            ColourDescription { primaries: 1, transfer: 1, matrix: 1, full_range: true }, // BT.709 full
            ColourDescription { primaries: 12, transfer: 17, matrix: 6, full_range: false }, // P3 / SMPTE 428 / 601
        ];
        let (base, g) = geom(64, 64, ChromaFormat::Yuv420);
        for c in colours {
            let cfg = Config { colour: Some(c), ..base.clone() };
            let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None)))
                .unwrap_or_else(|e| panic!("{c:?}: SPS rejected: {e}"));
            let vui = sps.vui.as_ref().unwrap_or_else(|| panic!("{c:?}: no VUI"));
            let (p, t, m) = vui.colour_description.unwrap_or_else(|| panic!("{c:?}: no colour description"));
            assert_eq!(p, c.primaries, "{c:?}: primaries");
            assert_eq!(t, c.transfer, "{c:?}: transfer");
            assert_eq!(m, c.matrix, "{c:?}: matrix");
            assert_eq!(vui.full_range, c.full_range, "{c:?}: range");
            assert_eq!(vui.chroma_loc, None, "{c:?}: no siting asked for, none written");
            assert_eq!(vui.timing, None, "{c:?}: no buffer, no clock");
            assert_eq!(vui.nal_hrd, None, "{c:?}: no buffer, no HRD");
            assert!(!vui.bitstream_restriction);
        }
        // Beside a buffer: both halves present, neither disturbing the other.
        let cpb = Cpb::new(64_000, 125).expect("representable");
        let cfg = Config {
            colour: Some(colours[0]),
            rate: crate::encode::RateControl::Bitrate { bps: 64_000 },
            cpb_ms: 125,
            ..base.clone()
        };
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, Some(&cpb))))
            .expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, Some((9, 16, 9)));
        assert_eq!(vui.timing, Some((1, 60)));
        assert_eq!(vui.nal_hrd.map(|h| h.bit_rate), Some(cpb.bit_rate));
        // A buffer alone says nothing about colour.
        let cfg = Config { colour: None, ..cfg };
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, Some(&cpb))))
            .expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, None, "a buffer alone must not invent a colour");
        assert!(!vui.full_range);
        assert_eq!(vui.timing, Some((1, 60)));
        // The chroma siting: alone it is a VUI that says nothing about
        // colour, and every code comes back for both fields.
        for t in 0..=5u8 {
            let cfg = Config { chroma_loc: Some(t), ..base.clone() };
            let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None)))
                .expect("SPS");
            let vui = sps.vui.as_ref().expect("a siting alone is a VUI");
            assert_eq!(vui.chroma_loc, Some((t, t)), "chroma_sample_loc_type {t}");
            assert_eq!(vui.colour_description, None, "a siting alone must not invent a colour");
        }
        // Beside a colour: both there, neither disturbing the other.
        let cfg = Config { colour: Some(colours[0]), chroma_loc: Some(1), ..base.clone() };
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None)))
            .expect("SPS");
        let vui = sps.vui.as_ref().expect("VUI");
        assert_eq!(vui.colour_description, Some((9, 16, 9)));
        assert_eq!(vui.chroma_loc, Some((1, 1)));
        // Neither: the bytes from before either existed.
        let plain = write_sps(&base, &g, 16, 16, None);
        let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&plain)).expect("SPS");
        assert!(sps.vui.is_none(), "no buffer, no colour and no siting, no VUI");
        assert_ne!(plain, write_sps(&Config { colour: Some(colours[0]), ..base.clone() }, &g, 16, 16, None));
        assert_ne!(plain, write_sps(&Config { chroma_loc: Some(0), ..base }, &g, 16, 16, None));
    }

    /// A chroma siting describes a 4:2:0 grid and nothing else: E.2.1
    /// wants `chroma_loc_info_present_flag` 0 for any other format, and
    /// libavcodec reports no siting there whatever is written — so the
    /// configuration is refused by name rather than written into a VUI
    /// no reader will honour. A code above 5 is refused likewise.
    #[test]
    fn a_chroma_siting_is_refused_off_420_and_above_5() {
        let (base, _) = geom(64, 64, ChromaFormat::Yuv420);
        assert!(Config { chroma_loc: Some(2), ..base.clone() }.validate().is_ok(), "4:2:0 takes a siting");
        assert!(Config { chroma_loc: None, chroma: ChromaFormat::Yuv444, ..base.clone() }.validate().is_ok());
        for c in [ChromaFormat::Monochrome, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
            let e = Config { chroma_loc: Some(0), chroma: c, ..base.clone() }
                .validate()
                .expect_err("a siting off 4:2:0 must be refused");
            assert!(e.to_string().contains("chroma_loc"), "{c:?}: {e}");
        }
        let e = Config { chroma_loc: Some(6), ..base }.validate().expect_err("6 is not a chroma_sample_loc_type");
        assert!(e.to_string().contains("0..=5"), "{e}");
    }

    #[test]
    fn annexb_prefixes_a_start_code_and_the_header_byte() {
        let n = annexb(NAL_SPS, 3, &[0xaa]);
        assert_eq!(&n[..4], &[0, 0, 0, 1]);
        assert_eq!(n[4] & 0x1f, NAL_SPS);
        assert_eq!(n[4] >> 5, 3);
    }
    /// A P slice header carrying a `pred_weight_table` — gains either side
    /// of the identity and at both ends of the range, negative offsets, a
    /// luma-only entry, a chroma-only entry, and a table of defaults — comes
    /// back through the production slice parser as the table written, with
    /// the fields after it (the marking, the quantiser) intact, at 8 and 10
    /// bits, for 4:2:0, 4:4:4 and monochrome (where no chroma half exists);
    /// and the PPS offers the flag only when asked, keeping the historical
    /// bytes otherwise.
    #[test]
    fn a_pred_weight_table_round_trips_through_the_slice_parser() {
        use crate::h264::slice::WeightEntry;
        let entries = [
            WeightEntry { luma: (48, -3), chroma: [(64, 0), (64, 0)], luma_flag: true, chroma_flag: false },
            WeightEntry { luma: (64, 0), chroma: [(70, 5), (60, -128)], luma_flag: false, chroma_flag: true },
            WeightEntry { luma: (127, 127), chroma: [(-128, -7), (64, 1)], luma_flag: true, chroma_flag: true },
            WeightEntry { luma: (-128, -128), chroma: [(64, 0), (0, 0)], luma_flag: true, chroma_flag: true },
            WeightEntry { luma: (64, 0), chroma: [(64, 0), (64, 0)], luma_flag: false, chroma_flag: false },
        ];
        for chroma in [ChromaFormat::Yuv420, ChromaFormat::Yuv444, ChromaFormat::Monochrome] {
            for depth in [8u32, 10] {
                let cfg = Config { width: 64, height: 64, chroma, bit_depth: depth, weighted_pred: true, ..Config::default() };
                let g = Geometry::new(&cfg);
                let sps = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, 16, None))).unwrap();
                let sps_look = |_id: u32| Some(sps.clone());
                let pps = crate::h264::pps::Pps::parse(&crate::nal::unescape_rbsp(&write_pps(&cfg, 26)), &sps_look).unwrap();
                assert!(pps.weighted_pred, "{chroma:?}: weighted_pred_flag");
                assert_eq!(pps.weighted_bipred_idc, 0, "{chroma:?}: B slices stay default-weighted");
                let pps_look = |_id: u32| Some(pps.clone());
                let has_chroma = chroma != ChromaFormat::Monochrome;
                for e in entries {
                    let e = if has_chroma { e } else { WeightEntry { chroma: [(64, 0); 2], chroma_flag: false, ..e } };
                    let table = PredWeightTable { luma_log2_denom: 6, chroma_log2_denom: 6, lists: [vec![e], Vec::new()] };
                    let mut w = BitWriter::new();
                    write_slice_header(
                        &SliceHeader {
                            kind: Kind::P,
                            frame_num: 3,
                            idr_pic_id: 0,
                            poc_lsb: 6,
                            qp: 31,
                            log2_max_frame_num: 16,
                            log2_max_poc_lsb: 16,
                            reference: true,
                            deblock: true,
                            cabac: true,
                            direct_spatial: false,
                            pred_weights: Some(PredWeights { table, chroma: has_chroma }),
                            interlaced: false,
                            bottom_field: None,
                            delta_poc_bottom: 0,
                        },
                        26,
                        &mut w,
                    );
                    w.rbsp_trailing_bits();
                    let nal = annexb(NAL_SLICE, 3, &w.into_nal());
                    let rbsp = crate::nal::unescape_rbsp(&nal[4..]);
                    let hdr = crate::nal::H264NalHeader::parse(&nal[4..]).unwrap();
                    let (parsed, _, _) = crate::h264::slice::SliceHeader::parse(&rbsp, hdr, &pps_look, &sps_look)
                        .unwrap_or_else(|err| panic!("{chroma:?} {depth}-bit {e:?}: slice header rejected: {err}"));
                    let got = parsed.pred_weights.as_ref().unwrap_or_else(|| panic!("{chroma:?} {depth}-bit: no table read"));
                    assert_eq!(got.luma_log2_denom, 6);
                    assert_eq!(got.lists[0].len(), 1, "one active reference, one entry");
                    assert!(got.lists[1].is_empty(), "a P slice has no list-1 half");
                    let r = got.lists[0][0];
                    assert_eq!((r.luma, r.luma_flag), (e.luma, e.luma_flag), "{chroma:?} {depth}-bit: luma of {e:?}");
                    if has_chroma {
                        assert_eq!(got.chroma_log2_denom, 6);
                        assert_eq!((r.chroma, r.chroma_flag), (e.chroma, e.chroma_flag), "{chroma:?} {depth}-bit: chroma of {e:?}");
                    } else {
                        assert!(!r.chroma_flag, "monochrome: no chroma half on the wire");
                    }
                    assert_eq!(parsed.slice_qp, 31, "{chroma:?} {depth}-bit: the quantiser after the table");
                    assert_eq!(parsed.frame_num, 3);
                }
            }
        }
        let plain = Config { width: 64, height: 64, ..Config::default() };
        assert_eq!(write_pps(&plain, 26).len(), 3, "no switch, the historical PPS");
        assert_ne!(write_pps(&plain, 26), write_pps(&Config { weighted_pred: true, ..plain.clone() }, 26));
    }
}
