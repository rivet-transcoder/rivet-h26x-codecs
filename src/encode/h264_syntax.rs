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
use crate::h264::cabac_mb::{CabacState, MB_TYPE_I_PCM, write_mb_type_i_cabac};
use crate::encode::{Config, Entropy};
use crate::picture::ChromaFormat;

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

/// `vui_parameters()` (E.1.1) carrying only what the buffer model needs:
/// the clock the removal delays are counted in, and the NAL HRD.
/// Everything else is absent by its own flag — a VUI is optional and this
/// encoder wrote none until a buffer needed one.
fn write_vui(w: &mut BitWriter, cpb: &Cpb, fps: u32) {
    w.flag(false); // aspect_ratio_info_present_flag
    w.flag(false); // overscan_info_present_flag
    w.flag(false); // video_signal_type_present_flag
    w.flag(false); // chroma_loc_info_present_flag
    w.flag(true); // timing_info_present_flag
    w.bits(32, 1); // num_units_in_tick
    w.bits(32, TICKS_PER_FRAME * fps.max(1)); // time_scale
    w.flag(true); // fixed_frame_rate_flag
    w.flag(true); // nal_hrd_parameters_present_flag
    write_hrd(w, cpb);
    w.flag(false); // vcl_hrd_parameters_present_flag
    w.flag(false); // low_delay_hrd_flag (present: a NAL HRD is)
    w.flag(false); // pic_struct_present_flag
    w.flag(false); // bitstream_restriction_flag
}

/// One SEI message wrapped as an SEI NAL payload: `payloadType`,
/// `payloadSize` in the standard's 255-at-a-time form, the payload bytes
/// (already byte-aligned by their own trailing bits), then the RBSP's.
///
/// The payload arrives as *raw* RBSP bytes and the emulation prevention
/// is applied once, here, to the whole NAL. A payload that had already
/// been escaped would be escaped again — a timing SEI is mostly zero
/// bytes, exactly the pattern the escape targets — and a reader would
/// find `0x03` where a delay's bits should be.
fn sei_nal(payload_type: u32, payload: &[u8]) -> Vec<u8> {
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
}

impl Geometry {
    /// Derive the coded geometry from a configuration.
    pub fn new(cfg: &Config) -> Self {
        let mbs_wide = cfg.width.div_ceil(16);
        let mbs_high = cfg.height.div_ceil(16);
        Self {
            mbs_wide,
            mbs_high,
            coded_width: mbs_wide * 16,
            coded_height: mbs_high * 16,
            width: cfg.width,
            height: cfg.height,
            chroma: cfg.chroma,
            bit_depth: cfg.bit_depth,
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
/// the coding tools: 4:2:2 and 4:4:4 and depths above 8 need High 4:2:2 or
/// High 4:4:4 Predictive, and monochrome needs High. Claiming a lower profile
/// than the stream needs is the kind of error a decoder is entitled to reject
/// the stream over, so this errs upwards.
fn profile_idc(g: &Geometry) -> u8 {
    match g.chroma {
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
/// carries a VUI — the frame clock and the NAL HRD — and without one no
/// VUI at all, so a stream that declares no buffer is byte-identical to
/// one from before the buffer model existed.
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
        w.ue(g.bit_depth - 8); // bit_depth_luma_minus8
        w.ue(if g.chroma == ChromaFormat::Monochrome { 0 } else { g.bit_depth - 8 });
        w.flag(false); // qpprime_y_zero_transform_bypass_flag
        w.flag(false); // seq_scaling_matrix_present_flag
    }
    w.ue(log2_max_frame_num - 4);
    w.ue(0); // pic_order_cnt_type 0
    w.ue(log2_max_poc_lsb - 4);
    w.ue(cfg.max_refs); // max_num_ref_frames
    w.flag(false); // gaps_in_frame_num_value_allowed_flag
    w.ue(g.mbs_wide - 1);
    w.ue(g.mbs_high - 1); // frame_mbs_only_flag is 1, so map units are MBs
    w.flag(true); // frame_mbs_only_flag
    w.flag(true); // direct_8x8_inference_flag
    // Cropping, because the coded size is rounded up to whole macroblocks and
    // the displayed size is not. The units are chroma samples horizontally
    // and, for frame pictures, chroma samples vertically.
    let (cw, ch) = match g.chroma {
        ChromaFormat::Monochrome => (1, 1),
        ChromaFormat::Yuv420 => (2, 2),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv444 => (1, 1),
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
    match cpb {
        Some(cpb) => {
            w.flag(true); // vui_parameters_present_flag
            write_vui(&mut w, cpb, cfg.fps);
        }
        None => w.flag(false), // vui_parameters_present_flag
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
    w.flag(false); // bottom_field_pic_order_in_frame_present_flag
    w.ue(0); // num_slice_groups_minus1
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.flag(false); // weighted_pred_flag
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

/// What a slice header needs that is not in the parameter sets.
#[derive(Debug, Clone, Copy)]
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
    // frame_mbs_only_flag is 1, so no field_pic_flag here.
    if h.kind == Kind::Idr {
        w.ue(h.idr_pic_id);
    }
    w.bits(h.log2_max_poc_lsb, h.poc_lsb);
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

/// Write one macroblock as `I_PCM`, taking its samples from the source and
/// leaving the same samples in `dst`, the padded reconstruction.
///
/// Samples outside the displayed picture — the padding a non-multiple-of-16
/// size implies — are filled by edge replication. Any value would be legal,
/// since the cropping rectangle excludes them from display, but replication
/// keeps the coded picture free of edges that would cost bits once this
/// encoder predicts and transforms rather than copying.
pub fn write_pcm_macroblock(
    w: &mut BitWriter,
    g: &Geometry,
    mb_x: u32,
    mb_y: u32,
    planes: &[Plane<'_>],
    dst: &mut [Recon],
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
pub fn write_pcm_slice_data_cabac(
    w: &mut BitWriter,
    g: &Geometry,
    qp: u8,
    planes: &[Plane<'_>],
    dst: &mut [Recon],
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
fn write_pcm_samples(
    w: &mut BitWriter,
    g: &Geometry,
    mb_x: u32,
    mb_y: u32,
    planes: &[Plane<'_>],
    dst: &mut [Recon],
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
                let v = src.data[syy as usize * src.stride + sxx as usize] as u32;
                w.bits(bd, v);
                let d = &mut dst[p];
                let i = ((sy + y) as usize + d.pad) * d.stride + (sx + x) as usize + d.pad;
                d.data[i] = v as u8;
            }
        }
    }
}

/// A source plane: samples, stride, and the size actually present.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    /// Samples, row-major.
    pub data: &'a [u8],
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
pub type Recon = crate::h264::frame::PaddedPlane<u8>;

/// A zeroed reconstruction plane of the given coded size.
pub fn recon_plane(width: u32, height: u32, pad: usize) -> Recon {
    Recon::new(width as usize, height as usize, pad)
}

/// Copy the displayed top-left rectangle out of a padded plane, which is what
/// a decoder emits and therefore what the SELF check compares against.
pub fn crop_into(p: &Recon, w: u32, h: u32, out: &mut Vec<u8>) {
    for y in 0..h as usize {
        let row = (y + p.pad) * p.stride + p.pad;
        out.extend_from_slice(&p.data[row..row + w as usize]);
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

    #[test]
    fn annexb_prefixes_a_start_code_and_the_header_byte() {
        let n = annexb(NAL_SPS, 3, &[0xaa]);
        assert_eq!(&n[..4], &[0, 0, 0, 1]);
        assert_eq!(n[4] & 0x1f, NAL_SPS);
        assert_eq!(n[4] >> 5, 3);
    }
}
