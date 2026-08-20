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
use crate::encode::Config;
use crate::picture::ChromaFormat;

/// Video parameter set.
pub const NAL_VPS: u8 = 32;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 33;
/// Picture parameter set.
pub const NAL_PPS: u8 = 34;
/// Coded slice of a non-IRAP picture, trailing, referenced.
pub const NAL_TRAIL_R: u8 = 1;
/// Coded slice of an IDR picture with no leading pictures.
pub const NAL_IDR_N_LP: u8 = 20;

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
    /// The CTU size is chosen rather than configured: 64x64 unless the
    /// picture is smaller than that, because a CTU larger than the picture is
    /// legal but wastes header bits describing a tree that cannot split.
    pub fn new(cfg: &Config) -> Self {
        let log2_min_cb = 3;
        // The coded picture is a whole number of CTUs, with the conformance
        // window cropping the rest — not the minimal legal size, which only
        // needs a multiple of the minimum coding block. Whole CTUs because
        // the intra decision machinery codes exactly one CU per CTU and
        // cannot express the quadtree shapes a partial edge CTU needs; the
        // padding costs a few edge blocks of replicated content, and the
        // window hides them. The standard's CTB floor is 16 (an 8x8 CTB is
        // illegal — this crate's own SPS parser rejects it, which is how
        // that constraint was rediscovered), and the decision machinery's
        // ceiling is 32, so the choice is 16 or 32: whichever pads less,
        // the larger on a tie.
        let (log2_ctb, coded_width, coded_height) = [5u32, 4]
            .into_iter()
            .map(|v| {
                let n = 1u32 << v;
                let w = cfg.width.div_ceil(n) * n;
                let h = cfg.height.div_ceil(n) * n;
                (v, w, h)
            })
            .min_by_key(|&(v, w, h)| (w * h, u32::MAX - v))
            .unwrap();
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

fn chroma_idc(c: ChromaFormat) -> u32 {
    match c {
        ChromaFormat::Monochrome => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// `profile_tier_level`, shared by the VPS and SPS.
///
/// Twelve bytes, most of them reserved and required to be zero, plus a
/// 43-bit reserved field that has to be written as two pieces because it does
/// not fit a single call. Main for 8-bit 4:2:0, Main 10 for deeper, Rext for
/// anything else — claiming a profile that does not admit the format is a
/// stream a decoder may refuse.
fn write_ptl(w: &mut BitWriter, g: &Geometry) {
    let profile = if g.chroma != ChromaFormat::Yuv420 || g.bit_depth > 10 {
        4 // Range extensions
    } else if g.bit_depth > 8 {
        2 // Main 10
    } else {
        1 // Main
    };
    w.bits(2, 0); // general_profile_space
    w.flag(false); // general_tier_flag
    w.bits(5, profile); // general_profile_idc
    // general_profile_compatibility_flag[32]
    for i in 0..32 {
        w.flag(i == profile);
    }
    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(false); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag
    // 43 reserved zero bits, in two writes because one call takes at most 32.
    w.zeros(43);
    w.flag(false); // general_inbld_flag / reserved
    w.bits(8, 120); // general_level_idc: level 4.0, which admits 1080p
}

/// Video parameter set.
pub fn write_vps(g: &Geometry) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(32);
    w.bits(4, 0); // vps_video_parameter_set_id
    w.flag(true); // vps_base_layer_internal_flag
    w.flag(true); // vps_base_layer_available_flag
    w.bits(6, 0); // vps_max_layers_minus1
    w.bits(3, 0); // vps_max_sub_layers_minus1
    w.flag(true); // vps_temporal_id_nesting_flag
    w.bits(16, 0xffff); // vps_reserved_0xffff_16bits
    write_ptl(&mut w, g);
    w.flag(true); // vps_sub_layer_ordering_info_present_flag
    w.ue(1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.bits(6, 0); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.flag(false); // vps_timing_info_present_flag
    w.flag(false); // vps_extension_flag
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// Sequence parameter set.
pub fn write_sps(cfg: &Config, g: &Geometry, log2_max_poc_lsb: u32) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(64);
    w.bits(4, 0); // sps_video_parameter_set_id
    w.bits(3, 0); // sps_max_sub_layers_minus1
    w.flag(true); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, g);
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
    w.ue(g.bit_depth - 8); // bit_depth_luma_minus8
    w.ue(if g.chroma == ChromaFormat::Monochrome { 0 } else { g.bit_depth - 8 });
    w.ue(log2_max_poc_lsb - 4);
    w.flag(true); // sps_sub_layer_ordering_info_present_flag
    w.ue(cfg.max_refs.max(1)); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(if cfg.bframes > 0 { cfg.bframes } else { 0 });
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(g.log2_min_cb - 3); // log2_min_luma_coding_block_size_minus3
    w.ue(g.log2_ctb - g.log2_min_cb); // log2_diff_max_min_luma_coding_block_size
    w.ue(0); // log2_min_luma_transform_block_size_minus2 -> 4x4
    // The maximum transform size equals the CTB size (the CTB is at most 32,
    // which is also the standard's largest transform), so a 2Nx2N CU can
    // carry a single CU-sized TU — which is the only transform tree the
    // intra decision module produces. The previous value, one below the CTB,
    // would have forced an inferred transform split under every whole-CTU
    // CU and made that shape unrepresentable.
    w.ue(g.log2_ctb - 2); // log2_diff_max_min_luma_transform_block_size
    w.ue(2); // max_transform_hierarchy_depth_inter
    w.ue(2); // max_transform_hierarchy_depth_intra
    w.flag(false); // scaling_list_enabled_flag
    w.flag(false); // amp_enabled_flag
    w.flag(false); // sample_adaptive_offset_enabled_flag
    w.flag(false); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.flag(false); // long_term_ref_pics_present_flag
    w.flag(false); // sps_temporal_mvp_enabled_flag
    w.flag(false); // strong_intra_smoothing_enabled_flag
    w.flag(false); // vui_parameters_present_flag
    w.flag(false); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.into_nal()
}

/// Picture parameter set.
///
/// `bypass` writes `transquant_bypass_enabled_flag` — the lossless switch.
/// It changes nothing else here or in the slice header (the parser reads no
/// other syntax conditionally on it); what it changes is the coding tree,
/// where every CU then carries a `cu_transquant_bypass_flag`.
pub fn write_pps(qp: u8, bypass: bool) -> Vec<u8> {
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
    w.se(qp as i32 - 26); // init_qp_minus26
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // transform_skip_enabled_flag
    w.flag(false); // cu_qp_delta_enabled_flag
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
    w.flag(false); // weighted_pred_flag
    w.flag(false); // weighted_bipred_flag
    w.flag(bypass); // transquant_bypass_enabled_flag
    w.flag(false); // tiles_enabled_flag
    w.flag(false); // entropy_coding_sync_enabled_flag
    w.flag(true); // pps_loop_filter_across_slices_enabled_flag
    // The deblocking filter is disabled picture-wide, and the reason is the
    // SELF property, not taste: the encoder does not yet run the filter over
    // its own reconstruction, and a decoder that filters against an encoder
    // that does not desyncs on exactly the samples the filter touches — a
    // first H.265 stream failed SELF on 24 luma samples, every one within
    // three of the 8-sample deblocking grid, while libavcodec and our
    // decoder agreed with each other perfectly. When the encoder learns to
    // deblock its reconstruction, these three bits flip for the quality the
    // filter buys.
    w.flag(true); // deblocking_filter_control_present_flag
    w.flag(false); // deblocking_filter_override_enabled_flag
    w.flag(true); // pps_deblocking_filter_disabled_flag
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
    /// Quantiser for the slice.
    pub qp: u8,
    /// Width of the `poc_lsb` field, from the SPS.
    pub log2_max_poc_lsb: u32,
    /// POC deltas of the reference pictures, relative to this slice's POC:
    /// negative for the past (list 0), positive for the future (list 1).
    /// Empty for an I or IDR slice. These become the slice's inline short
    /// term reference picture set, in the order the reader expects —
    /// negatives nearest-first, then positives nearest-first.
    pub ref_deltas: Vec<i32>,
}

/// Slice segment header, up to but not including the coded tree.
pub fn write_slice_header(h: &SliceHeader, pps_qp: u8, nal_type: u8, w: &mut BitWriter) {
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
        let mut negative: Vec<i32> = h.ref_deltas.iter().copied().filter(|d| *d < 0).collect();
        let mut positive: Vec<i32> = h.ref_deltas.iter().copied().filter(|d| *d > 0).collect();
        // Nearest first, as the deltas are coded as successive differences.
        negative.sort_by_key(|d| -d);
        positive.sort();
        w.ue(negative.len() as u32);
        w.ue(positive.len() as u32);
        let mut prev = 0i32;
        for d in &negative {
            w.ue((prev - d - 1) as u32); // delta_poc_s0_minus1
            w.flag(true); // used_by_curr_pic_s0_flag
            prev = *d;
        }
        let mut prev = 0i32;
        for d in &positive {
            w.ue((d - prev - 1) as u32); // delta_poc_s1_minus1
            w.flag(true); // used_by_curr_pic_s1_flag
            prev = *d;
        }
        // slice_temporal_mvp_enabled_flag is absent: the SPS disables
        // temporal MVP, so the reader never reads the slice-level flag —
        // which is also what makes the spatial candidate derivation the
        // complete one rather than a subset of it.
    }
    // slice_sao_luma_flag / slice_sao_chroma_flag are absent: SAO is
    // disabled in the SPS. (A flag was once written here anyway — one
    // spurious bit that shifted everything after it, unnoticed because
    // nothing could decode past the header until the coding tree existed.)
    if matches!(h.kind, Kind::P | Kind::B) {
        // num_ref_idx_active_override_flag: 0, taking the PPS defaults.
        // Both num_ref_idx_lX_default_active_minus1 are 0 there, so the
        // reader resolves [1, 0] for P and [1, 1] for B — exactly the one
        // reference per list the decision modules search. Verified against
        // the header parser, not inferred: it copies the PPS defaults and
        // overwrites them only when this flag is set.
        //
        // The coupling this buys is worth naming: change those PPS
        // defaults and every slice header written this way silently means
        // something else. Both are written from one contract in this file,
        // which is why the shorter form wins.
        w.flag(false);
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
        // slice_temporal_mvp_enabled is off; the weighted prediction
        // tables are absent because weighted_pred_flag and
        // weighted_bipred_flag are both 0 — default weighting is not a
        // simplification, it is the only combination this bitstream can
        // ask the reader for.
        w.ue(0); // five_minus_max_num_merge_cand -> MaxNumMergeCand 5
    }
    w.se(h.qp as i32 - pps_qp as i32); // slice_qp_delta
    // slice_loop_filter_across_slices_enabled_flag is NOT written: the
    // reader reads it only when SAO is on for the slice or deblocking is
    // not disabled, and this encoder's PPS disables deblocking while its
    // SPS disables SAO. Writing it anyway shifts every bit after it — the
    // same one-spurious-bit shape as the SAO flag this header once wrote,
    // caught the same way, by the production parser refusing the header.
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
            let vps = write_vps(&g);
            assert!(!vps.is_empty());
            let sps = write_sps(&cfg, &g, 8);
            let parsed = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps))
                .unwrap_or_else(|e| panic!("{w}x{h} {c:?}: SPS rejected: {e}"));
            assert_eq!(parsed.width, g.coded_width, "{w}x{h} {c:?}");
            assert_eq!(parsed.height, g.coded_height, "{w}x{h} {c:?}");
            assert_eq!(parsed.chroma_format_idc, chroma_idc(c), "{w}x{h} {c:?}");
            assert_eq!(parsed.bit_depth_luma, cfg.bit_depth, "{w}x{h} {c:?}");
        }
    }

    #[test]
    fn the_conformance_window_recovers_the_requested_size() {
        let (cfg, g) = geom(50, 34, ChromaFormat::Yuv420);
        assert_eq!((g.coded_width, g.coded_height), (64, 48));
        let sps = write_sps(&cfg, &g, 8);
        let parsed = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps)).unwrap();
        let (l, r, t, b) = parsed.conf_win;
        assert_eq!(g.coded_width - l - r, 50);
        assert_eq!(g.coded_height - t - b, 34);
    }

    /// The coded picture is a whole number of CTUs, CTB 16 or 32 by least
    /// padding — see `Geometry::new` for why partial edge CTUs are avoided.
    #[test]
    fn the_coded_picture_is_whole_ctus_with_least_padding() {
        let g = Geometry::new(&geom(64, 64, ChromaFormat::Yuv420).0);
        assert_eq!((g.log2_ctb, g.coded_width, g.coded_height), (5, 64, 64));
        let g = Geometry::new(&geom(48, 48, ChromaFormat::Yuv420).0);
        assert_eq!((g.log2_ctb, g.coded_width, g.coded_height), (4, 48, 48));
        // 24x24 pads to 32x32 under either CTB size; the tie goes to 32.
        let g = Geometry::new(&geom(24, 24, ChromaFormat::Yuv420).0);
        assert_eq!((g.log2_ctb, g.coded_width, g.coded_height), (5, 32, 32));
        // 50x34: CTB 16 pads to 64x48, CTB 32 to 64x64 — 16 pads less.
        let g = Geometry::new(&geom(50, 34, ChromaFormat::Yuv420).0);
        assert_eq!((g.log2_ctb, g.coded_width, g.coded_height), (4, 64, 48));
    }

    #[test]
    fn the_nal_header_is_two_bytes_and_carries_temporal_id_plus_one() {
        let n = annexb(NAL_SPS, &[0xaa]);
        assert_eq!(&n[..4], &[0, 0, 0, 1]);
        assert_eq!(n[4] >> 1, NAL_SPS);
        assert_eq!(n[5], 1, "nuh_temporal_id_plus1 must be 1, not 0");
    }
}
