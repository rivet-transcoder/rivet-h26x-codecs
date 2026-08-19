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

/// Coded slice of a non-IDR picture.
pub const NAL_SLICE: u8 = 1;
/// Coded slice of an IDR picture.
pub const NAL_IDR: u8 = 5;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 7;
/// Picture parameter set.
pub const NAL_PPS: u8 = 8;

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

/// Sequence parameter set.
pub fn write_sps(cfg: &Config, g: &Geometry, log2_max_frame_num: u32, log2_max_poc_lsb: u32) -> Vec<u8> {
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
    w.flag(false); // vui_parameters_present_flag
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
    /// Whether the deblocking filter runs over this slice.
    ///
    /// The PCM and all-skip paths leave it on, where it provably does
    /// nothing (PCM macroblocks average to a qP of zero, all-skip edges
    /// have boundary strength zero). The transform intra path turns it
    /// off, because the encoder does not yet run the filter over its own
    /// reconstruction — a filtered decode against an unfiltered
    /// reconstruction would fail SELF on every coded edge. When the
    /// encoder learns to deblock, this flips on for the quality it buys.
    pub deblock: bool,
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
        w.flag(false); // direct_spatial_mv_pred_flag
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
            let sps = write_sps(&cfg, &g, 4, 4);
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
        let sps = write_sps(&cfg, &g, 4, 4);
        let parsed = crate::h264::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps)).unwrap();
        // The property that matters is not the field values but what a
        // decoder ends up displaying: the size the caller asked for.
        let (left, right, top, bottom) = parsed.crop;
        assert_eq!(g.coded_width - left - right, 50);
        assert_eq!(g.coded_height - top - bottom, 34);
    }

    #[test]
    fn annexb_prefixes_a_start_code_and_the_header_byte() {
        let n = annexb(NAL_SPS, 3, &[0xaa]);
        assert_eq!(&n[..4], &[0, 0, 0, 1]);
        assert_eq!(n[4] & 0x1f, NAL_SPS);
        assert_eq!(n[4] >> 5, 3);
    }
}
