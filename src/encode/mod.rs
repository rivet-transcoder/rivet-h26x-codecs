//! The encoding side: H.264 and H.265 bitstreams produced from raw pictures.
//!
//! The decoders in this crate are bit-exact against the JVT and JCT-VC
//! conformance suites, and that shapes how the encoders are built and how they
//! are verified. An encoder has no conformance suite — there is no set of
//! reference bitstreams it must reproduce, because a standard constrains what
//! a *decoder* must do with a bitstream and leaves an encoder free to choose
//! any legal one. So "is the encoder correct" is not a question with a golden
//! answer, and the temptation is to answer a weaker question instead and call
//! it verified.
//!
//! # What correctness means here
//!
//! Three properties, in the order they are worth checking. Each is exact —
//! none is a measurement, and none has a noise floor:
//!
//! 1. **The bitstream decodes to what the encoder thinks it encoded.** The
//!    encoder reconstructs every picture as it goes, because prediction
//!    depends on reconstructed samples; running the decoder over its output
//!    must produce byte-identical pictures to those reconstructions. A
//!    mismatch is a desync — the encoder and decoder disagreed about state —
//!    and it is always a bug, never a quality question. This is the property
//!    that catches the largest class of encoder faults, and it needs no
//!    reference data at all.
//!
//! 2. **Another decoder agrees.** libavcodec decoding our output must produce
//!    the same pictures our decoder does. Property 1 is self-consistent and
//!    would pass happily if both sides shared a misreading of the standard;
//!    this is what makes the bitstream *legal* rather than merely
//!    self-compatible. It is also the property that matters commercially,
//!    since the output has to play elsewhere.
//!
//! 3. **The reconstruction is close to the source**, which is the only one of
//!    the three that is a quality question rather than a correctness one, and
//!    the only one with a knob attached. Reported as PSNR against the input,
//!    at a stated bitrate. Lossless mode makes it exact and therefore checkable
//!    like the other two.
//!
//! `tools/verify_encode.sh` gates 1 and 2 and reports 3. It also gates a
//! fourth exact property the first three structurally cannot see, because
//! both decoders read Annex-B: **one parameter set of each kind per
//! stream**, byte for byte. A re-sent PPS replaces the old one in Annex-B
//! and passes SELF and CROSS; in an MP4 `avc1` box the sets live out of
//! band, so a PPS that changed between the I and P pictures decodes the
//! pictures under the other one to garbage — which is how rivet's first
//! H.264 file failed with the whole gate green. See `tools/param_sets.py`.
//!
//! One more exact property applies only to what the samples cannot show:
//! **the stream says what colour it is**. A [`ColourDescription`], a
//! chroma siting or the HDR10 static metadata change no sample, so SELF
//! and CROSS pass whether the VUI and SEIs carry them or not, and the
//! crate's own parsers are the writers' inverses — a shared misreading of
//! E.1.1 round-trips cleanly. The gate therefore asks a *third* reader:
//! `tools/vui_probe.py` has ffprobe name every field, and a `--color` row
//! is green only when the names are exactly the codes the encoder was
//! handed (`VUI-FAIL` otherwise). A player showing BT.2020 PQ as washed-out
//! BT.709 is the failure that row exists to prevent.
//!
//! # Shape
//!
//! Deliberately the mirror of the decoders: an `H264Encoder` takes pictures
//! in and hands NAL units out, the way [`crate::h264::H264Decoder`] takes NAL
//! units in and hands pictures out. The same pixel kernels serve
//! both directions — the encoder's reconstruction loop *is* a decoder, and
//! reusing the conformance-proven inverse transform, prediction and
//! deblocking there is what makes property 1 achievable rather than
//! aspirational.
//!
//! The encode-only kernels — forward transforms, quantisation, and the
//! distortion metrics that motion search lives in — sit behind the same
//! runtime-dispatched table as the decode kernels, so the instruction-set
//! ladder covers them without a second mechanism.

use crate::Result;
use crate::picture::ChromaFormat;

pub mod gop;
pub mod h264;
pub mod h264_cabac_mb;
pub mod h264_cavlc_mb;
pub mod h264_deblock;
pub mod h264_intra;
pub mod h264_me;
pub mod h264_pic;
pub mod h264_syntax;
pub mod h265;
pub mod h265_deblock;
pub mod h265_intra;
pub mod h265_me;
pub(crate) mod rc;
pub(crate) mod aq;
pub(crate) mod h265_sao;
pub(crate) mod h265_wp;
pub mod hrd;
pub mod h265_syntax;

/// How lossy, and by what means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    /// Fixed quantiser. The simplest thing that produces a legal stream, and
    /// the one every other mode is built on top of and compared against.
    ConstantQp(u8),
    /// Mathematically lossless: transform bypass where the standard offers it.
    /// Worth having early and permanently, because it is the one configuration
    /// whose output can be checked *exactly* against the source rather than
    /// scored, which turns quality into a pass/fail.
    Lossless,
    /// Average bitrate: the encoder picks a quantiser per picture to spend
    /// roughly this many bits per second, given [`Config::fps`].
    ///
    /// The first mode whose correctness is not a property of the bitstream.
    /// A controller that ignores this number entirely still produces a
    /// perfectly legal stream that decodes identically on every decoder —
    /// see the module documentation of `encode::rc` for what is checked
    /// instead, and how.
    Bitrate {
        /// Target, in bits per second.
        bps: u32,
    },
}

/// Entropy coder, where the standard offers a choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entropy {
    /// H.264 only: variable-length coding. Simpler, and the sensible first
    /// target because it does not need an arithmetic coder to be correct.
    Cavlc,
    /// Context-adaptive arithmetic coding. H.265 has nothing else.
    Cabac,
}

/// The colour a stream's samples are to be interpreted in: the H.273
/// code points a display needs to show BT.2020 PQ as HDR rather than as
/// washed-out BT.709, carried in the SPS VUI (`video_signal_type_present_flag`,
/// H.264 E.1.1 / H.265 E.2.1). A stream without one says nothing, which
/// every player reads as BT.709 limited range.
///
/// The codes are the standard's own, not an enum: the writer copies them
/// into three 8-bit fields, the reader (`h264::sps::Vui`, `hevc::sps::Vui`)
/// hands them back as the same three numbers, and an enum in between would
/// be a place for a value to fail to round-trip. BT.2020 PQ is `9, 16, 9`;
/// HLG is `9, 18, 9`; SDR BT.709 is `1, 1, 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourDescription {
    /// `colour_primaries` (H.273 table 2): 1 BT.709, 9 BT.2020.
    pub primaries: u8,
    /// `transfer_characteristics` (H.273 table 3): 1 BT.709, 16 PQ (SMPTE
    /// ST 2084), 18 HLG (ARIB STD-B67).
    pub transfer: u8,
    /// `matrix_coefficients` (H.273 table 4): 1 BT.709, 9 BT.2020
    /// non-constant luminance.
    pub matrix: u8,
    /// `video_full_range_flag`: false is studio range (16..235 at 8
    /// bits), true is full range.
    pub full_range: bool,
}

/// HDR10 static metadata, first half: the colour volume of the display
/// the content was mastered on (SMPTE ST 2086), carried as the
/// `mastering_display_colour_volume` SEI — payloadType 137, H.264 D.1.29
/// and H.265 D.2.28, the same twelve fields in the same order. A player
/// tone-maps against these; without them Apple's fall back to BT.709 even
/// when the VUI says BT.2020 PQ.
///
/// Chromaticities are CIE 1931 (x, y) in units of 0.00002 (so BT.2020's
/// red is `(34000, 16000)`), luminances in units of 0.0001 cd/m² (so 1000
/// nits is `10_000_000`) — the SEI's own units, copied into it unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplay {
    /// Red primary (x, y) — `display_primaries_x/y[2]` in the SEI's order.
    pub red: (u16, u16),
    /// Green primary (x, y) — `display_primaries_x/y[0]`.
    pub green: (u16, u16),
    /// Blue primary (x, y) — `display_primaries_x/y[1]`.
    pub blue: (u16, u16),
    /// White point (x, y).
    pub white_point: (u16, u16),
    /// `max_display_mastering_luminance`, 0.0001 cd/m².
    pub max_luminance: u32,
    /// `min_display_mastering_luminance`, 0.0001 cd/m².
    pub min_luminance: u32,
}

/// HDR10 static metadata, second half: how bright the content itself gets
/// (CTA-861.3), carried as the `content_light_level_info` SEI —
/// payloadType 144, H.264 D.1.31 and H.265 D.2.35. Both in cd/m².
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLightLevel {
    /// `max_content_light_level`: the brightest pixel in the stream.
    pub max_cll: u16,
    /// `max_pic_average_light_level`: the brightest picture average.
    pub max_fall: u16,
}

/// Which field of an interlaced frame is earlier in time — the order the
/// two fields were captured in, which is the order they are coded and
/// displayed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// The top field (the frame's even rows, counting from 0) first.
    TopFirst,
    /// The bottom field (the odd rows) first.
    BottomFirst,
}

/// How an interlaced H.264 frame is coded (see [`Config::interlace`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldCoding {
    /// Every frame as two field pictures (`field_pic_flag` 1), in the
    /// frame's field order — picture-adaptive frame/field coding with the
    /// choice pinned to fields.
    Field,
    /// Picture-adaptive frame/field coding: each frame as one frame
    /// picture or as two field pictures, chosen by cost.
    Paff,
    /// Macroblock-adaptive frame/field coding (`mb_adaptive_frame_field_flag`
    /// 1): frame pictures whose macroblock pairs are each coded as frame or
    /// field macroblocks, chosen by cost.
    Mbaff,
}

/// How an H.264 B slice weights its predictions (8.4.2.3) — the PPS's
/// `weighted_bipred_idc` — see [`Config::b_weighting`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BWeighting {
    /// `weighted_bipred_idc` 0: a one-list prediction as it is, a two-list
    /// one the plain average of the two.
    Default,
    /// `weighted_bipred_idc` 2: a two-list prediction weighted by the
    /// picture's distances in display order to its two references
    /// (8.4.2.3.1) — two thirds and one third for a B picture a third of
    /// the way between its anchors — a one-list one as it is. Nothing is
    /// written per slice: the weights follow from the picture order counts.
    Implicit,
    /// `weighted_bipred_idc` 1: a fitted `pred_weight_table` in every B
    /// slice, priced against a table of defaults — what
    /// [`Config::weighted_pred`] gives B slices, and only beside it.
    Explicit,
}

/// Everything the encoder needs that is not a picture.
#[derive(Debug, Clone)]
pub struct Config {
    /// Luma dimensions. Not required to be a multiple of the coding block
    /// size; the encoder pads and signals the crop.
    pub width: u32,
    /// See `width`.
    pub height: u32,
    /// 8 to 14. The decoders handle the whole range and so must these.
    pub bit_depth: u32,
    /// 4:0:0 through 4:4:4.
    pub chroma: ChromaFormat,
    /// Pictures between IDRs. 0 means every picture is an IDR.
    pub gop: u32,
    /// Consecutive B pictures between references. 0 disables B pictures.
    pub bframes: u32,
    /// The most a slice may reference.
    pub max_refs: u32,
    /// See [`RateControl`].
    pub rate: RateControl,
    /// See [`Entropy`]. Ignored by H.265, which is always CABAC.
    pub entropy: Entropy,
    /// H.264: offer the 8x8 transform (`transform_8x8_mode_flag` in the
    /// PPS, and the per-macroblock `transform_size_8x8_flag` the encoder
    /// may then set). Off by default, so a stream that does not ask for
    /// it is byte-identical to one from an encoder that never had it.
    ///
    /// It needs a High profile, which every profile this encoder claims
    /// already is, and it is ignored by H.265 — whose transform sizes are
    /// a different mechanism entirely.
    pub transform_8x8: bool,
    /// H.264: offer inter partitions smaller than 16x16 — 16x8, 8x16 and
    /// the 8x8 sub-macroblock tree. Off by default, so a stream that does
    /// not ask for them is byte-identical to one from an encoder that
    /// never had them.
    ///
    /// Unlike [`Config::transform_8x8`] nothing in a parameter set
    /// announces this: every profile admits the shapes, so it is purely
    /// the encoder's own switch. Ignored by H.265, whose prediction units
    /// are a different mechanism.
    pub subparts: bool,
    /// Worker threads; 0 asks for one per core, matching the decoders.
    pub threads: usize,
    /// Coded picture buffer to declare, in milliseconds of the target
    /// bitrate. 0 declares no buffer at all, which is what every stream
    /// this encoder wrote before the buffer model existed.
    ///
    /// Only meaningful with [`RateControl::Bitrate`]: a buffer is a
    /// constraint on a rate, and there is no rate to constrain at a fixed
    /// quantiser. Asking for one anyway refuses by name.
    pub cpb_ms: u32,
    /// Frames per second. Nothing in either bitstream carries it — H.265
    /// puts frame rate in the optional VUI, which this encoder does not
    /// write — so it exists for exactly one reason: a target in bits per
    /// *second* is meaningless without it. It is declared rather than
    /// assumed so that a caller who cares can set it.
    pub fps: u32,
    /// Sample adaptive offset, the second in-loop filter (H.265 only).
    ///
    /// Off by default and a switch rather than something always applied,
    /// unlike deblocking: SAO costs bits per CTB and only pays where there
    /// is quantisation noise to shape, so a caller coding at a low
    /// quantiser wants it off. Setting it writes
    /// `sample_adaptive_offset_enabled_flag` in the SPS, which makes one
    /// or two more flags appear in *every* slice header.
    pub sao: bool,
    /// Adaptive quantisation strength, both codecs: 0 is off, which is
    /// the default. Above 0 every block — an H.265 coding tree block, an
    /// H.264 macroblock — is quantised at its own offset from the picture
    /// quantiser, chosen from its luma variance: flat blocks finer,
    /// textured blocks coarser, zero-mean over the picture, and at most
    /// six steps either way — one model, `encode::aq`. H.265 sets
    /// `cu_qp_delta_enabled_flag` in the PPS and carries each offset as a
    /// `cu_qp_delta`; H.264 has no switch to set and carries it as the
    /// `mb_qp_delta` every macroblock with a residual already has room
    /// for. In both, a block with no residual can carry no delta and holds
    /// the predicted quantiser. 1.0 is the strength the measurements in
    /// `encode::aq` were taken at. A lossless stream has no quantiser to
    /// adapt, and both codecs refuse the combination by name.
    ///
    /// A switch rather than always-on for the reason SAO is: it costs a
    /// delta per coded block and it trades global PSNR for a more even
    /// distribution of error, which a caller measuring PSNR does not
    /// want. Off, the stream is byte-identical to one from an encoder
    /// that never had it.
    pub aq_strength: f32,
    /// Rate-control lookahead (H.265 only): how many pictures the encoder
    /// holds back before coding one, so the controller can place bits by
    /// what is coming. 0 is off, which is the default; every picture is
    /// then coded as soon as the picture typing allows, and the stream is
    /// byte-identical to one from an encoder that never had it.
    ///
    /// Only meaningful with [`RateControl::Bitrate`]: a lookahead informs
    /// a rate controller, and a fixed quantiser has none to inform, so
    /// asking for one anyway refuses by name — as a coded picture buffer
    /// does. Each held picture is measured once (an 8x8 SATD sum, intra
    /// and against the previous picture) and the controller allocates the
    /// window's budget by those measurements; see `encode::rc`'s lookahead
    /// section for exactly what changes. Costs `lookahead` pictures of
    /// output delay and their source samples in memory.
    ///
    /// H.264 refuses it by name ("rate lookahead is not calibrated for
    /// H.264"). It was wired and measured on the branch
    /// `agent/h264tools-lookahead`, first with H.265's constants and then
    /// with H.264's own calibration of them (bits per cost of P and B
    /// against intra pictures, the reference-noise floor, the insensitivity
    /// band, each measured on the corpus), and it did not beat the
    /// past-only controller. Mean `|achieved / target - 1|` over the ten
    /// 8-bit clips, one binary, 2026-09-14:
    ///
    /// ```text
    ///   row                without   H.265 constants   H.264 calibration
    ///   abr 64k             0.177        0.195             0.183
    ///   abr 128k            0.152        0.151             0.121
    ///   CAVLC 64k           0.193        0.233             0.202
    ///   IPB 64k             0.186        0.261             0.229
    ///   AQ 64k              0.165        0.195             0.173
    ///   10-bit 128k         0.075        0.193             0.214
    /// ```
    ///
    /// Calibrated, the buffer row (64k, 125 ms, `src_cut`) was still refused
    /// — keyframes planned at QP 42..49 left P pictures that cost more than
    /// the per-picture rate at QP 51 — and PSNR at 64k fell 4.97 dB on
    /// average, because the past-only controller's first keyframe overshoots
    /// at the seed's QP 26 and carries a short clip, where the lookahead
    /// plans it at its share. On the held-out 256x160 clip the calibrated
    /// lookahead was the better controller at 128k..1280k (0.253 to 0.095 at
    /// 1280k), which is why the branch is kept rather than discarded.
    pub lookahead: u32,
    /// Weighted prediction, both codecs: off by default. On, the PPS sets
    /// `weighted_pred_flag` and every P slice carries a
    /// `pred_weight_table` — a gain and an offset per reference, fitted
    /// per picture to the source against the reference and used only
    /// where the fit lowers the residual (`encode::h265_wp`, one fit held
    /// to the weights each codec's table carries), the default weights
    /// otherwise. What it buys is a fade: motion compensation cannot
    /// change a reference's brightness, so without this every block of a
    /// fading picture carries the level change as residual.
    ///
    /// Both codecs weight B slices too when the GOP has B pictures: the
    /// PPS sets H.265's `weighted_bipred_flag` or H.264's
    /// `weighted_bipred_idc` 1, and every B slice's table carries an entry
    /// for each list's reference, fitted the same way, which the one-list
    /// and the bi predictions both apply (8.5.3.3.4.3, 8.4.2.3). Each
    /// codec prices a B picture's fitted table against a table of defaults
    /// and keeps the cheaper, and H.264 a P picture's where no fit in it is
    /// strong (`encode::h264`'s `code_attempt`). A lossless H.264 stream
    /// refuses it, its inter pictures being exact copies.
    /// Off, the stream is byte-identical to one from an encoder that never
    /// had it.
    pub weighted_pred: bool,
    /// Colour description to write into the SPS VUI, or `None` to write
    /// nothing about colour — which is what every stream this encoder
    /// wrote before the field existed, so an unset field keeps them all
    /// byte-identical. Set for HDR: without it a BT.2020 PQ picture is
    /// displayed as BT.709 by every player that does not read the
    /// container's colour box, and some that do.
    pub colour: Option<ColourDescription>,
    /// Where the 4:2:0 chroma samples sit relative to the luma grid, as
    /// H.273's `chroma_sample_loc_type` (0 left — the siting every decoder
    /// assumes when nothing is said; 1 centre — JPEG / MPEG-1, what a 2x2
    /// box average produces; 2 top-left; 3 top; 4 bottom-left; 5 bottom),
    /// written into the SPS VUI's `chroma_loc_info_present_flag` group for
    /// both fields — or `None` to write nothing, which keeps every stream
    /// from before the field existed byte-identical. A consumer that
    /// upsamples at the wrong siting loses about a decibel of chroma on
    /// detail; the field is what lets it not. 4:2:0 only: the siting
    /// describes a subsampled grid, E.2.1 says the flag should be 0 for
    /// any other format, and libavcodec reports none there whatever the
    /// VUI says — so a siting beside another format is refused by name.
    pub chroma_loc: Option<u8>,
    /// HDR10 mastering display colour volume, written as an SEI in every
    /// IDR / IRAP access unit — or `None` for no such SEI, which is what
    /// every stream before the field existed had. Meaningful beside a
    /// BT.2020 PQ [`colour`](Self::colour); the encoder does not insist.
    pub mastering_display: Option<MasteringDisplay>,
    /// HDR10 content light level, likewise an SEI in every IDR / IRAP
    /// access unit, or `None` for none.
    pub content_light: Option<ContentLightLevel>,
    /// H.265 only: how many levels the coding quadtree may split a coding
    /// tree block into smaller coding units, or `None` for the encoder's
    /// default — [`h265::DEFAULT_CU_DEPTH`], 2. `Some(0)` codes one unit
    /// per CTB, the geometry every stream had before the quadtree existed
    /// and byte-identical to it; `Some(1)` lets a unit halve once, `Some(2)`
    /// twice. A split never goes below the 8x8 minimum coding block the SPS
    /// declares, so a 16x16 CTB (the encoder chooses one for small or oddly
    /// sized pictures) splits at most once whatever this asks, and the
    /// census line reports the depths each picture kind actually took.
    ///
    /// Every node is a rate-distortion decision, which is what the default
    /// buys and costs: against `Some(0)`, depth 2 measured -29% BD-rate
    /// all-intra and -36% IP for 2.8-3.1x the CPU all-intra and 4.5-6x
    /// IPB; depth 1 half the saving for about twice the CPU (see
    /// `encode::h265`). A caller that needs throughput asks for less.
    ///
    /// An `Option` rather than a number because H.264 has no quadtree: the
    /// H.264 encoder refuses `Some(n)` with `n > 0` by name, and a number
    /// defaulting to 2 would have made every default configuration one it
    /// refuses.
    pub max_cu_depth: Option<u32>,
    /// H.264: the source pictures are interlaced frames, in this field
    /// order, and are coded as interlaced video (`frame_mbs_only_flag` 0)
    /// the way [`Config::field_coding`] says. `None`, the default, codes
    /// progressive frames and every stream is byte-identical to one from
    /// an encoder that never had the switch.
    ///
    /// A frame's top field is its even rows and its bottom field its odd
    /// ones, chroma included. The height must be a whole number of
    /// cropping units — four rows in 4:2:0, two otherwise — because an
    /// interlaced stream crops in field rows (7.4.2.1.1's `CropUnitY`).
    /// H.265 has no interlaced coding tools and refuses the switch by
    /// name.
    pub interlace: Option<FieldOrder>,
    /// H.264, with [`Config::interlace`]: field pictures, picture-adaptive
    /// or macroblock-adaptive frame/field coding. Ignored without it.
    pub field_coding: FieldCoding,
    /// H.264: how B slices weight their predictions, or `None` for the
    /// encoder's choice — [`BWeighting::Explicit`] under
    /// [`Config::weighted_pred`], [`BWeighting::Default`] otherwise.
    ///
    /// `Some(Explicit)` needs `weighted_pred` (the explicit table is its
    /// fit); `Some(Default)` beside `weighted_pred` weights P slices and
    /// leaves B slices to the plain average; `Some(Implicit)` weights B
    /// slices by distance whether or not P slices are weighted. Interlaced
    /// and lossless streams code default-weighted B pictures and refuse
    /// `Some(Implicit)` by name. H.265 has no implicit mode: it takes
    /// `None`, or `Some` of what it does anyway (explicit B under
    /// `weighted_pred`, default without), and refuses the rest by name.
    ///
    /// Implicit weighting is asked for, not chosen, because it is not a
    /// gain everywhere. Against default weighting over every clip of the
    /// corpus and rivet's 640x360 set, at one to three B pictures and QP
    /// 22..40, it gains where the picture changes between its anchors — a
    /// fade -2.4 to -16.8% BD-rate, a gradient -1.4 to -3.4%, motion -0.3 to
    /// -1.2% — and loses where it does not: detail +0.1 to +0.4%, testsrc2
    /// +0.1 to +0.2%, combed interlaced frames coded progressive +0.5 to
    /// +0.8%, with 48 of 504 cells both larger and worse. Two equally good
    /// anchors average their noise best at equal weights. With one B
    /// picture it is the default weighting: halfway, the weights are 32
    /// and 32.
    pub b_weighting: Option<BWeighting>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            gop: 250,
            bframes: 0,
            max_refs: 1,
            rate: RateControl::ConstantQp(26),
            entropy: Entropy::Cabac,
            transform_8x8: false,
            subparts: false,
            threads: 0,
            sao: false,
            fps: 30,
            cpb_ms: 0,
            aq_strength: 0.0,
            lookahead: 0,
            weighted_pred: false,
            colour: None,
            chroma_loc: None,
            mastering_display: None,
            content_light: None,
            max_cu_depth: None,
            interlace: None,
            field_coding: FieldCoding::Paff,
            b_weighting: None,
        }
    }
}

impl Config {
    /// Reject what the encoder cannot legally or sensibly produce, before it
    /// has written a byte. An encoder that fails late has usually already
    /// emitted a header describing something it then cannot deliver.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(crate::Error::unsupported("encode: zero-sized picture"));
        }
        if !(8..=14).contains(&self.bit_depth) {
            return Err(crate::Error::unsupported("encode: bit depth outside 8..=14"));
        }
        if self.max_refs == 0 {
            return Err(crate::Error::unsupported("encode: max_refs must be at least 1"));
        }
        if !(self.aq_strength >= 0.0) || self.aq_strength > 4.0 {
            return Err(crate::Error::unsupported("encode: aq_strength outside 0.0..=4.0"));
        }
        if self.lookahead > 0 && !matches!(self.rate, RateControl::Bitrate { .. }) {
            return Err(crate::Error::unsupported(
                "encode: a lookahead without a bitrate target (a lookahead informs a rate controller; a fixed quantiser has none)",
            ));
        }
        if self.lookahead > 250 {
            return Err(crate::Error::unsupported("encode: lookahead above 250 pictures"));
        }
        if self.chroma_loc.is_some_and(|t| t > 5) {
            return Err(crate::Error::unsupported("encode: chroma_loc outside 0..=5 (H.273 chroma_sample_loc_type)"));
        }
        if self.chroma_loc.is_some() && self.chroma != ChromaFormat::Yuv420 {
            return Err(crate::Error::unsupported(
                "encode: chroma_loc is a 4:2:0 siting (E.2.1: chroma_loc_info_present_flag should be 0 for any other format)",
            ));
        }
        if self.max_cu_depth.is_some_and(|d| d > 2) {
            return Err(crate::Error::unsupported(
                "encode: max_cu_depth above 2 (a coding tree block of at most 32x32 reaches the 8x8 minimum coding block in two splits)",
            ));
        }
        Ok(())
    }
}

/// Unpack one source picture from the caller's bytes into samples: the
/// bytes themselves at 8 bits, little-endian pairs deeper — the layout
/// [`crate::Picture::into_packed`] emits, so the two sides of SELF agree
/// without a conversion in between. `codec` names the encoder in the
/// refusal.
///
/// A sample above the declared depth is refused rather than coded: the
/// prediction and transform arithmetic assume `0..2^BitDepth`, and a
/// 10-bit stream carrying a 12-bit value would not fail, it would wrap
/// somewhere in the reconstruction and desync. At 8 bits every byte is
/// in range and nothing is checked.
pub(crate) fn unpack_samples<S: crate::sample::Sample>(bytes: &[u8], bit_depth: u32, codec: &str) -> Result<Vec<S>> {
    if S::BYTES == 1 {
        return Ok(bytes.iter().map(|&b| S::from_i32(i32::from(b))).collect());
    }
    let max = (1u32 << bit_depth) - 1;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let v = u16::from_le_bytes([pair[0], pair[1]]);
        if u32::from(v) > max {
            return Err(crate::Error::bitstream(format!(
                "{codec} encode: source sample {v} exceeds the declared {bit_depth}-bit depth"
            )));
        }
        out.push(S::from_i32(i32::from(v)));
    }
    Ok(out)
}

/// The inverse of [`unpack_samples`] for one row of a reconstruction.
pub(crate) fn pack_row<S: crate::sample::Sample>(row: &[S], out: &mut Vec<u8>) {
    if S::BYTES == 1 {
        out.extend(row.iter().map(|s| s.to_i32() as u8));
    } else {
        for s in row {
            out.extend_from_slice(&(s.to_i32() as u16).to_le_bytes());
        }
    }
}

/// One coded picture, and what the caller needs to know about it.
#[derive(Debug)]
pub struct Access {
    /// The Annex B byte stream for this picture: start codes included, ready
    /// to concatenate.
    pub data: Vec<u8>,
    /// Whether a decoder may begin here.
    pub keyframe: bool,
    /// Display order *within the GOP*: picture order count, reset to zero
    /// at every IDR. Two per picture (see `gop.rs`).
    pub poc: i32,
    /// Coding order.
    pub encode_index: u64,
    /// Display order across the whole stream: the index, counted from the
    /// first picture ever pushed, of the picture this access unit codes.
    ///
    /// With B pictures coding order is not display order, and `poc` cannot
    /// recover it because it restarts at each IDR. A caller that hands out
    /// timestamps needs exactly this: the packet for the picture pushed
    /// `display`-th carries that picture's timestamp, whatever position it
    /// was coded at.
    pub display: u64,
}
