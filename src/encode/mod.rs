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
//! `tools/verify_encode.sh` gates 1 and 2 and reports 3.
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
pub(crate) mod h265_rc;
pub(crate) mod h265_sao;
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
    /// see the module documentation of `encode::h265_rc` for what is checked
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
    /// Worker threads; 0 asks for one per core, matching the decoders.
    pub threads: usize,
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
            threads: 0,
            sao: false,
            fps: 30,
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
        Ok(())
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
    /// Display order.
    pub poc: i32,
    /// Coding order.
    pub encode_index: u64,
}
