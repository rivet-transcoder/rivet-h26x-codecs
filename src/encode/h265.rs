//! The H.265 encoder.
//!
//! Mirrors [`crate::hevc::HevcDecoder`]: pictures in, access units out, and
//! the same envelope-first discipline that brought the H.264 encoder up — the
//! configuration, geometry, picture typing and coding order are all built and
//! exercised before any entropy coding exists, so the gate can tell "not
//! built" from "wrong" while the coding-tree serialiser is written.
//!
//! # Why this one cannot take the I_PCM shortcut
//!
//! The H.264 encoder's first legal bitstream was all-I_PCM through CAVLC,
//! because H.264's PCM macroblock is reachable through a bit-level entropy
//! path simple enough to write in an afternoon. H.265 has no CAVLC: *every*
//! slice payload is CABAC, PCM included, and a PCM coding unit still sits
//! inside an arithmetic-coded quadtree. So the simplest legal H.265 stream
//! already needs the coding-tree writer, and this module refuses at exactly
//! that point until it exists. The parameter sets above it are written and
//! proven against the crate's own conformance-tested parsers.
//!
//! # The coding quadtree
//!
//! `Config::max_cu_depth` ([`DEFAULT_CU_DEPTH`], 2, when unset) lets a CTB
//! split into smaller coding units, down to the 8x8 minimum coding block. Every node is a
//! rate-distortion decision — the node coded whole against the node coded
//! as four children, each side's cost its reconstruction's SSD plus the
//! Lagrangian times the bits the production writer counts for its syntax
//! (`IntraPicture::code_ctu_tree`, `InterPicture::code_ctu_tree`) — and at
//! 8x8 an intra unit also weighs `PART_NxN`. The units are kept placed, in
//! decode order (`TreeCu`); the quadtree is read back from the placements
//! to be written (`write_tree`) and walked by the quantiser chain
//! (`QgChain`), which follows the reader's quantisation groups at any group
//! and unit size. At `Some(0)` every CTB is one unit, and the stream is the
//! one this encoder wrote before the quadtree existed.
//!
//! ## Measured (2026-09-14), and the default it decided
//!
//! One binary, `--cu-depth` the only difference, BD-rate by
//! `tools/bd_rate.py`'s method (QP 22/27/32/37, luma PSNR of the encoder's
//! own reconstruction) over the eleven 8-bit clips of the encode corpus.
//! Control: depth 0 encoded twice is byte-identical at every point.
//!
//! ```text
//!   BD-rate against depth 0, mean of 11 clips
//!                  depth 1    depth 2
//!   all-intra      -19.0%     -29.2%
//!   IP             -15.8%     -36.5%
//!   IPB            -15.7%     -36.3%
//!
//!   depth 2, IP, per clip: big (256x160) -58.2%, motion -55.6%,
//!   detail 400/420/422/444 -47.3/-45.7/-45.1/-40.2%, cut -42.4%,
//!   fade -31.0%, static -27.6%, odd -8.4% (16x16 CTBs: depth 1 is its
//!   limit), grad -0.1% (smooth gradients split almost nowhere)
//! ```
//!
//! The gate's rows at their own quantisers agree: `hevc-cu2-ipb` is 30.4%
//! smaller than `hevc-cqp-ipb` at +1.95 dB (mean over eleven clips),
//! `hevc-cu2-intra` 24.9% smaller at +1.41 dB, `hevc-cu2-40-ip` 5.9%
//! smaller at +0.53 dB, lossless IPB 33.0% smaller. The model check holds:
//! what the split decisions priced the coded units at is within +4.5% to
//! +9.7% of the slice data they took on every lossy configuration (census
//! `model_bits` over `coded_bits`; the neutral-context prices run a little
//! high, never low).
//!
//! Encode time, per-process CPU seconds on one pinned core, five
//! interleaved rounds, median of paired ratios against depth 0, on
//! workloads long enough for the CPU clock to resolve (the detail and cut
//! clips repeated to 960 frames, the 256x160 clip to 256). The control, a
//! second depth-0 run in every round, came out at 0.94–1.02: on a shared
//! machine, differences under about 10% are not resolved.
//!
//! ```text
//!                      depth 1   depth 2
//!   detail  all-intra   1.97x     2.91x
//!   cut     all-intra   2.01x     3.09x
//!   big     all-intra   1.73x     2.75x
//!   detail  IPB         2.61x     5.96x
//!   cut     IPB         2.27x     4.50x
//!   big     IPB         2.33x     5.09x
//! ```
//!
//! Inter pictures pay more than intra ones because a whole-CTB inter unit
//! is cheap — one motion search, mostly skips — while every node below it
//! runs a search of its own.
//!
//! **Decided: the default is 2** ([`DEFAULT_CU_DEPTH`]). It buys a third of
//! the bits at equal quality (-29% all-intra, -36% IP) for three to six
//! times the CPU — what a full CU quadtree buys in any H.265 encoder, and a
//! larger saving than every other tool this encoder has put together, on
//! every clip but the flat gradient, where it costs nothing but time. A
//! caller whose encode throughput binds asks for `Some(1)` — half the
//! saving (-19% / -16%) for about twice the CPU — or `Some(0)`.
//!
//! ## Chroma (2026-09-14)
//!
//! Per-plane BD-rate against depth 0 — Y, Cb, Cr and YUV weighted 6:1:1 —
//! mean of ten clips (eight 4:2:0, detail 4:2:2 and 4:4:4), over QP 22-40
//! and again over QP 34-43, where an equal-QP chroma floor first flagged
//! the quadtree:
//!
//! ```text
//!                   QP 22-40 (Y Cb Cr YUV)     QP 34-43 (Y Cb Cr YUV)
//!   all-intra      -27.3 -23.0 -23.5 -26.3    -19.8 -12.1 -13.6 -18.1
//!   IP             -32.6 -31.7 -32.3 -32.5    -16.1 -17.0 -10.6 -16.9
//!   IP before      -32.5 -31.3 -31.9 -32.3    -16.0 -16.6  -7.9 -16.6
//! ```
//!
//! The IP rows are the one change the measurement led to: pictures that may
//! be predicted from weigh chroma in the split cost (`h265_intra::cu_ssd`),
//! and every plane of both ranges gained; intra streams are unchanged by
//! it. What stayed: at QP 40 all-intra the quadtree codes chroma 1 to 1.7 dB
//! below depth 0 at the same QP, for 15-23% fewer bytes — `rdoq_trim`
//! prices chroma at the luma Lagrangian, and 8x8 units bring many 4x4
//! chroma blocks to trim. Every re-weighting of that tried (HM's weight,
//! its square root, 4x4 blocks only, no chroma trims) moved BD-rate from
//! luma to chroma and lost on YUV. Still positive in a chroma plane, and
//! known: the flat gradient in IP (Cb +0.8% at QP 22-40, Cb and Cr far
//! above at 34-43), whose rate points lie within 1% of each other so that
//! BD-rate cannot rank them, and the 50x34 clip's IP Cr at QP 34-43
//! (+1.1%).
//!
//! ## CTB size (2026-09-14)
//!
//! Under the coding quadtree every picture 64 or more in either direction
//! codes 32x32 CTBs, partial along the right and bottom edges, and the
//! coded size is the smallest legal one — whole 8x8 minimum coding blocks
//! (`h265_syntax::Geometry::new`, `tree_steps`). The rule before picked 16
//! or 32, whichever padded less, so 1280x720, 3840x2160 and 640x360 coded
//! 16x16 CTBs, where the quadtree can split once and depth 2 buys nothing.
//! One binary, 16 frames, QP 22/27/32/37, against that rule (bytes summed
//! over the QPs; BD-rate of luma and of YUV 6:1:1; per-process CPU
//! seconds):
//!
//! ```text
//!                          CTB 32 padded to whole CTBs     CTB 32 partial at the edges
//!                          bytes  BD Y  BD YUV  CPU        bytes  BD Y  BD YUV  CPU
//!   1280x720  testsrc2 I   -3.3%  -5.4   -6.2  1.00x       -3.2%  -5.3   -6.2  0.96x
//!                      IP  -6.2%  -7.4   -8.5  0.82x       -6.1%  -7.4   -8.5  0.83x
//!             natural  I   -3.6%  -4.8   -5.7  1.41x       -4.6%  -5.8   -6.7  1.35x
//!                      IP -22.9% -28.5  -29.0  0.93x      -24.0% -29.4  -29.8  0.88x
//!   3840x2160 testsrc2 I   -4.4%  -7.4   -8.8  0.82x       -4.4%  -7.4   -8.8  0.82x
//!                      IP  -6.3%  -8.0   -9.8  0.65x       -6.3%  -8.0   -9.8  0.64x
//!             natural  I  -16.1% -18.8  -22.3  1.03x      -16.5% -19.1  -22.6  1.03x
//!                      IP -38.8% -45.2  -46.5  0.85x      -39.1% -45.4  -46.7  0.84x
//! ```
//!
//! Partial CTBs code the fewest bytes of the three everywhere — by up to a
//! point on the natural 720p clip, whose padded bottom rows were replicated
//! content the coder still paid for — at the same CPU. The 1280x720 CPU
//! figures overlap other encoding on the machine and carry that noise; the
//! 3840x2160 ones do not. At `max_cu_depth` 0 a whole-CTB unit cannot be
//! partial, so that geometry keeps the old rule, with one change: it never
//! takes CTB 16 beyond level 4.1's picture limits, where no level admits
//! it. So 3840x2160 at depth 0 codes 32x32 CTBs, 3840x2176.
//!
//! So do pictures below 64 both ways, and that exception is fitted to one
//! clip. Partial CTB 32 against the old rule's whole CTBs (16x16 on both
//! clips; per-plane YUV BD-rate, QP 22-40 and in brackets 34-43, AQ at
//! `--aq 1.0`):
//!
//! ```text
//!              50x34 (coded 56x40, was 64x48)   88x44 (coded 88x48, was 96x48)
//!   I          -0.56%  (-0.62%)                  -0.80%  (-1.61%)
//!   IP         +1.08%  (+1.66%)                  -7.64%  (-9.67%)
//!   IPB        +1.33%  (+3.74%)                  -7.24%  (-9.80%)
//!   AQ IP      +8.58% (+11.75%)                  -5.98%  (-6.60%)
//!   AQ IPB     +7.51% (+11.40%)                  -4.94%  (-6.98%)
//! ```
//!
//! CTB 32 padded to whole CTBs (64x64) lost as much on 50x34 with AQ (IP
//! +11.1%, IPB +10.6%), and 8x8 groups on the partial CTBs still lost 8.2%
//! and 8.0%, so it is CTB 32 on a picture that small, not the partial
//! geometry or the group size. 50x34 keeps the old rule and codes as it
//! did; 88x44, 64 or more one way, takes partial CTBs. No size between the
//! two was measured, so 64 marks where the rule was drawn, not where the
//! loss ends.
//!
//! The quantisation group adaptive quantisation uses follows the stream:
//! the CTB in an all-intra stream, half the CTB where pictures are
//! predicted from others (`Core::new` records the measurement; CTB groups
//! in every stream regressed the fading clip in IP and IPB). It does so at
//! either CTB size: on 50x34's 16x16 CTBs the all-intra group of 16
//! against the 8 before gains 1.9% YUV at QP 22-40 (3.3% at 34-43), every
//! plane better.

use super::gop::{Coded, Kind, Scheduler};
use super::h265_deblock::{deblock_inter_picture, deblock_picture};
use super::h265_intra::{CuDecision, IntraCtx, IntraPicture, MIN_CB_LOG2, Srcs, TreeCu, ssd_lambda};
use super::h265_me::{InterCuDecision, InterCuKind, InterPicture, PCuDecision, TreeRefs, MAX_MERGE_CAND};
use super::rc::{Insensitivity, PicKind, RateController};
use super::h265_sao::{SaoPlan, sao_picture};
use super::aq;
use super::h265_wp;
use super::h265_syntax::{self as syn, Cpb, PpsOptions};
use crate::hevc::ctu::explicit_weighting;
use super::{Access, Config, RateControl};
use crate::bitwriter::BitWriter;
use crate::cabac_enc::CabacEncoder;
use crate::dsp::distortion::DistortionDsp;
use crate::dsp::hevc::HevcDsp;
use crate::dsp::hevc_enc::HevcEncDsp;
use crate::dsp::Cpu;
use crate::hevc::ctx::Contexts;
use crate::hevc::pic::PicInfo;
use crate::sample::Sample;
use crate::hevc::ctu::{
    SaoCtx, SaoMergeNb, SplitCuNb, qp_y_from_pred, qp_y_pred_from, write_cbf_chroma, write_cbf_luma, write_cu_qp_delta,
    write_cu_skip_flag, write_sao,
    write_cu_transquant_bypass_flag, write_merge_flag, write_merge_idx, write_mvd,
    write_inter_pred_idc, write_mvp_flag, write_part_mode_inter, write_pred_mode_flag,
    write_ref_idx, write_rqt_root_cbf,
    write_intra_chroma_pred_mode, write_mpm_idx, write_part_mode_intra, write_prev_intra_luma_pred_flag,
    write_rem_intra_luma_pred_mode, write_split_cu_flag, write_split_transform_flag,
};
use crate::hevc::residual::{ResidualParams, residual_scan_idx, write_residual};
use crate::{Error, Result};

/// H.265 encoder. See the module documentation for what is and is not built.
///
/// A thin face over the private `Core<S>`, instantiated at the sample
/// width the configuration's bit depth needs — the same split the decoder
/// makes at its NAL layer: everything below is generic over the crate's
/// `Sample` trait, `u8`
/// for 8-bit streams and `u16` for 9- to 14-bit ones, so an 8-bit encode
/// moves half the bytes and runs the 8-bit SIMD tiers, and a deeper one
/// runs the decoder's own 16-bit kernels for prediction, transform,
/// deblocking and SAO rather than a second set written for the encoder.
///
/// Pictures cross this face as bytes in both directions, in the layout
/// [`crate::Picture::into_packed`] uses: one byte per sample at 8 bits,
/// little-endian `u16` pairs deeper — so a source picture and the
/// decoder's output of it compare byte for byte at every depth, which is
/// what the SELF check reads. [`H265Encoder::frame_bytes`] says how many.
pub struct H265Encoder {
    inner: Inner,
}

/// The two sample widths an encoder can be built at.
enum Inner {
    /// 8-bit samples.
    Eight(Core<u8>),
    /// 9 to 14 bits.
    Wide(Core<u16>),
}

/// Run one expression against whichever width the encoder was built at.
macro_rules! with_core {
    ($inner:expr, $e:ident => $body:expr) => {
        match $inner {
            Inner::Eight($e) => $body,
            Inner::Wide($e) => $body,
        }
    };
}

/// The encoder proper, at one sample width.
struct Core<S: Sample> {
    cfg: Config,
    sched: Scheduler,
    /// Source pictures held in display order, so a B picture kept back by the
    /// scheduler still has its samples when its anchor arrives — already
    /// unpacked from the caller's bytes into samples, so the coding
    /// paths never see a byte layout.
    held: std::collections::BTreeMap<u64, Vec<S>>,
    /// Reconstructions in coding order, for the SELF check.
    recon: Vec<Vec<u8>>,
    frame_bytes: usize,
    /// Display index of the next picture offered — an explicit counter, for
    /// the reason recorded on the H.264 encoder: `held` empties as pictures
    /// code, so inferring the index from it fails the moment the scheduler
    /// releases pictures as fast as they arrive.
    next_display: u64,
    /// Display index of the next picture to offer the *scheduler*. Equal
    /// to `next_display` without a lookahead — every picture is offered
    /// as it arrives — and behind it by up to `cfg.lookahead` with one,
    /// which is what holds pictures back for the rate controller to see.
    offered: u64,
    /// Each held picture's lookahead cost by display index, measured once
    /// when it was pushed and dropped when it is coded. Empty without a
    /// lookahead.
    costs: std::collections::BTreeMap<u64, PicCost>,
    /// The display-size luma of the last picture pushed, which the next
    /// one's inter cost is measured against. `None` without a lookahead.
    last_luma: Option<Vec<S>>,
    /// The quantiser the last kept picture was coded at, which is what
    /// the lookahead's inter-cost floor ([`PicCost::inter_floor`]) is
    /// scaled by: a picture predicted from a reference carries that
    /// reference's quantisation noise as residual however still the
    /// content is. `None` before the first picture.
    last_qp: Option<u8>,
    geom: syn::Geometry,
    /// Reference pictures as the decoder holds them — full `Frame`s with
    /// their motion grids and extended borders, not cropped bytes: the
    /// inter decision predicts through the decoder's own MC, which reads
    /// padded planes, and derives candidates from stored motion.
    refs: Vec<crate::hevc::frame::Frame<S>>,
    /// The rate controller, when the configuration asked for a bitrate.
    /// `None` at a constant quantiser, which is the mode every other one
    /// is measured against.
    rc: Option<RateController>,
    /// `init_qp_minus26 + 26`: the quantiser the PPS declares, **fixed for
    /// the whole stream**.
    ///
    /// It has to be fixed, because there is one PPS and every slice refers
    /// to it; a per-picture quantiser is carried by `slice_qp_delta`
    /// instead, which `write_slice_header` computes as `slice_qp -
    /// pps_qp`. At a constant quantiser this equals that quantiser and
    /// every delta is zero, which is exactly what the streams before rate
    /// control carried — that is why enabling this changed no bytes.
    pps_qp: i32,
    /// Bytes emitted so far, to hold the controller's ledger to.
    emitted: u64,
    /// How many pictures had to be coded more than once to fit the
    /// declared buffer, and how many extra codings that cost.
    ///
    /// Reported rather than hidden: re-coding is the encoder doing a
    /// picture's work twice, and a caller choosing a buffer size deserves
    /// to see what that choice costs before it shows up as a slow encode
    /// nobody can account for.
    recoded: u64,
    /// The coded picture buffer this stream declares, when it declares one.
    /// The *declared* values, snapped to what the syntax carries — the
    /// controller is handed these rather than the caller's request, so what
    /// it aims at and what the stream promises are one number.
    cpb: Option<Cpb>,
    /// The PPS switches this stream declares beyond the quantiser and the
    /// two flags `write_pps` takes by position: per-CTB quantiser deltas
    /// when adaptive quantisation is on. Fixed for the stream, like the
    /// PPS itself.
    pps_opts: PpsOptions,
    /// What every kept picture coded, by picture kind. See [`Census`].
    census: Census,
}

/// One coded picture, before anything about it has been kept.
///
/// A picture that will not fit the declared buffer is coded again at a
/// higher quantiser, and the attempt that lost must leave no trace: it
/// must not appear in the reconstructions the SELF check reads, and above
/// all it must not become the reference the *next* picture predicts from,
/// which would make every later picture depend on bytes no decoder will
/// ever see.
///
/// Keeping the commit out of the coding functions is what makes that
/// impossible rather than merely avoided. They hand back what they made;
/// only [`H265Encoder::code_picture`] decides to keep it.
///
/// That the coding functions keep *nothing* is the claim the byte-identity
/// of every non-buffer stream rests on, so it is checked rather than
/// assumed: `code_attempt` and `code_inter_picture` contain **no writes to
/// `self` at all** — no counter, no scratch buffer reused across pictures,
/// no cached context. Every write lives in [`H265Encoder::commit`] and
/// `retain_reference`, which run only for an attempt that is kept. A
/// single stray write in either coding path would leave a trace, make the
/// second attempt something other than a clean re-run, and surface as a
/// byte moving in a cell that had no reason to move.
struct Attempt<S: Sample> {
    access: Access,
    /// The reconstruction, cropped to display size and packed to bytes
    /// (little-endian `u16` above 8 bits).
    rec: Vec<u8>,
    /// The reconstruction as a reference frame, uncropped.
    frame: crate::hevc::frame::Frame<S>,
    /// Whether this picture empties the reference buffer first, which is
    /// what makes an IDR a random access point.
    clears_refs: bool,
    /// What this picture's CUs were coded as, added to the encoder's
    /// census only if the attempt is kept.
    census: KindCensus,
    /// A B picture coded under a fitted `pred_weight_table` that weights
    /// something: the attempt `code_attempt` prices against the same
    /// picture under a table of defaults. False for every other attempt.
    b_fitted: bool,
}

/// The POC LSB width the SPS declares. Fixed and generous, as on the H.264
/// side: a wrap that never happens is a class of bug that never happens.
const LOG2_MAX_POC_LSB: u32 = 16;

impl H265Encoder {
    /// Fails rather than starting if the configuration cannot produce a legal
    /// stream — an encoder that fails late has usually already emitted a
    /// header describing something it then cannot deliver.
    ///
    /// The bit depth picks the sample width once, here: 8 bits codes in
    /// `u8`, anything deeper in `u16`, exactly as [`crate::hevc::HevcDecoder`]
    /// chooses on reading the SPS. `Config::validate` bounds the depth to
    /// the 8..=14 both decoders and the transform arithmetic admit.
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        let inner = if cfg.bit_depth > 8 { Inner::Wide(Core::new(cfg)?) } else { Inner::Eight(Core::new(cfg)?) };
        Ok(H265Encoder { inner })
    }

    /// How many bytes one source picture must be: one per sample at 8
    /// bits, two (little-endian) deeper.
    pub fn frame_bytes(&self) -> usize {
        with_core!(&self.inner, e => e.frame_bytes)
    }

    /// What the rate controller achieved against what it was asked for,
    /// in bits per second — `None` at a constant quantiser, where there
    /// was no target to miss.
    ///
    /// Reported by the encoder rather than recomputed by whoever is
    /// watching: the encoder knows the frame count, the frame rate and the
    /// exact bytes emitted, and a second implementation of that division
    /// somewhere else is a second thing that can be wrong. The gate reads
    /// this line rather than doing the arithmetic itself.
    pub fn rate_report(&self) -> Option<(f64, f64)> {
        with_core!(&self.inner, e => e.rate_report())
    }

    /// How many extra codings the declared buffer cost: pictures that came
    /// out too large for it and had to be coded again at a higher
    /// quantiser. Zero when no buffer was declared, because then nothing
    /// can fail to fit.
    pub fn recodes(&self) -> u64 {
        with_core!(&self.inner, e => e.recoded)
    }

    /// The rate controller's model check: the mean distance, in quantiser
    /// steps of its law, between what each picture was planned at and
    /// what it cost — see `RateController::plan_error`. `None` at a
    /// constant quantiser, where nothing was planned.
    pub fn plan_error(&self) -> Option<f64> {
        with_core!(&self.inner, e => e.plan_error())
    }

    /// What the rate controller's insensitivity rule did — verdicts,
    /// probes, releases; see `RateController::insensitivity`. `None` at a
    /// constant quantiser.
    pub fn rate_insensitivity(&self) -> Option<Insensitivity> {
        with_core!(&self.inner, e => e.rc.as_ref().map(RateController::insensitivity))
    }

    /// The reconstructions produced so far, in coding order, packed as
    /// the source pictures were handed in (see [`H265Encoder`]).
    pub fn reconstructions(&self) -> &[Vec<u8>] {
        with_core!(&self.inner, e => &e.recon)
    }

    /// What the pictures coded so far were made of — see [`Census`]. A
    /// configuration row turns a code path on; this is what says whether
    /// the clip took it.
    pub fn census(&self) -> &Census {
        with_core!(&self.inner, e => &e.census)
    }

    /// Offer the next picture in display order.
    pub fn push(&mut self, picture: &[u8]) -> Result<Vec<Access>> {
        with_core!(&mut self.inner, e => e.push(picture))
    }

    /// Code everything still held back.
    pub fn flush(&mut self) -> Result<Vec<Access>> {
        with_core!(&mut self.inner, e => e.flush())
    }

    /// Make the next picture pushed an IDR, restarting the GOP there. See
    /// [`Scheduler::force_idr`] for who needs this and what it does to any
    /// B pictures held back at the time.
    pub fn force_idr(&mut self) {
        with_core!(&mut self.inner, e => e.sched.force_idr())
    }
}

use super::{pack_row, unpack_samples};

/// Replicate a `sw` by `sh` plane out to `tw` by `th` — sources at coded
/// size, edge-extended: the coded picture is a whole number of CTUs, the
/// display size usually is not, and the conformance window hides the
/// difference.
fn pad_plane<S: Sample>(src: &[S], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<S> {
    let mut out = vec![S::default(); tw * th];
    for y in 0..th {
        let sy = y.min(sh - 1);
        for x in 0..tw {
            out[y * tw + x] = src[sy * sw + x.min(sw - 1)];
        }
    }
    out
}

impl<S: Sample> Core<S> {
    fn new(cfg: Config) -> Result<Self> {
        if cfg.interlace.is_some() {
            // Not "in progress": H.265 has no interlaced coding tools. What
            // it has is signalling — field_seq_flag and a pic_struct SEI
            // over pictures that are each one field — which this encoder
            // does not write.
            return Err(Error::unsupported(
                "H.265 encode: interlaced coding (H.265 has no field or MBAFF tools; field_seq_flag / pic_struct signalling is not written)",
            ));
        }
        if cfg.sao && matches!(cfg.rate, RateControl::Lossless) {
            // Every CU of a lossless picture is transquant-bypass, every
            // bypass sample is exempt from both loop filters, and SAO
            // would therefore be a declared no-op: two flags in every
            // slice header and parameters in every CTB, buying nothing.
            // Refusing names that rather than shipping it.
            return Err(Error::unsupported(
                "H.265 encode: sample adaptive offset on a lossless picture (every sample is filter-exempt)",
            ));
        }
        if cfg.aq_strength > 0.0 && matches!(cfg.rate, RateControl::Lossless) {
            // Every CU of a lossless picture is transquant-bypass and
            // scaling never runs, so a per-CTB quantiser would be a
            // `cu_qp_delta` in every coded CU steering nothing.
            return Err(Error::unsupported(
                "H.265 encode: adaptive quantisation on a lossless picture (no quantiser to adapt)",
            ));
        }
        let g = syn::Geometry::new(&cfg);
        // Adaptive quantisation is the one thing that varies the
        // quantiser below the slice, so it is what turns the PPS switch
        // on. The group is the CTB where every CTB is one coding unit — the
        // granularity that decision has — and in an all-intra stream; it
        // is half the CTB where the coding quadtree codes pictures that
        // others predict from. Measured 2026-09-14 on the 32x32-CTB
        // geometry, per-plane BD-rate against half-CTB groups, ten clips at
        // `--aq 1.0`: CTB groups gain 6.3% YUV all-intra at QP 22-40 (5.5%
        // at 34-43) with no clip worse, but in IP and IPB streams they
        // regress the fading clip (+0.9% and +2.1% YUV at QP 22-40, +3.7%
        // IP at 34-43). The anchor I picture's groups cause it: CTB-level
        // groups on the I picture alone, half-CTB ones on its P and B
        // pictures, regressed that clip too. So the choice is per stream.
        // Weighted prediction sets the P-slice flag, and the B-slice flag
        // when the GOP codes B pictures: every P and B slice then carries
        // a table. Without B pictures the second flag would be a PPS bit
        // no slice reads, and one every such stream coded before B slices
        // were weighted does not have.
        let pps_opts = PpsOptions {
            cu_qp_delta_depth: (cfg.aq_strength > 0.0).then_some(u32::from(tree_depth(&cfg, &g) > 0 && cfg.gop != 0)),
            weighted_pred: cfg.weighted_pred,
            weighted_bipred: cfg.weighted_pred && cfg.bframes > 0,
        };
        let (sw, sh) = cfg.chroma.subsampling();
        let luma = cfg.width as usize * cfg.height as usize;
        let chroma = if cfg.chroma == crate::ChromaFormat::Monochrome {
            0
        } else {
            2 * (cfg.width as usize).div_ceil(sw as usize)
                * (cfg.height as usize).div_ceil(sh as usize)
        };
        // The PPS quantiser: the constant one where there is one, and the
        // middle of the road where the controller will vary it per picture.
        // Its only effect on the stream is the size of each
        // `slice_qp_delta`, since both sides derive everything else from
        // the slice quantiser.
        let pps_qp: i32 = match cfg.rate {
            RateControl::ConstantQp(q) => i32::from(q.min(51)),
            RateControl::Lossless => 26,
            RateControl::Bitrate { .. } => 26,
        };
        let cpb = match (cfg.cpb_ms, cfg.rate) {
            (0, _) => None,
            (ms, RateControl::Bitrate { bps }) => match Cpb::new(bps, ms) {
                Some(c) => Some(c),
                None => {
                    return Err(Error::unsupported(
                        "H.265 encode: a coded picture buffer this size needs a bit-rate or buffer scale (encoder writes both as 0)",
                    ));
                }
            },
            _ => {
                return Err(Error::unsupported(
                    "H.265 encode: a coded picture buffer without a bitrate target (a buffer constrains a rate; a fixed quantiser has none)",
                ));
            }
        };
        let rc = match cfg.rate {
            // The controller aims at the *declared* rate where a buffer
            // was declared, so the two cannot disagree by the rounding.
            RateControl::Bitrate { bps } => {
                let bps = cpb.map_or(bps, |c| c.bit_rate as u32);
                Some(RateController::with_cpb(bps, cfg.fps, cfg.width, cfg.height, cfg.gop, cfg.bframes, cpb.map(|c| c.size)))
            }
            _ => None,
        };
        Ok(Self {
            geom: g,
            cpb,
            pps_opts,
            census: Census::default(),
            sched: Scheduler::new(cfg.gop, cfg.bframes),
            rc,
            pps_qp,
            emitted: 0,
            recoded: 0,
            cfg,
            held: std::collections::BTreeMap::new(),
            recon: Vec::new(),
            frame_bytes: (luma + chroma) * S::BYTES,
            next_display: 0,
            offered: 0,
            costs: std::collections::BTreeMap::new(),
            last_luma: None,
            last_qp: None,
            refs: Vec::new(),
        })
    }

    /// See [`H265Encoder::plan_error`].
    fn plan_error(&self) -> Option<f64> {
        self.rc.as_ref()?.plan_error()
    }

    /// See [`H265Encoder::rate_report`].
    fn rate_report(&self) -> Option<(f64, f64)> {
        let rc = self.rc.as_ref()?;
        let target = match self.cfg.rate {
            RateControl::Bitrate { bps } => bps as f64,
            _ => return None,
        };
        Some((rc.achieved_bps(self.cfg.fps), target))
    }

    /// See [`H265Encoder::push`].
    fn push(&mut self, picture: &[u8]) -> Result<Vec<Access>> {
        if picture.len() != self.frame_bytes {
            return Err(Error::bitstream(format!(
                "H.265 encode: picture is {} bytes, expected {}",
                picture.len(),
                self.frame_bytes
            )));
        }
        let samples = unpack_samples::<S>(picture, self.cfg.bit_depth, "H.265")?;
        let display = self.next_display;
        self.next_display += 1;
        if self.cfg.lookahead > 0 {
            // Measured once, here, against the picture pushed before it —
            // which may already have been coded and left `held`, so its
            // luma is kept aside for exactly this.
            let (dw, dh) = (self.cfg.width as usize, self.cfg.height as usize);
            let cost = PicCost::measure(&samples[..dw * dh], dw, dh, self.last_luma.as_deref(), self.cfg.bit_depth);
            self.costs.insert(display, cost);
            self.last_luma = Some(samples[..dw * dh].to_vec());
        }
        self.held.insert(display, samples);
        // Offer the scheduler everything but the last `lookahead` pictures.
        // Without a lookahead that is the picture just pushed, once, which
        // is the path every stream took before one existed.
        let mut out = Vec::new();
        while self.next_display - self.offered > u64::from(self.cfg.lookahead) {
            self.offered += 1;
            let ready = self.sched.push();
            out.extend(self.code(ready)?);
        }
        Ok(out)
    }

    /// See [`H265Encoder::flush`].
    fn flush(&mut self) -> Result<Vec<Access>> {
        let mut out = Vec::new();
        // Whatever the lookahead was still holding back goes to the
        // scheduler first, in order, so it types them as it would have.
        while self.offered < self.next_display {
            self.offered += 1;
            let ready = self.sched.push();
            out.extend(self.code(ready)?);
        }
        let ready = self.sched.flush();
        out.extend(self.code(ready)?);
        Ok(out)
    }

    fn code(&mut self, ready: Vec<Coded>) -> Result<Vec<Access>> {
        let mut out = Vec::with_capacity(ready.len());
        for (i, &c) in ready.iter().enumerate() {
            let src = self.held.remove(&c.display).ok_or_else(|| {
                Error::bitstream("H.265 encode: scheduler released an absent picture")
            })?;
            // The pictures released alongside this one and not yet coded
            // are part of what a lookahead can see.
            let access = self.code_picture(c, &src, &ready[i + 1..])?;
            self.costs.remove(&c.display);
            // The ledger closes here, at the one place every picture of
            // every kind passes through — an accounting call inside each
            // coding path could be forgotten in one of them, and the
            // symptom would be a controller that quietly believes it has
            // spent less than it has.
            //
            // What is counted is the whole access unit: start codes, NAL
            // headers, parameter sets and slice payload, because that is
            // what the target is measured against. Counting the payload
            // alone would run about a percent low on these clips and
            // rather more on small pictures, and nothing else here would
            // notice.
            self.emitted += access.data.len() as u64 * 8;
            let emitted = self.emitted;
            if let Some(rc) = self.rc.as_mut() {
                rc.account(access.data.len());
                debug_assert_eq!(
                    rc.bits_spent, emitted,
                    "rate-control ledger drifted: the controller has {} bits, the encoder emitted {emitted}",
                    rc.bits_spent
                );
            }
            out.push(access);
        }
        Ok(out)
    }

    /// Code one picture, re-coding it at a higher quantiser if it will not
    /// fit the buffer this stream declares.
    ///
    /// The quantiser is chosen **once**. What the loop does is escalate
    /// from it, using the same law the controller steers by, and the
    /// controller is told afterwards which quantiser the picture was
    /// actually coded at — so the complexity model learns from the picture
    /// that shipped rather than from one that was thrown away.
    ///
    /// Without a declared buffer there is nothing to fit and the loop runs
    /// exactly once, which is why every stream that does not ask for a
    /// buffer is byte-identical to what it was before this existed.
    fn code_picture(&mut self, c: Coded, src: &[S], upcoming: &[Coded]) -> Result<Access> {
        let mut qp = self.pick_picture_qp(&c, upcoming)?;
        let bypass = matches!(self.cfg.rate, RateControl::Lossless);
        for attempt in 0..super::rc::MAX_ATTEMPTS {
            let a = self.code_attempt(c, src, qp, bypass)?;
            let bits = a.access.data.len() as u64 * 8;
            // What the buffer can hand over at this picture's removal time.
            // `None` means no buffer was declared and nothing can fail.
            let affordable = self.rc.as_ref().and_then(|rc| rc.affordable_bits());
            let Some(afford) = affordable else {
                return Ok(self.commit(&c, a, qp));
            };
            if bits <= afford {
                if let Some(rc) = self.rc.as_mut() {
                    rc.note_recode(qp);
                }
                return Ok(self.commit(&c, a, qp));
            }
            if attempt + 1 == super::rc::MAX_ATTEMPTS || qp >= 51 {
                // The declared buffer is smaller than this content can be
                // coded into. That is a configuration error, and emitting
                // a stream that declares a buffer it violates is the one
                // outcome this whole feature exists to prevent — so it
                // refuses by name rather than shipping it.
                return Err(Error::unsupported(format!(
                    "H.265 encode: picture {} needs {bits} bits and the declared buffer affords {afford}                      even at quantiser {qp} (the coded picture buffer is too small for this content)",
                    c.poc
                )));
            }
            self.recoded += 1;
            qp = RateController::escalate(qp, bits, afford);
        }
        unreachable!("the loop returns or errors on its last attempt")
    }

    /// Keep what an attempt made: the reconstruction the SELF check reads,
    /// and the reference the next picture predicts from.
    fn commit(&mut self, c: &Coded, a: Attempt<S>, qp: u8) -> Access {
        self.recon.push(a.rec);
        self.last_qp = Some(qp);
        self.census.by_kind[Census::slot(c.kind)].add(&a.census);
        if a.clears_refs {
            self.refs.clear();
        }
        self.retain_reference(c, a.frame);
        a.access
    }

    /// The quantiser this picture starts at, before any buffer escalation.
    /// `upcoming` is what was released with it and is still to code.
    fn pick_picture_qp(&mut self, c: &Coded, upcoming: &[Coded]) -> Result<u8> {
        Ok(match self.cfg.rate {
            RateControl::Lossless => 26,
            RateControl::ConstantQp(q) => q.min(51),
            RateControl::Bitrate { .. } => {
                let kind = pic_kind(c.kind);
                if self.cfg.lookahead == 0 {
                    self.rc.as_mut().expect("a bitrate configuration builds a controller").pick_qp(kind)
                } else {
                    let (cost, window) = self.lookahead_window(c, upcoming);
                    self.rc.as_mut().expect("a bitrate configuration builds a controller").pick_qp_ahead(kind, cost, &window)
                }
            }
        })
    }

    /// What the rate controller may plan this picture against: its own
    /// cost and the window — this picture, the ones released with it and
    /// still to code, and everything the scheduler has not released yet,
    /// each with the kind it will be coded as ([`Scheduler::preview`])
    /// and the cost that kind pays: an intra picture its intra cost, an
    /// inter picture [`PicCost::inter_cost`] at the quantiser the last
    /// picture was coded at.
    fn lookahead_window(&self, c: &Coded, upcoming: &[Coded]) -> (f64, Vec<(PicKind, f64)>) {
        let qp_ref = self.last_qp.map_or(26, i32::from);
        let cost_of = |kind: Kind, display: u64| -> f64 {
            let pc = self.costs.get(&display).copied().unwrap_or_default();
            let cost = if kind.is_intra() { pc.intra } else { pc.inter_cost(qp_ref, self.cfg.weighted_pred) };
            (cost as f64).max(1.0)
        };
        let mine = cost_of(c.kind, c.display);
        let mut window = vec![(pic_kind(c.kind), mine)];
        window.extend(upcoming.iter().map(|u| (pic_kind(u.kind), cost_of(u.kind, u.display))));
        let ahead = self.next_display - self.offered;
        window.extend(self.sched.preview(ahead).iter().map(|p| (pic_kind(p.kind), cost_of(p.kind, p.display))));
        (mine, window)
    }

    /// Sum of squared differences between the display area of `f` and the
    /// source picture `src` (planar, display size), over every component.
    fn display_ssd(&self, f: &crate::hevc::frame::Frame<S>, src: &[S]) -> u64 {
        let (dw, dh) = (self.cfg.width as usize, self.cfg.height as usize);
        let (sw, sh) = match self.cfg.chroma {
            crate::ChromaFormat::Yuv420 => (2usize, 2usize),
            crate::ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cdw, cdh) = (dw.div_ceil(sw), dh.div_ceil(sh));
        let plane = |p: &crate::hevc::frame::Plane16<S>, s: &[S], w: usize, h: usize| -> u64 {
            let o = p.origin();
            (0..h)
                .map(|y| {
                    let row = &p.data[o + y * p.stride..];
                    (0..w).map(|x| u64::from(row[x].to_i32().abs_diff(s[y * w + x].to_i32())).pow(2)).sum::<u64>()
                })
                .sum()
        };
        let mut ssd = plane(&f.y, &src[..dw * dh], dw, dh);
        if self.cfg.chroma != crate::ChromaFormat::Monochrome {
            ssd += plane(&f.cb, &src[dw * dh..dw * dh + cdw * cdh], cdw, cdh);
            ssd += plane(&f.cr, &src[dw * dh + cdw * cdh..], cdw, cdh);
        }
        ssd
    }

    /// Code one picture at a given quantiser, keeping nothing.
    fn code_attempt(&mut self, c: Coded, src: &[S], qp: u8, bypass: bool) -> Result<Attempt<S>> {
        let g = self.geom;
        let pps_qp = self.pps_qp;
        // Lossless is transquant bypass: the PPS enables it, every CU says
        // it, and the residuals travel raw. The QP still appears in the
        // headers because the syntax demands one, and it still matters to
        // exactly one thing — the CABAC context initialisation, which both
        // sides derive from the slice QP — while scaling never runs: the
        // decoder's residual path skips dequantisation and the transform for
        // a bypassed CU, so any legal value would decode identically. 26 is
        // the middle of the road and needs no explanation in a debugger.
        //
        // Both arrive as arguments: the quantiser is chosen once by the
        // caller and escalated by it, so an attempt cannot quietly pick a
        // different one than the buffer arithmetic is reasoning about.
        if c.kind != Kind::Idr {
            let fitted = self.code_inter_picture(c, src, qp, bypass, true)?;
            if !fitted.b_fitted {
                return Ok(fitted);
            }
            // A B picture's fitted table is priced, not trusted. The fit is
            // judged at zero motion one list at a time, while a bi unit
            // predicts from the average of both weighted anchors — which on
            // a fade the default average is often already as near — so the
            // table can cost its bits and steer the search for little. The
            // picture is coded again under a table of defaults, and the
            // cheaper of the two is kept: the SSD of its reconstruction plus
            // the bits of its access unit, at the Lagrangian every
            // SSD-against-bits choice in this encoder uses.
            let mut plain = self.code_inter_picture(c, src, qp, bypass, false)?;
            let lam = ssd_lambda(i32::from(qp), self.cfg.bit_depth);
            let cost = |a: &Attempt<S>| self.display_ssd(&a.frame, src) as f64 + lam * (a.access.data.len() * 8) as f64;
            if cost(&plain) < cost(&fitted) {
                plain.census.wp_rd_default += 1;
                return Ok(plain);
            }
            return Ok(fitted);
        }

        // Sources at coded size, edge-replicated: the coded picture is a
        // whole number of CTUs, the display size usually is not, and the
        // conformance window hides the difference.
        let (dw, dh) = (self.cfg.width as usize, self.cfg.height as usize);
        let (cw, ch) = (g.coded_width as usize, g.coded_height as usize);
        // Per-format chroma geometry: SubWidthC/SubHeightC divide the luma
        // dimensions, and monochrome has no chroma planes at all — the
        // decision module's chroma slices are then empty and never indexed.
        let chroma = self.cfg.chroma;
        let cat = match chroma {
            crate::ChromaFormat::Monochrome => 0u32,
            crate::ChromaFormat::Yuv420 => 1,
            crate::ChromaFormat::Yuv422 => 2,
            crate::ChromaFormat::Yuv444 => 3,
        };
        let (sw, sh) = match chroma {
            crate::ChromaFormat::Yuv420 => (2usize, 2usize),
            crate::ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cdw, cdh) = (dw.div_ceil(sw), dh.div_ceil(sh));
        let (ccw, cch) = (cw / sw, ch / sh);
        let py = pad_plane(&src[..dw * dh], dw, dh, cw, ch);
        let (pcb, pcr) = if cat != 0 {
            (
                pad_plane(&src[dw * dh..dw * dh + cdw * cdh], cdw, cdh, ccw, cch),
                pad_plane(&src[dw * dh + cdw * cdh..], cdw, cdh, ccw, cch),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // The decision machinery, on the decoder's own kernels — the
        // table the decoder builds for this sample width, SIMD tiers
        // included.
        let cpu = Cpu::detect_honouring_env();
        let dsp = HevcDsp::<S>::new(cpu);
        let enc = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<S>::new(cpu);
        let bit_depth = self.cfg.bit_depth;
        let ictx = IntraCtx {
            dsp: &dsp,
            enc: &enc,
            dist: &dist,
            qp: qp as i32,
            bit_depth,
            strong_smoothing: false,
            bypass,
            // An all-intra stream has no inter pictures at all, so no
            // picture here is ever predicted from and each may be
            // quantised for its own optimum. With a GOP, this IDR is the
            // anchor every P picture reads, and trimming it costs them
            // more than it saves — see `rdoq_trim`.
            free_to_trim: self.cfg.gop == 0,
        };
        let mut pic = IntraPicture::<S>::new_with_chroma(cw, ch, g.log2_ctb, bit_depth, chroma);
        // The transform-split search is on: a CU may carry four quarter-size
        // TUs where that wins the decision module's cost comparison, and the
        // writer below spells both shapes.
        pic.split_depth = 1;

        // Parameter sets, then the one slice.
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(syn::NAL_VPS, &syn::write_vps(&self.cfg, &g)));
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB, self.cpb.as_ref()),
        ));
        // Every picture is filtered, so the picture-wide PPS flag can
        // declare it unconditionally: intra pictures through
        // `deblock_picture`, P pictures through `deblock_inter_picture`.
        // The flag was `self.cfg.gop == 0` — all-intra streams only —
        // for as long as a P picture had no filter to apply, because
        // declaring one the encoder does not run is exactly the failure
        // that first turned it off: two decoders filter, the encoder
        // does not, SELF fails on every coded edge while CROSS stays
        // green.
        let deblock = true;
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, &syn::write_pps_opts(self.pps_qp, bypass, deblock, &self.pps_opts)));
        // A buffering period may begin at any IRAP, and this encoder makes
        // every one of them one: the message carries the initial removal
        // delay, which is the single number the schedule cannot derive
        // from the frame rate.
        if let Some(cpb) = self.cpb.as_ref() {
            out.extend_from_slice(&syn::annexb(syn::NAL_PREFIX_SEI, &syn::write_buffering_period_sei(cpb)));
        }
        // HDR10 static metadata, with every IRAP so that a stream joined at
        // any of them carries it (as x265 does with repeated headers).
        if let Some(m) = self.cfg.mastering_display.as_ref() {
            out.extend_from_slice(&syn::annexb(syn::NAL_PREFIX_SEI, &syn::write_mastering_display_sei(m)));
        }
        if let Some(c) = self.cfg.content_light.as_ref() {
            out.extend_from_slice(&syn::annexb(syn::NAL_PREFIX_SEI, &syn::write_content_light_level_sei(c)));
        }

        let mut w = BitWriter::with_capacity(cw * ch / 2);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp: i32::from(qp),
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                // An IDR references nothing, so its reference picture set
                // is empty.
                ref_deltas: Vec::new(),
                // Present exactly when the SPS enabled SAO, and the chroma
                // flag only outside monochrome - the reader's two gates.
                sao: sao_flags(self.cfg.sao, cat),
                // An I slice carries no prediction weights whatever the
                // PPS says.
                pred_weights: None,
            },
            pps_qp,
            syn::NAL_IDR_N_LP,
            deblock,
            &mut w,
        );
        // byte_alignment(): one, then zeros to the byte.
        w.flag(true);
        w.align_zero();

        let (wc, hc) = (g.ctbs_wide as usize, g.ctbs_high as usize);
        let mut cx = Contexts::new(0, qp as i32);
        // Decide first, serialise second. The two passes exist because of
        // SAO: its parameters are not known until the whole picture has
        // reconstructed *and* deblocked, yet the reader takes them at the
        // START of each CTU, ahead of the coding quadtree. Splitting the
        // walk costs nothing — no decision here ever depended on the
        // bitstream — and it is what lets one pass write both.
        //
        // The decisions also outlive the loop for the deblocker, which
        // derives its boundary strengths from them exactly as a decoder
        // derives them from what it just parsed.
        // Per-unit quantisers, when adaptive quantisation asks for them:
        // each unit codes at the picture quantiser plus the offset its
        // quantisation group wants, and the chain settles what a decoder
        // will actually hold for it — the offset only where a unit of the
        // group carries a cbf to hang the delta on.
        //
        // Without a tree (`max_cu_depth` 0) every CTB is one unit and one
        // group, coded by `code_ctu` exactly as before the quadtree
        // existed; with one, `code_ctu_tree` decides the split at every
        // node.
        let max_depth = self.tree_depth();
        let log2_qg = self.log2_qg();
        let offsets = self.aq_offsets(&py, cw, ch, log2_qg);
        let want = |x: usize, y: usize, log2: u32| cu_want(qp, offsets.as_deref(), cw, log2_qg, x, y, log2);
        let src = Srcs { y: &py, y_stride: cw, cb: &pcb, cr: &pcr, c_stride: ccw };
        let mut cus: Vec<TreeCu<CuDecision>> = Vec::with_capacity(wc * hc);
        let mut ctu_start = Vec::with_capacity(wc * hc + 1);
        for cy in 0..hc {
            for cxu in 0..wc {
                ctu_start.push(cus.len());
                if max_depth > 0 {
                    cus.extend(pic.code_ctu_tree(&ictx, &want, max_depth, cxu, cy, &src));
                    continue;
                }
                let (x0, y0) = (cxu << g.log2_ctb, cy << g.log2_ctb);
                let cctx = IntraCtx { qp: want(x0, y0, g.log2_ctb), ..ictx };
                let d = pic.code_ctu(&cctx, cxu, cy, &py, cw, &pcb, &pcr, ccw);
                cus.push(TreeCu { x0, y0, log2: g.log2_ctb, depth: 0, bits: 0.0, d });
            }
        }
        ctu_start.push(cus.len());
        if offsets.is_some() {
            let mut chain = QgChain::new(i32::from(qp), bit_depth, &g, log2_qg);
            settle_tree(&mut chain, &mut cus, &ctu_start, &g, &want);
        }
        // After the whole picture reconstructs — intra prediction reads
        // unfiltered neighbours — and before the crop, because the
        // filtered planes are what a decoder emits and therefore what SELF
        // compares against. Bypass CUs are exempt sample for sample, so
        // lossless stays exact with the filter on.
        let mut info = deblock_picture(&ictx, &mut pic, &cus);
        // Then SAO, over the deblocked samples, which is the order 8.7
        // fixes and the order `decoder.rs` applies them in.
        let plan = self.cfg.sao.then(|| {
            let (sps, pps) = parsed_sets(&self.cfg, &g, i32::from(qp), bypass, deblock, self.cpb.as_ref(), &self.pps_opts);
            sao_picture(&ictx, &mut pic.recon, &mut info, &sps, &pps, &py, cw, &pcb, &pcr, ccw)
        });
        let mut census = KindCensus::of_intra(&cus, i32::from(qp));
        {
            let mut e = CabacEncoder::new(&mut w);
            // The writer's own copy of the quantiser chain: it must spell
            // the delta against the same prediction the decision pass
            // settled with, and a decoder derives that prediction from
            // the stream alone.
            let mut chain = offsets.is_some().then(|| QgChain::new(i32::from(qp), bit_depth, &g, log2_qg));
            let mut tc = TreeCtx::new(&g);
            let start = e.position();
            for cy in 0..hc {
                for cxu in 0..wc {
                    let addr = cy * wc + cxu;
                    write_sao_for(&mut e, &mut cx, plan.as_ref(), addr, cxu, cy, bit_depth, cat);
                    let ctu = &cus[ctu_start[addr]..ctu_start[addr + 1]];
                    census.qp_delta += write_tree(&mut e, &mut cx, ctu, (cxu << g.log2_ctb, cy << g.log2_ctb), g.log2_ctb, &mut tc, chain.as_mut(), &mut |e, cx, cu, _, _, delta| {
                        write_cu_intra_i(e, cx, &cu.d, bypass, cat, delta)
                    });
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                }
            }
            if max_depth > 0 {
                census.coded_bits += e.position() - start;
            }
        }
        w.align_zero();
        out.extend_from_slice(&syn::annexb(syn::NAL_IDR_N_LP, &w.into_nal()));

        // The reconstruction, cropped to display size and packed to bytes
        // — what a decoder emits, and therefore what SELF compares
        // against.
        let mut rec = Vec::with_capacity(self.frame_bytes);
        let crop = |p: &crate::hevc::frame::Plane16<S>, tw: usize, th: usize, out: &mut Vec<u8>| {
            let o = p.origin();
            for y in 0..th {
                let row = o + y * p.stride;
                pack_row(&p.data[row..row + tw], out);
            }
        };
        crop(&pic.recon.y, dw, dh, &mut rec);
        if cat != 0 {
            crop(&pic.recon.cb, cdw, cdh, &mut rec);
            crop(&pic.recon.cr, cdw, cdh, &mut rec);
        }
        // Handed back rather than kept: an attempt that does not fit the
        // buffer is coded again, and it must leave nothing behind. An IDR
        // empties the reference buffer when it is *committed* — everything
        // before it is discarded, which is what makes it a random access
        // point — and not before.
        Ok(Attempt {
            access: Access {
                data: out,
                keyframe: true,
                poc: c.poc,
                encode_index: c.encode,
                display: c.display,
            },
            rec,
            frame: pic.recon,
            clears_refs: true,
            census,
            b_fitted: false,
        })
    }

    /// Keep a coded picture as a reference the way a decoder keeps it —
    /// borders extended for motion compensation, picture order count set,
    /// motion grid intact — and drop what can no longer be referenced.
    ///
    /// A picture the scheduler marked non-reference is not kept at all: in
    /// a non-pyramid group of pictures nothing refers to a B picture, and
    /// keeping one would let a later search predict from a picture the
    /// bitstream never told a decoder to hold.
    ///
    /// Of the rest, two are enough for the geometry this encoder codes:
    /// list 0 takes the nearest past picture and list 1 the nearest
    /// future one, and the anchors either side of a group of B pictures
    /// are exactly those two. Keeping the two most recently coded is not
    /// the same as keeping the two nearest in display order, which is why
    /// the selection above searches by picture order count rather than
    /// taking the last.
    fn retain_reference(&mut self, c: &Coded, mut frame: crate::hevc::frame::Frame<S>) {
        if !c.reference {
            return;
        }
        frame.poc = c.poc as i32;
        frame.extend_rows(0, frame.height);
        self.refs.push(frame);
        // Two for the B geometry above, or as many past pictures as
        // `max_refs` lets a P picture choose between — whichever is more.
        let keep = (self.cfg.max_refs as usize).max(2);
        while self.refs.len() > keep {
            self.refs.remove(0);
        }
    }

    /// Code one inter picture — P or B: every CTU one coding unit, or a
    /// coding quadtree of them under `max_cu_depth`, decided against the
    /// references and serialised through the coding-tree
    /// writers that live beside their readers.
    ///
    /// The two halves meet here and nowhere else. The decision module
    /// chooses skip / merge / AMVP by calling the decoder's own candidate
    /// derivation, so what this writes is what that decoder will rebuild;
    /// the writers spell each shape in the reader's element order. The
    /// shapes and their inference traps are documented on `InterCuKind` -
    /// most sharply that a non-skip 2Nx2N merge CU never codes
    /// `rqt_root_cbf` (the reader infers it true), which is why a merge
    /// with nothing left to code must be spelled as a skip instead.
    ///
    /// `fit_b` false codes a B picture under a table of defaults whatever
    /// its fit says: the alternative `code_attempt` prices a fitted table
    /// against. It changes nothing for a P picture.
    fn code_inter_picture(&mut self, c: Coded, src: &[S], qp: u8, bypass: bool, fit_b: bool) -> Result<Attempt<S>> {
        let g = self.geom;
        let pps_qp = self.pps_qp;
        let (dw, dh) = (self.cfg.width as usize, self.cfg.height as usize);
        let (cw, ch) = (g.coded_width as usize, g.coded_height as usize);
        // Per-format chroma geometry, exactly as the intra path derives it:
        // SubWidthC/SubHeightC divide the luma dimensions, and monochrome
        // has no chroma planes at all - the source carries none and the
        // decision module never indexes the empty slices.
        let chroma = self.cfg.chroma;
        let cat = match chroma {
            crate::ChromaFormat::Monochrome => 0u32,
            crate::ChromaFormat::Yuv420 => 1,
            crate::ChromaFormat::Yuv422 => 2,
            crate::ChromaFormat::Yuv444 => 3,
        };
        let (sw, sh) = match chroma {
            crate::ChromaFormat::Yuv420 => (2usize, 2usize),
            crate::ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cdw, cdh) = (dw.div_ceil(sw), dh.div_ceil(sh));
        let (ccw, cch) = (cw / sw, ch / sh);
        let py = pad_plane(&src[..dw * dh], dw, dh, cw, ch);
        let (pcb, pcr) = if cat != 0 {
            (
                pad_plane(&src[dw * dh..dw * dh + cdw * cdh], cdw, cdh, ccw, cch),
                pad_plane(&src[dw * dh + cdw * cdh..], cdw, cdh, ccw, cch),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // The parameter sets this picture is coded against, parsed back
        // through the decoder's own parsers: the candidate derivation the
        // decision module calls reads decoder structures, and building
        // them from the very bytes the stream carries is what keeps the
        // encoder's idea of the geometry and the decoder's identical.
        let sps_rbsp = syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB, self.cpb.as_ref());
        // The very bytes the IDR access unit carried: `code_picture`
        // writes one PPS for the stream, with the deblocking filter on.
        // The very bytes the IDR access unit carried — which means the
        // *stream's* PPS quantiser, not this picture's.
        //
        // This line used to pass `qp`, under a comment making the same
        // claim. That was true only while every picture shared one
        // quantiser; the moment rate control varied it per picture the
        // comment would have become a lie and this struct would have
        // disagreed with the PPS the stream actually carries. Nothing
        // downstream reads `init_qp_minus26` out of it, so it was never
        // going to break — it was going to sit here being wrong, which is
        // how the last two stale comments started.
        let pps_rbsp = syn::write_pps_opts(self.pps_qp, bypass, true, &self.pps_opts);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps_rbsp))?;
        let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps_rbsp))?;
        pps.resolve_tiles(&sps)?;

        // List 0 is the nearest reference in the past; a B picture's list 1
        // is the nearest in the future. The scheduler codes both anchors
        // before releasing the B pictures between them, so both are here by
        // the time one of those codes — and the retention below keeps them
        // until the last picture that can reference them has been coded.
        let cur = c.poc as i32;
        // RefPicList0: the past references, nearest first, capped by what
        // the configuration asks for (`max_refs`, 1 by default) and what
        // the SPS sized the decoded picture buffer to hold. Nearest first
        // is not cosmetic — it is the order the reader builds the list in
        // from the reference picture set, so index 0 must be the nearest.
        // A B picture takes one past reference: its second list is the
        // future anchor, and `max_refs` names the P choice only.
        let mut l0: Vec<&crate::hevc::frame::Frame<S>> = self.refs.iter().filter(|f| f.poc < cur).collect();
        l0.sort_by_key(|f| -f.poc);
        l0.truncate(if c.kind == Kind::B { 1 } else { (self.cfg.max_refs.max(1) as usize).min(l0.len()) });
        let past = *l0.first().ok_or_else(|| Error::bitstream("H.265 encode: an inter picture with no past reference"))?;
        let future = self.refs.iter().filter(|f| f.poc > cur).min_by_key(|f| f.poc);
        if c.kind == Kind::B && future.is_none() {
            return Err(Error::bitstream(
                "H.265 encode: a B picture with no future reference",
            ));
        }
        let future_poc = future.map(|f| f.poc);

        let cpu = Cpu::detect_honouring_env();
        let dsp = HevcDsp::<S>::new(cpu);
        let enc = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<S>::new(cpu);
        let bit_depth = self.cfg.bit_depth;
        let mctx = IntraCtx {
            dsp: &dsp,
            enc: &enc,
            dist: &dist,
            qp: qp as i32,
            bit_depth,
            strong_smoothing: false,
            bypass,
            // Every inter picture this encoder codes is kept as the next
            // one's reference, so none of them is free to trim.
            free_to_trim: false,
        };
        // The reconstruction takes its sample depth from the parsed SPS —
        // the same field a decoder of this stream sizes its frames by.
        let mut pic = InterPicture::<S>::new(&sps, &pps, c.poc as i32);

        // Weighted prediction, for a P slice under `weighted_pred_flag` and
        // a B slice under `weighted_bipred_flag`: one entry per reference
        // in each list the slice uses, each component fitted to the source
        // against that reference's reconstruction and kept only where it
        // lowers the zero-motion residual (`h265_wp`). The table travels
        // in the slice header; the walk predicts with exactly what a
        // decoder derives from that table, through the reader's own
        // `explicit_weighting`.
        let weighted = match c.kind {
            Kind::P => self.pps_opts.weighted_pred,
            Kind::B => self.pps_opts.weighted_bipred,
            Kind::Idr | Kind::I => false,
        };
        let wp = weighted.then(|| {
            let identity = h265_wp::PlaneFit::identity(0);
            let fit = |rf: &&crate::hevc::frame::Frame<S>| -> [h265_wp::PlaneFit; 3] {
                // A B picture's default-weighted alternative: no fit, so
                // every entry is the default and the walk predicts with
                // default weighting.
                if c.kind == Kind::B && !fit_b {
                    return [identity; 3];
                }
                let luma = h265_wp::fit_plane(&src[..dw * dh], dw, &rf.y, dw, dh, bit_depth);
                let (cb, cr) = if cat != 0 {
                    (
                        h265_wp::fit_plane(&src[dw * dh..dw * dh + cdw * cdh], cdw, &rf.cb, cdw, cdh, bit_depth),
                        h265_wp::fit_plane(&src[dw * dh + cdw * cdh..], cdw, &rf.cr, cdw, cdh, bit_depth),
                    )
                } else {
                    (identity, identity)
                };
                [luma, cb, cr]
            };
            // One entry per reference in each list, each its own fit: an
            // older reference of a fade is further down the ramp and
            // wants a different gain, and a B picture's two anchors sit on
            // either side of it, so each list's gain is its own too.
            let l1: Vec<&crate::hevc::frame::Frame<S>> = if c.kind == Kind::B { future.into_iter().collect() } else { Vec::new() };
            let fits: [Vec<[h265_wp::PlaneFit; 3]>; 2] = [l0.iter().map(&fit).collect(), l1.iter().map(&fit).collect()];
            let entries = |list: &[[h265_wp::PlaneFit; 3]]| list.iter().map(|f| h265_wp::entry_for(*f, bit_depth, bit_depth)).collect();
            (h265_wp::table_for([entries(&fits[0]), entries(&fits[1])]), fits)
        });
        match (&wp, c.kind) {
            (Some((t, _)), Kind::P) => pic.wp = (0..l0.len()).map(|r| explicit_weighting(t, bit_depth, bit_depth, [r as i8, -1])).collect(),
            // A B slice's three predictions — list 0, list 1, both — each
            // weighted as the reader derives it for reference 0 of the
            // lists it uses. A table whose every entry is the default
            // predicts exactly the samples default weighting does, uni and
            // bi alike (`w = 1 << denom` and `o = 0` reduce 8.5.3.3.4.3 to
            // 8.5.3.3.4.2), so it leaves the walk on default weighting and
            // its fused kernels.
            (Some((t, fits)), Kind::B) if fits.iter().flatten().any(|f| f.iter().any(h265_wp::PlaneFit::used)) => {
                pic.set_b_weights(t, bit_depth, bit_depth, past, future.expect("a B picture has a future anchor (checked above)"));
            }
            _ => {}
        }

        let mut w = BitWriter::with_capacity(cw * ch / 4);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp: i32::from(qp),
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                // The inline short term reference picture set: one entry
                // per reference this slice will use — every picture in
                // RefPicList0, and for a B picture the future anchor. The
                // header writer counts these to decide
                // `num_ref_idx_active_override_flag`, so the set and the
                // declared counts cannot disagree.
                ref_deltas: {
                    let mut d: Vec<i32> = l0.iter().map(|f| f.poc - cur).collect();
                    if let (Some(f), Kind::B) = (future_poc, c.kind) {
                        d.push(f - cur);
                    }
                    d
                },
                // As in `code_picture`, and from the same switch.
                sao: sao_flags(self.cfg.sao, cat),
                // The table a P slice must carry under `weighted_pred_flag`
                // and a B slice under `weighted_bipred_flag`, defaults and
                // all.
                pred_weights: wp.as_ref().map(|(t, _)| syn::PredWeights {
                    table: t.clone(),
                    chroma: cat != 0,
                    bit_depth_luma: bit_depth,
                    bit_depth_chroma: bit_depth,
                }),
            },
            pps_qp,
            syn::NAL_TRAIL_R,
            // Filtered, like every other picture, and the header must
            // agree with the PPS this stream carries.
            true,
            &mut w,
        );
        w.flag(true); // byte_alignment()
        w.align_zero();

        let (wc, hc) = (g.ctbs_wide as usize, g.ctbs_high as usize);
        // The initialisation type the decoder derives when cabac_init_flag
        // is absent, which the PPS guarantees: 1 for a P slice, 2 for a B.
        let mut cx = Contexts::new(if c.kind == Kind::B { 2 } else { 1 }, qp as i32);
        // The decisions outlive the loop: the deblocker derives its
        // boundary strengths from them, as a decoder does from what it
        // has just parsed. Decide and reconstruct first; serialise below.
        // See the same split in `code_picture` for why SAO forces it.
        // Per-unit quantisers, exactly as the intra path derives and
        // settles them.
        let max_depth = self.tree_depth();
        let log2_qg = self.log2_qg();
        let offsets = self.aq_offsets(&py, cw, ch, log2_qg);
        let want = |x: usize, y: usize, log2: u32| cu_want(qp, offsets.as_deref(), cw, log2_qg, x, y, log2);
        let src = Srcs { y: &py, y_stride: cw, cb: &pcb, cr: &pcr, c_stride: ccw };
        let refs = match future {
            Some(r1) if c.kind == Kind::B => TreeRefs::B(past, r1),
            _ => TreeRefs::P(&l0),
        };
        let mut cus: Vec<TreeCu<PCuDecision>> = Vec::with_capacity(wc * hc);
        let mut ctu_start = Vec::with_capacity(wc * hc + 1);
        for cy in 0..hc {
            for cxu in 0..wc {
                ctu_start.push(cus.len());
                if max_depth > 0 {
                    cus.extend(pic.code_ctu_tree(&mctx, &want, max_depth, refs, cxu, cy, &src));
                    continue;
                }
                let (x0, y0) = (cxu << g.log2_ctb, cy << g.log2_ctb);
                let cctx = IntraCtx { qp: want(x0, y0, g.log2_ctb), ..mctx };
                let d = match future {
                    Some(r1) if c.kind == Kind::B => {
                        pic.code_ctu_b(&cctx, past, r1, cxu, cy, &py, cw, &pcb, &pcr, ccw)
                    }
                    _ => pic.code_ctu(&cctx, &l0, cxu, cy, &py, cw, &pcb, &pcr, ccw),
                };
                // The decision module answers `UseIntra` when its
                // flatness proxy says inter has lost. The CU is then
                // coded by the intra decision over *this* picture's
                // reconstruction - the same `code_cu_2nx2n_intra` an
                // I slice runs, reading the inter neighbours already
                // reconstructed beside it, which the PPS's
                // `constrained_intra_pred_flag` 0 makes references.
                let coded = if matches!(d.kind, InterCuKind::UseIntra) {
                    PCuDecision::Intra(Box::new(pic.code_ctu_intra(&cctx, cxu, cy, &py, cw, &pcb, &pcr, ccw)))
                } else {
                    PCuDecision::Inter(d)
                };
                cus.push(TreeCu { x0, y0, log2: g.log2_ctb, depth: 0, bits: 0.0, d: coded });
            }
        }
        ctu_start.push(cus.len());
        if offsets.is_some() {
            let mut chain = QgChain::new(i32::from(qp), bit_depth, &g, log2_qg);
            settle_tree(&mut chain, &mut cus, &ctu_start, &g, &want);
        }
        // The weighting's model check, counted rather than asserted: the
        // fit predicted that scaling the reference lowers the residual
        // over the picture at zero motion; here is whether it lowered the
        // luma SATD of the vectors the search actually chose, CU by CU.
        // `wp_lost` above `wp_won` on a picture says the fit was wrong
        // for it — a test holds that on a fade, and the census line
        // reports it on every clip.
        let mut wp_stats = (0u64, 0u64, 0u64);
        if let Some((_, fits)) = &wp {
            if fits.iter().flatten().any(|f| f[0].used()) {
                wp_stats.0 = 1;
                for cu in &cus {
                    let PCuDecision::Inter(d) = &cu.d else { continue };
                    // A CU counts when a list it predicts from carries a
                    // chosen luma fit; a bi CU is scored as the pair.
                    let ref_idx = [d.ref_idx, d.ref_idx_l1];
                    let fitted = |list: usize| ref_idx[list] >= 0 && fits[list][ref_idx[list] as usize][0].used();
                    if !fitted(0) && !fitted(1) {
                        continue;
                    }
                    let (plain, weighted) = match refs {
                        TreeRefs::B(r0, r1) => pic.weighting_gain_b(&mctx, r0, r1, cu.x0, cu.y0, cu.log2, &py, cw, [d.mv, d.mv_l1], ref_idx),
                        TreeRefs::P(_) => {
                            let r = d.ref_idx as usize;
                            pic.weighting_gain(&mctx, l0[r], r, cu.x0, cu.y0, cu.log2, &py, cw, d.mv)
                        }
                    };
                    wp_stats.1 += u64::from(weighted < plain);
                    wp_stats.2 += u64::from(weighted > plain);
                }
            }
        }
        // After the whole picture reconstructs and before the crop, for
        // the same reasons the intra path gives: intra prediction — which
        // a P slice now also performs — reads unfiltered neighbours, and
        // the filtered planes are what a decoder emits and therefore what
        // SELF compares against. This picture becomes the next one's
        // reference filtered, which is what a decoder's DPB holds.
        deblock_inter_picture(&mctx, &mut pic, &cus);
        // Then SAO, over the deblocked samples. `InterPicture` already
        // holds the decoder-grade state both filters read, so unlike the
        // intra path there is nothing to hand across.
        let plan = self.cfg.sao.then(|| {
            let InterPicture { info, recon, .. } = &mut pic;
            sao_picture(&mctx, recon, info, &sps, &pps, &py, cw, &pcb, &pcr, ccw)
        });
        let mut census = KindCensus::of_inter(&cus, i32::from(qp));
        census.wp_on += wp_stats.0;
        census.wp_won += wp_stats.1;
        census.wp_lost += wp_stats.2;
        {
            let mut e = CabacEncoder::new(&mut w);
            let mut chain = offsets.is_some().then(|| QgChain::new(i32::from(qp), bit_depth, &g, log2_qg));
            // cu_skip_flag's context counts *skipped* available
            // neighbours and split_cu_flag's counts deeper ones; the tree
            // context carries both, per 4x4, as the units are written.
            let mut tc = TreeCtx::new(&g);
            let nref = l0.len() as u32;
            let start = e.position();
            for cy in 0..hc {
                for cxu in 0..wc {
                    let addr = cy * wc + cxu;
                    write_sao_for(&mut e, &mut cx, plan.as_ref(), addr, cxu, cy, bit_depth, cat);
                    let ctu = &cus[ctu_start[addr]..ctu_start[addr + 1]];
                    census.qp_delta += write_tree(&mut e, &mut cx, ctu, (cxu << g.log2_ctb, cy << g.log2_ctb), g.log2_ctb, &mut tc, chain.as_mut(), &mut |e, cx, cu, left, above, delta| match &cu.d {
                        PCuDecision::Inter(d) => write_cu_inter(e, cx, d, left, above, cat, bypass, delta, nref, cu.depth),
                        PCuDecision::Intra(d) => write_cu_intra_in_p(e, cx, d, left, above, cat, bypass, delta),
                    });
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                }
            }
            if max_depth > 0 {
                census.coded_bits += e.position() - start;
            }
        }
        w.align_zero();
        let mut out = Vec::new();
        // A picture nothing will reference is a sub-layer non-reference
        // picture, and saying so lets a decoder discard it.
        let nal = if c.reference { syn::NAL_TRAIL_R } else { syn::NAL_TRAIL_N };
        out.extend_from_slice(&syn::annexb(nal, &w.into_nal()));

        let mut rec = Vec::with_capacity(self.frame_bytes);
        let crop = |p: &crate::hevc::frame::Plane16<S>, tw: usize, th: usize, out: &mut Vec<u8>| {
            let o = p.origin();
            for y in 0..th {
                let row = o + y * p.stride;
                pack_row(&p.data[row..row + tw], out);
            }
        };
        crop(&pic.recon.y, dw, dh, &mut rec);
        if cat != 0 {
            crop(&pic.recon.cb, cdw, cdh, &mut rec);
            crop(&pic.recon.cr, cdw, cdh, &mut rec);
        }
        // Handed back rather than kept — see the intra path.
        Ok(Attempt {
            access: Access {
                data: out,
                keyframe: false,
                poc: c.poc,
                encode_index: c.encode,
                display: c.display,
            },
            rec,
            frame: pic.recon,
            clears_refs: false,
            census,
            b_fitted: c.kind == Kind::B && wp.as_ref().is_some_and(|(_, fits)| fits.iter().flatten().any(|f| f.iter().any(h265_wp::PlaneFit::used))),
        })
    }

    /// The quantiser offsets adaptive quantisation wants for a picture
    /// whose padded luma plane is `py` (`cw` by `ch`), one per quantisation
    /// group of `1 << log2_qg` in raster order, or `None` when it is off — in which case nothing below varies the quantiser
    /// and the decisions keep the context's. Reads the configuration
    /// only: an attempt stays free of writes to `self`.
    fn aq_offsets(&self, py: &[S], cw: usize, ch: usize, log2_qg: u32) -> Option<Vec<i32>> {
        (self.cfg.aq_strength > 0.0).then(|| aq::ctb_offsets(py, cw, cw, ch, log2_qg, self.cfg.bit_depth, self.cfg.aq_strength))
    }

    /// How many quadtree levels below the CTB this stream's units may
    /// split — see [`tree_depth`].
    fn tree_depth(&self) -> u32 {
        tree_depth(&self.cfg, &self.geom)
    }

    /// log2 of the quantisation group this stream's PPS declares: the CTB
    /// less `diff_cu_qp_delta_depth` — the CTB itself when no depth is
    /// declared, where no unit carries a delta at all.
    fn log2_qg(&self) -> u32 {
        self.geom.log2_ctb - self.pps_opts.cu_qp_delta_depth.unwrap_or(0)
    }
}

/// The rate controller's kind for a scheduled picture.
fn pic_kind(kind: Kind) -> PicKind {
    match kind {
        Kind::Idr | Kind::I => PicKind::Intra,
        Kind::P => PicKind::Inter,
        Kind::B => PicKind::B,
    }
}

/// One source picture's lookahead cost: what the rate controller plans
/// its bits by before it is coded. See `encode::rc`'s lookahead section
/// for what the controller does with it.
///
/// Both are sums of 8x8 luma SATDs over the display-size picture (whole
/// blocks only; a partial edge block is not counted, which biases every
/// picture of a stream identically). `intra` is each block against its
/// own mean — the residual the DC predictor would leave, a bound on
/// what any intra mode leaves. `inter` is each block against the same
/// block of the previous source picture at zero motion — the residual
/// a skip would leave, a bound on what a motion search leaves. Bounds
/// rather than the encoder's real costs, deliberately: a lookahead that
/// ran the decision machinery would cost as much as coding, and what
/// the controller needs is a number that moves *with* the content,
/// which is calibrated once ([`SEED_BITS_PER_COST`](super::rc)) and
/// then pinned by every observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PicCost {
    /// Sum of 8x8 SATDs against each block's mean.
    pub intra: u64,
    /// Sum of 8x8 SATDs against the previous source picture; equal to
    /// `intra` for the first picture, which has none.
    pub inter: u64,
    /// `inter` with each block's mean difference taken out of the previous
    /// picture first: what is left of the change once the brightness has
    /// been matched. Equal to `intra` for the first picture.
    pub inter_ac: u64,
    /// The mean differences taken out for `inter_ac`, summed as absolute
    /// levels over the blocks: how far the brightness moved. 0 for the
    /// first picture.
    pub dc: u64,
}

/// The share of a picture's intra cost that predicting it from a
/// reference coded at quantiser 45 leaves as residual, however still the
/// content — the reference's own quantisation noise, which the
/// source-against-source inter cost cannot see.
///
/// Measured once: on the corpus's held-frame clip, whose inter cost is
/// exactly zero, the first P picture after a keyframe coded at 45 cost
/// 41928 bits at quantiser 0 against the keyframe's 85600 units of intra
/// cost — 41928 / 3.55 (the inter bits-per-cost median of the same run)
/// is 11.8k units, 0.14 of the intra cost. Without the floor that
/// picture was planned as free and coded at quantiser 0, twelve times
/// the keyframe; with it the plan sees what a decoder's reference
/// actually holds.
const REF_NOISE_AT_45: f64 = 0.14;

/// Quantiser steps per halving of the reference-noise floor as the
/// reference's quantiser falls. The step size itself halves every six,
/// and the first version of the floor followed it — which planned the
/// odd-sized clip's P pictures (reference at 28) from a floor three
/// times too low (0.02 of the intra cost predicted, 0.058 measured from
/// the bits a P at 26 actually cost) and walked their quantiser to 0
/// chasing bits the model said were not there. Twelve fits the two
/// points measured (0.14 at 45, 0.058 at 28: `0.14 * 2^(-17/12)` is
/// 0.052), and on the corpus took the odd clip's ABR ratio from 1.45 to
/// 1.16 and the held clip's from 1.21 to 0.78 with every other clip
/// unchanged. Two points is a fit, not a law; the plan error the
/// controller reports is where a third point would show.
const REF_NOISE_HALVING: f64 = 12.0;

/// What [`PicCost::inter_cost`] charges per level of a block's mean
/// difference when it asks whether a picture's inter cost is only a
/// brightness step: half of the 32 units the SATD itself charges (four 4x4
/// tiles, each taking a one-level shift of the block as 16 levels of DC and
/// halving its sum to 8).
///
/// The intra cost is taken against each block's own mean, so it has no DC
/// at all, while the inter cost carries the whole brightness step. As a
/// fade reaches black its texture goes and the intra cost falls under the
/// inter cost — and capping the inter cost there plans the picture at the
/// intra cost of a nearly flat frame while it still codes the fade: on the
/// gain-and-offset fade at 96 kbps the last three P pictures were planned
/// at 4206, 3504 and 2415 bits and spent 5136, 4176 and 7688, the last at
/// 3.18 times its plan. So a picture whose change, the brightness step
/// priced at this rate, still fits under its intra cost is planned at its
/// inter cost uncapped: it is predictable, and the cap only saw the step.
///
/// Measured over every rate row of the gate, 2026-09-18 (the module docs
/// of `encode::rc` hold the table): the cap binds on ten distinct pictures
/// of the corpus's lookahead cells — the cut clip's cut, which keeps it,
/// and the last one to three pictures of each fade — and no other rate
/// cell moves. At 16 the gain fade's three unweighted lookahead cells move
/// from 1.083, 1.030 and 1.109 of target to 1.070, 1.019 and 1.102, and
/// the native 10-bit fade from 1.003 to 0.995. At 8 that fade fell to
/// 0.888: its end pictures had been cancelling an under-spent keyframe,
/// and a cheaper price uncaps more of them.
const CAP_DC_PRICE: u64 = 16;

impl PicCost {
    /// The cost an inter picture is planned at, predicted from a reference
    /// coded at `qp_ref`: its inter cost held above the reference's
    /// quantisation noise ([`PicCost::inter_floor`]), and capped at its
    /// intra cost — a block the previous picture predicts badly is still
    /// coded intra — **unless** the difference is a brightness step and
    /// nothing more ([`CAP_DC_PRICE`]).
    ///
    /// Not under weighted prediction (`weighted`): there the encoder takes
    /// the brightness step out itself, with the weights, and a picture it
    /// weights is not the one the cap misjudges.
    fn inter_cost(&self, qp_ref: i32, weighted: bool) -> u64 {
        let predicted = self.inter.max(self.inter_floor(qp_ref));
        if !weighted && self.inter_ac + CAP_DC_PRICE * self.dc <= self.intra {
            predicted
        } else {
            self.intra.min(predicted)
        }
    }

    /// The least an inter picture predicted from a reference coded at
    /// `qp_ref` can cost: [`REF_NOISE_AT_45`] of the intra cost, scaled
    /// by the reference's step size.
    fn inter_floor(&self, qp_ref: i32) -> u64 {
        (self.intra as f64 * REF_NOISE_AT_45 * 2f64.powf((qp_ref - 45) as f64 / REF_NOISE_HALVING)) as u64
    }

    /// Measure a `w` by `h` luma plane (stride `w`) of `bit_depth`-bit
    /// samples, and `prev` — the picture pushed before it at the same size
    /// — when there is one.
    ///
    /// **In 8-bit sample units, whatever the depth.** A SATD grows with the
    /// sample scale, four times at 10 bits for the same picture, while the
    /// bits a quantiser buys do not: H.265 offsets the quantiser by the
    /// depth (`QpBdOffset`), so quantiser 30 at 10 bits spends what 30 does
    /// at 8. Unscaled, the calibrated seed ([`SEED_BITS_PER_COST`](super::rc))
    /// saw a 10-bit keyframe as four times the content it was and asked for
    /// twelve quantiser steps more than the same picture at 8 bits — motion10
    /// under `--lookahead 8` at 96 kbps seeded its keyframe at 41 against 29
    /// for the 8-bit clip, spent 3496 of 12287 planned bits and ended at
    /// 0.77x of target. So each sum is shifted down by `bit_depth - 8`.
    fn measure<S: Sample>(luma: &[S], w: usize, h: usize, prev: Option<&[S]>, bit_depth: u32) -> Self {
        let dist = DistortionDsp::<S>::new(Cpu::detect_honouring_env());
        let (bw, bh) = (w / 8, h / 8);
        let mut flat = [S::default(); 64];
        let mut shifted = [S::default(); 64];
        let top = (1i32 << bit_depth) - 1;
        let (mut intra, mut inter, mut inter_ac, mut dc) = (0u64, 0u64, 0u64, 0u64);
        for by in 0..bh {
            for bx in 0..bw {
                let at = by * 8 * w + bx * 8;
                let block = &luma[at..];
                let mut sum = 0i32;
                for y in 0..8 {
                    for x in 0..8 {
                        sum += block[y * w + x].to_i32();
                    }
                }
                flat.fill(S::from_i32((sum + 32) >> 6));
                intra += u64::from((dist.satd)(block, w, &flat, 8, 8, 8));
                if let Some(p) = prev {
                    inter += u64::from((dist.satd)(block, w, &p[at..], w, 8, 8));
                    // The block's mean difference, rounded half away from
                    // zero, and the previous block moved by it.
                    let mut diff = 0i32;
                    for y in 0..8 {
                        for x in 0..8 {
                            diff += block[y * w + x].to_i32() - p[at + y * w + x].to_i32();
                        }
                    }
                    let mean = (diff + if diff >= 0 { 32 } else { -32 }) / 64;
                    for y in 0..8 {
                        for x in 0..8 {
                            shifted[y * 8 + x] = S::from_i32((p[at + y * w + x].to_i32() + mean).clamp(0, top));
                        }
                    }
                    inter_ac += u64::from((dist.satd)(block, w, &shifted, 8, 8, 8));
                    dc += u64::from(mean.unsigned_abs());
                }
            }
        }
        let shift = bit_depth.saturating_sub(8);
        let (intra, inter, inter_ac, dc) = (intra >> shift, inter >> shift, inter_ac >> shift, dc >> shift);
        match prev {
            Some(_) => PicCost { intra, inter, inter_ac, dc },
            None => PicCost { intra, inter: intra, inter_ac: intra, dc: 0 },
        }
    }
}

/// The coding-quadtree depth this encoder codes when the configuration
/// leaves `max_cu_depth` unset: two splits below the CTB, down to 8x8 at a
/// 32x32 CTB. Decided on the measurement recorded in this module's docs;
/// `Some(0)` still codes one unit per CTB.
pub const DEFAULT_CU_DEPTH: u32 = 2;

/// How many quadtree levels below the CTB a stream coded under `cfg` at
/// geometry `g` may split: what `max_cu_depth` asks, or
/// [`DEFAULT_CU_DEPTH`] when it asks nothing, held above the 8x8 minimum
/// coding block — so a 16x16 CTB splits at most once.
fn tree_depth(cfg: &Config, g: &syn::Geometry) -> u32 {
    cfg.max_cu_depth.unwrap_or(DEFAULT_CU_DEPTH).min(g.log2_ctb - MIN_CB_LOG2)
}

/// The quantiser the unit of `1 << log2` at `(x0, y0)` codes at: the
/// picture's, plus — when the picture varies it — the offset of the
/// quantisation group the unit belongs to, held to the range the encoder's
/// quantiser takes. `offsets` holds one offset per group of `1 << log2_qg`
/// in raster order over a picture `width` samples wide.
///
/// A unit no larger than a group belongs to the group that contains it. A
/// unit larger than a group *is* its own group in the reader's walk (the
/// quantisation group restarts at every quadtree node at least the group
/// size), so it takes the rounded mean of the offsets it covers — the one
/// quantiser a single delta can give it.
fn cu_want(pic_qp: u8, offsets: Option<&[i32]>, width: usize, log2_qg: u32, x0: usize, y0: usize, log2: u32) -> i32 {
    let Some(o) = offsets else { return i32::from(pic_qp) };
    let wq = width.div_ceil(1 << log2_qg);
    let (gx, gy) = (x0 >> log2_qg, y0 >> log2_qg);
    let off = if log2 <= log2_qg {
        o[gy * wq + gx]
    } else {
        let k = 1usize << (log2 - log2_qg);
        let sum: i32 = (0..k * k).map(|i| o[(gy + i / k) * wq + gx + i % k]).sum();
        (f64::from(sum) / (k * k) as f64).round() as i32
    };
    (i32::from(pic_qp) + off).clamp(0, 51)
}

/// The encoder's mirror of the reader's quantisation-group state
/// (`coding_quadtree` / `transform_unit`, and 8.6.1), for any group size
/// and any unit size.
///
/// What the reader does, and what this therefore has to reproduce:
///
/// - A group starts at every quadtree node at least the group size
///   (`log2_cb >= Log2CtbSize - diff_cu_qp_delta_depth`): `IsCuQpDeltaCoded`
///   and `CuQpDeltaVal` reset, and the group's `qPY_PREV` is `SliceQpY` for
///   the first group of the slice and otherwise the `QpY` of the last unit
///   of the group before. That last value is taken where a node's
///   bottom-right corner lands on the group grid ([`QgChain::leave`]).
/// - Every unit's `qPY_PRED` averages the `QpY` of the units left of and
///   above its **group's** top-left corner, each where it lies inside the
///   same CTB, taking `qPY_PREV` for each that does not — the reader's own
///   [`qp_y_pred_from`]. Every unit of a group therefore shares one
///   prediction.
/// - The first unit of a group whose tree carries a coded cbf codes the
///   group's one `cu_qp_delta`; that unit and every later one of the group
///   hold `qPY_PRED + CuQpDeltaVal` wrapped ([`qp_y_from_pred`]).
/// - A unit before that one — no cbf, no delta yet — holds the prediction,
///   **whatever the encoder wanted for it**, and so does every unit of a
///   group that never codes a cbf. The deblocker must filter those units
///   at that value, because a decoder's will.
///
/// Two copies run per picture — one over the decided units ([`settle_tree`])
/// and one while writing ([`write_tree`]) — and both must land on the same
/// numbers; `settle` records what a decoder will hold and `spell` asserts
/// the writer's view agrees.
struct QgChain {
    slice_qp: i32,
    bit_depth: u32,
    log2_ctb: u32,
    log2_qg: u32,
    /// Width of `qp_map` in 4x4 blocks.
    w4: usize,
    /// No group has started yet: the first takes `SliceQpY`.
    first: bool,
    /// `QpY` of the last unit of the last group to end.
    prev: i32,
    /// The current group's top-left corner, and its `qPY_PREV`.
    qg: (usize, usize),
    qg_prev: i32,
    /// `IsCuQpDeltaCoded` and `CuQpDeltaVal` of the current group.
    coded: bool,
    delta: i32,
    /// `QpY` of the unit settled or spelled last.
    last: i32,
    /// `QpY` per 4x4 over the picture, filled unit by unit — what a later
    /// group's prediction reads to its left and above.
    qp_map: Vec<i8>,
}

impl QgChain {
    fn new(slice_qp: i32, bit_depth: u32, g: &syn::Geometry, log2_qg: u32) -> Self {
        let (w4, h4) = (g.coded_width as usize / 4, g.coded_height as usize / 4);
        QgChain {
            slice_qp,
            bit_depth,
            log2_ctb: g.log2_ctb,
            log2_qg,
            w4,
            first: true,
            prev: slice_qp,
            qg: (0, 0),
            qg_prev: slice_qp,
            coded: false,
            delta: 0,
            last: slice_qp,
            qp_map: vec![0; w4 * h4],
        }
    }

    /// A quadtree node of `1 << log2` at `(x0, y0)` begins.
    fn enter(&mut self, x0: usize, y0: usize, log2: u32) {
        if log2 >= self.log2_qg {
            self.coded = false;
            self.delta = 0;
            self.qg = (x0, y0);
            self.qg_prev = if self.first { self.slice_qp } else { self.prev };
            self.first = false;
        }
    }

    /// A quadtree node of `1 << log2` at `(x0, y0)` ends.
    fn leave(&mut self, x0: usize, y0: usize, log2: u32) {
        let mask = (1usize << self.log2_qg) - 1;
        let size = 1usize << log2;
        if (x0 + size) & mask == 0 && (y0 + size) & mask == 0 {
            self.prev = self.last;
        }
    }

    /// `qPY_PRED` for a unit at `(x0, y0)` of the current group.
    fn pred(&self, x0: usize, y0: usize) -> i32 {
        let (xq, yq) = self.qg;
        let ctb = |x: usize, y: usize| (x >> self.log2_ctb, y >> self.log2_ctb);
        let here = ctb(x0, y0);
        let at = |x: usize, y: usize| i32::from(self.qp_map[(y >> 2) * self.w4 + (x >> 2)]);
        let qa = (xq > 0 && ctb(xq - 1, yq) == here).then(|| at(xq - 1, yq));
        let qb = (yq > 0 && ctb(xq, yq - 1) == here).then(|| at(xq, yq - 1));
        qp_y_pred_from(qa, qb, self.qg_prev)
    }

    fn hold(&mut self, x0: usize, y0: usize, log2: u32, qp_y: i32) {
        let n = 1usize << log2;
        PicInfo::fill4(&mut self.qp_map, self.w4, x0, y0, n, n, qp_y as i8);
        self.last = qp_y;
    }

    /// Decision side: the unit of `1 << log2` at `(x0, y0)` was coded at
    /// `want` and carries a cbf or not. Returns the `QpY` a decoder will
    /// hold for it.
    fn settle(&mut self, x0: usize, y0: usize, log2: u32, want: i32, has_cbf: bool) -> i32 {
        let pred = self.pred(x0, y0);
        if has_cbf && !self.coded {
            self.coded = true;
            self.delta = want - pred;
        }
        let qp_y = qp_y_from_pred(pred, self.delta, self.bit_depth);
        debug_assert!(!has_cbf || qp_y == want, "a unit with a cbf must hold the quantiser it was coded at ({want}), not {qp_y}");
        self.hold(x0, y0, log2, qp_y);
        qp_y
    }

    /// Writer side: the `CuQpDeltaVal` to spell in the unit whose decision
    /// holds `qp_y`, or `None` where the reader reads none — and in that
    /// case the decision must already hold what the chain derives, which
    /// is checked rather than assumed.
    fn spell(&mut self, x0: usize, y0: usize, log2: u32, qp_y: i32, has_cbf: bool) -> Option<i32> {
        let pred = self.pred(x0, y0);
        let spelled = if has_cbf && !self.coded {
            self.coded = true;
            self.delta = qp_y - pred;
            Some(self.delta)
        } else {
            None
        };
        debug_assert_eq!(
            qp_y_from_pred(pred, self.delta, self.bit_depth),
            qp_y,
            "the writer's chain disagrees with the quantiser the decision settled at ({x0},{y0})"
        );
        self.hold(x0, y0, log2, qp_y);
        spelled
    }
}

/// What the coding-tree walk needs to know about a coded unit, whatever
/// its kind.
trait CodedUnit {
    fn qp_y(&self) -> i32;
    fn set_qp_y(&mut self, qp_y: i32);
    fn any_cbf(&self) -> bool;
    fn skipped(&self) -> bool;
}

impl CodedUnit for CuDecision {
    fn qp_y(&self) -> i32 {
        self.qp_y
    }
    fn set_qp_y(&mut self, qp_y: i32) {
        self.qp_y = qp_y;
    }
    fn any_cbf(&self) -> bool {
        CuDecision::any_cbf(self)
    }
    fn skipped(&self) -> bool {
        false
    }
}

impl CodedUnit for PCuDecision {
    fn qp_y(&self) -> i32 {
        PCuDecision::qp_y(self)
    }
    fn set_qp_y(&mut self, qp_y: i32) {
        PCuDecision::set_qp_y(self, qp_y)
    }
    fn any_cbf(&self) -> bool {
        PCuDecision::any_cbf(self)
    }
    fn skipped(&self) -> bool {
        matches!(self, PCuDecision::Inter(d) if matches!(d.kind, InterCuKind::Skip { .. }))
    }
}

/// One step of the reader's walk over a CTB's coding quadtree, in the
/// order `coding_quadtree` takes them.
enum TreeStep {
    /// A node begins: a quantisation group may start, and its
    /// `split_cu_flag` is coded here when `flag` — above the minimum coding
    /// block and wholly inside the picture; a node crossing the picture
    /// edge is split by inference.
    Enter { x0: usize, y0: usize, log2: u32, depth: u32, split: bool, flag: bool },
    /// The coding unit at this index of the CTB's units.
    Unit(usize),
    /// The node ends: a quantisation group may end.
    Leave { x0: usize, y0: usize, log2: u32 },
}

/// The walk over one CTB's units, read back from their placements: a node
/// is a leaf exactly when the next unit covers it whole. The units must
/// tile the CTB in z-scan order, which every tree decision produces.
///
/// `(pw, ph)` is the coded picture size. A CTB along its right or bottom
/// edge may be partial: a node crossing the edge is split with no coded
/// flag and a child starting outside the picture is not visited, which is
/// `coding_quadtree`'s inference (the flag is read only where `x0 + size <=
/// pic_width` and `y0 + size <= pic_height`), mirrored.
fn tree_steps<D>(ctu: &[TreeCu<D>], (x_ctb, y_ctb): (usize, usize), log2_ctb: u32, (pw, ph): (usize, usize)) -> Vec<TreeStep> {
    #[allow(clippy::too_many_arguments)]
    fn node<D>(ctu: &[TreeCu<D>], idx: &mut usize, x0: usize, y0: usize, log2: u32, depth: u32, pic: (usize, usize), out: &mut Vec<TreeStep>) {
        let size = 1usize << log2;
        let inside = x0 + size <= pic.0 && y0 + size <= pic.1;
        let cu = &ctu[*idx];
        let leaf = inside && cu.x0 == x0 && cu.y0 == y0 && cu.log2 == log2;
        debug_assert!(leaf || (cu.log2 < log2 && log2 > MIN_CB_LOG2), "units do not tile the node of {} at ({x0},{y0})", 1 << log2);
        out.push(TreeStep::Enter { x0, y0, log2, depth, split: !leaf, flag: inside && log2 > MIN_CB_LOG2 });
        if leaf {
            debug_assert_eq!(cu.depth, depth, "a unit's recorded depth disagrees with its place in the tree");
            out.push(TreeStep::Unit(*idx));
            *idx += 1;
        } else {
            let half = size / 2;
            for i in 0..4 {
                let (x, y) = (x0 + (i & 1) * half, y0 + (i >> 1) * half);
                if x < pic.0 && y < pic.1 {
                    node(ctu, idx, x, y, log2 - 1, depth + 1, pic, out);
                }
            }
        }
        out.push(TreeStep::Leave { x0, y0, log2 });
    }
    let mut out = Vec::with_capacity(3 * ctu.len() + 8);
    let mut idx = 0;
    node(ctu, &mut idx, x_ctb, y_ctb, log2_ctb, 0, (pw, ph), &mut out);
    assert_eq!(idx, ctu.len(), "units left over after the CTB's quadtree was walked");
    out
}

/// Settle every unit's `QpY` in decode order through `chain` — see
/// [`QgChain`]. `want` is the quantiser each unit was coded at.
fn settle_tree<D: CodedUnit>(chain: &mut QgChain, cus: &mut [TreeCu<D>], ctu_start: &[usize], g: &syn::Geometry, want: &dyn Fn(usize, usize, u32) -> i32) {
    let wc = g.ctbs_wide as usize;
    for (k, range) in ctu_start.windows(2).enumerate() {
        let ctu = &mut cus[range[0]..range[1]];
        let at = ((k % wc) << g.log2_ctb, (k / wc) << g.log2_ctb);
        for step in tree_steps(ctu, at, g.log2_ctb, (g.coded_width as usize, g.coded_height as usize)) {
            match step {
                TreeStep::Enter { x0, y0, log2, .. } => chain.enter(x0, y0, log2),
                TreeStep::Unit(i) => {
                    let cu = &mut ctu[i];
                    let q = chain.settle(cu.x0, cu.y0, cu.log2, want(cu.x0, cu.y0, cu.log2), cu.d.any_cbf());
                    cu.d.set_qp_y(q);
                }
                TreeStep::Leave { x0, y0, log2 } => chain.leave(x0, y0, log2),
            }
        }
    }
}

/// The per-4x4 facts the coding-tree syntax's neighbour contexts read, as
/// the writer accumulates them: `CtDepth` for `split_cu_flag`, and
/// `cu_skip_flag` for itself. In one slice and one tile the left and above
/// neighbours of any unit inside the picture are already written, so
/// availability is the picture edge.
struct TreeCtx {
    w4: usize,
    /// The coded picture size, where the reader's split inference begins.
    pic: (usize, usize),
    ct_depth: Vec<u8>,
    skip: Vec<u8>,
}

impl TreeCtx {
    fn new(g: &syn::Geometry) -> Self {
        let (w4, h4) = (g.coded_width as usize / 4, g.coded_height as usize / 4);
        TreeCtx { w4, pic: (g.coded_width as usize, g.coded_height as usize), ct_depth: vec![0; w4 * h4], skip: vec![0; w4 * h4] }
    }

    fn at(&self, grid: &[u8], x: usize, y: usize) -> u8 {
        grid[(y >> 2) * self.w4 + (x >> 2)]
    }
}

/// A leaf writer for [`write_tree`]: the unit, its left and above
/// neighbours' `cu_skip_flag` where available, and the `CuQpDeltaVal` to
/// spell in it.
type LeafWriter<'a, D> = dyn FnMut(&mut CabacEncoder<'_>, &mut Contexts, &TreeCu<D>, Option<bool>, Option<bool>, Option<i32>) + 'a;

/// Write one CTB's coding quadtree: every node's `split_cu_flag` in the
/// reader's order with the reader's neighbour-depth context, each unit
/// through `leaf`, and the quantiser chain walked alongside. Returns how
/// many `cu_qp_delta`s were spelled.
#[allow(clippy::too_many_arguments)]
fn write_tree<D: CodedUnit>(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    ctu: &[TreeCu<D>],
    at: (usize, usize),
    log2_ctb: u32,
    tc: &mut TreeCtx,
    mut chain: Option<&mut QgChain>,
    leaf: &mut LeafWriter<'_, D>,
) -> u64 {
    let mut deltas = 0;
    for step in tree_steps(ctu, at, log2_ctb, tc.pic) {
        match step {
            TreeStep::Enter { x0, y0, log2, depth, split, flag } => {
                if flag {
                    let nb = SplitCuNb {
                        left_depth: (x0 > 0).then(|| tc.at(&tc.ct_depth, x0 - 1, y0)),
                        above_depth: (y0 > 0).then(|| tc.at(&tc.ct_depth, x0, y0 - 1)),
                    };
                    write_split_cu_flag(e, cx, &nb, depth, split);
                }
                if let Some(ch) = chain.as_deref_mut() {
                    ch.enter(x0, y0, log2);
                }
            }
            TreeStep::Unit(i) => {
                let cu = &ctu[i];
                let delta = chain.as_deref_mut().and_then(|ch| ch.spell(cu.x0, cu.y0, cu.log2, cu.d.qp_y(), cu.d.any_cbf()));
                deltas += u64::from(delta.is_some());
                let left = (cu.x0 > 0).then(|| tc.at(&tc.skip, cu.x0 - 1, cu.y0) != 0);
                let above = (cu.y0 > 0).then(|| tc.at(&tc.skip, cu.x0, cu.y0 - 1) != 0);
                leaf(e, cx, cu, left, above, delta);
                let n = 1usize << cu.log2;
                PicInfo::fill4(&mut tc.ct_depth, tc.w4, cu.x0, cu.y0, n, n, cu.depth as u8);
                PicInfo::fill4(&mut tc.skip, tc.w4, cu.x0, cu.y0, n, n, u8::from(cu.d.skipped()));
            }
            TreeStep::Leave { x0, y0, log2 } => {
                if let Some(ch) = chain.as_deref_mut() {
                    ch.leave(x0, y0, log2);
                }
            }
        }
    }
    deltas
}

/// The price of one `split_cu_flag` bin, in fractional bits, under a
/// slice's initial contexts (`init_type` 0 for I, 1 for P, 2 for B) and the
/// neutral neighbour context — the terms every other counted price in this
/// encoder is taken in (see `h265_me::Rate`).
pub(crate) fn split_flag_bits(init_type: usize, qp: i32, split: bool) -> f32 {
    let mut cx = Contexts::new(init_type, qp);
    let mut e = CabacEncoder::counting();
    write_split_cu_flag(&mut e, &mut cx, &SplitCuNb { left_depth: None, above_depth: None }, 0, split);
    e.fractional_bits() as f32
}

/// What an I-slice intra unit's syntax costs through the production
/// writer, in fractional bits under the slice's initial contexts: the
/// quadtree's rate term, residual included.
pub(crate) fn intra_cu_bits(d: &CuDecision, cat: u32, qp: i32, pps_bypass: bool) -> f32 {
    let mut cx = Contexts::new(0, qp);
    let mut e = CabacEncoder::counting();
    write_cu_intra_i(&mut e, &mut cx, d, pps_bypass, cat, None);
    e.fractional_bits() as f32
}

/// The same for a unit of a P (`is_b` false) or B slice, inter or intra,
/// with neutral skip contexts, `nref` active list-0 references and the
/// unit at quadtree depth `depth`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_cu_bits(d: &PCuDecision, cat: u32, qp: i32, pps_bypass: bool, is_b: bool, nref: u32, depth: u32) -> f32 {
    let mut cx = Contexts::new(if is_b { 2 } else { 1 }, qp);
    let mut e = CabacEncoder::counting();
    match d {
        PCuDecision::Inter(d) => write_cu_inter(&mut e, &mut cx, d, None, None, cat, pps_bypass, None, nref, depth),
        PCuDecision::Intra(d) => write_cu_intra_in_p(&mut e, &mut cx, d, None, None, cat, pps_bypass, None),
    }
    e.fractional_bits() as f32
}

/// What the H.265 encoder's pictures were made of, by picture kind — the
/// H.265 twin of the H.264 encoder's shape census, and for the same
/// reason: a gate row turns a code path on, and only the clip decides
/// whether anything takes it. A row proves the syntax; this says whether
/// the feature was exercised, so a green cell over a feature no CU chose
/// can be told from one that proved something.
#[derive(Debug, Clone, Copy, Default)]
pub struct Census {
    /// Indexed by [`Census::slot`]: intra pictures, P, B.
    pub by_kind: [KindCensus; 3],
}

impl Census {
    /// Which of the three tallies a picture kind lands in.
    pub fn slot(kind: Kind) -> usize {
        match kind {
            Kind::Idr | Kind::I => 0,
            Kind::P => 1,
            Kind::B => 2,
        }
    }
}

/// The tally for one picture kind. Every field is a count of coding
/// units.
#[derive(Debug, Clone, Copy, Default)]
pub struct KindCensus {
    /// Coding units coded, all kinds.
    pub cus: u64,
    /// Skipped.
    pub skip: u64,
    /// Merged with residual.
    pub merge: u64,
    /// AMVP, either list or both.
    pub amvp: u64,
    /// Bi-predicted (both lists), merge or AMVP.
    pub bi: u64,
    /// Intra — every CU of an intra picture, and the intra CUs a P or B
    /// picture chose.
    pub intra: u64,
    /// Intra CUs whose transform tree split.
    pub split_tu: u64,
    /// CUs that coded a `cu_qp_delta` — carried a cbf under a per-CTB
    /// quantiser.
    pub qp_delta: u64,
    /// CUs whose `QpY` differs from the picture quantiser, which is what
    /// proves adaptive quantisation moved something rather than coding
    /// zero deltas.
    pub qp_moved: u64,
    /// Pictures whose `pred_weight_table` carried a luma weighting other
    /// than the default — weighted prediction chosen, not merely
    /// enabled. (A picture, not a CU: the table is per slice.)
    pub wp_on: u64,
    /// Under a chosen weighting, inter CUs whose luma SATD at the chosen
    /// vector was lower weighted than plain — the fit's prediction
    /// holding, CU by CU.
    pub wp_won: u64,
    /// The same, higher weighted than plain — the fit's prediction
    /// failing.
    pub wp_lost: u64,
    /// B pictures whose fitted table lost the picture-level check to a
    /// table of defaults (SSD plus λ·bits), and were kept default-weighted.
    pub wp_rd_default: u64,
    /// Inter CUs predicted from a list-0 reference other than the
    /// nearest (`ref_idx` 1 or more) — the choice multi-reference
    /// prediction exists for, taken.
    pub ref_older: u64,
    /// Intra units coded `PART_NxN`: four 4x4 blocks at the 8x8 minimum.
    pub nxn: u64,
    /// Coding units at quadtree depth 1: one split below the CTB.
    pub depth1: u64,
    /// Coding units at quadtree depth 2: two splits below — 8x8 at a 32x32
    /// CTB.
    pub depth2: u64,
    /// The model check on the quadtree's rate term: the bits its decisions
    /// priced the coded units at (each unit's syntax and its own
    /// `split_cu_flag`, under the slice's initial contexts), rounded — to
    /// set beside `coded_bits`. The split nodes' own flags, one per split,
    /// and the SAO parameters are not in it. 0 without a tree.
    pub model_bits: u64,
    /// What the slice data of those pictures actually took, in bits, SAO
    /// and quantiser deltas included. 0 without a tree.
    pub coded_bits: u64,
}

impl KindCensus {
    fn of_intra(cus: &[TreeCu<CuDecision>], pic_qp: i32) -> Self {
        let mut c = KindCensus::default();
        for cu in cus {
            let d = &cu.d;
            c.cus += 1;
            c.intra += 1;
            c.split_tu += u64::from(d.split_tu);
            c.nxn += u64::from(d.nxn);
            c.qp_moved += u64::from(d.qp_y != pic_qp);
            c.count_depth(cu.depth);
        }
        c.model_bits = model_bits(cus);
        c
    }

    fn of_inter(cus: &[TreeCu<PCuDecision>], pic_qp: i32) -> Self {
        let mut c = KindCensus::default();
        for cu in cus {
            let pd = &cu.d;
            c.cus += 1;
            c.qp_moved += u64::from(pd.qp_y() != pic_qp);
            c.count_depth(cu.depth);
            match pd {
                PCuDecision::Intra(d) => {
                    c.intra += 1;
                    c.split_tu += u64::from(d.split_tu);
                    c.nxn += u64::from(d.nxn);
                }
                PCuDecision::Inter(d) => {
                    match d.kind {
                        InterCuKind::Skip { .. } => c.skip += 1,
                        InterCuKind::Merge { .. } => c.merge += 1,
                        InterCuKind::Amvp { .. } | InterCuKind::BAmvp { .. } => c.amvp += 1,
                        InterCuKind::UseIntra => unreachable!("replaced by the intra decision"),
                    }
                    c.bi += u64::from(d.ref_idx >= 0 && d.ref_idx_l1 >= 0);
                    c.ref_older += u64::from(d.ref_idx >= 1);
                }
            }
        }
        c.model_bits = model_bits(cus);
        c
    }

    fn count_depth(&mut self, depth: u32) {
        self.depth1 += u64::from(depth == 1);
        self.depth2 += u64::from(depth == 2);
    }

    /// Fold `other` into this tally.
    fn add(&mut self, other: &KindCensus) {
        self.cus += other.cus;
        self.skip += other.skip;
        self.merge += other.merge;
        self.amvp += other.amvp;
        self.bi += other.bi;
        self.intra += other.intra;
        self.split_tu += other.split_tu;
        self.qp_delta += other.qp_delta;
        self.qp_moved += other.qp_moved;
        self.wp_on += other.wp_on;
        self.wp_won += other.wp_won;
        self.wp_lost += other.wp_lost;
        self.wp_rd_default += other.wp_rd_default;
        self.ref_older += other.ref_older;
        self.nxn += other.nxn;
        self.depth1 += other.depth1;
        self.depth2 += other.depth2;
        self.model_bits += other.model_bits;
        self.coded_bits += other.coded_bits;
    }

    /// The nonzero counters, named, for a census line.
    pub fn taken(&self) -> Vec<(&'static str, u64)> {
        [
            ("cus", self.cus),
            ("skip", self.skip),
            ("merge", self.merge),
            ("amvp", self.amvp),
            ("bi", self.bi),
            ("intra", self.intra),
            ("split_tu", self.split_tu),
            ("qp_delta", self.qp_delta),
            ("qp_moved", self.qp_moved),
            ("wp_on", self.wp_on),
            ("wp_won", self.wp_won),
            ("wp_lost", self.wp_lost),
            ("wp_rd_default", self.wp_rd_default),
            ("ref_older", self.ref_older),
            ("nxn", self.nxn),
            ("depth1", self.depth1),
            ("depth2", self.depth2),
            ("model_bits", self.model_bits),
            ("coded_bits", self.coded_bits),
        ]
        .into_iter()
        .filter(|&(_, n)| n != 0)
        .collect()
    }
}

/// The rate the tree decisions priced a picture's units at, rounded to
/// bits — see [`KindCensus::model_bits`].
fn model_bits<D>(cus: &[TreeCu<D>]) -> u64 {
    cus.iter().map(|cu| f64::from(cu.bits)).sum::<f64>().round() as u64
}

/// The slice header's SAO switches for a picture coded with `sao` set:
/// both components on, or `None` when SAO is off and the reader takes no
/// bit at all. The chroma flag is itself conditional — the reader's gate
/// is `chroma_format_idc != 0` — so monochrome carries only the luma one.
///
/// Both switches go on together because the decision module decides per
/// CTB per component and can turn any of them off there for free, by
/// choosing `type_idx` 0; a cleared slice flag would instead forbid the
/// choice picture-wide for one bit.
fn sao_flags(sao: bool, cat: u32) -> Option<syn::SaoFlags> {
    sao.then(|| syn::SaoFlags { luma: true, chroma: (cat != 0).then_some(true) })
}

/// The parameter sets a picture is coded against, parsed back through the
/// decoder's own parsers — the pattern `code_p_picture` established: the
/// filters and the candidate derivations read decoder structures, and
/// building them from the very bytes the stream carries is what keeps the
/// encoder's idea of the geometry and the decoder's identical.
#[allow(clippy::too_many_arguments)]
fn parsed_sets(cfg: &Config, g: &syn::Geometry, qp: i32, bypass: bool, deblock: bool, cpb: Option<&Cpb>, opts: &PpsOptions) -> (crate::hevc::sps::Sps, crate::hevc::pps::Pps) {
    let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(cfg, g, LOG2_MAX_POC_LSB, cpb)))
        .expect("the encoder's own SPS parses");
    let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps_opts(qp, bypass, deblock, opts)))
        .expect("the encoder's own PPS parses");
    pps.resolve_tiles(&sps).expect("one tile covering the picture");
    (sps, pps)
}

/// Write one CTB's `sao()`, or nothing when the picture carries no SAO —
/// in which case the reader takes no bin here either, because the slice
/// header's flags are both clear.
///
/// Called at the top of every CTU, ahead of the coding quadtree, which is
/// where `decode_ctu` reads it.
#[allow(clippy::too_many_arguments)]
fn write_sao_for(e: &mut CabacEncoder, cx: &mut Contexts, plan: Option<&SaoPlan>, addr: usize, cxu: usize, cy: usize, bit_depth: u32, cat: u32) {
    let Some(plan) = plan else { return };
    let sctx = SaoCtx {
        sao_luma: true,
        sao_chroma: cat != 0,
        cat,
        // `cMax` of `sao_offset_abs` (7.4.9.3.2): `(1 << (Min(bitDepth,
        // 10) - 5)) - 1` — 7 at 8 bits, 31 at 10 and above. The reader
        // derives it from the SPS depth (`ctu.rs`), and so must the
        // decision (`h265_sao::sao_picture` clamps to the same cMax), or
        // an offset of 12 at 10 bits would be spelled as a truncated
        // unary with the wrong terminator and desync the slice.
        cmax: (1u32 << (bit_depth.min(10) - 5)) - 1,
        // No PPS range extension, so `log2_sao_offset_scale` is 0 at
        // every depth: offsets are carried unscaled, and `sao_picture`
        // chose them in the same units.
        shift: (0, 0),
    };
    // One slice, one tile: the reader's availability test for the merge
    // flags is exactly the picture edge.
    let nb = SaoMergeNb { left: cxu > 0, up: cy > 0 };
    write_sao(e, cx, &sctx, &nb, plan.merges[addr], &plan.params[addr]);
}

/// Serialise one inter coding unit - a `PART_2Nx2N` CU of whatever size the
/// coding quadtree gave it - in the
/// reader's element order (`coding_unit` / `prediction_unit`).
///
/// Which elements exist depends on the shape, and two of them the decoder
/// reads without anybody writing:
///
/// - A skipped CU codes `cu_skip_flag` and `merge_idx`, then stops:
///   `rqt_root_cbf` is inferred 0 and no transform tree follows.
/// - A non-skip 2Nx2N *merge* CU does not code `rqt_root_cbf` either - the
///   reader infers it **true** - so its transform tree always follows, and
///   a merge whose residual quantised away is unspellable. The decision
///   module spells that case as a skip instead.
/// - An AMVP CU codes `rqt_root_cbf` explicitly, and the tree follows only
///   when it is set.
///
/// `ref_idx_l0` is absent because the slice declares one active reference,
/// and `inter_pred_idc` is absent because a P slice forces list 0 - both
/// reader-side conditions rather than simplifications.
///
/// `cat` is `ChromaArrayType`, and the transform tree below is the only
/// part of this walk that depends on it - see the comments there for the
/// per-format cbf and residual shapes, which are the inter mirror of what
/// `write_ctu_intra`'s unsplit branch spells.
#[allow(clippy::too_many_arguments)]
fn write_cu_inter(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    d: &InterCuDecision,
    left_skip: Option<bool>,
    above_skip: Option<bool>,
    cat: u32,
    pps_bypass: bool,
    qp_delta: Option<i32>,
    nref: u32,
    depth: u32,
) {
    let log2 = d.log2_cu;
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    debug_assert!(u32::from(d.ref_idx.max(0) as u8) < nref.max(1), "a CU naming a reference beyond the active list");
    // The unit's `split_cu_flag`, and the quadtree above it, are the
    // tree walk's to write (`write_tree`); this spells the coding unit.
    // cu_transquant_bypass_flag is the CU's VERY FIRST bin - `coding_unit`
    // reads it before cu_skip_flag, so even a skipped CU spells one, and
    // it is present exactly when the PPS sets
    // transquant_bypass_enabled_flag. Writing it after the skip flag, or
    // omitting it on a skip, desyncs from the first lossless CU onward.
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }

    let skip = matches!(d.kind, InterCuKind::Skip { .. });
    write_cu_skip_flag(e, cx, left_skip, above_skip, skip);
    // A quantiser delta rides in the transform tree, so a CU without one
    // — skipped, or root cbf 0 below — cannot have been handed a delta:
    // the reader would take no bin for it.
    debug_assert!(qp_delta.is_none() || d.rqt_root_cbf, "a quantiser delta was handed to a CU with no transform tree");
    if let InterCuKind::Skip { merge_idx } = d.kind {
        write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
        return;
    }

    write_pred_mode_flag(e, cx, false);
    write_part_mode_inter(e, cx, crate::hevc::ctu::PartMode::P2Nx2N);
    match d.kind {
        InterCuKind::Merge { merge_idx } => {
            write_merge_flag(e, cx, true);
            write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
            // No rqt_root_cbf: the reader infers it true, so the tree
            // below is not optional here.
            debug_assert!(d.rqt_root_cbf, "a merge CU with no residual must be spelled as a skip");
        }
        InterCuKind::Amvp { mvp_flag, mvd } => {
            write_merge_flag(e, cx, false);
            // `ref_idx_l0` exists only when the slice declares more than
            // one active reference — the reader's `if nref > 1` — so a
            // single-reference stream writes nothing here and is
            // unchanged. `nref` is the count the slice header derived
            // from its own reference picture set.
            if nref > 1 {
                write_ref_idx(e, cx, nref, u32::from(d.ref_idx.max(0) as u8));
            }
            write_mvd(e, cx, mvd);
            write_mvp_flag(e, cx, mvp_flag != 0);
            write_rqt_root_cbf(e, cx, d.rqt_root_cbf);
            if !d.rqt_root_cbf {
                return;
            }
        }
        InterCuKind::BAmvp { idc, mvd, mvp_flag } => {
            write_merge_flag(e, cx, false);
            // inter_pred_idc, then per list -- L0's mvd and mvp_flag, then
            // L1's, interleaved as `prediction_unit` reads them rather
            // than grouped by element. No ref_idx in either list: each
            // declares exactly one active reference. See
            // `write_inter_pred_idc`'s docblock for the `w + h != 12`
            // reading; a 2Nx2N unit of 8x8 or more is never 12, and the
            // first bin's context is the unit's CtDepth.
            let n = 1i32 << log2;
            write_inter_pred_idc(e, cx, n, n, depth, u32::from(idc));
            for list in 0..2usize {
                let uses = match idc {
                    0 => list == 0,
                    1 => list == 1,
                    _ => true,
                };
                if !uses {
                    continue;
                }
                write_mvd(e, cx, mvd[list]);
                write_mvp_flag(e, cx, mvp_flag[list] != 0);
            }
            write_rqt_root_cbf(e, cx, d.rqt_root_cbf);
            if !d.rqt_root_cbf {
                return;
            }
        }
        InterCuKind::Skip { .. } | InterCuKind::UseIntra => unreachable!("handled above"),
    }

    // The transform tree: one CU-sized TU, no split.
    write_split_transform_flag(e, cx, log2, false);
    // Chroma cbfs, per component. Monochrome codes none at all - the
    // reader's `cat != 0` gate in `transform_tree` - and 4:2:2 codes the
    // stacked pair's second bin immediately after the first, on this
    // unsplit node, per its `cat == 2 && (!split || log2 == 3)` arm. The
    // node is above 4x4 in every geometry this encoder produces, so the
    // `log2 == 2` chroma-at-the-parent case never arises.
    if cat != 0 {
        for comp in 0..2 {
            write_cbf_chroma(e, cx, 0, d.cbf_chroma[comp]);
            if cat == 2 {
                write_cbf_chroma(e, cx, 0, d.cbf_chroma_bot[comp]);
            }
        }
    }
    // cbf_luma is coded only because a chroma cbf is set or the depth is
    // nonzero - for an inter leaf at depth 0 with every chroma cbf clear
    // the reader infers cbf_luma 1 and reads no bin, so writing one would
    // desync. Monochrome has no chroma cbf to set, so its inter leaves
    // never carry the bin at all and must genuinely have luma
    // coefficients; the decision module guarantees that by spelling a
    // residual-free CU as a skip or as rqt_root_cbf 0.
    let any_chroma_cbf =
        cat != 0 && (d.cbf_chroma[0] || d.cbf_chroma[1] || d.cbf_chroma_bot[0] || d.cbf_chroma_bot[1]);
    if any_chroma_cbf {
        write_cbf_luma(e, cx, 0, d.cbf_luma);
    } else {
        debug_assert!(d.cbf_luma, "an inter leaf with no chroma cbf has cbf_luma inferred 1");
    }
    // cu_qp_delta_abs / sign, where `transform_unit` reads it: after the
    // cbfs and before any residual. This single depth-0 unit is the first
    // of its group and carries a cbf whenever the tree exists, so the
    // reader reads a delta here exactly when the PPS enables one — and
    // the caller hands one over exactly then.
    if let Some(v) = qp_delta {
        write_cu_qp_delta(e, cx, v);
    }

    let n = 1usize << log2;
    // Inter blocks always scan diagonally: the mode-dependent scans are an
    // intra rule (7.4.9.11), and `residual_scan_idx` returns 0 for every
    // non-intra block regardless of size or component.
    let params = |log2_size: u32, c_idx: usize| ResidualParams {
        log2_size,
        c_idx,
        scan_idx: 0,
        bypass: d.bypass,
        transform_skip_allowed: false,
        sign_hiding: false,
        intra: false,
        pred_mode_intra: 0,
        ts_context: false,
        implicit_rdpcm: false,
        explicit_rdpcm: false,
        persistent_rice: false,
        trace: false,
    };
    if d.cbf_luma {
        write_residual(e, cx, &params(log2, 0), &d.luma[..n * n]);
    }
    if cat != 0 {
        // The chroma TB is the luma's own size at 4:4:4 and half of it
        // elsewhere; 4:2:2 carries two of them stacked, top then bottom.
        // The reader walks components outermost and the stacked pair
        // within (`transform_unit`'s `for c` around `for t`), and the
        // decision module packs slot `t` at `t * nc2` - the same layout
        // and the same order as the intra writer above.
        let log2c = if cat == 3 { log2 } else { log2 - 1 };
        let nc2 = 1usize << (2 * log2c);
        for comp in 0..2 {
            let pair = if cat == 2 { 2 } else { 1 };
            for t in 0..pair {
                let cbf = if t == 0 { d.cbf_chroma[comp] } else { d.cbf_chroma_bot[comp] };
                if cbf {
                    write_residual(e, cx, &params(log2c, comp + 1), &d.chroma[comp][t * nc2..(t + 1) * nc2]);
                }
            }
        }
    }
}

/// Serialise one CTU of an **I** slice, holding exactly one `PART_2Nx2N`
/// CU whose transform tree is either a single CU-sized TU or one level of
/// splitting into four quarter TUs — the two shapes the decision machinery
/// produces at CTB 16 or 32, and the geometry guarantees no partial CTUs.
///
/// This is the I-slice envelope; the CU itself is
/// [`write_cu_intra_body`], shared with the P-slice envelope
/// [`write_cu_intra_in_p`]. Here `coding_quadtree` reads one
/// `split_cu_flag` (the CTB is above the minimum CU size, so the flag is
/// coded, false), and then `coding_unit` starts straight at the intra
/// syntax: an I slice reads neither `cu_skip_flag` nor `pred_mode_flag`,
/// both being gated on `slice_type != I`.
///
/// `pps_bypass` mirrors the PPS's `transquant_bypass_enabled_flag`: when
/// set, `coding_unit` reads a `cu_transquant_bypass_flag` as its very first
/// bin, so this writer spells one — the CU's own choice, `d.bypass` — and
/// when clear, nothing is written and the CU must not claim bypass.
///
/// `qp_delta` is the `CuQpDeltaVal` to spell inside the transform tree,
/// `Some` exactly when the PPS enables per-group quantisers *and* this
/// CU carries a cbf for the reader to read it under — the encoder's
/// quantiser chain decides both; see [`write_cu_intra_body`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ctu_intra(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, ctu_x: usize, ctu_y: usize, pps_bypass: bool, cat: u32, qp_delta: Option<i32>) {
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // A whole-CTB unit: every coded neighbour has depth 0, and in a single
    // slice availability is picture geometry. (The quadtree walk writes its
    // own flags with the real neighbour depths; this serves the rate model.)
    let nb = SplitCuNb {
        left_depth: (ctu_x > 0).then_some(0),
        above_depth: (ctu_y > 0).then_some(0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);
    write_cu_intra_i(e, cx, d, pps_bypass, cat, qp_delta);
}

/// One intra coding unit of an **I** slice, without the quadtree around
/// it: what [`write_ctu_intra`] spells after its `split_cu_flag`, and what
/// the quadtree walk spells for every unit of an I picture.
pub(crate) fn write_cu_intra_i(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, pps_bypass: bool, cat: u32, qp_delta: Option<i32>) {
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // An I slice reads no `cu_skip_flag` and no `pred_mode_flag` — both
    // are gated on `slice_type != I` (ctu.rs:405, ctu.rs:434) — so the
    // CU starts at the bypass flag.
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }
    // `part_mode` exists for an intra unit only at the minimum coding
    // block — the reader's `!intra || log2_cb == MinCbLog2SizeY` gate —
    // where it says 2Nx2N or NxN; above it 2Nx2N is inferred.
    if d.log2_cu == MIN_CB_LOG2 {
        write_part_mode_intra(e, cx, d.nxn);
    }
    write_cu_intra_body(e, cx, d, cat, qp_delta);
}

/// Serialise one intra coding unit inside a **P** slice.
///
/// Same CU, different envelope. Ahead of the intra syntax a P slice reads
/// three more things, and the reader's own gates say which
/// (`coding_unit`, `src/hevc/ctu.rs:395`):
///
/// - `cu_skip_flag` — coded whenever `slice_type != I` (ctu.rs:405), 0
///   here, with the same left/above skipped-neighbour context increment
///   the inter writer uses.
/// - `pred_mode_flag` — likewise coded when `slice_type != I`
///   (ctu.rs:434), and **1**: this is the element that makes the CU
///   intra, and the one whose absence made intra-in-P unspellable.
/// - `part_mode` — *not* coded. The reader's gate is
///   `!intra || log2_cb == log2_min_cb_size` (ctu.rs:437), and these CUs
///   are intra at the whole CTB, 16 or 32, while `write_sps` fixes the
///   minimum coding block at 8 (`log2_min_cb = 3`, `h265_syntax.rs`).
///   Writing one would desync; `PART_2Nx2N` is inferred.
///
/// No `cu_transquant_bypass_flag` either: `code_p_picture` writes its PPS
/// with `transquant_bypass_enabled_flag` clear (lossless inter refuses by
/// name upstream), so the reader takes no such bin.
#[allow(clippy::too_many_arguments)]
fn write_cu_intra_in_p(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    d: &CuDecision,
    left_skip: Option<bool>,
    above_skip: Option<bool>,
    cat: u32,
    pps_bypass: bool,
    qp_delta: Option<i32>,
) {
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // `coding_unit` reads cu_transquant_bypass_flag BEFORE cu_skip_flag,
    // so it comes first here too.
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }
    write_cu_skip_flag(e, cx, left_skip, above_skip, false);
    write_pred_mode_flag(e, cx, true);
    // As in an I slice: `part_mode` at the minimum coding block only.
    if d.log2_cu == MIN_CB_LOG2 {
        write_part_mode_intra(e, cx, d.nxn);
    }
    write_cu_intra_body(e, cx, d, cat, qp_delta);
}

/// The intra coding unit proper: everything from `prev_intra_luma_pred_flag`
/// to the last residual block, which is byte for byte the same syntax in an
/// I slice and a P slice — the reader reaches it from both through the same
/// `coding_unit` tail (`src/hevc/ctu.rs:448` onward), and nothing in it
/// consults the slice type. One spelling, so the two cannot drift.
///
/// The walk is the reader's, specialised to the two shapes the decision
/// machinery produces: `coding_unit` reads the luma mode syntax for one
/// prediction block and the chroma mode; `transform_tree` reads one coded
/// `split_transform_flag` (the SPS makes the maximum transform equal the
/// CTB precisely so the unsplit shape is expressible, and declares
/// hierarchy depth 2 so the split one is too), the chroma cbfs at depth 0,
/// and then per leaf `cbf_luma` (always coded for intra) and the residual
/// blocks — see the split branch below for the per-child ordering.
/// Anything that stops matching the reader here desyncs the arithmetic
/// coder and fails SELF wholesale, which is exactly the property the
/// encode gate checks.
///
/// `qp_delta`, when given, is spelled in the **first transform unit that
/// carries a coded cbf** — luma, or the chroma bins that unit holds — and
/// nowhere else, which is where `transform_unit` reads it
/// (`IsCuQpDeltaCoded` gates the rest). The caller guarantees such a
/// unit exists (`CuDecision::any_cbf`); a delta left unspelled at the
/// end is a caller bug and is asserted.
fn write_cu_intra_body(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, cat: u32, qp_delta: Option<i32>) {
    let log2 = d.log2_cu;
    debug_assert!((MIN_CB_LOG2..=5).contains(&log2), "a coding unit is 8x8 to 32x32");
    debug_assert!(!d.nxn || log2 == MIN_CB_LOG2, "PART_NxN exists only at the minimum CU size");
    if d.nxn {
        write_cu_intra_nxn_body(e, cx, d, cat, qp_delta);
        return;
    }
    let mut pending = qp_delta;

    let syn0 = d.luma_syntax[0];
    write_prev_intra_luma_pred_flag(e, cx, syn0.prev_flag);
    if syn0.prev_flag {
        write_mpm_idx(e, u32::from(syn0.mpm_idx));
    } else {
        write_rem_intra_luma_pred_mode(e, u32::from(syn0.rem));
    }
    // Monochrome has no chroma syntax at all: `coding_unit` reads the mode,
    // `transform_tree` the cbfs and `transform_unit` the residuals only
    // when `chroma_array_type != 0`, so the writer emits nothing chroma.
    if cat != 0 {
        write_intra_chroma_pred_mode(e, cx, u32::from(d.chroma_syntax));
    }

    let n = 1usize << log2;
    let params = |log2_size: u32, c_idx: usize, mode: u8| ResidualParams {
        log2_size,
        c_idx,
        scan_idx: residual_scan_idx(true, log2_size, c_idx, cat, u32::from(mode)),
        bypass: d.bypass,
        transform_skip_allowed: false,
        sign_hiding: false,
        intra: true,
        pred_mode_intra: u32::from(mode),
        ts_context: false,
        implicit_rdpcm: false,
        explicit_rdpcm: false,
        persistent_rice: false,
        trace: false,
    };
    if d.split_tu {
        // Transform tree: one level of splitting — four quarter-size TUs in
        // z-order. The walk is the reader's `transform_tree`, specialised:
        // at depth 0 the split flag is coded (log2 <= max_tb, > min_tb,
        // depth < the SPS's max_transform_hierarchy_depth_intra of 2) and
        // says split; the chroma cbfs are coded ONCE here, at depth 0; each
        // child then codes its own split flag (still coded — child sizes 8
        // and 16 are above the 4x4 minimum and depth 1 < 2), saying no
        // further split, its chroma cbfs gated on the parent's per-component
        // bins (the reader reads a child bin only where the parent's was
        // set, inferring zero otherwise), its always-coded intra cbf_luma,
        // and its residuals: luma at the child size, chroma at half that —
        // both child sizes keep log2 > 2, so chroma splits alongside and the
        // chroma-at-the-parent rule for 4x4 luma children never triggers.
        // Prediction is per-PU (one mode for the whole 2Nx2N CU), so every
        // child TB scans by the same luma mode.
        write_split_transform_flag(e, cx, log2, true);
        if cat != 0 {
            // Depth-0 chroma cbfs, per component — the gate bins. A 4:2:2
            // split parent codes only the first bin of each pair (the
            // reader's `!split || log2 == 3` arm skips the second above the
            // 8x8 node), so the single stored bin gates all of that
            // component's child squares, top and bottom alike.
            for comp in 0..2 {
                write_cbf_chroma(e, cx, 0, d.cbf_chroma[comp]);
            }
        }
        let q = (n / 2) * (n / 2);
        // The child chroma TB: the luma child's own size at 4:4:4, half of
        // it elsewhere.
        let log2c = if cat == 3 { log2 - 1 } else { log2 - 2 };
        let qc = 1usize << (2 * log2c);
        for i in 0..4 {
            write_split_transform_flag(e, cx, log2 - 1, false);
            if cat != 0 {
                for comp in 0..2 {
                    if d.cbf_chroma[comp] {
                        write_cbf_chroma(e, cx, 1, d.cbf_chroma_tu[comp][i]);
                        if cat == 2 {
                            write_cbf_chroma(e, cx, 1, d.cbf_chroma_tu_bot[comp][i]);
                        }
                    }
                }
            }
            // Positional cbf_luma: quadrant `i` owns `[4i..4i+4]`, and a
            // quadrant that is a single leaf — which every quadrant is at
            // this one split level — uses its first slot. The decision
            // module's layout is positional precisely so no slot depends
            // on a sibling's structure.
            write_cbf_luma(e, cx, 1, d.cbf_luma[4 * i]);
            // The delta, at the first child with a cbf: its luma bin, or
            // the chroma bins it coded just above (which exist only under
            // a set parent bin — the reader infers a clear one otherwise).
            let child_chroma = cat != 0 && (0..2).any(|comp| d.cbf_chroma[comp] && (d.cbf_chroma_tu[comp][i] || d.cbf_chroma_tu_bot[comp][i]));
            if pending.is_some() && (d.cbf_luma[4 * i] || child_chroma) {
                write_cu_qp_delta(e, cx, pending.take().expect("checked"));
            }
            if d.cbf_luma[4 * i] {
                write_residual(e, cx, &params(log2 - 1, 0, d.luma_modes[0]), &d.luma[i * q..(i + 1) * q]);
            }
            if cat != 0 {
                for comp in 0..2 {
                    if !d.cbf_chroma[comp] {
                        continue;
                    }
                    // 4:2:2: the child's stacked pair, top then bottom;
                    // one square everywhere else. The reader walks
                    // components outermost, squares within.
                    let pair = if cat == 2 { 2 } else { 1 };
                    for t in 0..pair {
                        let cbf = if t == 0 { d.cbf_chroma_tu[comp][i] } else { d.cbf_chroma_tu_bot[comp][i] };
                        if cbf {
                            let slot = if cat == 2 { 2 * i + t } else { i };
                            write_residual(
                                e,
                                cx,
                                &params(log2c, comp + 1, d.chroma_mode),
                                &d.chroma[comp][slot * qc..(slot + 1) * qc],
                            );
                        }
                    }
                }
            }
        }
        debug_assert!(pending.is_none(), "a quantiser delta was handed to a split CU with no coded cbf");
        return;
    }

    // Transform tree: a single TU the size of the CU.
    write_split_transform_flag(e, cx, log2, false);
    if cat != 0 {
        // Per component: the cbf, and at 4:2:2 the stacked pair's second
        // bin right after it (the reader's `!split || log2 == 3` arm — an
        // unsplit node always codes both halves).
        for comp in 0..2 {
            write_cbf_chroma(e, cx, 0, d.cbf_chroma[comp]);
            if cat == 2 {
                write_cbf_chroma(e, cx, 0, d.cbf_chroma_bot[comp]);
            }
        }
    }
    write_cbf_luma(e, cx, 0, d.cbf_luma[0]);
    if let Some(v) = pending.take() {
        debug_assert!(
            d.cbf_luma[0] || (cat != 0 && (0..2).any(|comp| d.cbf_chroma[comp] || d.cbf_chroma_bot[comp])),
            "a quantiser delta was handed to a CU with no coded cbf"
        );
        write_cu_qp_delta(e, cx, v);
    }
    if d.cbf_luma[0] {
        write_residual(e, cx, &params(log2, 0, d.luma_modes[0]), &d.luma[..n * n]);
    }
    if cat != 0 {
        // The chroma TB is the luma's own size at 4:4:4, half elsewhere;
        // 4:2:2 stacks two squares per component, top then bottom.
        let log2c = if cat == 3 { log2 } else { log2 - 1 };
        let nc2 = 1usize << (2 * log2c);
        for comp in 0..2 {
            let pair = if cat == 2 { 2 } else { 1 };
            for t in 0..pair {
                let cbf = if t == 0 { d.cbf_chroma[comp] } else { d.cbf_chroma_bot[comp] };
                if cbf {
                    write_residual(
                        e,
                        cx,
                        &params(log2c, comp + 1, d.chroma_mode),
                        &d.chroma[comp][t * nc2..(t + 1) * nc2],
                    );
                }
            }
        }
    }
}

/// The `PART_NxN` intra coding unit proper, at the 8x8 minimum coding
/// block: [`write_cu_intra_body`]'s twin for the four-block shape, in the
/// reader's element order.
///
/// - `prev_intra_luma_pred_flag` for all four blocks, and only then each
///   block's `mpm_idx` or `rem_intra_luma_pred_mode` — `coding_unit` reads
///   the flags in one loop and the payloads in the next.
/// - `intra_chroma_pred_mode` once, or at 4:4:4 four times in z-order
///   (`nc = if cat == 3 { npu } else { 1 }`); never in monochrome.
/// - The transform tree's root, 8x8, splits by inference (`IntraSplitFlag`)
///   and codes no split flag. Its chroma cbfs are coded there — both 4:2:2
///   bins, the `log2 == 3` arm.
/// - Four 4x4 children, which code no split flag (the 4x4 minimum). At
///   4:4:4 each codes its own chroma cbfs under the root's gate; at 4:2:0
///   and 4:2:2 they inherit the root's (`log2 == 2` inherits). Each codes
///   `cbf_luma`; then the quantiser delta, if it is the first unit with a
///   coded cbf — which at 4:2:0 and 4:2:2 counts the INHERITED chroma bins,
///   so a set root chroma bin puts the delta in the first child whatever
///   its luma holds; then its luma residual; then chroma — its own 4x4
///   blocks at 4:4:4, Cb then Cr under its own block's chroma mode, and at
///   4:2:0 and 4:2:2 the CU's chroma once, after the fourth child
///   (`blk_idx == 3`).
fn write_cu_intra_nxn_body(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, cat: u32, qp_delta: Option<i32>) {
    let mut pending = qp_delta;
    for pb in 0..4 {
        write_prev_intra_luma_pred_flag(e, cx, d.luma_syntax[pb].prev_flag);
    }
    for pb in 0..4 {
        let syn = d.luma_syntax[pb];
        if syn.prev_flag {
            write_mpm_idx(e, u32::from(syn.mpm_idx));
        } else {
            write_rem_intra_luma_pred_mode(e, u32::from(syn.rem));
        }
    }
    match cat {
        0 => {}
        3 => {
            for pb in 0..4 {
                write_intra_chroma_pred_mode(e, cx, u32::from(d.chroma_syntax_nxn[pb]));
            }
        }
        _ => write_intra_chroma_pred_mode(e, cx, u32::from(d.chroma_syntax)),
    }
    let params = |c_idx: usize, mode: u8| ResidualParams {
        log2_size: 2,
        c_idx,
        scan_idx: residual_scan_idx(true, 2, c_idx, cat, u32::from(mode)),
        bypass: d.bypass,
        transform_skip_allowed: false,
        sign_hiding: false,
        intra: true,
        pred_mode_intra: u32::from(mode),
        ts_context: false,
        implicit_rdpcm: false,
        explicit_rdpcm: false,
        persistent_rice: false,
        trace: false,
    };
    if cat != 0 {
        for comp in 0..2 {
            write_cbf_chroma(e, cx, 0, d.cbf_chroma[comp]);
            if cat == 2 {
                write_cbf_chroma(e, cx, 0, d.cbf_chroma_bot[comp]);
            }
        }
    }
    let inherited = cat != 0 && cat != 3 && (0..2).any(|comp| d.cbf_chroma[comp] || d.cbf_chroma_bot[comp]);
    for i in 0..4 {
        if cat == 3 {
            for comp in 0..2 {
                if d.cbf_chroma[comp] {
                    write_cbf_chroma(e, cx, 1, d.cbf_chroma_tu[comp][i]);
                }
            }
        }
        write_cbf_luma(e, cx, 1, d.cbf_luma[4 * i]);
        let child_chroma = if cat == 3 { (0..2).any(|comp| d.cbf_chroma_tu[comp][i]) } else { inherited };
        if pending.is_some() && (d.cbf_luma[4 * i] || child_chroma) {
            write_cu_qp_delta(e, cx, pending.take().expect("checked"));
        }
        if d.cbf_luma[4 * i] {
            write_residual(e, cx, &params(0, d.luma_modes[i]), &d.luma[16 * i..16 * i + 16]);
        }
        if cat == 3 {
            for comp in 0..2 {
                if d.cbf_chroma_tu[comp][i] {
                    write_residual(e, cx, &params(comp + 1, d.chroma_mode_nxn[i]), &d.chroma[comp][16 * i..16 * i + 16]);
                }
            }
        } else if cat != 0 && i == 3 {
            for comp in 0..2 {
                let pair = if cat == 2 { 2 } else { 1 };
                for t in 0..pair {
                    let cbf = if t == 0 { d.cbf_chroma[comp] } else { d.cbf_chroma_bot[comp] };
                    if cbf {
                        write_residual(e, cx, &params(comp + 1, d.chroma_mode), &d.chroma[comp][t * 16..(t + 1) * 16]);
                    }
                }
            }
        }
    }
    debug_assert!(pending.is_none(), "a quantiser delta was handed to an NxN CU with no coded cbf");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaFormat;
    use crate::dsp::hevc::install_simd_u8;

    fn cfg(w: u32, h: u32, chroma: ChromaFormat) -> Config {
        Config { width: w, height: h, chroma, ..Config::default() }
    }

    #[test]
    fn frame_size_matches_every_chroma_format() {
        for (chroma, per_px) in [
            (ChromaFormat::Monochrome, 1.0),
            (ChromaFormat::Yuv420, 1.5),
            (ChromaFormat::Yuv422, 2.0),
            (ChromaFormat::Yuv444, 3.0),
        ] {
            let e = H265Encoder::new(cfg(64, 64, chroma)).unwrap();
            assert_eq!(e.frame_bytes(), (64.0 * 64.0 * per_px) as usize, "{chroma:?}");
        }
    }

    /// The lookahead cost is in 8-bit sample units at every depth: a
    /// 10-bit picture that is an 8-bit one shifted up two costs what the
    /// 8-bit one does, intra and inter, to within the rounding of the
    /// shift. Unscaled it cost four times as much, and the rate
    /// controller seeded a 10-bit keyframe twelve quantiser steps coarser
    /// than its 8-bit twin.
    #[test]
    fn a_ten_bit_picture_costs_what_its_eight_bit_twin_does() {
        let (w, h) = (64usize, 64usize);
        let pic = |t: usize| -> Vec<u8> {
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    ((x * 3 + y * 5 + t * 7) % 97 + ((x * y + t) % 13) * 9) as u8
                })
                .collect()
        };
        let (a8, b8) = (pic(0), pic(1));
        let widen = |p: &[u8]| p.iter().map(|&s| u16::from(s) << 2).collect::<Vec<u16>>();
        let (a10, b10) = (widen(&a8), widen(&b8));
        let c8 = PicCost::measure(&b8, w, h, Some(&a8[..]), 8);
        let c10 = PicCost::measure(&b10, w, h, Some(&a10[..]), 10);
        assert!(c8.intra > 10_000 && c8.inter > 10_000, "the pictures must have content to cost: {c8:?}");
        for (name, a, b) in [("intra", c8.intra, c10.intra), ("inter", c8.inter, c10.inter)] {
            let (lo, hi) = (a.min(b) as f64, a.max(b) as f64);
            assert!(hi / lo < 1.01, "{name}: 8-bit {a} against 10-bit {b}");
        }
    }

    /// The lookahead tells a brightness step from the rest of a change. A
    /// picture that is the one before it five levels darker everywhere
    /// costs exactly the SATD's price of the step — 32 units per level per
    /// 8x8 block — and nothing once the step is matched; the same picture
    /// moved one sample sideways keeps most of its cost with the step
    /// matched.
    /// At 10 bits, in 8-bit units like every other cost.
    #[test]
    fn a_brightness_step_is_measured_apart_from_the_rest_of_the_change() {
        let (w, h) = (64usize, 64usize);
        let blocks = (w / 8 * (h / 8)) as u64;
        let pic = |dx: usize| -> Vec<u8> { (0..w * h).map(|i| (40 + ((i % w + dx) * 7 + (i / w) * 3) % 150) as u8).collect() };
        let prev = pic(0);
        let darker: Vec<u8> = prev.iter().map(|&s| s - 5).collect();
        let fade = PicCost::measure(&darker, w, h, Some(&prev[..]), 8);
        assert_eq!((fade.inter, fade.inter_ac, fade.dc), (32 * 5 * blocks, 0, 5 * blocks), "a uniform step: {fade:?}");

        let moved = PicCost::measure(&pic(1), w, h, Some(&prev[..]), 8);
        assert!(moved.inter_ac > 10_000 && 32 * moved.dc < moved.inter_ac / 4, "motion is mostly not a brightness step: {moved:?}");

        let widen = |p: &[u8]| p.iter().map(|&s| u16::from(s) << 2).collect::<Vec<u16>>();
        let fade10 = PicCost::measure(&widen(&darker), w, h, Some(&widen(&prev)[..]), 10);
        assert_eq!((fade10.inter, fade10.inter_ac, fade10.dc), (fade.inter, fade.inter_ac, fade.dc), "10-bit: {fade10:?}");

        let first = PicCost::measure(&prev, w, h, None, 8);
        assert_eq!((first.inter, first.inter_ac, first.dc), (first.intra, first.intra, 0), "no previous picture: {first:?}");
    }

    /// The intra cap on an inter picture's planning cost is dropped
    /// exactly when the change, its brightness step priced at 16 per level,
    /// fits under the intra cost — and never under weighted prediction,
    /// which takes the step out itself. A picture cheaper to predict than
    /// to code intra is planned at its inter cost either way, held above
    /// the reference's noise.
    #[test]
    fn the_intra_cap_is_dropped_exactly_when_the_change_is_a_brightness_step() {
        // qp_ref 21: the noise floor is 0.14 * 2^-2 = 0.035 of intra, far
        // under every inter cost here.
        let step = |inter_ac: u64, dc: u64| PicCost { intra: 1000, inter: 3000, inter_ac, dc };
        assert_eq!(step(200, 50).inter_cost(21, false), 3000, "200 + 16 * 50 = 1000: a step, uncapped");
        assert_eq!(step(200, 51).inter_cost(21, false), 1000, "200 + 16 * 51 = 1016: more than a step, capped");
        assert_eq!(step(1001, 0).inter_cost(21, false), 1000, "no step at all, and dearer than intra: capped");
        assert_eq!(step(0, 62).inter_cost(21, false), 3000, "all step: uncapped");
        assert_eq!(step(0, 63).inter_cost(21, false), 1000, "16 * 63 = 1008: capped");
        for (ac, dc) in [(200, 50), (0, 62)] {
            assert_eq!(step(ac, dc).inter_cost(21, true), 1000, "weighted prediction keeps the cap ({ac}, {dc})");
        }
        let cheap = PicCost { intra: 5000, inter: 3000, inter_ac: 2900, dc: 100 };
        for weighted in [false, true] {
            assert_eq!(cheap.inter_cost(21, weighted), 3000, "cheaper than intra: the inter cost (weighted {weighted})");
        }
        let held = PicCost { intra: 100_000, inter: 0, inter_ac: 0, dc: 0 };
        assert_eq!(held.inter_cost(45, false), 14_000, "a held picture is planned at the reference's noise");
    }

    /// Pictures whose four CTBs differ sharply in variance — flat, a
    /// gradient, noise, a checkerboard — so adaptive quantisation has
    /// something to move, in `chroma` at `bit_depth`, `count` frames
    /// that drift a little each so inter pictures carry residual.
    fn aq_frames(chroma: ChromaFormat, bit_depth: u32, count: usize) -> Vec<Vec<u8>> {
        let (w, h) = (64usize, 64usize);
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = if chroma == ChromaFormat::Monochrome { (0, 0) } else { (w / sw, h / sh) };
        let shift = bit_depth - 8;
        (0..count)
            .map(|i| {
                let mut seed = 0x9e37u32.wrapping_add(i as u32 * 7919);
                let mut samples: Vec<u32> = Vec::with_capacity(w * h + 2 * cw * ch);
                for y in 0..h {
                    for x in 0..w {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let v: i32 = match (x >= 32, y >= 32) {
                            (false, false) => 110 + i as i32,
                            (true, false) => ((x + y + i) % 96) as i32 + 60,
                            (false, true) => (seed >> 24) as i32,
                            (true, true) => if ((x / 4) + (y / 4) + i) % 2 == 0 { 40 } else { 200 },
                        };
                        samples.push((v.clamp(0, 255) as u32) << shift);
                    }
                }
                for _ in 0..2 {
                    for y in 0..ch {
                        for x in 0..cw {
                            samples.push((((120 + x / 2 + y / 3 + i) & 0xff) as u32) << shift);
                        }
                    }
                }
                if shift == 0 {
                    samples.iter().map(|&v| v as u8).collect()
                } else {
                    samples.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
                }
            })
            .collect()
    }

    /// Adaptive quantisation — a quantiser per CTB, carried by
    /// `cu_qp_delta` — round-trips through the production decoder for
    /// intra, P and B pictures at every chroma format and at a deep
    /// depth, and the census proves it moved something: a stream whose
    /// every delta was zero, or whose every CTB kept the picture
    /// quantiser, would pass SELF while proving only the syntax.
    ///
    /// SELF is the right check here rather than a QP map comparison: a
    /// residual coded at one quantiser and scaled at another desyncs
    /// the reconstruction, and a residual-free CTB filtered at the wrong
    /// quantiser moves the deblocked samples — both surface as a
    /// picture that differs from the encoder's own.
    #[test]
    fn adaptive_quantisation_round_trips_and_moves_the_quantiser() {
        for (chroma, bit_depth, bframes) in [
            (ChromaFormat::Yuv420, 8u32, 0u32),
            (ChromaFormat::Yuv420, 8, 2),
            (ChromaFormat::Yuv422, 8, 0),
            (ChromaFormat::Yuv444, 8, 2),
            (ChromaFormat::Monochrome, 8, 0),
            (ChromaFormat::Yuv420, 10, 2),
        ] {
            let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes}");
            let frames = aq_frames(chroma, bit_depth, 6);
            for qp in [22u8, 40] {
                let mut e = H265Encoder::new(Config {
                    gop: 8,
                    bframes,
                    bit_depth,
                    rate: super::super::RateControl::ConstantQp(qp),
                    aq_strength: 2.0,
                    ..cfg(64, 64, chroma)
                })
                .unwrap();
                let mut units = Vec::new();
                for f in &frames {
                    units.extend(e.push(f).expect("an adaptively quantised picture should code"));
                }
                units.extend(e.flush().unwrap());
                assert_eq!(units.len(), frames.len(), "{tag}");

                let census = e.census();
                let deltas: u64 = census.by_kind.iter().map(|k| k.qp_delta).sum();
                let moved: u64 = census.by_kind.iter().map(|k| k.qp_moved).sum();
                assert!(deltas > 0, "{tag} qp {qp}: no CU coded a cu_qp_delta, so the syntax was never exercised");
                assert!(moved > 0, "{tag} qp {qp}: no CTB left the picture quantiser, so the feature did nothing");
                if bframes > 0 {
                    assert!(census.by_kind[2].cus > 0, "{tag}: no B picture was coded");
                }

                // The decoder emits display order; the reconstructions
                // are in coding order, matched through each access unit's
                // POC as `deep_pictures_round_trip_through_the_decoder`
                // does.
                let mut dec = crate::hevc::HevcDecoder::new();
                for u in &units {
                    dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag} qp {qp}: {err}"));
                }
                dec.flush().unwrap();
                let mut by_display = vec![None; units.len()];
                for u in &units {
                    by_display[(u.poc / 2) as usize] = Some(u.encode_index as usize);
                }
                for (i, coded) in by_display.iter().enumerate() {
                    let want = &e.reconstructions()[coded.unwrap_or_else(|| panic!("{tag} qp {qp}: display index {i} never coded"))];
                    let got = dec.next_picture().unwrap_or_else(|| panic!("{tag} qp {qp}: picture {i} missing"));
                    assert!(got.into_packed() == *want, "{tag} qp {qp}: picture {i} differs from the reconstruction");
                }
            }
        }

        // Strength 0 is off: the stream is what the encoder writes with
        // the switch absent — the PPS bit clear and no delta anywhere.
        let frames = aq_frames(ChromaFormat::Yuv420, 8, 3);
        let encode = |strength: f32| -> Vec<u8> {
            let mut e = H265Encoder::new(Config { gop: 8, aq_strength: strength, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
            let mut out = Vec::new();
            for f in &frames {
                for u in e.push(f).unwrap() {
                    out.extend_from_slice(&u.data);
                }
            }
            for u in e.flush().unwrap() {
                out.extend_from_slice(&u.data);
            }
            assert_eq!(e.census().by_kind.iter().map(|k| k.qp_delta).sum::<u64>() > 0, strength > 0.0);
            out
        };
        assert_eq!(encode(0.0), encode(Config::default().aq_strength));
        assert_ne!(encode(0.0), encode(1.0), "strength 1 must change the stream");

        // Lossless has no quantiser to adapt: refused by name.
        let err = H265Encoder::new(Config {
            rate: super::super::RateControl::Lossless,
            aq_strength: 1.0,
            ..cfg(64, 64, ChromaFormat::Yuv420)
        })
        .err()
        .expect("adaptive quantisation on a lossless stream must refuse");
        assert!(format!("{err}").contains("adaptive quantisation"), "{err}");
    }

    /// A busy picture scaled down a step per frame — a luma gain fade,
    /// chroma untouched — in `chroma` at `bit_depth`, `count` frames.
    fn fade_frames(chroma: ChromaFormat, bit_depth: u32, count: usize) -> Vec<Vec<u8>> {
        let (w, h) = (64usize, 64usize);
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = if chroma == ChromaFormat::Monochrome { (0, 0) } else { (w / sw, h / sh) };
        let shift = bit_depth - 8;
        (0..count)
            .map(|i| {
                let gain = 1.0 - i as f64 / 16.0;
                let mut samples: Vec<u32> = Vec::with_capacity(w * h + 2 * cw * ch);
                let mut seed = 0x51edu32;
                for y in 0..h {
                    for x in 0..w {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        // Texture with structure and a little noise, so the
                        // search has something to match and the fit has
                        // variance to work with.
                        let base = 40 + ((x * 3 + y * 5 + (x * y) / 7) % 150) as i32 + ((seed >> 28) as i32 - 8);
                        let v = (f64::from(base) * gain).round().clamp(0.0, 255.0) as u32;
                        samples.push(v << shift);
                    }
                }
                for _ in 0..2 {
                    for y in 0..ch {
                        for x in 0..cw {
                            samples.push((((110 + x / 3 + y / 2) & 0xff) as u32) << shift);
                        }
                    }
                }
                if shift == 0 { samples.iter().map(|&v| v as u8).collect() } else { samples.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect() }
            })
            .collect()
    }

    /// Weighted prediction on a fade: every P slice carries a
    /// `pred_weight_table`, the fit is chosen (census `wp_on`), it holds
    /// CU by CU (`wp_won` above `wp_lost`, the model check), the stream
    /// is markedly smaller than the same encode without it, and it
    /// round-trips through the decoder — at every chroma format, with B
    /// pictures (weighted from both anchors, chosen and holding CU by CU
    /// like the P ones), and at 10 bits. On a held
    /// clip the fit is the identity: the table is all defaults, `wp_on`
    /// is 0, and the stream is the unweighted one plus a few table bits
    /// per slice.
    #[test]
    fn weighted_prediction_pays_on_a_fade_and_round_trips() {
        for (chroma, bit_depth, bframes) in [
            (ChromaFormat::Yuv420, 8u32, 0u32),
            (ChromaFormat::Yuv420, 8, 2),
            (ChromaFormat::Yuv422, 8, 0),
            (ChromaFormat::Yuv444, 8, 0),
            (ChromaFormat::Monochrome, 8, 0),
            (ChromaFormat::Yuv420, 10, 2),
        ] {
            let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes}");
            let frames = fade_frames(chroma, bit_depth, 8);
            let encode = |weighted_pred: bool| -> (Vec<Access>, Vec<Vec<u8>>, Census) {
                let mut e = H265Encoder::new(Config { gop: 8, bframes, bit_depth, weighted_pred, ..cfg(64, 64, chroma) })
                    .unwrap_or_else(|err| panic!("{tag}: {err}"));
                let mut units = Vec::new();
                for f in &frames {
                    units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                }
                units.extend(e.flush().unwrap());
                (units, e.reconstructions().to_vec(), *e.census())
            };
            let (with, recon, census) = encode(true);
            let (without, _, _) = encode(false);
            let bytes = |u: &[Access]| u.iter().map(|a| a.data.len()).sum::<usize>();
            let p = &census.by_kind[1];
            assert!(p.wp_on > 0, "{tag}: no P picture chose a weighting: {p:?}");
            assert!(p.wp_won > p.wp_lost, "{tag}: the fit lost more CUs than it won: {p:?}");
            if bframes > 0 {
                // The B pictures between the fade's anchors are weighted
                // too, or the B round trip below proves default weighting.
                let b = &census.by_kind[2];
                assert!(b.wp_on > 0, "{tag}: no B picture chose a weighting: {b:?}");
                assert!(b.wp_won > b.wp_lost, "{tag}: the B fit lost more CUs than it won: {b:?}");
            }
            assert!(
                (bytes(&with) as f64) < (bytes(&without) as f64) * 0.9,
                "{tag}: weighting saved little on a fade: {} against {} bytes",
                bytes(&with),
                bytes(&without)
            );
            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &with {
                dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: {err}"));
            }
            dec.flush().unwrap();
            let mut by_display = vec![None; with.len()];
            for u in &with {
                by_display[(u.poc / 2) as usize] = Some(u.encode_index as usize);
            }
            for (i, coded) in by_display.iter().enumerate() {
                let want = &recon[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                assert!(got.into_packed() == *want, "{tag}: picture {i} differs from the reconstruction");
            }
        }

        // A held clip: the identity fits, nothing is chosen, and the
        // table costs its flags and nothing more.
        let held: Vec<Vec<u8>> = std::iter::repeat_n(fade_frames(ChromaFormat::Yuv420, 8, 1).remove(0), 6).collect();
        let encode = |weighted_pred: bool| -> (usize, Census) {
            let mut e = H265Encoder::new(Config { gop: 8, weighted_pred, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
            let mut bytes = 0;
            for f in &held {
                bytes += e.push(f).unwrap().iter().map(|a| a.data.len()).sum::<usize>();
            }
            bytes += e.flush().unwrap().iter().map(|a| a.data.len()).sum::<usize>();
            (bytes, *e.census())
        };
        let (with, census) = encode(true);
        let (without, _) = encode(false);
        assert_eq!(census.by_kind[1].wp_on, 0, "a held clip chose a weighting: {:?}", census.by_kind[1]);
        assert!(with >= without && with <= without + 2 * held.len(), "the table of defaults should cost bits, not bytes: {with} against {without}");
    }

    /// The picture-level check between a B picture's fitted table and a
    /// table of defaults (`code_attempt`) keeps each where it pays. On the
    /// fade with two B pictures between anchors — a third and two thirds of
    /// the way, where default bi-prediction's even average is at the wrong
    /// level — every fitted table is kept at QP 26. With one B picture, at
    /// the midpoint default bi-prediction already averages to, and at QP 40,
    /// where the table's bits weigh most, some B pictures are coded
    /// default-weighted. A check with its comparison inverted fails both.
    #[test]
    fn a_b_pictures_table_is_kept_only_where_it_pays() {
        let frames = fade_frames(ChromaFormat::Yuv420, 8, 8);
        let run = |bframes: u32, qp: u8| -> KindCensus {
            let mut e = H265Encoder::new(Config {
                gop: 8,
                bframes,
                weighted_pred: true,
                rate: super::super::RateControl::ConstantQp(qp),
                ..cfg(64, 64, ChromaFormat::Yuv420)
            })
            .unwrap();
            for f in &frames {
                e.push(f).unwrap();
            }
            e.flush().unwrap();
            e.census().by_kind[2]
        };
        let kept = run(2, 26);
        assert!(kept.wp_on > 0, "bframes=2 QP 26: no B picture took a fitted table: {kept:?}");
        assert_eq!(kept.wp_rd_default, 0, "bframes=2 QP 26: a fitted table lost to the defaults: {kept:?}");
        let mid = run(1, 40);
        assert!(mid.wp_rd_default > 0, "bframes=1 QP 40: every fitted table was kept: {mid:?}");
    }

    /// Two references: every P slice declares two active references in
    /// list 0, `ref_idx` is coded on every AMVP unit, and the stream
    /// round-trips through the decoder — with B pictures (whose lists
    /// stay one each), under weighted prediction (an entry per
    /// reference), and at 10 bits. The census says how often the older
    /// reference was chosen, which is reported rather than asserted:
    /// on whole-CTB units the honest answer is usually zero.
    /// At the default of one reference the stream is what it always was.
    #[test]
    fn two_references_round_trip_and_the_default_is_unchanged() {
        for (chroma, bit_depth, bframes, weighted_pred) in [
            (ChromaFormat::Yuv420, 8u32, 0u32, false),
            (ChromaFormat::Yuv420, 8, 2, false),
            (ChromaFormat::Yuv444, 8, 0, true),
            (ChromaFormat::Yuv420, 10, 0, false),
        ] {
            let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes} wp={weighted_pred}");
            // Drifting content, so the older reference is a genuinely
            // different picture; a fade under weighting, so each
            // reference wants its own gain.
            let frames = if weighted_pred { fade_frames(chroma, bit_depth, 8) } else { aq_frames(chroma, bit_depth, 8) };
            let encode = |max_refs: u32| -> (Vec<Access>, Vec<Vec<u8>>, Census) {
                let mut e = H265Encoder::new(Config { gop: 8, bframes, bit_depth, max_refs, weighted_pred, ..cfg(64, 64, chroma) })
                    .unwrap_or_else(|err| panic!("{tag}: {err}"));
                let mut units = Vec::new();
                for f in &frames {
                    units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                }
                units.extend(e.flush().unwrap());
                (units, e.reconstructions().to_vec(), *e.census())
            };
            let (two, recon, census) = encode(2);
            let (one, _, _) = encode(1);
            assert_eq!(two.len(), frames.len(), "{tag}");
            let bytes = |u: &[Access]| u.iter().flat_map(|a| a.data.iter().copied()).collect::<Vec<u8>>();
            assert_ne!(bytes(&two), bytes(&one), "{tag}: a second reference changed nothing — the header must at least declare it");
            let p = &census.by_kind[1];
            assert!(p.cus > 0, "{tag}: no P picture");
            // Reported, not required: `ref_older` is how many CUs took
            // the older picture.
            eprintln!("{tag}: {} of {} P CUs chose the older reference", p.ref_older, p.cus);

            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &two {
                dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: {err}"));
            }
            dec.flush().unwrap();
            let mut by_display = vec![None; two.len()];
            for u in &two {
                by_display[(u.poc / 2) as usize] = Some(u.encode_index as usize);
            }
            for (i, coded) in by_display.iter().enumerate() {
                let want = &recon[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                assert!(got.into_packed() == *want, "{tag}: picture {i} differs from the reconstruction");
            }
        }
    }

    /// A lookahead holds pictures back and hands the controller their
    /// costs: the first `lookahead` pushes return nothing, every picture
    /// is still coded exactly once with the typing and order it would
    /// have had without one, the stream round-trips through the decoder,
    /// the ledger holds — and the stream *differs* from the one coded
    /// without a lookahead, or the feature was inert. A lookahead without
    /// a bitrate target refuses by name.
    #[test]
    fn a_lookahead_delays_output_and_codes_every_picture_once() {
        // One GOP exactly, so that `poc / 2` is the display index — a
        // second IDR would restart POC and alias display 0.
        let frames = aq_frames(ChromaFormat::Yuv420, 8, 8);
        for (bframes, lookahead) in [(0u32, 3u32), (2, 2), (2, 8), (0, 12)] {
            let tag = format!("bframes={bframes} lookahead={lookahead}");
            let run = |lookahead: u32| -> (Vec<Access>, Vec<Vec<u8>>) {
                let mut e = H265Encoder::new(Config {
                    gop: 8,
                    bframes,
                    lookahead,
                    fps: 25,
                    rate: super::super::RateControl::Bitrate { bps: 400_000 },
                    ..cfg(64, 64, ChromaFormat::Yuv420)
                })
                .unwrap_or_else(|err| panic!("{tag}: {err}"));
                let mut units = Vec::new();
                for (i, f) in frames.iter().enumerate() {
                    let out = e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}"));
                    if (i as u32) < lookahead {
                        assert!(out.is_empty(), "{tag}: picture {i} was released before the lookahead filled");
                    }
                    units.extend(out);
                }
                units.extend(e.flush().unwrap());
                assert!(e.rate_report().is_some(), "{tag}: no rate report");
                (units, e.reconstructions().to_vec())
            };
            let (with, recon) = run(lookahead);
            let (without, _) = run(0);
            assert_eq!(with.len(), frames.len(), "{tag}: one access unit per picture");
            let typing = |u: &[Access]| u.iter().map(|a| (a.poc, a.keyframe, a.encode_index)).collect::<Vec<_>>();
            assert_eq!(typing(&with), typing(&without), "{tag}: the lookahead changed the picture typing or order");
            let bytes = |u: &[Access]| u.iter().flat_map(|a| a.data.iter().copied()).collect::<Vec<u8>>();
            assert_ne!(bytes(&with), bytes(&without), "{tag}: the lookahead changed nothing about the stream");

            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &with {
                dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: {err}"));
            }
            dec.flush().unwrap();
            let mut by_display = vec![None; with.len()];
            for u in &with {
                by_display[(u.poc / 2) as usize] = Some(u.encode_index as usize);
            }
            for (i, coded) in by_display.iter().enumerate() {
                let want = &recon[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                assert!(got.into_packed() == *want, "{tag}: picture {i} differs from the reconstruction");
            }
        }

        let err = H265Encoder::new(Config { lookahead: 4, ..cfg(64, 64, ChromaFormat::Yuv420) })
            .err()
            .expect("a lookahead at a constant quantiser must refuse");
        assert!(format!("{err}").contains("lookahead"), "{err}");
    }

    /// Intra 4:2:0 codes for real now; everything else still refuses by
    /// the name of its missing piece, never by the name of the codec.
    #[test]
    fn intra_codes_and_the_remaining_holes_name_themselves() {
        // The real path: every picture an IDR, 4:2:0, constant QP.
        let mut e = H265Encoder::new(Config { gop: 0, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
        let frame = vec![64u8; 64 * 64 * 3 / 2];
        let out = e.push(&frame).unwrap();
        assert_eq!(out.len(), 1, "an all-intra picture should code");
        assert!(out[0].keyframe);
        assert!(!out[0].data.is_empty());
        assert_eq!(e.reconstructions().len(), 1);

        // Every chroma format codes now — a picture per format, each
        // producing a stream and a reconstruction of the right size.
        for (chroma, per) in [
            (ChromaFormat::Monochrome, 64 * 64),
            (ChromaFormat::Yuv422, 64 * 64 * 2),
            (ChromaFormat::Yuv444, 64 * 64 * 3),
        ] {
            let mut e = H265Encoder::new(Config { gop: 0, ..cfg(64, 64, chroma) }).unwrap();
            let out = e.push(&vec![64u8; per]).unwrap();
            assert_eq!(out.len(), 1, "{chroma:?} should code");
            assert!(!out[0].data.is_empty());
            assert_eq!(e.reconstructions()[0].len(), per, "{chroma:?} recon size");
        }

        // P pictures code, in every chroma format: a GOP produces one
        // access unit per picture, the first a keyframe and the rest not,
        // and the reconstruction is the size that format implies.
        for (chroma, per) in [
            (ChromaFormat::Monochrome, 64 * 64),
            (ChromaFormat::Yuv420, 64 * 64 * 3 / 2),
            (ChromaFormat::Yuv422, 64 * 64 * 2),
            (ChromaFormat::Yuv444, 64 * 64 * 3),
        ] {
            let mut e = H265Encoder::new(Config { gop: 8, ..cfg(64, 64, chroma) }).unwrap();
            let frame = vec![64u8; per];
            let mut units = Vec::new();
            for _ in 0..3 {
                units.extend(e.push(&frame).expect("a P picture should code"));
            }
            units.extend(e.flush().unwrap());
            assert_eq!(units.len(), 3, "{chroma:?}: one access unit per picture");
            assert!(units[0].keyframe, "{chroma:?}: the first is an IDR");
            assert!(!units[1].keyframe && !units[2].keyframe, "{chroma:?}: the rest are P");
            assert!(units.iter().all(|u| !u.data.is_empty()), "{chroma:?}");
            assert!(e.reconstructions().iter().all(|r| r.len() == per), "{chroma:?}: recon size");
        }

        // B pictures code too, in every chroma format: a group with two of
        // them per anchor produces one access unit per picture, and only
        // the first is a keyframe.
        for (chroma, per) in [
            (ChromaFormat::Yuv420, 64 * 64 * 3 / 2),
            (ChromaFormat::Yuv444, 64 * 64 * 3),
        ] {
            let mut e =
                H265Encoder::new(Config { gop: 8, bframes: 2, ..cfg(64, 64, chroma) }).unwrap();
            let frame = vec![64u8; per];
            let mut units = Vec::new();
            for _ in 0..6 {
                units.extend(e.push(&frame).expect("a B group should code"));
            }
            units.extend(e.flush().unwrap());
            assert_eq!(units.len(), 6, "one access unit per picture for {chroma:?}");
            assert!(units[0].keyframe, "the first is an IDR");
            assert!(units[1..].iter().all(|u| !u.keyframe), "the rest are not");
            assert!(units.iter().all(|u| !u.data.is_empty()));
        }

        // No named holes remain in the H.265 envelope: intra, P and B all
        // code, in every chroma format, lossy and lossless. This array is
        // deliberately kept — empty — rather than deleted, because the
        // loop below is the shape that proves a refusal is reached BY the
        // configuration that asks for it rather than merely present in the
        // source, and the next exclusion should be added here.
        let holes: [(Config, usize, &str); 0] = [];
        for (config, per, want) in holes {
            let mut e = H265Encoder::new(config).unwrap();
            let frame = vec![64u8; per];
            let mut named = false;
            for _ in 0..6 {
                if let Err(err) = e.push(&frame) {
                    let s = format!("{err}");
                    assert!(s.contains(want), "expected {want:?} in: {s}");
                    named = true;
                    break;
                }
            }
            if !named {
                if let Err(err) = e.flush() {
                    assert!(format!("{err}").contains(want));
                    named = true;
                }
            }
            assert!(named, "never reached the {want:?} hole");
        }

        // Lossless inter, the last hole to close, codes in every chroma
        // format and reconstructs the source EXACTLY — for P and for B.
        // Exactness is the whole point of the mode, so it is asserted
        // here and not merely that a stream came out.
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for bframes in [0u32, 2] {
                let frames = moving_frames_n(64, 64, chroma, 6);
                let mut e = H265Encoder::new(Config {
                    rate: super::super::RateControl::Lossless,
                    gop: 8,
                    bframes,
                    ..cfg(64, 64, chroma)
                })
                .unwrap();
                let mut units = Vec::new();
                for f in &frames {
                    units.extend(e.push(f).expect("lossless inter should code"));
                }
                units.extend(e.flush().unwrap());
                assert_eq!(units.len(), frames.len(), "{chroma:?} bframes={bframes}");
                assert!(
                    units[1..].iter().any(|u| !u.keyframe),
                    "{chroma:?} bframes={bframes}: no inter picture was coded, so lossless INTER is untested"
                );
                // The reconstructions come back in coding order; every one
                // must equal its source picture exactly.
                assert_eq!(e.reconstructions().len(), frames.len());
                // POC advances by TWO per picture (`gop.rs`: poc = display
                // * 2), so the display index of an access unit is poc / 2 —
                // not poc. Getting that wrong reads a neighbouring source
                // and reports a lossless stream as lossy, which is exactly
                // what it did while this test was being written.
                // With B pictures the scheduler must actually have held
                // one back, or the bframes arm proves nothing beyond the
                // bframes=0 one.
                if bframes > 0 {
                    assert!(
                        units.iter().any(|u| u.encode_index as usize != (u.poc / 2) as usize),
                        "{chroma:?}: coding order never differed from display order, so no B picture was coded"
                    );
                }
                for u in &units {
                    // `display` is the stream-wide display index, which in
                    // this single GOP is poc / 2 as well; the two must agree
                    // or a caller's timestamp table is read at the wrong row.
                    assert_eq!(
                        u.display,
                        (u.poc / 2) as u64,
                        "{chroma:?} bframes={bframes}: display index disagrees with poc"
                    );
                    let rec = &e.reconstructions()[u.encode_index as usize];
                    assert_eq!(
                        rec, &frames[(u.poc / 2) as usize],
                        "{chroma:?} bframes={bframes}: picture poc {} is not lossless",
                        u.poc
                    );
                }
            }
        }
    }

    /// The acceptance property of the rate model: what [`Rate`] says a
    /// shape costs is EXACTLY the number of bits `write_cu_inter` emits
    /// for it.
    ///
    /// This is the whole point of counting rather than estimating. The
    /// numbers it replaced — `tr_bins`, and an `mvd_cost` approximating
    /// exponential-Golomb as `5 + 2 * log2(a - 1)` — could not be checked
    /// by anything the project had: SELF and CROSS pass whatever the
    /// decision picks, and PSNR moves by fractions. A counted cost has a
    /// right answer, and this asserts it against the production writer
    /// rather than against a second opinion about the writer.
    ///
    /// Two shapes are compared, and they are the two for which
    /// `write_cu_inter` emits signalling and stops: a skip (the reader
    /// infers the whole transform tree away) and an AMVP CU whose
    /// `rqt_root_cbf` is 0 (the writer returns there). A merge CU has no
    /// such boundary — its `rqt_root_cbf` is inferred TRUE, so a transform
    /// tree always follows — which is the same reader-side rule that makes
    /// a residual-free merge unspellable.
    #[test]
    fn counted_rate_equals_the_bits_the_writer_emits() {
        use super::super::h265_me::Rate;
        use crate::hevc::frame::Mv;
        let log2 = 5u32;
        // Bits `write_cu_inter` emits for `d`, counted rather than
        // written, against the neutral neighbour context `Rate` prices in.
        // Fractional bits, the figure the decision actually compares on.
        // Comparing emitted bits here would assert almost nothing: a skip
        // is short enough that the coder emits none of them.
        let emitted = |d: &InterCuDecision, qp: i32| -> f32 {
            let mut cx = Contexts::new(1, qp);
            let mut e = CabacEncoder::counting();
            // The unit's `split_cu_flag`, which the tree walk writes
            // ahead of every unit above the minimum coding block, at the
            // neutral neighbour context `Rate` prices it in.
            write_split_cu_flag(&mut e, &mut cx, &SplitCuNb { left_depth: None, above_depth: None }, 0, false);
            write_cu_inter(&mut e, &mut cx, d, None, None, 1, false, None, 1, 0);
            e.fractional_bits() as f32
        };

        for qp in [22i32, 26, 34, 40] {
            let rate = Rate::new(qp, false, log2);

            for idx in 0..MAX_MERGE_CAND as u8 {
                let d = InterCuDecision {
                    log2_cu: log2,
                    kind: InterCuKind::Skip { merge_idx: idx },
                    ..InterCuDecision::default()
                };
                assert_eq!(
                    rate.skip(idx),
                    emitted(&d, qp),
                    "qp {qp} skip idx {idx}: counted cost is not the bits written"
                );
            }

            // Vectors that reach every arm of write_mvd: zero, one, the
            // Golomb remainder, and both ends of the component range.
            for &(x, y) in &[
                (0i16, 0i16),
                (1, 0),
                (0, -1),
                (2, 2),
                (-3, 5),
                (17, -9),
                (100, -1000),
                (32767, -32767),
                (-32768, 32767),
            ] {
                for flag in [0u8, 1] {
                    let d = InterCuDecision {
                        log2_cu: log2,
                        kind: InterCuKind::Amvp { mvp_flag: flag, mvd: Mv::new(x, y) },
                        rqt_root_cbf: false,
                        ..InterCuDecision::default()
                    };
                    assert_eq!(
                        rate.amvp(Mv::new(x, y), flag, false),
                        emitted(&d, qp),
                        "qp {qp} amvp mvd ({x},{y}) flag {flag}: counted cost is not the bits written"
                    );
                }
            }
        }
    }

    /// A lossless inter picture over STATIC content, which is the only way
    /// this suite reaches a bypassed **skip** CU.
    ///
    /// Why it needs its own test. `cu_transquant_bypass_flag` is the CU's
    /// very first bin — `coding_unit` reads it before `cu_skip_flag` — so
    /// a skipped CU spells one too. Every moving clip in the encode gate
    /// codes lossless CUs that all carry residual (with no quantiser to
    /// round it away, any imperfect prediction survives), so no skip ever
    /// occurs and the ordering rule is never exercised: seeding the
    /// mutation that omits the flag on a skip leaves the whole
    /// `hevc-lossless-ip` row green. On identical frames the prediction is
    /// exact, the residual really is zero, skips appear, and that same
    /// mutation fails SELF immediately — which is how this test was
    /// shown to be able to fail rather than assumed to be.
    #[test]
    fn lossless_inter_over_static_content_reaches_a_bypassed_skip() {
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            // One detailed picture, repeated: motion is exactly zero and a
            // merge candidate predicts it perfectly.
            let one = moving_frames_n(64, 64, chroma, 1).remove(0);
            let frames = vec![one; 4];

            // The decision must actually produce a skip, or the ordering
            // rule this test exists for is untouched.
            assert!(
                static_lossless_skips(&frames, chroma),
                "{chroma:?}: no CU skipped on identical frames, so no bypassed skip was coded"
            );

            let mut e = H265Encoder::new(Config {
                rate: super::super::RateControl::Lossless,
                gop: 8,
                ..cfg(64, 64, chroma)
            })
            .unwrap();
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).expect("lossless inter should code"));
            }
            units.extend(e.flush().unwrap());
            assert!(units[1..].iter().any(|u| !u.keyframe), "{chroma:?}: no inter picture");

            // Exact, and what a decoder rebuilds.
            for u in &units {
                assert_eq!(
                    &e.reconstructions()[u.encode_index as usize],
                    &frames[(u.poc / 2) as usize],
                    "{chroma:?}: poc {} is not lossless",
                    u.poc
                );
            }
            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &units {
                dec.push_annexb(&u.data).unwrap();
            }
            dec.flush().unwrap();
            for (i, want) in e.reconstructions().iter().enumerate() {
                let got = dec.next_picture().unwrap_or_else(|| panic!("{chroma:?}: picture {i} missing"));
                assert_eq!(&got.into_packed(), want, "{chroma:?}: picture {i} differs from the reconstruction");
            }
        }
    }

    /// Re-run the inter decision over identical frames and report whether
    /// any CU came out a skip. Same inputs and context as the encoder, so
    /// it reads the decision the encoder made.
    fn static_lossless_skips(frames: &[Vec<u8>], chroma: ChromaFormat) -> bool {
        let (w, h) = (64usize, 64usize);
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let mono = chroma == ChromaFormat::Monochrome;
        let (cw, ch) = if mono { (0, 0) } else { (w / sw, h / sh) };
        let config =
            Config { rate: super::super::RateControl::Lossless, gop: 8, ..cfg(w as u32, h as u32, chroma) };
        let g = syn::Geometry::new(&config);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB, None)))
            .unwrap();
        let mut pps =
            crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(26, true, true))).unwrap();
        pps.resolve_tiles(&sps).unwrap();

        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc_dsp = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        // Bypass, as `code_picture` builds it for a lossless stream: QP 26
        // is what the headers carry and scaling never runs.
        let ctx = IntraCtx {
            dsp: &dsp,
            enc: &enc_dsp,
            dist: &dist,
            qp: 26,
            bit_depth: 8,
            strong_smoothing: false,
            bypass: true, free_to_trim: false,
        };
        let split = |f: &[u8]| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            let (y, c) = f.split_at(w * h);
            let (cb, cr) = c.split_at(cw * ch);
            (y.to_vec(), cb.to_vec(), cr.to_vec())
        };
        let (wc, hc) = (w >> g.log2_ctb, h >> g.log2_ctb);

        // Bypass is carried by the context, not the picture.
        let mut ip = IntraPicture::<u8>::new_with_chroma(w, h, g.log2_ctb, 8, chroma);
        let (py, pcb, pcr) = split(&frames[0]);
        for cy in 0..hc {
            for cx in 0..wc {
                ip.code_ctu(&ctx, cx, cy, &py, w, &pcb, &pcr, cw);
            }
        }
        let mut refp = ip.recon;
        refp.poc = 0;
        refp.extend_rows(0, h);

        let mut pic = InterPicture::<u8>::new(&sps, &pps, 2);
        let (py, pcb, pcr) = split(&frames[1]);
        let mut any_skip = false;
        for cy in 0..hc {
            for cx in 0..wc {
                let d = pic.code_ctu(&ctx, &[&refp], cx, cy, &py, w, &pcb, &pcr, cw);
                any_skip |= matches!(d.kind, InterCuKind::Skip { .. });
            }
        }
        any_skip
    }

    /// Lossless (transquant bypass) reconstructs the source exactly: the
    /// encoder-held reconstruction — what SELF compares the decode against —
    /// must equal the input byte for byte, on content with real detail in
    /// it, not only on flat frames whose residual is zero everywhere.
    #[test]
    fn lossless_reconstruction_equals_the_source() {
        let mut e = H265Encoder::new(Config {
            rate: super::super::RateControl::Lossless,
            gop: 0,
            ..cfg(48, 32, ChromaFormat::Yuv420)
        })
        .unwrap();
        let mut frame = vec![0u8; 48 * 32 * 3 / 2];
        let mut seed = 0x5eedu32;
        for v in frame.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 24) as u8;
        }
        let out = e.push(&frame).unwrap();
        assert_eq!(out.len(), 1);
        assert!(!out[0].data.is_empty());
        assert_eq!(e.reconstructions()[0], frame, "bypass reconstruction differs from the source");
    }

    /// Inter pictures round-trip in every chroma format, in process:
    /// SELF without leaving the harness. This is the serialiser side of
    /// the contract — the decision module has its own replay test for the
    /// reconstruction it holds, and this proves the bits spell that
    /// reconstruction, which is the half a wrong cbf shape or a misplaced
    /// chroma residual breaks.
    ///
    /// The vacuity guard matters as much as the round trip. Content whose
    /// every CU codes as a skip would round-trip through any cbf shape at
    /// all, because no chroma bin would ever be written; so the pictures
    /// move and carry detail, and the test then asserts that chroma
    /// residual really was coded. A stream without it proves nothing about
    /// the chroma path this test exists to hold.
    #[test]
    fn inter_pictures_round_trip_in_every_chroma_format() {
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            let (w, h) = (64usize, 64usize);
            let frames = moving_frames(w, h, chroma);
            let per = frames[0].len();

            let mut e = H265Encoder::new(Config {
                rate: super::super::RateControl::ConstantQp(30),
                gop: 8,
                ..cfg(w as u32, h as u32, chroma)
            })
            .unwrap();
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).expect("should code"));
            }
            units.extend(e.flush().unwrap());
            assert_eq!(units.len(), frames.len(), "{chroma:?}");
            assert!(!units[1].keyframe, "{chroma:?}: the second picture should be a P picture");
            assert!(e.reconstructions().iter().all(|r| r.len() == per), "{chroma:?}: recon size");

            // SELF: the production decoder rebuilds every picture exactly
            // as the encoder holds it.
            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &units {
                dec.push_annexb(&u.data).unwrap();
            }
            dec.flush().unwrap();
            for (i, want) in e.reconstructions().iter().enumerate() {
                let got = dec.next_picture().unwrap_or_else(|| panic!("{chroma:?}: picture {i} missing"));
                assert_eq!(
                    &got.into_packed(),
                    want,
                    "{chroma:?}: picture {i} differs from the encoder-held reconstruction"
                );
            }

            // Vacuity guard, at the decision level.
            let (luma_coded, chroma_coded) = inter_traffic(&frames, chroma);
            assert!(luma_coded, "{chroma:?}: no P CU carried luma residual");
            if chroma != ChromaFormat::Monochrome {
                assert!(
                    chroma_coded,
                    "{chroma:?}: no P CU carried a chroma residual; the round trip proves nothing about chroma"
                );
            } else {
                assert!(!chroma_coded, "monochrome carried a chroma cbf");
            }
        }
    }

    /// Three pictures of detailed content, each translated a little
    /// further, in the packed layout the encoder takes. Real motion plus
    /// real detail is what makes a P picture carry residual rather than
    /// coding as a field of skips.
    fn moving_frames(w: usize, h: usize, chroma: ChromaFormat) -> Vec<Vec<u8>> {
        moving_frames_n(w, h, chroma, 3)
    }

    /// The same, with a chosen picture count — a B group needs more than
    /// three before the scheduler actually holds one back.
    fn moving_frames_n(w: usize, h: usize, chroma: ChromaFormat, count: usize) -> Vec<Vec<u8>> {
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let mono = chroma == ChromaFormat::Monochrome;
        let (cw, ch) = if mono { (0, 0) } else { (w / sw, h / sh) };
        let per = w * h + 2 * cw * ch;
        (0..count)
            .map(|f| {
                let mut frame = vec![0u8; per];
                let (dx, dy) = (3 * f, f);
                for y in 0..h {
                    for x in 0..w {
                        let tx = ((x + dx) as i32 % 25 - 12).abs();
                        let ty = ((y + dy) as i32 % 27 - 13).abs();
                        frame[y * w + x] = (40 + 4 * tx + 3 * ty) as u8;
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        let (sx, sy) = (x + dx / sw, y + dy / sh);
                        let r2 = (sx as i32 % 17 - 8).abs() * (sy as i32 % 19 - 9).abs();
                        frame[w * h + y * cw + x] = (110 + r2.min(90)) as u8;
                        frame[w * h + cw * ch + y * cw + x] = (150 - r2.min(90)) as u8;
                    }
                }
                frame
            })
            .collect()
    }

    /// Re-run the inter decision over the same content, reporting whether
    /// any CU carried luma and chroma residual. It reads the decision the
    /// encoder made, because the inputs and the context are identical —
    /// cheaper and more direct than threading a counter out of the
    /// encoder, and it cannot report traffic the encoder did not have.
    fn inter_traffic(frames: &[Vec<u8>], chroma: ChromaFormat) -> (bool, bool) {
        let (w, h) = (64usize, 64usize);
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let mono = chroma == ChromaFormat::Monochrome;
        let (cw, ch) = if mono { (0, 0) } else { (w / sw, h / sh) };
        let config =
            Config { rate: super::super::RateControl::ConstantQp(30), gop: 8, ..cfg(w as u32, h as u32, chroma) };
        let g = syn::Geometry::new(&config);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB, None,
        )))
        .unwrap();
        let mut pps =
            crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(30, false, false))).unwrap();
        pps.resolve_tiles(&sps).unwrap();

        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc_dsp = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let ctx =
            IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 30, bit_depth: 8, strong_smoothing: false, bypass: false, free_to_trim: false };
        let split = |f: &[u8]| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            let (y, c) = f.split_at(w * h);
            let (cb, cr) = c.split_at(cw * ch);
            (y.to_vec(), cb.to_vec(), cr.to_vec())
        };
        let (wc, hc) = (w >> g.log2_ctb, h >> g.log2_ctb);

        // The reference is picture 0 coded as intra, as the encoder builds it.
        let mut ip = IntraPicture::<u8>::new_with_chroma(w, h, g.log2_ctb, 8, chroma);
        ip.split_depth = 1;
        let (py, pcb, pcr) = split(&frames[0]);
        for cy in 0..hc {
            for cx in 0..wc {
                ip.code_ctu(&ctx, cx, cy, &py, w, &pcb, &pcr, cw);
            }
        }
        let mut refp = ip.recon;
        refp.poc = 0;
        refp.extend_rows(0, h);

        let mut pic = InterPicture::<u8>::new(&sps, &pps, 1);
        let (py, pcb, pcr) = split(&frames[1]);
        let (mut luma, mut chr) = (false, false);
        for cy in 0..hc {
            for cx in 0..wc {
                let d = pic.code_ctu(&ctx, &[&refp], cx, cy, &py, w, &pcb, &pcr, cw);
                luma |= d.cbf_luma && d.rqt_root_cbf;
                chr |= d.cbf_chroma[0] || d.cbf_chroma[1] || d.cbf_chroma_bot[0] || d.cbf_chroma_bot[1];
            }
        }
        (luma, chr)
    }

    /// The transform split carries live traffic and round-trips. Content
    /// built to make the split win — flat CTUs with one busy quadrant —
    /// must split at the decision level first: that is the guard that keeps
    /// this test from silently exercising only the single-TU path (a wired
    /// writer nobody reaches is the vacuity class this crate keeps
    /// rediscovering). Then the full encoder's stream must decode, in
    /// process through the production decoder, to the encoder-held
    /// reconstruction byte for byte — SELF without leaving the harness.
    #[test]
    fn a_split_transform_carries_traffic_and_round_trips() {
        let (w, h) = (64usize, 64usize);
        let mut frame = vec![128u8; w * h * 3 / 2];
        // One busy 16x16 quadrant per 32x32 CTU (bottom-right), luma only.
        let mut seed = 0xb1a5u32;
        for cty in 0..2usize {
            for ctx_ in 0..2usize {
                for y in 16..32 {
                    for x in 16..32 {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        frame[(cty * 32 + y) * w + ctx_ * 32 + x] = (seed >> 24) as u8;
                    }
                }
            }
        }
        let config = Config {
            rate: super::super::RateControl::ConstantQp(30),
            gop: 0,
            ..cfg(64, 64, ChromaFormat::Yuv420)
        };

        // Decision-level guard: this content actually splits.
        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc_dsp = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let ictx = IntraCtx {
            dsp: &dsp,
            enc: &enc_dsp,
            dist: &dist,
            qp: 30,
            bit_depth: 8,
            strong_smoothing: false,
            bypass: false, free_to_trim: false,
        };
        let mut pic = IntraPicture::<u8>::new(64, 64, 5, 8);
        pic.split_depth = 1;
        let (py, pc) = frame.split_at(w * h);
        let (pcb, pcr) = pc.split_at(w * h / 4);
        let mut splits = 0usize;
        for cy in 0..2 {
            for cx in 0..2 {
                let d = pic.code_ctu(&ictx, cx, cy, py, w, pcb, pcr, w / 2);
                splits += usize::from(d.split_tu);
            }
        }
        assert!(
            splits > 0,
            "the construction was meant to make splitting win; it did not, and the round trip below would be vacuous"
        );

        // Full-encoder SELF, in process.
        let mut e = H265Encoder::new(config).unwrap();
        let out = e.push(&frame).unwrap();
        assert_eq!(out.len(), 1);
        let mut dec = crate::hevc::HevcDecoder::new();
        dec.push_annexb(&out[0].data).unwrap();
        dec.flush().unwrap();
        let decoded = dec.next_picture().expect("one picture");
        assert_eq!(
            decoded.into_packed(),
            e.reconstructions()[0],
            "decoded bytes differ from the encoder-held reconstruction"
        );
    }

    /// An intra coding unit inside a P slice: decided by the intra module
    /// over the P picture's *own* reconstruction, spelled with
    /// `cu_skip_flag` 0 and `pred_mode_flag` 1, and round-tripped through
    /// the production decoder.
    ///
    /// The construction forces the choice rather than hoping for it. The
    /// reference is noise; the P picture repeats it except for one flat
    /// CTU, where `prefer_intra`'s DC proxy costs nothing and no vector
    /// into a noisy reference can compete. The decision-level guard runs
    /// the real decision module against the encoder's own reconstruction
    /// of the IDR and fails loudly if no CU chooses intra — a round trip
    /// that never reaches the new path is the vacuity class this crate
    /// keeps rediscovering.
    ///
    /// What the round trip proves is the whole chain at once: the syntax
    /// (a wrong element count desyncs CABAC and the picture comes out
    /// garbage), the prediction (an intra CU reading inter neighbours
    /// differently than the decoder does drifts), and the deblocker
    /// (which sees bS 2 at this CU's edges and nowhere else in the
    /// picture).
    #[test]
    fn an_intra_cu_inside_a_p_slice_round_trips() {
        use crate::hevc::frame::Frame;
        let (w, h) = (64usize, 64usize);
        let mut noise = vec![0u8; w * h * 3 / 2];
        let mut seed = 0x51deu32;
        for v in noise.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 24) as u8;
        }
        // The P picture: the same content, except the bottom-right 32x32
        // CTU is flat in all three planes.
        let mut flat = noise.clone();
        for y in 32..h {
            for x in 32..w {
                flat[y * w + x] = 128;
            }
        }
        for y in 16..h / 2 {
            for x in 16..w / 2 {
                flat[w * h + y * (w / 2) + x] = 128;
                flat[w * h + w * h / 4 + y * (w / 2) + x] = 128;
            }
        }

        let config = Config {
            rate: super::super::RateControl::ConstantQp(26),
            gop: 8,
            ..cfg(w as u32, h as u32, ChromaFormat::Yuv420)
        };
        let mut e = H265Encoder::new(config.clone()).unwrap();
        let mut units = e.push(&noise).unwrap();
        units.extend(e.push(&flat).unwrap());
        units.extend(e.flush().unwrap());
        assert_eq!(units.len(), 2, "one access unit per picture");

        // Decision-level guard, on the real modules: the encoder's own
        // reconstruction of the IDR, rebuilt as the reference frame the
        // P picture predicts from.
        let g = syn::Geometry::new(&config);
        assert_eq!(g.log2_ctb, 5, "the guard assumes the writer's 32x32 CTB choice");
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB, None))).unwrap();
        let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(26, false, true))).unwrap();
        pps.resolve_tiles(&sps).unwrap();
        let mut refp = Frame::<u8>::new(w, h, ChromaFormat::Yuv420, 8);
        refp.poc = 0;
        let rec0 = &e.reconstructions()[0];
        for (plane, (src, pw, ph)) in [&mut refp.y, &mut refp.cb, &mut refp.cr].into_iter().zip([
            (&rec0[..w * h], w, h),
            (&rec0[w * h..w * h + w * h / 4], w / 2, h / 2),
            (&rec0[w * h + w * h / 4..], w / 2, h / 2),
        ]) {
            let o = plane.origin();
            for y in 0..ph {
                plane.data[o + y * plane.stride..o + y * plane.stride + pw].copy_from_slice(&src[y * pw..y * pw + pw]);
            }
        }
        refp.extend_rows(0, h);

        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc_dsp = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let ctx = IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 26, bit_depth: 8, strong_smoothing: false, bypass: false, free_to_trim: false };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, 1);
        let (py, pc) = flat.split_at(w * h);
        let (pcb, pcr) = pc.split_at(w * h / 4);
        let mut intra_cus = 0usize;
        for cy in 0..2 {
            for cx in 0..2 {
                let d = pic.code_ctu(&ctx, &[&refp], cx, cy, py, w, pcb, pcr, w / 2);
                if matches!(d.kind, InterCuKind::UseIntra) {
                    intra_cus += 1;
                    // The marks `coding_unit` records before it parses any
                    // intra syntax, which every later derivation reads.
                    let i = pic.info.idx4(cx * 32, cy * 32);
                    assert_eq!(pic.info.pred_mode[i], 1, "an intra CU must record pred_mode 1");
                    assert_eq!(pic.info.skip[i], 0, "an intra CU is never skipped");
                    let _ = pic.code_ctu_intra(&ctx, cx, cy, py, w, pcb, pcr, w / 2);
                }
            }
        }
        assert!(intra_cus > 0, "the construction was meant to make an intra CU win in the P slice; it did not, and the round trip below would be vacuous");

        // SELF, in process: both pictures decode to the reconstructions
        // the encoder holds.
        let mut dec = crate::hevc::HevcDecoder::new();
        for u in &units {
            dec.push_annexb(&u.data).unwrap();
        }
        dec.flush().unwrap();
        for i in 0..2 {
            let decoded = dec.next_picture().unwrap_or_else(|| panic!("picture {i} missing"));
            assert_eq!(decoded.into_packed(), e.reconstructions()[i], "picture {i}: decoded bytes differ from the encoder-held reconstruction");
        }
    }

    /// An intra CU inside a P slice whose **transform tree splits** —
    /// the shape the flat-CU case above can never produce, and the one
    /// that makes the deblocker's per-TB edge derivation load-bearing:
    /// a split intra CU has interior transform-block edges, every one of
    /// them boundary strength 2, and a state builder that marked only the
    /// CU's own boundary would leave them unfiltered while both decoders
    /// filtered them.
    ///
    /// `prefer_intra` is not a flatness test in absolute terms — it asks
    /// whether a DC prediction beats the best inter one — so a *textured*
    /// CU wins it outright when the reference has nothing like it. Here
    /// the reference is noise and the CU is flat with one busy quadrant:
    /// cheap against DC, hopeless against noise, and busy enough in one
    /// corner that four quarter-size TUs beat one.
    ///
    /// Both facts are asserted before the round trip, because either one
    /// silently ceasing to hold would leave this test green and empty.
    #[test]
    fn a_split_intra_cu_inside_a_p_slice_round_trips() {
        use crate::hevc::frame::Frame;
        let (w, h) = (64usize, 64usize);
        let mut noise = vec![0u8; w * h * 3 / 2];
        let mut seed = 0x9e37u32;
        for v in noise.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 24) as u8;
        }
        // The P picture repeats the reference except in the bottom-right
        // CTU, which is flat but for its own bottom-right 16x16 quadrant.
        let mut split = noise.clone();
        for y in 32..h {
            for x in 32..w {
                split[y * w + x] = 128;
            }
        }
        for y in 48..h {
            for x in 48..w {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                split[y * w + x] = (seed >> 24) as u8;
            }
        }
        for y in 16..h / 2 {
            for x in 16..w / 2 {
                split[w * h + y * (w / 2) + x] = 128;
                split[w * h + w * h / 4 + y * (w / 2) + x] = 128;
            }
        }

        let config = Config {
            rate: super::super::RateControl::ConstantQp(30),
            gop: 8,
            ..cfg(w as u32, h as u32, ChromaFormat::Yuv420)
        };
        let mut e = H265Encoder::new(config.clone()).unwrap();
        let mut units = e.push(&noise).unwrap();
        units.extend(e.push(&split).unwrap());
        units.extend(e.flush().unwrap());
        assert_eq!(units.len(), 2);

        // Decision-level guards on the real modules, against the
        // encoder's own reconstruction of the IDR.
        let g = syn::Geometry::new(&config);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB, None))).unwrap();
        let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(30, false, true))).unwrap();
        pps.resolve_tiles(&sps).unwrap();
        let mut refp = Frame::<u8>::new(w, h, ChromaFormat::Yuv420, 8);
        refp.poc = 0;
        let rec0 = &e.reconstructions()[0];
        for (plane, (src, pw, ph)) in [&mut refp.y, &mut refp.cb, &mut refp.cr].into_iter().zip([
            (&rec0[..w * h], w, h),
            (&rec0[w * h..w * h + w * h / 4], w / 2, h / 2),
            (&rec0[w * h + w * h / 4..], w / 2, h / 2),
        ]) {
            let o = plane.origin();
            for y in 0..ph {
                plane.data[o + y * plane.stride..o + y * plane.stride + pw].copy_from_slice(&src[y * pw..y * pw + pw]);
            }
        }
        refp.extend_rows(0, h);

        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc_dsp = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let ctx = IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 30, bit_depth: 8, strong_smoothing: false, bypass: false, free_to_trim: false };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, 1);
        let (py, pc) = split.split_at(w * h);
        let (pcb, pcr) = pc.split_at(w * h / 4);
        let (mut intra_cus, mut split_cus) = (0usize, 0usize);
        for cy in 0..2 {
            for cx in 0..2 {
                let d = pic.code_ctu(&ctx, &[&refp], cx, cy, py, w, pcb, pcr, w / 2);
                if matches!(d.kind, InterCuKind::UseIntra) {
                    intra_cus += 1;
                    let id = pic.code_ctu_intra(&ctx, cx, cy, py, w, pcb, pcr, w / 2);
                    split_cus += usize::from(id.split_tu);
                }
            }
        }
        assert!(intra_cus > 0, "no CU chose intra; the round trip below would be vacuous");
        assert!(split_cus > 0, "the intra CU never split its transform, so the per-TB edge derivation stays untested");

        let mut dec = crate::hevc::HevcDecoder::new();
        for u in &units {
            dec.push_annexb(&u.data).unwrap();
        }
        dec.flush().unwrap();
        for i in 0..2 {
            let decoded = dec.next_picture().unwrap_or_else(|| panic!("picture {i} missing"));
            assert_eq!(decoded.into_packed(), e.reconstructions()[i], "picture {i}: decoded bytes differ from the encoder-held reconstruction");
        }
    }

    /// SAO end to end: the stream a decoder reads must reconstruct to the
    /// picture the encoder filtered, for an intra picture and a P picture
    /// alike.
    ///
    /// This is where the ordering invariant is actually held. SAO runs on
    /// deblocked samples and its output is the picture; the parameters are
    /// written at the START of each CTU but decided only after the whole
    /// picture has reconstructed and deblocked. Get any of that wrong —
    /// filter before deblocking, decide from unfiltered samples, write the
    /// parameters in the wrong place — and the decoder rebuilds something
    /// else. Nothing here asserts the order directly; the byte comparison
    /// does it.
    ///
    /// The guard below keeps the test honest: on content flat enough for
    /// SAO to decline every CTB, this would pass while exercising nothing,
    /// so the QP is high and the content textured, and the reconstruction
    /// must actually differ from what the same stream produces with SAO
    /// off.
    #[test]
    fn sao_round_trips_for_intra_and_p_pictures() {
        let (w, h) = (64usize, 64usize);
        let mut frames: Vec<Vec<u8>> = Vec::new();
        let mut seed = 0x5a01u32;
        for f in 0..3 {
            let mut fr = vec![0u8; w * h * 3 / 2];
            for y in 0..h {
                for x in 0..w {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let tex = (x * 7 + y * 11 + f * 3) % 160;
                    fr[y * w + x] = (30 + tex + ((seed >> 28) as usize % 12)) as u8;
                }
            }
            for c in 0..2 {
                for y in 0..h / 2 {
                    for x in 0..w / 2 {
                        fr[w * h + c * w * h / 4 + y * (w / 2) + x] = (100 + (x * 3 + y * 5 + c * 7) % 40) as u8;
                    }
                }
            }
            frames.push(fr);
        }

        for gop in [0u32, 8] {
            let base = Config { rate: super::super::RateControl::ConstantQp(40), gop, ..cfg(w as u32, h as u32, ChromaFormat::Yuv420) };
            let mut recons = Vec::new();
            for sao in [false, true] {
                let mut e = H265Encoder::new(Config { sao, ..base.clone() }).unwrap();
                let mut units = Vec::new();
                for fr in &frames {
                    units.extend(e.push(fr).unwrap());
                }
                units.extend(e.flush().unwrap());
                assert_eq!(units.len(), frames.len(), "gop={gop} sao={sao}: one access unit per picture");

                // SELF, in process, through the production decoder.
                let mut dec = crate::hevc::HevcDecoder::new();
                for u in &units {
                    dec.push_annexb(&u.data).unwrap();
                }
                dec.flush().unwrap();
                for i in 0..frames.len() {
                    let got = dec.next_picture().unwrap_or_else(|| panic!("gop={gop} sao={sao}: picture {i} missing"));
                    assert_eq!(
                        got.into_packed(),
                        e.reconstructions()[i],
                        "gop={gop} sao={sao}: picture {i} decoded differently than the encoder reconstructed it"
                    );
                }
                recons.push(e.reconstructions().to_vec());
            }
            assert_ne!(recons[0], recons[1], "gop={gop}: SAO changed nothing, so the round trip above proved nothing about it");
        }
    }

    /// `--sao` on a lossless picture refuses by name rather than shipping a
    /// filter that cannot touch a single sample.
    #[test]
    fn sao_on_a_lossless_picture_refuses_rather_than_doing_nothing() {
        let r = H265Encoder::new(Config {
            rate: super::super::RateControl::Lossless,
            gop: 0,
            sao: true,
            ..cfg(64, 64, ChromaFormat::Yuv420)
        });
        let Err(err) = r else { panic!("lossless + SAO must refuse") };
        let s = format!("{err}");
        assert!(s.contains("sample adaptive offset"), "{s}");
        assert!(s.contains("filter-exempt"), "the refusal should say why: {s}");
    }

    /// `count` pictures of detailed, moving content at `bit_depth`, packed
    /// as little-endian `u16` the way the encoder takes them above 8
    /// bits. The texture uses the whole sample range and every low bit —
    /// an 8-bit picture shifted up by two would leave the low bits zero
    /// and a depth bug that only touched them invisible.
    fn deep_frames(w: usize, h: usize, chroma: ChromaFormat, bit_depth: u32, count: usize) -> Vec<Vec<u8>> {
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let mono = chroma == ChromaFormat::Monochrome;
        let (cw, ch) = if mono { (0, 0) } else { (w / sw, h / sh) };
        let max = (1u32 << bit_depth) - 1;
        let mut seed = 0x10b1u32;
        (0..count)
            .map(|f| {
                let mut out = Vec::with_capacity(2 * (w * h + 2 * cw * ch));
                let (dx, dy) = (3 * f, f);
                let mut push = |v: u32| out.extend_from_slice(&(v.min(max) as u16).to_le_bytes());
                for y in 0..h {
                    for x in 0..w {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let tx = ((x + dx) as i32 % 25 - 12).unsigned_abs();
                        let ty = ((y + dy) as i32 % 27 - 13).unsigned_abs();
                        // A ramp over the full range plus a few low bits of noise.
                        push((max / 16) + (max / 30) * tx + (max / 40) * ty + (seed >> 29));
                    }
                }
                for c in 0..2 {
                    for y in 0..ch {
                        for x in 0..cw {
                            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                            let (sx, sy) = (x + dx / sw, y + dy / sh);
                            let r2 = ((sx as i32 % 17 - 8).abs() * (sy as i32 % 19 - 9).abs()) as u32;
                            let base = if c == 0 { max / 3 } else { max * 2 / 3 };
                            push(base + (r2.min(90) * max / 255) + (seed >> 30));
                        }
                    }
                }
                out
            })
            .collect()
    }

    /// Deep pictures code and decode: for 10 and 12 bits, every chroma
    /// format, intra / P / B, lossy and lossless — the production decoder
    /// rebuilds each picture byte for byte from the stream (SELF, in
    /// process), a lossless stream reproduces the source exactly, and the
    /// pictures really are deep.
    ///
    /// The last clause is the vacuity guard: a 10-bit source whose every
    /// sample fitted 8 bits would round-trip through an encoder that
    /// silently narrowed, so the source is built to use the whole range
    /// and the test asserts the reconstruction does too.
    #[test]
    fn deep_pictures_round_trip_through_the_decoder() {
        for bit_depth in [10u32, 12] {
            for chroma in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
                let frames = deep_frames(64, 64, chroma, bit_depth, 5);
                let max = (1u32 << bit_depth) - 1;
                let deep = |bytes: &[u8]| bytes.chunks_exact(2).any(|p| u32::from(u16::from_le_bytes([p[0], p[1]])) > 255);
                assert!(deep(&frames[0]), "{bit_depth}-bit {chroma:?}: the source never leaves 8 bits");
                assert!(frames[0].chunks_exact(2).all(|p| u32::from(u16::from_le_bytes([p[0], p[1]])) <= max));

                for (rate, bframes, sao) in [
                    (super::super::RateControl::ConstantQp(26), 0u32, false),
                    (super::super::RateControl::ConstantQp(40), 2, true),
                    (super::super::RateControl::Lossless, 2, false),
                ] {
                    let tag = format!("{bit_depth}-bit {chroma:?} {rate:?} bframes={bframes} sao={sao}");
                    let mut e = H265Encoder::new(Config { bit_depth, rate, gop: 8, bframes, sao, ..cfg(64, 64, chroma) })
                        .unwrap_or_else(|err| panic!("{tag}: {err}"));
                    assert_eq!(e.frame_bytes(), frames[0].len(), "{tag}: two bytes per sample");
                    let mut units = Vec::new();
                    for f in &frames {
                        units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                    }
                    units.extend(e.flush().unwrap());
                    assert_eq!(units.len(), frames.len(), "{tag}: one access unit per picture");
                    assert!(units[1..].iter().any(|u| !u.keyframe), "{tag}: no inter picture was coded");
                    if bframes > 0 {
                        assert!(units.iter().any(|u| u.encode_index as usize != (u.poc / 2) as usize), "{tag}: no B picture was held back");
                    }

                    // SELF, through the production decoder. It emits
                    // display order; the reconstructions are in coding
                    // order, so each decoded picture is matched to the
                    // reconstruction whose access unit carries its POC
                    // (display index `poc / 2`, as `gop.rs` counts it).
                    let mut dec = crate::hevc::HevcDecoder::new();
                    for u in &units {
                        dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: decoder rejected the stream: {err}"));
                    }
                    dec.flush().unwrap();
                    let mut by_display = vec![None; units.len()];
                    for u in &units {
                        by_display[(u.poc / 2) as usize] = Some(u.encode_index as usize);
                    }
                    for (i, coded) in by_display.iter().enumerate() {
                        let want = &e.reconstructions()[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                        let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                        assert_eq!(got.bit_depth, bit_depth, "{tag}: decoded depth");
                        assert!(got.into_packed() == *want, "{tag}: picture {i} decoded differently than the encoder reconstructed it");
                    }
                    assert!(deep(&e.reconstructions()[0]), "{tag}: the reconstruction never leaves 8 bits");
                    if rate == super::super::RateControl::Lossless {
                        for u in &units {
                            assert!(
                                e.reconstructions()[u.encode_index as usize] == frames[(u.poc / 2) as usize],
                                "{tag}: picture poc {} is not lossless",
                                u.poc
                            );
                        }
                    }
                }
            }
        }
    }

    /// A hashed noise value for a sample position — content that is busy
    /// everywhere yet the same wherever the same position is asked for,
    /// so a moving picture can carry it.
    fn hash2(x: i32, y: i32) -> u32 {
        let mut v = (x as u32).wrapping_mul(0x9e37_79b1) ^ (y as u32).wrapping_mul(0x85eb_ca77);
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^= v >> 12;
        v
    }

    /// Pictures built for the coding quadtree to take every depth. Each
    /// 32x32 region is one of four kinds, walking with its position: busy
    /// throughout (nothing for a split to isolate), flat with a busy
    /// 16x16 island in one quadrant (one split isolates it), flat with a
    /// busy 8x8 island inside a quadrant (only the second split does),
    /// and a gradient crossed by a moving edge. The busy content drifts a
    /// sample per picture so inter pictures carry residual, and chroma
    /// follows luma's structure so the chroma trees vary too. At depth
    /// above 8 every low bit is used.
    fn tree_frames(w: usize, h: usize, chroma: ChromaFormat, bit_depth: u32, count: usize) -> Vec<Vec<u8>> {
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = if chroma == ChromaFormat::Monochrome { (0, 0) } else { (w / sw, h / sh) };
        let shift = bit_depth - 8;
        let luma = |x: usize, y: usize, f: usize| -> u32 {
            let (lx, ly) = (x % 32, y % 32);
            let busy = || 40 + hash2(x as i32 - f as i32, y as i32) % 150;
            let v = match ((x / 32) + 2 * (y / 32)) % 4 {
                0 => busy(),
                1 if lx >= 16 && ly >= 16 => busy(),
                1 => 120,
                2 if (24..32).contains(&lx) && (8..16).contains(&ly) => busy(),
                2 => 90,
                _ => 40 + (x as u32 + y as u32 / 2) + if (x + 2 * f) % 32 < 12 { 60 } else { 0 },
            };
            v.min(255)
        };
        (0..count)
            .map(|f| {
                let mut samples: Vec<u32> = Vec::with_capacity(w * h + 2 * cw * ch);
                let low = |x: usize, y: usize, c: u32| hash2(x as i32 + 7 * c as i32, y as i32 + f as i32) & ((1u32 << shift) - 1);
                for y in 0..h {
                    for x in 0..w {
                        samples.push((luma(x, y, f) << shift) | low(x, y, 0));
                    }
                }
                for c in 1..=2u32 {
                    for y in 0..ch {
                        for x in 0..cw {
                            let base = 96 + (luma(x * sw, y * sh, f) >> 3) + 8 * c;
                            samples.push((base.min(255) << shift) | low(x, y, c));
                        }
                    }
                }
                if shift == 0 {
                    samples.iter().map(|&v| v as u8).collect()
                } else {
                    samples.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
                }
            })
            .collect()
    }

    /// Partial edge CTBs round-trip. Where the picture is not whole 32x32
    /// CTBs (and not below 64 both ways, see `Geometry::new`) the coded
    /// picture is the smallest legal one and the CTBs along the right and
    /// bottom edges are partial, their splits inferred rather than coded:
    /// 1280x720 leaves a bottom row of 16, 1366x768 a right column of 24
    /// behind a conformance window of 2, and the small sizes take the other
    /// remainders (8, 16, 24) in each direction, intra, P and B, through the
    /// production decoder.
    #[test]
    fn partial_edge_ctbs_round_trip() {
        for (w, h, gop, bframes, frames) in [
            (1280usize, 720usize, 8u32, 0u32, 2usize),
            (1366, 768, 8, 0, 2),
            (88, 44, 8, 2, 4),
            (72, 56, 8, 2, 4),
            (80, 34, 0, 0, 2),
        ] {
            let tag = format!("{w}x{h} gop={gop} bframes={bframes}");
            let config = Config { gop, bframes, ..cfg(w as u32, h as u32, ChromaFormat::Yuv420) };
            let g = syn::Geometry::new(&config);
            assert_eq!(g.log2_ctb, 5, "{tag}");
            assert!(!g.coded_width.is_multiple_of(32) || !g.coded_height.is_multiple_of(32), "{tag}: no partial CTB to test");
            let frames = tree_frames(w, h, ChromaFormat::Yuv420, 8, frames);
            let mut e = H265Encoder::new(config).unwrap_or_else(|err| panic!("{tag}: {err}"));
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
            }
            units.extend(e.flush().unwrap());
            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &units {
                dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: the decoder rejected the stream: {err}"));
            }
            dec.flush().unwrap();
            let mut by_display = vec![None; units.len()];
            for u in &units {
                by_display[u.display as usize] = Some(u.encode_index as usize);
            }
            for (i, coded) in by_display.iter().enumerate() {
                let want = &e.reconstructions()[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                assert!(got.into_packed() == *want, "{tag}: picture {i} decoded differently than the encoder reconstructed it");
            }
        }
    }

    /// Adaptive quantisation's group follows the stream: the CTB in an
    /// all-intra stream and wherever every CTB is one unit, half the CTB
    /// where the coding quadtree codes pictures that others predict from —
    /// the measured split recorded in `Core::new` — and the quantiser moves
    /// under each.
    #[test]
    fn adaptive_quantisation_groups_follow_the_stream() {
        for (gop, max_cu_depth, want) in [(0u32, None, 0u32), (8, None, 1), (8, Some(1), 1), (0, Some(0), 0), (8, Some(0), 0)] {
            let tag = format!("gop {gop} max_cu_depth {max_cu_depth:?}");
            let frames = tree_frames(64, 64, ChromaFormat::Yuv420, 8, 3);
            let config = Config { gop, aq_strength: 2.0, max_cu_depth, ..cfg(64, 64, ChromaFormat::Yuv420) };
            let mut e = H265Encoder::new(config).unwrap_or_else(|err| panic!("{tag}: {err}"));
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
            }
            units.extend(e.flush().unwrap());
            let depths: Vec<u32> = units
                .iter()
                .flat_map(|u| crate::nal::annexb_nals(&u.data))
                .filter(|nal| (nal[0] >> 1) & 0x3f == syn::NAL_PPS)
                .map(|nal| crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&nal[2..])).unwrap().diff_cu_qp_delta_depth)
                .collect();
            assert!(!depths.is_empty(), "{tag}: no PPS in the stream");
            assert!(depths.iter().all(|&d| d == want), "{tag}: diff_cu_qp_delta_depth {depths:?}, want {want}");
            let moved: u64 = e.census().by_kind.iter().map(|k| k.qp_moved).sum();
            assert!(moved > 0, "{tag}: no unit left the picture quantiser");
        }
    }

    /// The coding quadtree codes and round-trips — intra, P and B, every
    /// chroma format, 10 bits, adaptive quantisation over quantisation
    /// groups smaller than the CTB, lossless, and a 16x16-CTB picture —
    /// and takes both split depths in every picture kind: a round trip
    /// over units that never split would prove only the unsplit syntax it
    /// already had.
    ///
    /// The model check rides along: what the split decisions priced the
    /// coded units at must be within a factor of two of the slice data
    /// the pictures actually took.
    #[test]
    fn the_coding_quadtree_takes_every_depth_and_round_trips() {
        use super::super::RateControl::{ConstantQp, Lossless};
        let mut total = [KindCensus::default(); 3];
        // PART_NxN units per chroma format, by ChromaArrayType.
        let mut nxn_by_format = [0u64; 4];
        for (w, h, chroma, bit_depth, gop, bframes, rate, aq_strength, max_cu_depth) in [
            (64usize, 64usize, ChromaFormat::Yuv420, 8u32, 0u32, 0u32, ConstantQp(30), 0.0f32, 2u32),
            (64, 64, ChromaFormat::Yuv420, 8, 8, 2, ConstantQp(30), 0.0, 2),
            (64, 64, ChromaFormat::Yuv420, 8, 8, 2, ConstantQp(30), 0.0, 1),
            (64, 64, ChromaFormat::Yuv422, 8, 8, 0, ConstantQp(40), 0.0, 2),
            (64, 64, ChromaFormat::Yuv422, 8, 0, 0, ConstantQp(26), 0.0, 2),
            (64, 64, ChromaFormat::Yuv444, 8, 8, 2, ConstantQp(30), 0.0, 2),
            (64, 64, ChromaFormat::Monochrome, 8, 8, 0, ConstantQp(30), 0.0, 2),
            (64, 64, ChromaFormat::Yuv420, 10, 8, 2, ConstantQp(30), 0.0, 2),
            (64, 64, ChromaFormat::Yuv420, 8, 8, 2, ConstantQp(30), 2.0, 2),
            (64, 64, ChromaFormat::Yuv420, 8, 0, 0, ConstantQp(30), 2.0, 2),
            (64, 64, ChromaFormat::Yuv444, 8, 8, 0, ConstantQp(26), 2.0, 1),
            (64, 64, ChromaFormat::Yuv420, 8, 8, 2, Lossless, 0.0, 2),
            (48, 40, ChromaFormat::Yuv420, 8, 8, 0, ConstantQp(30), 2.0, 2),
        ] {
            let tag = format!("{w}x{h} {chroma:?} {bit_depth}-bit gop={gop} bframes={bframes} {rate:?} aq={aq_strength} depth={max_cu_depth}");
            let frames = tree_frames(w, h, chroma, bit_depth, 6);
            let config = Config { gop, bframes, bit_depth, rate, aq_strength, max_cu_depth: Some(max_cu_depth), ..cfg(w as u32, h as u32, chroma) };
            let mut e = H265Encoder::new(config.clone()).unwrap_or_else(|err| panic!("{tag}: {err}"));
            let mut units = Vec::new();
            for f in &frames {
                units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
            }
            units.extend(e.flush().unwrap());
            assert_eq!(units.len(), frames.len(), "{tag}");
            let census = *e.census();

            // SELF, through the production decoder, display order against
            // coding order through each access unit's stream-wide display
            // index — not `poc / 2`, which an all-intra stream resets at
            // every picture (each is an IDR at POC 0).
            let mut dec = crate::hevc::HevcDecoder::new();
            for u in &units {
                dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: the decoder rejected the stream: {err}"));
            }
            dec.flush().unwrap();
            let mut by_display = vec![None; units.len()];
            for u in &units {
                by_display[u.display as usize] = Some(u.encode_index as usize);
            }
            for (i, coded) in by_display.iter().enumerate() {
                let want = &e.reconstructions()[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                assert!(got.into_packed() == *want, "{tag}: picture {i} decoded differently than the encoder reconstructed it");
                if rate == Lossless {
                    assert!(*want == frames[i], "{tag}: picture {i} is not lossless");
                }
            }

            if aq_strength > 0.0 {
                assert!(census.by_kind.iter().map(|k| k.qp_delta).sum::<u64>() > 0, "{tag}: no unit coded a cu_qp_delta");
                assert!(census.by_kind.iter().map(|k| k.qp_moved).sum::<u64>() > 0, "{tag}: no unit left the picture quantiser");
            }
            let g = syn::Geometry::new(&config);
            if g.log2_ctb == 4 || max_cu_depth == 1 {
                assert!(census.by_kind.iter().all(|k| k.depth2 == 0), "{tag}: split twice where one split reaches the limit: {census:?}");
                assert!(census.by_kind.iter().any(|k| k.depth1 > 0), "{tag}: never split at all: {census:?}");
            }
            for (t, k) in total.iter_mut().zip(census.by_kind.iter()) {
                t.add(k);
            }
            let cat = match chroma {
                ChromaFormat::Monochrome => 0,
                ChromaFormat::Yuv420 => 1,
                ChromaFormat::Yuv422 => 2,
                ChromaFormat::Yuv444 => 3,
            };
            nxn_by_format[cat] += census.by_kind.iter().map(|k| k.nxn).sum::<u64>();
        }
        assert!(nxn_by_format.iter().all(|&n| n > 0), "PART_NxN was not taken in every chroma format (by ChromaArrayType): {nxn_by_format:?}");
        assert!(total[0].nxn > 0 && total[1].nxn + total[2].nxn > 0, "PART_NxN never taken in an I picture or never inside a P/B one: {total:?}");
        for (slot, name) in [(0usize, "I"), (1, "P"), (2, "B")] {
            let t = &total[slot];
            assert!(t.depth1 > 0 && t.depth2 > 0, "{name} pictures never took both split depths: {t:?}");
            let (m, c) = (t.model_bits as f64, t.coded_bits as f64);
            assert!(m > 0.5 * c && m < 2.0 * c, "{name} pictures: the split decisions priced {m} bits for {c} coded");
        }

        // A tree changes the stream, and the default is the depth-2 tree.
        let frames = tree_frames(64, 64, ChromaFormat::Yuv420, 8, 3);
        let encode = |max_cu_depth: Option<u32>| -> Vec<u8> {
            let mut e = H265Encoder::new(Config { gop: 8, max_cu_depth, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
            let mut out = Vec::new();
            for f in &frames {
                for u in e.push(f).unwrap() {
                    out.extend_from_slice(&u.data);
                }
            }
            for u in e.flush().unwrap() {
                out.extend_from_slice(&u.data);
            }
            out
        };
        assert_eq!(Config::default().max_cu_depth, None, "the default asks nothing, and the encoder's default answers");
        assert_eq!(encode(None), encode(Some(DEFAULT_CU_DEPTH)), "an unset depth must code the default depth");
        assert_eq!(DEFAULT_CU_DEPTH, 2);
        assert_ne!(encode(Some(0)), encode(Some(2)), "a depth-2 tree changed nothing on content built to split");

        // Deeper than the minimum coding block allows at any CTB is refused
        // by name rather than clamped.
        let err = H265Encoder::new(Config { max_cu_depth: Some(3), ..cfg(64, 64, ChromaFormat::Yuv420) }).err().expect("depth 3 must refuse");
        assert!(format!("{err}").contains("max_cu_depth"), "{err}");
    }

    /// The coding quadtree is H.265's: the H.264 encoder refuses a depth
    /// asked for on purpose by name, and takes the default (`None`) and an
    /// explicit `Some(0)` — which is what keeps every H.264 caller building
    /// its configuration from `Config::default()` working after the H.265
    /// default became a tree.
    #[test]
    fn the_h264_encoder_refuses_a_quadtree_depth_by_name() {
        for depth in [1u32, 2] {
            let err = crate::encode::h264::H264Encoder::new(Config { max_cu_depth: Some(depth), ..cfg(64, 64, ChromaFormat::Yuv420) })
                .err()
                .unwrap_or_else(|| panic!("H.264 accepted max_cu_depth Some({depth})"));
            let msg = format!("{err}");
            assert!(msg.contains("max_cu_depth") && msg.contains("H.264"), "{msg}");
        }
        for ok in [None, Some(0)] {
            assert!(crate::encode::h264::H264Encoder::new(Config { max_cu_depth: ok, ..cfg(64, 64, ChromaFormat::Yuv420) }).is_ok(), "H.264 refused {ok:?}");
        }
    }

    /// A source sample above the declared depth is refused by name, not
    /// coded: nothing downstream checks the range, and a wrapped sample
    /// would be a desync far from its cause.
    #[test]
    fn a_sample_above_the_declared_depth_refuses() {
        let mut e = H265Encoder::new(Config { bit_depth: 10, gop: 0, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
        let mut frame = vec![0u8; e.frame_bytes()];
        frame[..2].copy_from_slice(&1024u16.to_le_bytes());
        let err = e.push(&frame).expect_err("1024 does not fit 10 bits");
        assert!(format!("{err}").contains("exceeds the declared 10-bit depth"), "{err}");
    }

    /// Fifteen bits and up stay refused: `Config::validate` bounds the
    /// depth to what the decoders and the 16-bit transform path admit.
    #[test]
    fn deeper_than_fourteen_bits_refuses() {
        let Err(err) = H265Encoder::new(Config { bit_depth: 16, ..cfg(64, 64, ChromaFormat::Yuv420) }) else {
            panic!("16-bit was accepted")
        };
        assert!(format!("{err}").contains("bit depth outside 8..=14"), "{err}");
    }
}
