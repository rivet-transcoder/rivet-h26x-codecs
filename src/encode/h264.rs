//! The H.264 encoder.
//!
//! Mirrors [`crate::h264::H264Decoder`]: pictures in, access units out. The
//! reconstruction loop is a decoder — it runs the same conformance-proven
//! inverse transform, prediction and deblocking the decoder does — which is
//! what makes the SELF property in `tools/verify_encode.sh` achievable rather
//! than aspirational, and it is why this module reaches into `crate::h264`
//! rather than reimplementing anything it can borrow.
//!
//! # State of it
//!
//! Configuration, picture typing and coding order, the access-unit envelope,
//! and real compression on every picture type through both entropy coders:
//! intra, P and B pictures go through prediction, transform, quantisation
//! and the loop filter, decided once in the shared walks of
//! [`super::h264_pic`] and spelled by the CAVLC
//! ([`super::h264_cavlc_mb`]) or CABAC ([`super::h264_cabac_mb`]) writers.
//! What still codes as `I_PCM` does so for a stated reason: lossless,
//! because PCM *is* the exact mode and the transform path is lossy — and
//! the inter pictures of a lossless stream are all-skip, because a
//! lossless inter picture has no exact spelling but PCM either.
//!
//! # Sample depth
//!
//! Eight to fourteen bits, the decoder's own range. [`H264Encoder`] is a
//! thin face over a private `Core<S>` instantiated at the sample width the
//! depth needs — `u8` at 8 bits, `u16` above — exactly the split
//! [`super::h265::H265Encoder`] makes and the decoder makes on reading the
//! SPS. Everything below the face is generic: the decision walks, the
//! predictors and kernels (`H264Dsp<S>`, the decoder's), the writers. What
//! the depth changes is spelled in one place each — `QP'` for the scaling
//! tables ([`super::h264_intra::IntraCtx::qp_prime`]), the PCM sample
//! width, the SPS depth fields and profile, and the loop filter's
//! thresholds (which `deblock_mb_rows` scales itself from the frame's
//! depth) — and an 8-bit stream is byte for byte what it was before any
//! of it existed. What the depth deliberately does *not* change is the
//! mode-decision multiplier: `satd_lambda` in [`super::h264_intra`]
//! records the measurement that decided it.

use super::gop::{Coded, Kind, Scheduler};
use super::rc::{PicKind, RateController};
use super::h264_syntax as syn;
use super::h265_wp;
use crate::h264::slice::{PredWeightTable, WeightEntry};
use super::{Access, Config, Entropy, FieldCoding, FieldOrder, RateControl};
use crate::bitwriter::BitWriter;
use crate::h264::dpb::{DecodedPic, Dpb, PocState, RefMark};
use crate::h264::frame::{BlockMotion, Frame, PARITY_FRAME, SharedFrame};
use crate::sample::Sample;
use crate::{Error, Result};
use std::sync::Arc;

/// H.264 encoder. See the module documentation for what is and is not built.
///
/// A thin face over the private `Core<S>`, instantiated at the sample
/// width the configuration's bit depth needs. Pictures cross this face as
/// bytes in both directions, in the layout [`crate::Picture::into_packed`]
/// uses: one byte per sample at 8 bits, little-endian `u16` pairs deeper —
/// so a source picture and the decoder's output of it compare byte for
/// byte at every depth, which is what the SELF check reads.
/// [`H264Encoder::frame_bytes`] says how many.
pub struct H264Encoder {
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
    /// The rate controller, when the configuration asked for a bitrate.
    /// The *same* controller H.265 drives — see [`super::rc`] for why
    /// essentially all of it turned out to be codec-agnostic.
    rc: Option<RateController>,
    /// Bytes emitted so far, to hold the controller's ledger to.
    emitted: u64,
    /// Source pictures held in display order, indexed by display position, so
    /// that a B picture held back by the scheduler still has its samples when
    /// its anchor arrives — already unpacked from the caller's bytes into
    /// samples, so the coding paths never see a byte layout.
    held: std::collections::BTreeMap<u64, Vec<S>>,
    /// Reconstructions, in coding order, for the SELF check — packed as
    /// the source pictures were handed in.
    recon: Vec<Vec<u8>>,
    frame_bytes: usize,
    /// Display index of the next picture offered. Counted here rather than
    /// inferred from `held`, because `held` empties as pictures are coded —
    /// with every picture an IDR it is empty on every call, and inferring
    /// would hand picture two the index of picture one.
    next_display: u64,
    geom: syn::Geometry,
    /// Reference pictures, at *coded* size, newest last. Kept because inter
    /// prediction reads reconstructed samples rather than source ones — that
    /// identity is what SELF checks, and predicting from the source instead
    /// is the classic way to make an encoder only its author can decode.
    ///
    /// A B picture needs one reference on each side of it in display order,
    /// so this holds several and picks by POC rather than keeping only the
    /// most recent.
    ///
    /// Beside each picture's planes, its motion in the decoder's own
    /// layout ([`super::h264_pic::PicMotion`]): per-4x4 `BlockMotion` and
    /// the per-macroblock `MbInfo`. That is what a later B picture's
    /// spatial direct derivation reads as colocated motion, at whatever
    /// granularity 8.4.1.2.1 asks for — which is why it is stored whole
    /// rather than summarised.
    refs: Vec<(i32, Vec<syn::Recon<S>>, super::h264_pic::PicMotion)>,
    /// `frame_num`, which counts *reference* pictures and wraps.
    frame_num: u32,
    idr_pic_id: u32,
    /// Plane sizes of one source picture, derived once.
    plane_dims: Vec<(u32, u32)>,
    /// Kernels and derived tables for the transform intra path, built once.
    tools: super::h264_pic::IntraTools<S>,
    /// The quantiser the PPS declares, for the whole stream. Every slice
    /// carries its own as a delta against it — see `code_picture` for why
    /// the PPS must not follow the picture.
    pps_qp: u8,
    /// What shapes the pictures took — see [`ShapeCensus`].
    census: ShapeCensus,
    /// The coded picture buffer this stream declares, when it declares
    /// one: the *declared* values, snapped to what the syntax carries, so
    /// what the controller aims at and what the stream promises are one
    /// number. `None` writes no VUI and no SEI at all.
    cpb: Option<syn::Cpb>,
    /// Coding index of the last access unit that carried a buffering
    /// period — every `cpb_removal_delay` counts clock ticks from its
    /// removal.
    last_bp_encode: u64,
    /// How many extra codings the declared buffer cost — pictures that
    /// came out too large for it and were coded again at a higher
    /// quantiser. Reported rather than hidden, as on the H.265 side.
    recoded: u64,
    /// The interlaced state, when the configuration asked for interlaced
    /// coding: `None` for a progressive stream, whose every path above is
    /// what it was before interlacing existed.
    fields: Option<Fields<S>>,
}

/// What an interlaced encoder keeps between frames: the decoder's own
/// reference model, and the reconstructed fields that model names.
///
/// Field reference lists are where an encoder that reasons its way to the
/// answer goes wrong: a P field's list alternates parities through the
/// frames ordered by `FrameNumWrap` (8.2.4.2.5), the second field of a
/// frame may name the first, and whether it does depends on what the
/// sliding window unmarked after that first field. So nothing here derives
/// a list. Each field's slice header is written, read back through the
/// production parser, and handed to the decoder's `compute_poc`,
/// `build_ref_lists` and `Dpb::store` — the functions the conformance
/// suite's field streams run through — and the encoder predicts from
/// whichever reconstructed field index 0 of those lists names.
struct Fields<S: Sample> {
    /// The field coded, and displayed, first.
    order: FieldOrder,
    /// The SPS every access unit carries, as the decoder parses it.
    sps: crate::h264::Sps,
    /// The PPS likewise.
    pps: crate::h264::Pps,
    /// The decoder's DPB and POC state after the last committed frame.
    model: RefModel,
    /// Reconstructed reference frames, by the id of the `SharedFrame` the
    /// model's entry for them carries.
    stored: Vec<StoredFrame<S>>,
}

/// The decoder-side reference state an interlaced encoder runs. Forked
/// into each attempt ([`RefModel::fork`]) and kept only by the attempt that
/// is committed, so a re-coded frame leaves no trace in it.
struct RefModel {
    /// The decoded picture buffer, its entries plane-less stand-ins.
    dpb: Dpb<u8>,
    /// POC bookkeeping across pictures.
    poc: PocState,
    /// The id the next frame's `SharedFrame` takes.
    next_id: u64,
}

impl RefModel {
    /// A copy to run an attempt against: the same entries (the stand-in
    /// frames shared, which is what the DPB compares), the same marking and
    /// POC state, no pending output.
    fn fork(&self) -> RefModel {
        let d = &self.dpb;
        let mut dpb = Dpb::new();
        dpb.pics = d
            .pics
            .iter()
            .map(|p| DecodedPic {
                frame: p.frame.clone(),
                poc: p.poc,
                field_poc: p.field_poc,
                fields: p.fields,
                frame_num: p.frame_num,
                frame_num_wrap: p.frame_num_wrap,
                long_term_frame_idx: p.long_term_frame_idx,
                mark: p.mark,
                needed_for_output: p.needed_for_output,
                awaiting_field: p.awaiting_field,
                non_existing: p.non_existing,
                decode_index: p.decode_index,
            })
            .collect();
        dpb.capacity = d.capacity;
        dpb.num_reorder = d.num_reorder;
        dpb.max_long_term_frame_idx = d.max_long_term_frame_idx;
        dpb.crop = d.crop;
        RefModel { dpb, poc: self.poc.clone(), next_id: self.next_id }
    }
}

/// One reconstructed frame of an interlaced stream, field by field.
struct StoredFrame<S: Sample> {
    /// The id of the model's stand-in for it.
    id: u64,
    /// Top and bottom field, once coded — or, for a frame coded as a frame
    /// picture, extracted from it ([`extract_field`]).
    fields: [Option<StoredField<S>>; 2],
    /// The whole frame at coded size, borders replicated — a frame
    /// picture's reconstruction, or a field pair's interleaved
    /// ([`interleave_fields`]): what a later frame picture predicts from.
    frame: Vec<syn::Recon<S>>,
    /// The frame's motion in the decoder's frame-row layout, once both
    /// fields are coded ([`field_pair_motion`]) — what a later B picture's
    /// colocated derivation reads.
    col: Frame<u8>,
}

/// A field-coded frame's motion in the decoder's frame-row layout: each
/// field's macroblock rows at frame rows `2r + parity`, flagged field
/// macroblocks, the frame marked field-coded with its fields' POCs — built
/// with the decoder's own `take_field_motion_row`, which is how the decoder
/// lays out the colocated frame its direct derivation reads.
fn field_pair_motion<S: Sample>(frame: &StoredFrame<S>, g: &syn::Geometry, field_poc: [i32; 2]) -> Frame<u8> {
    let (mbw, mbh) = (g.mbs_wide as usize, g.mbs_high as usize);
    let n = mbw * mbh;
    let mut col = Frame::<u8>::empty();
    col.mb_width = mbw;
    col.mb_height = mbh;
    col.motion = [vec![BlockMotion::default(); n * 16], vec![BlockMotion::default(); n * 16]];
    col.mb_intra = vec![false; n];
    col.mb_field = vec![false; n];
    col.field_coded = true;
    col.field_poc = field_poc;
    for (p, field) in frame.fields.iter().enumerate() {
        let field = field.as_ref().expect("both fields are coded before the frame's motion is laid out");
        for r in 0..mbh / 2 {
            col.take_field_motion_row(&field.motion.frame, r, p);
        }
    }
    col
}

/// One reconstructed field: its planes at field size, borders replicated,
/// and its motion in the decoder's layout — what a later B field reads as
/// colocated motion.
struct StoredField<S: Sample> {
    planes: Vec<syn::Recon<S>>,
    motion: super::h264_pic::PicMotion,
}

/// What a field attempt leaves for `commit`: the forked model after both
/// fields, the frame it coded, and each field's kind and macroblock record
/// for the census.
struct FieldsOut<S: Sample> {
    model: RefModel,
    frame: StoredFrame<S>,
    census: Vec<(Kind, Vec<crate::h264::mb::MbInfo>)>,
    /// The frame was coded as one frame picture rather than two fields.
    frame_coded: bool,
}

impl<S: Sample> Fields<S> {
    /// A slice header as the decoder reads it: written alone, closed, and
    /// parsed by the production parser against this stream's own parameter
    /// sets.
    fn read_back(&self, header: &syn::SliceHeader, pps_qp: u8, nal_type: u8, nal_ref_idc: u8) -> Result<crate::h264::SliceHeader> {
        let mut hw = BitWriter::new();
        syn::write_slice_header(header, pps_qp, &mut hw);
        hw.rbsp_trailing_bits();
        let nal = syn::annexb(nal_type, nal_ref_idc, &hw.into_nal());
        let nh = crate::nal::H264NalHeader::parse(&nal[4..])
            .ok_or_else(|| Error::bitstream("H.264 encode: a picture's own NAL header does not parse"))?;
        let rbsp = crate::nal::unescape_rbsp(&nal[4..]);
        let (h, _, _) =
            crate::h264::SliceHeader::parse(&rbsp, nh, &|_id: u32| Some(self.pps.clone()), &|_id: u32| Some(self.sps.clone()))?;
        Ok(h)
    }
}

/// The decoder's picture end (`finish_picture`, src/h264/decoder.rs) on
/// the encoder's reference model: the `frame_num` bookkeeping, then the
/// picture stored and marked by `Dpb::store` — a field into the entry it
/// shares with its pair, a frame whole. `parity` is 0 / 1 for a field,
/// `PARITY_FRAME` for a frame; `field_poc` is what the decoder records.
#[allow(clippy::too_many_arguments)]
fn model_store(
    model: &mut RefModel,
    sps: &crate::h264::Sps,
    hdr: &crate::h264::SliceHeader,
    shared: &Arc<SharedFrame<u8>>,
    frame_num: u32,
    parity: u8,
    poc: i32,
    field_poc: [i32; 2],
    decode_index: u64,
) -> Result<()> {
    model.poc.prev_frame_num = frame_num;
    model.poc.prev_had_mmco5 = false;
    if hdr.is_reference() {
        model.poc.prev_ref_frame_num = frame_num;
    }
    let pic = DecodedPic {
        frame: shared.clone(),
        poc,
        field_poc,
        fields: if parity == PARITY_FRAME { 3 } else { 1 << parity },
        frame_num,
        frame_num_wrap: frame_num as i32,
        long_term_frame_idx: 0,
        mark: [RefMark::Unused; 2],
        needed_for_output: true,
        awaiting_field: false,
        non_existing: false,
        decode_index,
    };
    model.dpb.store(pic, hdr, sps, false, parity)?;
    model.dpb.output.clear();
    Ok(())
}

/// The frame a field pair makes: each plane's rows interleaved — the top
/// field's on even rows, the bottom's on odd — at the frame's coded size,
/// borders replicated. A decoder's frame is exactly this, which is why a
/// frame picture may predict from it.
fn interleave_fields<S: Sample>(fields: &[Option<StoredField<S>>; 2]) -> Vec<syn::Recon<S>> {
    let top = &fields[0].as_ref().expect("both fields are coded").planes;
    let bottom = &fields[1].as_ref().expect("both fields are coded").planes;
    top.iter()
        .zip(bottom)
        .map(|(t, b)| {
            let mut p = syn::recon_plane(t.width as u32, (t.height * 2) as u32, t.pad);
            for y in 0..p.height {
                let src = if y % 2 == 0 { t } else { b };
                let s = src.offset(0, (y / 2) as isize);
                let d = p.offset(0, y as isize);
                p.data[d..d + p.width].copy_from_slice(&src.data[s..s + src.width]);
            }
            p.extend_edges(false);
            p
        })
        .collect()
}

/// Field `parity` of a frame's planes, as planes of half the height with
/// borders of their own — what a field picture predicts from when the
/// frame its list names was coded as a frame picture.
fn extract_field<S: Sample>(frame: &[syn::Recon<S>], parity: usize) -> Vec<syn::Recon<S>> {
    frame
        .iter()
        .map(|f| {
            let mut p = syn::recon_plane(f.width as u32, (f.height / 2) as u32, f.pad);
            for r in 0..p.height {
                let s = f.offset(0, (2 * r + parity) as isize);
                let d = p.offset(0, r as isize);
                p.data[d..d + p.width].copy_from_slice(&f.data[s..s + f.width]);
            }
            p.extend_edges(false);
            p
        })
        .collect()
}

/// Sum of squared differences between source samples and a packed
/// reconstruction of the same picture (one byte per sample at 8 bits,
/// little-endian pairs deeper).
fn ssd_packed<S: Sample>(src: &[S], rec: &[u8]) -> u64 {
    src.iter()
        .enumerate()
        .map(|(i, s)| {
            let r = if S::BYTES == 1 { i64::from(rec[i]) } else { i64::from(u16::from_le_bytes([rec[2 * i], rec[2 * i + 1]])) };
            let d = i64::from(s.to_i32()) - r;
            (d * d) as u64
        })
        .sum()
}

/// One coded picture, before anything about it has been kept — the
/// H.265 side's own pattern, for the same reason: a picture that will
/// not fit the declared buffer is coded again, and the attempt that lost
/// must leave no trace — not in the reconstructions the SELF check reads,
/// and above all not as the reference the next picture predicts from.
///
/// `code_attempt` takes `&self` and so *cannot* write: every per-picture
/// write lives in `Core::commit`, which runs only for an attempt that is
/// kept. That is the claim the byte-identity of every non-buffer stream
/// rests on, enforced by the borrow checker rather than by care.
struct Attempt<S: Sample> {
    access: Access,
    /// The reconstruction, cropped to display size and packed to bytes.
    rec: Vec<u8>,
    /// The reconstruction at coded size, borders not yet replicated.
    recon: Vec<syn::Recon<S>>,
    /// The picture's motion in the decoder's layout.
    motion: super::h264_pic::PicMotion,
    /// An interlaced frame's two fields, when it was coded as field
    /// pictures (`recon` and `motion` are then empty).
    fields: Option<FieldsOut<S>>,
}

/// How many macroblocks of each kind the stream's pictures took, per
/// picture type — read off each coded picture's own `MbInfo` record
/// (what the loop filter and the next picture's direct derivation read),
/// so it counts what was committed and nothing the decision walks
/// tried.
///
/// A configuration row can switch a shape *on*; only the content decides
/// whether anything *takes* it, and a gate cell whose new shape was never
/// chosen proves the syntax and nothing else. This is how that is told
/// apart from a cell that exercised the shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShapeCensus {
    /// `[picture type][macroblock kind]`: picture types 0 intra, 1 P, 2 B;
    /// kinds in [`ShapeCensus::KINDS`]' order.
    pub counts: [[u64; 11]; 3],
    /// Pictures coded, per picture type.
    pub pictures: [u64; 3],
    /// Macroblocks whose `QP_Y` differs from their picture's quantiser, per
    /// picture type — what proves adaptive quantisation moved something
    /// rather than coding zero deltas.
    pub qp_moved: [u64; 3],
    /// Macroblocks that coded a non-zero `mb_qp_delta`, per picture type.
    pub qp_delta: [u64; 3],
    /// Pictures with at least one non-zero `mb_qp_delta`, per picture type.
    pub qp_delta_pictures: [u64; 3],
    /// Pictures whose `pred_weight_table` weights something — weighted
    /// prediction chosen, not merely enabled — per picture type (only P
    /// pictures carry a table).
    pub wp_on: [u64; 3],
    /// Under a luma weighting, inter macroblocks whose luma SATD at the
    /// vectors chosen was lower weighted than plain: the fit's model check
    /// holding.
    pub wp_won: [u64; 3],
    /// The same, higher weighted than plain: the fit's model check failing.
    pub wp_lost: [u64; 3],
    /// Field pictures coded — two per frame of an interlaced stream coded
    /// as fields, and what proves a field row coded fields rather than
    /// declaring an interlaced sequence over frames.
    pub field_pictures: u64,
    /// Frame pictures of an interlaced stream: frames picture-adaptive
    /// coding chose to code whole. With [`ShapeCensus::field_pictures`],
    /// what proves the decision chose both ways.
    pub frame_pictures: u64,
}

impl ShapeCensus {
    /// Every macroblock kind, in the order `counts` is indexed.
    pub const KINDS: [crate::h264::mb::MbKind; 11] = {
        use crate::h264::mb::MbKind::*;
        [I4x4, I8x8, I16x16, IPcm, Inter16x16, Inter16x8, Inter8x16, Inter8x8, PSkip, BSkip, BDirect16x16]
    };

    /// Count one coded picture's macroblocks, coded at picture quantiser
    /// `pic_qp`: their kinds, and — read off the same committed `MbInfo`
    /// the loop filter reads — how many left that quantiser and how many
    /// coded a delta to do it.
    fn add(&mut self, kind: Kind, mbs: &[crate::h264::mb::MbInfo], pic_qp: u8, weighting: super::h264_pic::WeightCensus) {
        let pic = match kind {
            Kind::Idr | Kind::I => 0,
            Kind::P => 1,
            Kind::B => 2,
        };
        self.pictures[pic] += 1;
        let mut deltas = 0u64;
        for m in mbs {
            let k = Self::KINDS.iter().position(|&k| k == m.kind).expect("every kind is listed");
            self.counts[pic][k] += 1;
            self.qp_moved[pic] += u64::from(i32::from(m.qp) != i32::from(pic_qp));
            deltas += u64::from(m.qp_delta_nonzero);
        }
        self.qp_delta[pic] += deltas;
        self.qp_delta_pictures[pic] += u64::from(deltas > 0);
        self.wp_on[pic] += u64::from(weighting.on);
        self.wp_won[pic] += weighting.won;
        self.wp_lost[pic] += weighting.lost;
    }

    /// The kinds that occurred in pictures of type `pic` (0 intra, 1 P,
    /// 2 B), as `(name, count)`, for reporting.
    pub fn taken(&self, pic: usize) -> Vec<(String, u64)> {
        Self::KINDS
            .iter()
            .zip(&self.counts[pic])
            .filter(|&(_, &n)| n != 0)
            .map(|(k, &n)| (format!("{k:?}"), n))
            .collect()
    }
}

/// The exponents the SPS declares. Fixed rather than derived: 16 bits of
/// `frame_num` and of POC LSB is more than any GOP this encoder produces
/// needs, and a wrap that never happens is a class of bug that never happens.
const LOG2_MAX_FRAME_NUM: u32 = 16;
const LOG2_MAX_POC_LSB: u32 = 16;

impl H264Encoder {
    /// Fails rather than starting if the configuration cannot produce a legal
    /// stream — an encoder that fails late has usually already emitted a
    /// header describing something it then cannot deliver.
    ///
    /// The bit depth picks the sample width once, here: 8 bits codes in
    /// `u8`, anything deeper in `u16`, exactly as [`crate::h264::H264Decoder`]
    /// chooses on reading the SPS. `Config::validate` bounds the depth to
    /// the 8..=14 both decoders admit.
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        let inner = if cfg.bit_depth > 8 { Inner::Wide(Core::new(cfg)?) } else { Inner::Eight(Core::new(cfg)?) };
        Ok(H264Encoder { inner })
    }

    /// How many extra codings the declared buffer cost. Zero when no
    /// buffer was declared, because then nothing can fail to fit.
    pub fn recodes(&self) -> u64 {
        with_core!(&self.inner, e => e.recoded)
    }

    /// How many bytes one source picture must be: one per sample at 8
    /// bits, two (little-endian) deeper.
    pub fn frame_bytes(&self) -> usize {
        with_core!(&self.inner, e => e.frame_bytes)
    }

    /// Which macroblock kinds the pictures coded so far took, per picture
    /// type.
    pub fn shape_census(&self) -> &ShapeCensus {
        with_core!(&self.inner, e => &e.census)
    }

    /// The reconstructions produced so far, in coding order, packed as
    /// the source pictures were handed in (see [`H264Encoder`]). The SELF
    /// property compares these against decoding the bitstream.
    pub fn reconstructions(&self) -> &[Vec<u8>] {
        with_core!(&self.inner, e => &e.recon)
    }

    /// Offer the next picture in display order. Returns whatever became
    /// codable — nothing when the picture is a B held for its anchor, several
    /// when an anchor releases the Bs behind it.
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

    /// What the rate controller achieved against what it was asked for,
    /// in bits per second — `None` at a constant quantiser. Reported by the
    /// encoder rather than recomputed by whoever is watching, for the
    /// reason given on the H.265 side: one division, in one place.
    pub fn rate_report(&self) -> Option<(f64, f64)> {
        with_core!(&self.inner, e => e.rate_report())
    }

    /// The quantiser this picture is coded at, before any adaptive
    /// adjustment. Lossless is signalled separately, so it has no QP of its
    /// own and reports the lowest.
    pub fn picture_qp(&self, kind: Kind) -> u8 {
        with_core!(&self.inner, e => e.picture_qp(kind))
    }
}

impl<S: Sample> Core<S> {
    fn new(cfg: Config) -> Result<Self> {
        if cfg.max_cu_depth.is_some_and(|d| d > 0) {
            // The coding quadtree is H.265's: H.264 codes 16x16
            // macroblocks and has no coding tree to split. A depth asked
            // for on purpose is refused by name rather than ignored;
            // `None` (the codec's default) and `Some(0)` ask for nothing.
            return Err(Error::unsupported(
                "H.264 encode: a coding quadtree depth (max_cu_depth, an H.265 tool; H.264 codes 16x16 macroblocks)",
            ));
        }
        if cfg.sao {
            // Not "in progress": H.264 has no sample adaptive offset at
            // all. Refusing names that rather than silently ignoring a
            // switch the caller set on purpose.
            return Err(Error::unsupported(
                "H.264 encode: sample adaptive offset (an H.265 tool; H.264 has none)",
            ));
        }
        if cfg.aq_strength > 0.0 && cfg.rate == RateControl::Lossless {
            // Every macroblock of a lossless picture is I_PCM, which has
            // no quantiser: a per-macroblock one would steer nothing.
            return Err(Error::unsupported(
                "H.264 encode: adaptive quantisation on a lossless picture (no quantiser to adapt)",
            ));
        }
        if cfg.lookahead > 0 {
            // Wired and measured on agent/h264tools-lookahead, and not
            // landed: H.264's own calibration of the lookahead still left a
            // buffer row refused and did not beat the past-only controller —
            // see `Config::lookahead`. Refused by name rather than coded by
            // a controller measured to be worse.
            return Err(Error::unsupported(
                "H.264 encode: rate lookahead is not calibrated for H.264 (the past-only controller is used; see Config::lookahead)",
            ));
        }
        if cfg.interlace.is_some() {
            // An interlaced stream crops in field rows: `CropUnitY` doubles
            // (7.4.2.1.1), so the displayed height has to be a whole number
            // of them — and each field is half the frame, so it has to be
            // even at the very least.
            let unit = match cfg.chroma {
                crate::ChromaFormat::Yuv420 => 4,
                _ => 2,
            };
            if !cfg.height.is_multiple_of(unit) {
                return Err(Error::unsupported(format!(
                    "H.264 encode: an interlaced height of {} rows (an interlaced {:?} stream crops in units of {unit} rows)",
                    cfg.height, cfg.chroma
                )));
            }
            match cfg.field_coding {
                FieldCoding::Field | FieldCoding::Paff => {}
                FieldCoding::Mbaff => {
                    return Err(Error::unsupported(
                        "H.264 encode: macroblock-adaptive frame/field coding (encoder in progress)",
                    ));
                }
            }
            if cfg.rate == RateControl::Lossless {
                return Err(Error::unsupported(
                    "H.264 encode: interlaced lossless coding (encoder in progress: the PCM and all-skip paths code frames)",
                ));
            }
            if cfg.cpb_ms > 0 {
                return Err(Error::unsupported(
                    "H.264 encode: a coded picture buffer over interlaced coding (encoder in progress: every field is an access unit to the HRD)",
                ));
            }
            if cfg.weighted_pred {
                return Err(Error::unsupported(
                    "H.264 encode: weighted prediction over interlaced coding (encoder in progress)",
                ));
            }
        }
        if cfg.weighted_pred && cfg.rate == RateControl::Lossless {
            // A lossless stream's inter pictures are all-skip copies of
            // their reference (PCM has no inter spelling), and a weighting
            // would scale the copy away from the source it must reproduce.
            return Err(Error::unsupported(
                "H.264 encode: weighted prediction on a lossless stream (its inter pictures are exact copies)",
            ));
        }
        let (sw, sh) = cfg.chroma.subsampling();
        let luma = cfg.width as usize * cfg.height as usize;
        let chroma = if cfg.chroma == crate::ChromaFormat::Monochrome {
            0
        } else {
            2 * (cfg.width as usize).div_ceil(sw as usize)
                * (cfg.height as usize).div_ceil(sh as usize)
        };
        debug_assert_eq!(S::BYTES, if cfg.bit_depth > 8 { 2 } else { 1 }, "the face picks the width");
        let sched = Scheduler::new(cfg.gop, cfg.bframes);
        let mut cfg = cfg;
        // A stream with B pictures keeps two marked references — the two
        // anchors a B predicts between. Declaring one in the SPS would let
        // the sliding window unmark the past anchor the moment the future
        // one arrives, and every list-0 reference in a B slice would point
        // at a picture the decoder no longer holds.
        if cfg.bframes > 0 {
            cfg.max_refs = cfg.max_refs.max(2);
        }
        // The 8x8 transform is a High-profile tool and every profile this
        // encoder claims is one, so nothing gates it but the caller —
        // except lossless, where the transform is bypassed entirely and
        // the picture codes as I_PCM, and the flag would describe a
        // transform nothing runs.
        cfg.transform_8x8 = cfg.transform_8x8 && cfg.rate != RateControl::Lossless;
        let geom = syn::Geometry::new(&cfg);
        let mut plane_dims = vec![(cfg.width, cfg.height)];
        if cfg.chroma != crate::ChromaFormat::Monochrome {
            let cw = (cfg.width as usize).div_ceil(sw as usize) as u32;
            let chh = (cfg.height as usize).div_ceil(sh as usize) as u32;
            plane_dims.push((cw, chh));
            plane_dims.push((cw, chh));
        }
        let tools = super::h264_pic::IntraTools::new(cfg.transform_8x8, cfg.subparts, cfg.bit_depth).with_aq(cfg.aq_strength);
        // The buffer to declare, snapped to what the syntax can carry —
        // the same rules, and the same refusals, as the H.265 side.
        let cpb = match (cfg.cpb_ms, cfg.rate) {
            (0, _) => None,
            (ms, RateControl::Bitrate { bps }) => match syn::Cpb::new(bps, ms) {
                Some(c) => Some(c),
                None => {
                    return Err(Error::unsupported(
                        "H.264 encode: a coded picture buffer this size needs a bit-rate or buffer scale (encoder writes both as 0)",
                    ));
                }
            },
            _ => {
                return Err(Error::unsupported(
                    "H.264 encode: a coded picture buffer without a bitrate target (a buffer constrains a rate; a fixed quantiser has none)",
                ));
            }
        };
        let rc = match cfg.rate {
            // The controller aims at the *declared* rate where a buffer
            // was declared, so the two cannot disagree by the rounding.
            RateControl::Bitrate { bps } => Some(match cpb {
                Some(c) => RateController::with_cpb(
                    c.bit_rate as u32, cfg.fps, cfg.width, cfg.height, cfg.gop, cfg.bframes, Some(c.size),
                ),
                None => RateController::new(bps, cfg.fps, cfg.width, cfg.height, cfg.gop, cfg.bframes),
            }),
            _ => None,
        };
        // The PPS quantiser: the constant one where there is one, and the
        // middle of the road where the controller varies it per picture or
        // no quantiser applies. Its only effect on the stream is the size of
        // each `slice_qp_delta`. The same rule as the H.265 side.
        let pps_qp = match cfg.rate {
            RateControl::ConstantQp(q) => q.min(51),
            RateControl::Lossless | RateControl::Bitrate { .. } => 26,
        };
        // The interlaced state: the parameter sets as the decoder reads
        // them, and an empty reference model. The SPS is the one every
        // access unit writes (no buffer: interlaced coding refuses one).
        let fields = match cfg.interlace {
            None => None,
            Some(order) => {
                let sps_nal = syn::write_sps(&cfg, &geom, LOG2_MAX_FRAME_NUM, LOG2_MAX_POC_LSB, cpb.as_ref());
                let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps_nal))?;
                let look = |_id: u32| Some(sps.clone());
                let pps = crate::h264::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(&cfg, pps_qp)), &look)?;
                Some(Fields {
                    order,
                    sps,
                    pps,
                    model: RefModel { dpb: Dpb::new(), poc: PocState::default(), next_id: 1 },
                    stored: Vec::new(),
                })
            }
        };
        Ok(Self {
            fields,
            rc,
            emitted: 0,
            cfg,
            sched,
            pps_qp,
            held: std::collections::BTreeMap::new(),
            recon: Vec::new(),
            frame_bytes: (luma + chroma) * S::BYTES,
            next_display: 0,
            geom,
            refs: Vec::new(),
            frame_num: 0,
            idr_pic_id: 0,
            plane_dims,
            tools,
            census: ShapeCensus::default(),
            cpb,
            last_bp_encode: 0,
            recoded: 0,
        })
    }

    /// See [`H264Encoder::push`].
    fn push(&mut self, picture: &[u8]) -> Result<Vec<Access>> {
        if picture.len() != self.frame_bytes {
            return Err(Error::bitstream(format!(
                "H.264 encode: picture is {} bytes, expected {}",
                picture.len(),
                self.frame_bytes
            )));
        }
        let samples = super::unpack_samples::<S>(picture, self.cfg.bit_depth, "H.264")?;
        // Insert before scheduling: the scheduler may release this very
        // picture (every picture is an IDR when `gop` is 0), and `code` looks
        // the samples up by display index.
        let display = self.next_display;
        self.next_display += 1;
        self.held.insert(display, samples);
        let ready = self.sched.push();
        self.code(ready)
    }

    /// See [`H264Encoder::flush`].
    fn flush(&mut self) -> Result<Vec<Access>> {
        let ready = self.sched.flush();
        self.code(ready)
    }

    fn code(&mut self, ready: Vec<Coded>) -> Result<Vec<Access>> {
        let mut out = Vec::with_capacity(ready.len());
        for c in ready {
            let src = self
                .held
                .remove(&c.display)
                .ok_or_else(|| Error::bitstream("H.264 encode: scheduler released an absent picture"))?;
            let access = self.code_picture(c, &src)?;
            // The ledger closes here, at the one place every picture of
            // every kind passes through, counting the whole access unit —
            // start codes, NAL headers, parameter sets and slice payload —
            // because that is what the target is measured against. Same
            // shape, and the same reasoning, as the H.265 side.
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
    /// fit the buffer this stream declares — the same loop, the same
    /// escalation law and the same refusal as the H.265 side.
    ///
    /// The quantiser is chosen **once**; the loop escalates from it and
    /// tells the controller afterwards which quantiser the picture was
    /// actually coded at. Without a declared buffer there is nothing to
    /// fit and the loop runs exactly once, which is why every stream that
    /// does not ask for a buffer is byte-identical to what it was before
    /// this existed.
    fn code_picture(&mut self, c: Coded, src: &[S]) -> Result<Access> {
        let mut qp = self.pick_picture_qp(&c);
        for attempt in 0..super::rc::MAX_ATTEMPTS {
            let a = match (self.fields.is_some(), self.cfg.field_coding) {
                (false, _) => self.code_attempt(&c, src, qp)?,
                (true, FieldCoding::Field) => self.code_attempt_fields(&c, src, qp)?,
                (true, _) => self.code_attempt_paff(&c, src, qp)?,
            };
            let bits = a.access.data.len() as u64 * 8;
            // What the buffer can hand over at this picture's removal time;
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
                // coded into: a configuration error, refused by name rather
                // than shipped as a stream that violates what it declares.
                return Err(Error::unsupported(format!(
                    "H.264 encode: picture {} needs {bits} bits and the declared buffer affords {afford} even at quantiser {qp} (the coded picture buffer is too small for this content)",
                    c.poc
                )));
            }
            self.recoded += 1;
            qp = RateController::escalate(qp, bits, afford);
        }
        unreachable!("the loop returns or errors on its last attempt")
    }

    /// The quantiser this picture starts at, before any buffer escalation:
    /// the controller's choice under a bitrate, the configured one
    /// otherwise.
    fn pick_picture_qp(&mut self, c: &Coded) -> u8 {
        match self.cfg.rate {
            RateControl::Bitrate { .. } => {
                let kind = match c.kind {
                    Kind::Idr | Kind::I => PicKind::Intra,
                    Kind::P => PicKind::Inter,
                    Kind::B => PicKind::B,
                };
                self.rc.as_mut().expect("a bitrate configuration builds a controller").pick_qp(kind)
            }
            _ => self.picture_qp(c.kind),
        }
    }

    /// Keep what an attempt made: the reconstruction the SELF check reads,
    /// the reference the next pictures predict from, and the counters the
    /// headers run on. Every write this encoder makes per picture is here
    /// — `code_attempt` makes none — so a re-coded attempt leaves no
    /// trace.
    fn commit(&mut self, c: &Coded, a: Attempt<S>, qp: u8) -> Access {
        let idr = c.kind == Kind::Idr;
        self.recon.push(a.rec);
        if let Some(out) = a.fields {
            // An interlaced frame: its fields' census, the model the
            // attempt ran, and the reconstruction kept while the model
            // still marks it — the frame counters exactly as a frame's.
            for (kind, mbs) in &out.census {
                self.census.add(*kind, mbs, qp, super::h264_pic::WeightCensus::default());
                if out.frame_coded {
                    self.census.frame_pictures += 1;
                } else {
                    self.census.field_pictures += 1;
                }
            }
            if idr {
                self.frame_num = 0;
                self.idr_pic_id ^= 1;
                self.last_bp_encode = c.encode;
            }
            let f = self.fields.as_mut().expect("a field attempt comes from an interlaced encoder");
            f.model = out.model;
            if c.reference {
                self.frame_num = (self.frame_num + 1) & ((1 << LOG2_MAX_FRAME_NUM) - 1);
                f.stored.push(out.frame);
            }
            let model = &f.model;
            f.stored.retain(|s| model.dpb.pics.iter().any(|p| p.frame.id == s.id && p.is_ref()));
            return a.access;
        }
        self.census.add(c.kind, &a.motion.info.mbs, qp, a.motion.weighting);
        if idr {
            // The attempt wrote `frame_num` 0 for an IDR; the count
            // restarts from there.
            self.frame_num = 0;
            self.idr_pic_id ^= 1;
            self.last_bp_encode = c.encode;
        }
        if c.reference {
            self.frame_num = (self.frame_num + 1) & ((1 << LOG2_MAX_FRAME_NUM) - 1);
            let mut recon = a.recon;
            // Replicate the borders motion compensation reads — once, here,
            // so every stored reference is search-ready and the P path never
            // has to wonder whether a plane's border is stale.
            crate::encode::h264_me::prepare_reference(&mut recon);
            self.refs.push((c.poc, recon, a.motion));
            // The DPB the SPS declares. Dropping the oldest keeps the encoder
            // inside what it told the decoder to allocate.
            let cap = (self.cfg.max_refs.max(1) as usize) + 1;
            while self.refs.len() > cap {
                self.refs.remove(0);
            }
        }
        a.access
    }

    /// Code one picture at a given quantiser, keeping nothing: parameter
    /// sets, the buffer SEI where a buffer is declared, slice header, then
    /// the slice data of whichever path the configuration selects — the
    /// transform writers (either entropy coder), PCM where exactness
    /// demands it, or the all-skip inter fallback.
    fn code_attempt(&self, c: &Coded, src: &[S], qp: u8) -> Result<Attempt<S>> {
        let g = self.geom;
        let idr = c.kind == Kind::Idr;
        // Reference lists, by picture order count: list0 runs backwards from
        // the current picture, list1 forwards. A P picture uses list0 only.
        let (past, future) = self.lists_for(c.poc);
        if !idr && past.is_none() {
            return Err(Error::bitstream(
                "H.264 encode: an inter picture with no earlier reference reconstructed",
            ));
        }
        if c.kind == Kind::B && future.is_none() {
            return Err(Error::bitstream(
                "H.264 encode: a B picture with no later reference reconstructed",
            ));
        }

        // Source planes, laid out consecutively as the caller supplies them.
        let mut planes = Vec::with_capacity(self.plane_dims.len());
        let mut off = 0usize;
        for &(w, h) in &self.plane_dims {
            let n = (w * h) as usize;
            planes.push(syn::Plane {
                data: &src[off..off + n],
                stride: w as usize,
                width: w,
                height: h,
            });
            off += n;
        }

        // Reconstruction at coded size; cropped to display size afterwards,
        // because that is what a decoder emits and therefore what the SELF
        // check compares against.
        let (cw, ch) = g.chroma_mb();
        // The same border the decoder gives its own frames, because these
        // planes are the decoder's type and its intra predictors read
        // neighbours out of that border.
        let mut recon: Vec<syn::Recon<S>> =
            vec![syn::recon_plane(g.coded_width, g.coded_height, crate::h264::frame::LUMA_PAD)];
        if cw != 0 {
            // 4:4:4 chroma is a luma-like plane: its motion compensation
            // runs the six-tap luma kernel, whose windows assume the luma
            // border (Frame::new gives its own 4:4:4 chroma the same).
            let pad = if self.cfg.chroma == crate::ChromaFormat::Yuv444 {
                crate::h264::frame::LUMA_PAD
            } else {
                crate::h264::frame::CHROMA_PAD
            };
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
        }

        // An IDR restarts the count, and the header below carries the reset
        // value; `commit` restarts the stored count to match.
        let frame_num = if idr { 0 } else { self.frame_num };
        // H.264 repeats its parameter sets with *every* access unit — see
        // the SPS/PPS emitted below — and this side once put each picture's
        // own quantiser straight into `pic_init_qp` with a zero
        // `slice_qp_delta`, on the grounds that a re-sent PPS replaces the
        // old one and the decoded quantiser is identical either way. Every
        // Annex-B check agreed: SELF, CROSS, the whole gate.
        //
        // What none of them could see is a container. An MP4 `avc1` sample
        // entry carries the parameter sets OUT OF BAND in `avcC` and strips
        // them from the samples, so a PPS that differs between the I and P
        // pictures leaves the box holding two under one id; a decoder keeps
        // whichever it parses last and reads the pictures written under the
        // other one as garbage from their first macroblock. libavcodec did
        // exactly that on rivet's first H.264 file (2026-08-27) — MB 0 0 of
        // the IDR, "top block unavailable for requested intra mode" —
        // while the same bytes as an Annex-B stream decoded clean.
        //
        // So the PPS is now the stream's, fixed at construction, and each
        // slice carries its own quantiser as a delta against it — the shape
        // the H.265 side has always had. (The note that this was "tried
        // and reverted for no gain" was true of the quantiser alone; the
        // gain is that the stream can be put in a box.)
        //
        // The quantiser arrives as an argument: chosen once by the caller
        // and escalated by it, so an attempt cannot quietly pick a
        // different one than the buffer arithmetic is reasoning about.
        //
        // Whether this picture takes the transform intra path rather than
        // I_PCM. Lossless stays PCM because PCM is the exactly-lossless mode
        // and the transform path quantises.
        // The envelope is about *quantising*, not about which mode picked
        // the quantiser: the transform path quantises, so lossless has to
        // stay PCM, and everything else may use it. It was spelled as
        // `ConstantQp` because for a long time that was the only lossy
        // mode there was — and when a bitrate target arrived it silently
        // fell outside, coding every picture as PCM. The symptom was a
        // controller that appeared to work and a stream whose size did not
        // move with the target at all, because nothing downstream was
        // reading the quantiser it chose.
        let lossy = !matches!(self.cfg.rate, RateControl::Lossless);
        let transform_intra = idr && lossy;
        // Whether a P picture takes the motion-search path rather than
        // all-skip: the same envelope as the intra transform path.
        let transform_p = !idr && c.kind == Kind::P && lossy;
        // B pictures share the envelope: inside it every picture type is
        // transform-coded, so a colocated picture's motion is always the
        // real record the direct derivation needs; outside it everything
        // is PCM or all-skip, where the zero-colocated assumption of the
        // temporal-direct fallback below still holds.
        let transform_b = !idr && c.kind == Kind::B && lossy;
        // Weighted prediction, for a P picture under the PPS flag: one entry
        // for list 0's one reference, each component fitted to the source
        // against that reference's reconstruction over the display area and
        // kept only where it lowers the zero-motion residual — the H.265
        // side's fit (`h265_wp`), held to the weights H.264's table carries.
        // The table travels in the slice header; the walk predicts with what
        // the reader derives from it.
        let wp = (self.cfg.weighted_pred && transform_p).then(|| {
            let rf = &self.refs[past.expect("checked above")].1;
            let fits: Vec<h265_wp::PlaneFit> = (0..self.plane_dims.len())
                .map(|p| {
                    let (pw, ph) = self.plane_dims[p];
                    let refs = h265_wp::RefSamples { data: &rf[p].data, origin: rf[p].origin(), stride: rf[p].stride };
                    h265_wp::fit_samples(planes[p].data, planes[p].stride, refs, pw as usize, ph as usize, g.bit_depth, h265_wp::H264_WEIGHTS)
                })
                .collect();
            let default = (1i32 << h265_wp::LOG2_DENOM, 0i32);
            let comp = |f: &h265_wp::PlaneFit| if f.used() { (f.weight, f.offset) } else { default };
            let luma = comp(&fits[0]);
            let chroma = if fits.len() == 3 { [comp(&fits[1]), comp(&fits[2])] } else { [default; 2] };
            let entry = WeightEntry { luma, chroma, luma_flag: luma != default, chroma_flag: chroma != [default; 2] };
            syn::PredWeights {
                table: PredWeightTable {
                    luma_log2_denom: h265_wp::LOG2_DENOM,
                    chroma_log2_denom: h265_wp::LOG2_DENOM,
                    lists: [vec![entry], Vec::new()],
                },
                chroma: g.chroma != crate::ChromaFormat::Monochrome,
            }
        });
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            3,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_FRAME_NUM, LOG2_MAX_POC_LSB, self.cpb.as_ref()),
        ));
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, 3, &syn::write_pps(&self.cfg, self.pps_qp)));
        if let Some(cpb) = self.cpb.as_ref() {
            // A buffering period begins at every IDR, carrying the initial
            // removal delay; and every access unit of a stream with a NAL
            // HRD carries its timing — `cpb_removal_delay`, clock ticks
            // since the last buffering period's removal, which is what
            // fixes this picture's removal time (C.1.2; H.265 infers it
            // from a fixed picture rate, H.264 has no such inference), and
            // `dpb_output_delay`, ticks from removal to output: the
            // reorder depth plus this picture's own displacement, never
            // negative because a picture is never displayed more than
            // `bframes` positions before it is coded.
            if idr {
                out.extend_from_slice(&syn::annexb(
                    syn::NAL_SEI,
                    0,
                    &syn::write_buffering_period_sei(cpb),
                ));
            }
            let removal = syn::TICKS_PER_FRAME as u64 * (c.encode - self.last_bp_encode);
            let output = syn::TICKS_PER_FRAME as i64
                * (c.display as i64 + self.cfg.bframes as i64 - c.encode as i64);
            out.extend_from_slice(&syn::annexb(
                syn::NAL_SEI,
                0,
                &syn::write_pic_timing_sei(cpb, removal as u32, output.max(0) as u32),
            ));
        }
        // HDR10 static metadata, with every IDR so that a stream joined at
        // any of them carries it.
        if idr {
            if let Some(m) = self.cfg.mastering_display.as_ref() {
                out.extend_from_slice(&syn::annexb(syn::NAL_SEI, 0, &syn::write_mastering_display_sei(m)));
            }
            if let Some(c) = self.cfg.content_light.as_ref() {
                out.extend_from_slice(&syn::annexb(syn::NAL_SEI, 0, &syn::write_content_light_level_sei(c)));
            }
        }

        let cabac = self.cfg.entropy == Entropy::Cabac;
        let mut w = BitWriter::with_capacity(self.frame_bytes + 256);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                frame_num,
                idr_pic_id: self.idr_pic_id,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp,
                log2_max_frame_num: LOG2_MAX_FRAME_NUM,
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                reference: c.reference,
                // Always on. The transform picture writers run the
                // decoder's own loop filter over their reconstruction
                // (`h264_deblock`), so the header may finally say so; the
                // PCM and all-skip pictures always could — an all-I_PCM
                // picture filters at qP 0 (below every threshold) and an
                // all-skip one is bS 0 on every edge, so the filter leaves
                // both untouched.
                deblock: true,
                cabac,
                direct_spatial: transform_b,
                pred_weights: wp.clone(),
                interlaced: false,
                bottom_field: None,
                delta_poc_bottom: 0,
            },
            self.pps_qp,
            &mut w,
        );
        // Every coded picture leaves its motion in the decoder's layout:
        // the transform walks return the real thing; the PCM and all-skip
        // paths synthesize what a decoder would store for them (intra; a
        // skip at reference 0 with zero vectors — both lists for a B
        // skip), because a later B picture reads it as colocated motion.
        let synth = |kind: crate::h264::mb::MbKind, l0: bool, l1: bool| {
            use crate::h264::frame::{BlockMotion, Mv, PARITY_FRAME};
            let mut pm = super::h264_pic::PicMotion::new(
                g.mbs_wide as usize,
                g.mbs_high as usize,
            );
            let mut mot = [[BlockMotion::default(); 16]; 2];
            for (l, used) in [(0usize, l0), (1usize, l1)] {
                if used {
                    mot[l] = [BlockMotion {
                        mv: Mv::ZERO,
                        ref_idx: 0,
                        ref_parity: PARITY_FRAME,
                        ref_id: 1 + l as u16,
                    }; 16];
                }
            }
            // The chroma QP as the decoder stores it beside `QP_Y`: the
            // 8.5.8 map, clipping at `-QpBdOffset_C` below at depth.
            let bd_off = 6 * (g.bit_depth as i32 - 8);
            let qpc = crate::h264::mb::chroma_qp(qp as i32, 0, bd_off) as i8;
            for addr in 0..(g.mbs_wide * g.mbs_high) as usize {
                pm.commit(
                    addr,
                    crate::h264::mb::MbInfo {
                        kind,
                        decoded: true,
                        slice: 0,
                        qp: qp as i8,
                        qpc: [qpc; 2],
                        nz_mask: if kind == crate::h264::mb::MbKind::IPcm { 0xffff } else { 0 },
                        ..crate::h264::mb::MbInfo::default()
                    },
                    &mot,
                );
            }
            pm
        };
        let motion;
        if idr {
            // A CABAC slice's final terminate flushes the codeword and its
            // last one *is* the rbsp_stop_one_bit, so the CABAC writers
            // close the slice themselves and no `rbsp_trailing_bits`
            // follows them (9.3.4.6) — on both the transform and PCM paths.
            if transform_intra && cabac {
                motion = super::h264_cabac_mb::write_intra_picture_cabac(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon,
                );
            } else if transform_intra {
                motion = super::h264_cavlc_mb::write_intra_picture(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon,
                );
                w.rbsp_trailing_bits();
            } else if cabac {
                syn::write_pcm_slice_data_cabac(&mut w, &g, qp, &planes, &mut recon);
                motion = synth(crate::h264::mb::MbKind::IPcm, false, false);
            } else {
                for mb_y in 0..g.mbs_high {
                    for mb_x in 0..g.mbs_wide {
                        syn::write_pcm_macroblock(&mut w, &g, mb_x, mb_y, &planes, &mut recon);
                    }
                }
                w.rbsp_trailing_bits();
                motion = synth(crate::h264::mb::MbKind::IPcm, false, false);
            }
        } else if transform_p {
            // A real P picture: motion search, skip where it is legal and
            // free, the intra decision where inter loses. One decision
            // walk, two spellings (`encode::h264_pic`).
            let p0 = past.expect("checked above");
            if cabac {
                motion = super::h264_cabac_mb::write_p_picture_cabac(
                    &mut w,
                    &g,
                    &self.tools,
                    qp,
                    &planes,
                    &mut recon,
                    &self.refs[p0].1,
                    wp.as_ref().map(|p| &p.table),
                );
            } else {
                motion = super::h264_cavlc_mb::write_p_picture(
                    &mut w,
                    &g,
                    &self.tools,
                    qp,
                    &planes,
                    &mut recon,
                    &self.refs[p0].1,
                    wp.as_ref().map(|p| &p.table),
                );
                w.rbsp_trailing_bits();
            }
        } else if transform_b {
            // A real B picture: per-list search, bi-prediction, spatial
            // direct off the stored colocated motion, B_Skip where nothing
            // survives. The scheduler delivered both anchors before this
            // picture — checked above — and the colocated record is the
            // list-1 reference's.
            let p0 = past.expect("checked above");
            let p1 = future.expect("checked above");
            let refs2 = [&self.refs[p0].1[..], &self.refs[p1].1[..]];
            let col = super::h264_pic::Colocated::progressive(&self.refs[p1].2);
            if cabac {
                motion = super::h264_cabac_mb::write_b_picture_cabac(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, &col,
                );
            } else {
                motion = super::h264_cavlc_mb::write_b_picture(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, &col,
                );
                w.rbsp_trailing_bits();
            }
        } else {
            // Every macroblock skipped — what the inter pictures of a
            // lossless stream do, since PCM has no inter spelling.
            // `P_Skip` carries no motion vector difference and no residual:
            // the vector is the median prediction of its neighbours, which
            // in an all-skip picture is zero everywhere, so the
            // reconstruction is the reference unchanged.
            //
            // Deblocking does not disturb it either: every edge has matching
            // motion, the same reference and no coefficients, so every
            // boundary strength is zero.
            if cabac {
                super::h264_cabac_mb::write_skip_picture_cabac(&mut w, &g, qp, c.kind == Kind::B);
            } else {
                w.ue(g.mbs_wide * g.mbs_high); // mb_skip_run
            }
            motion = if c.kind == Kind::B {
                synth(crate::h264::mb::MbKind::BSkip, true, true)
            } else {
                synth(crate::h264::mb::MbKind::PSkip, true, false)
            };
            let p0 = past.expect("checked above");
            match c.kind {
                Kind::B => {
                    // B_Skip is direct prediction. The colocated picture motion
                    // is zero throughout an all-skip stream, so temporal direct
                    // derives zero vectors on both lists and the prediction is
                    // the default bi-predictive average, (a + b + 1) >> 1.
                    let p1 = future.expect("checked above");
                    for i in 0..recon.len() {
                        let (a, b) = (&self.refs[p0].1[i].data, &self.refs[p1].1[i].data);
                        for (d, (&x, &y)) in recon[i].data.iter_mut().zip(a.iter().zip(b.iter())) {
                            *d = S::from_i32((x.to_i32() + y.to_i32() + 1) >> 1);
                        }
                    }
                }
                _ => {
                    for i in 0..recon.len() {
                        recon[i].data.copy_from_slice(&self.refs[p0].1[i].data);
                    }
                }
            }
            if !cabac {
                w.rbsp_trailing_bits();
            }
        }
        out.extend_from_slice(&syn::annexb(
            if idr { syn::NAL_IDR } else { syn::NAL_SLICE },
            if c.reference { 3 } else { 0 },
            &w.into_nal(),
        ));

        let mut cropped = Vec::with_capacity(self.frame_bytes);
        syn::crop_into(&recon[0], g.width, g.height, &mut cropped);
        for (i, p) in recon.iter().enumerate().skip(1) {
            let (dw, dh) = self.plane_dims[i];
            syn::crop_into(p, dw, dh, &mut cropped);
        }

        Ok(Attempt {
            access: Access {
                data: out,
                keyframe: idr,
                poc: c.poc,
                encode_index: c.encode,
                display: c.display,
            },
            rec: cropped,
            recon,
            motion,
            fields: None,
        })
    }

    /// The rows of field `parity` (0 top, 1 bottom) of a source frame, plane
    /// by plane, laid out as a picture of half the height — row `r` of the
    /// field is row `2r + parity` of the frame, in every plane and every
    /// chroma format (an interlaced frame's chroma rows alternate fields
    /// exactly as its luma rows do).
    fn field_source(&self, src: &[S], parity: usize) -> Vec<S> {
        let mut out = Vec::with_capacity(src.len() / 2);
        let mut off = 0usize;
        for &(w, h) in &self.plane_dims {
            let (w, h) = (w as usize, h as usize);
            for r in 0..h / 2 {
                let row = off + (2 * r + parity) * w;
                out.extend_from_slice(&src[row..row + w]);
            }
            off += w * h;
        }
        out
    }

    /// Code one interlaced frame as its two field pictures, keeping
    /// nothing: the parameter sets, then each field — in the frame's field
    /// order — as a slice NAL of its own.
    ///
    /// Each field is typed from the frame: an IDR frame's first field is
    /// the IDR and its second a non-IDR P field predicting from the first
    /// (an IDR marks every other picture unused, so a second IDR field
    /// would unpair the frame); a P or B frame's fields are both P or both
    /// B. Both fields carry the frame's `frame_num`; the first takes the
    /// frame's POC and the second the next one, which is what orders them.
    ///
    /// The reference each field predicts from is index 0 of the lists the
    /// decoder builds for it (see [`Fields`]), and a 4:2:0 field predicting
    /// from the other parity takes Table 8-10's chroma offset.
    fn code_attempt_fields(&self, c: &Coded, src: &[S], qp: u8) -> Result<Attempt<S>> {
        let f = self.fields.as_ref().expect("an interlaced encoder carries its field state");
        let g = self.geom;
        let idr = c.kind == Kind::Idr;
        let cabac = self.cfg.entropy == Entropy::Cabac;
        let mut model = f.model.fork();
        let mut out = self.interlaced_prefix(idr);
        let frame_num = if idr { 0 } else { self.frame_num };
        let id = model.next_id;
        model.next_id += 1;
        // The model's stand-in for this frame: both fields' entries are one
        // DPB entry, found by this shared frame.
        let shared = Arc::new(SharedFrame::new(Frame::<u8>::empty(), id, true));
        let mut cur = StoredFrame::<S> { id, fields: [None, None], frame: Vec::new(), col: Frame::empty() };
        let mut census = Vec::with_capacity(2);
        let mut field_poc = [0i32; 2];
        for k in 0..2usize {
            let parity = match f.order {
                FieldOrder::TopFirst => k,
                FieldOrder::BottomFirst => 1 - k,
            };
            let kind = match (k, c.kind) {
                (1, Kind::Idr | Kind::I) => Kind::P,
                (_, kind) => kind,
            };
            let nal_type = if kind == Kind::Idr { syn::NAL_IDR } else { syn::NAL_SLICE };
            let nal_ref_idc = if c.reference { 3 } else { 0 };
            let header = syn::SliceHeader {
                kind,
                frame_num,
                idr_pic_id: self.idr_pic_id,
                poc_lsb: ((c.poc + k as i32) as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp,
                log2_max_frame_num: LOG2_MAX_FRAME_NUM,
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                reference: c.reference,
                deblock: true,
                cabac,
                direct_spatial: kind == Kind::B,
                pred_weights: None,
                interlaced: true,
                bottom_field: Some(parity == 1),
                delta_poc_bottom: 0,
            };
            // The header as the decoder reads it: written alone, closed, and
            // parsed by the production slice header parser.
            let hdr = f.read_back(&header, self.pps_qp, nal_type, nal_ref_idc)?;
            // The decoder's picture start: an IDR, or an empty buffer,
            // (re)sizes the DPB from the SPS; then the field's POC.
            if hdr.is_idr() || model.dpb.pics.is_empty() {
                model.dpb.configure(&f.sps);
            }
            let (top, bottom) = crate::h264::dpb::compute_poc(&f.sps, &hdr, &mut model.poc);
            let poc = if parity == 0 { top } else { bottom };
            field_poc[parity] = poc;
            // Index 0 of each list the slice predicts from, as (frame id,
            // field parity).
            let mut ref0: [Option<(u64, usize)>; 2] = [None, None];
            if !kind.is_intra() {
                let rl = crate::h264::dpb::build_ref_lists(&mut model.dpb, &f.sps, &hdr, poc, parity as u8)?;
                for (l, r) in ref0.iter_mut().enumerate().take(if kind == Kind::B { 2 } else { 1 }) {
                    let (i, par) = rl.lists[l].first().copied().unwrap_or(crate::h264::dpb::MISSING_REF);
                    if i >= model.dpb.pics.len() {
                        return Err(Error::bitstream(format!(
                            "H.264 encode: field {k} of picture {} has no reference at index 0 of list {l}",
                            c.poc
                        )));
                    }
                    *r = Some((model.dpb.pics[i].frame.id, par as usize));
                }
            }
            let field_of = |r: Option<(u64, usize)>| -> Result<&StoredField<S>> {
                let (rid, par) = r.expect("an inter field has its list 0 (and a B field its list 1)");
                let frame = if rid == cur.id { Some(&cur) } else { f.stored.iter().find(|s| s.id == rid) };
                frame.and_then(|fr| fr.fields[par].as_ref()).ok_or_else(|| {
                    Error::bitstream("H.264 encode: the reference model names a field that was never reconstructed")
                })
            };
            // Table 8-10: a 4:2:0 field predicting from the other parity
            // offsets its vertical chroma vector by a quarter chroma sample.
            let dy = |r: Option<(u64, usize)>| match r {
                Some((_, rp)) if g.chroma == crate::ChromaFormat::Yuv420 && rp != parity => {
                    if parity == 1 { 2 } else { -2 }
                }
                _ => 0,
            };
            let gf = syn::Geometry { chroma_mv_dy: [dy(ref0[0]), dy(ref0[1])], ..g.field() };

            let fsrc = self.field_source(src, parity);
            let mut planes = Vec::with_capacity(self.plane_dims.len());
            let mut off = 0usize;
            for &(w, h) in &self.plane_dims {
                let n = (w * (h / 2)) as usize;
                planes.push(syn::Plane { data: &fsrc[off..off + n], stride: w as usize, width: w, height: h / 2 });
                off += n;
            }
            let (cw, ch) = gf.chroma_mb();
            let mut recon: Vec<syn::Recon<S>> =
                vec![syn::recon_plane(gf.coded_width, gf.coded_height, crate::h264::frame::LUMA_PAD)];
            if cw != 0 {
                let pad = if self.cfg.chroma == crate::ChromaFormat::Yuv444 {
                    crate::h264::frame::LUMA_PAD
                } else {
                    crate::h264::frame::CHROMA_PAD
                };
                recon.push(syn::recon_plane(gf.mbs_wide * cw, gf.mbs_high * ch, pad));
                recon.push(syn::recon_plane(gf.mbs_wide * cw, gf.mbs_high * ch, pad));
            }

            let mut w = BitWriter::with_capacity(self.frame_bytes / 2 + 256);
            syn::write_slice_header(&header, self.pps_qp, &mut w);
            let motion = match kind {
                Kind::Idr | Kind::I => {
                    if cabac {
                        super::h264_cabac_mb::write_intra_picture_cabac(&mut w, &gf, &self.tools, qp, &planes, &mut recon)
                    } else {
                        let m = super::h264_cavlc_mb::write_intra_picture(&mut w, &gf, &self.tools, qp, &planes, &mut recon);
                        w.rbsp_trailing_bits();
                        m
                    }
                }
                Kind::P => {
                    let r = field_of(ref0[0])?;
                    if cabac {
                        super::h264_cabac_mb::write_p_picture_cabac(&mut w, &gf, &self.tools, qp, &planes, &mut recon, &r.planes, None)
                    } else {
                        let m = super::h264_cavlc_mb::write_p_picture(&mut w, &gf, &self.tools, qp, &planes, &mut recon, &r.planes, None);
                        w.rbsp_trailing_bits();
                        m
                    }
                }
                Kind::B => {
                    let (r0, r1) = (field_of(ref0[0])?, field_of(ref0[1])?);
                    let refs2 = [&r0.planes[..], &r1.planes[..]];
                    // Direct prediction reads the list-1 reference's frame
                    // in the decoder's frame-row layout, through the
                    // decoder's colocated mapping for a field picture.
                    let (rid, rpar) = ref0[1].expect("a B field has list 1");
                    let colf = f.stored.iter().find(|s| s.id == rid).ok_or_else(|| {
                        Error::bitstream("H.264 encode: a B field's list-1 reference is not a stored frame")
                    })?;
                    let col = super::h264_pic::Colocated {
                        frame: &colf.col,
                        map: crate::h264::recon::ColMap {
                            cur_parity: parity as u8,
                            col_parity: rpar as u8,
                            cur_poc: poc,
                            cur_mbaff: false,
                            mb_width: g.mbs_wide as usize,
                        },
                    };
                    if cabac {
                        super::h264_cabac_mb::write_b_picture_cabac(&mut w, &gf, &self.tools, qp, &planes, &mut recon, refs2, &col)
                    } else {
                        let m = super::h264_cavlc_mb::write_b_picture(&mut w, &gf, &self.tools, qp, &planes, &mut recon, refs2, &col);
                        w.rbsp_trailing_bits();
                        m
                    }
                }
            };
            out.extend_from_slice(&syn::annexb(nal_type, nal_ref_idc, &w.into_nal()));

            // The decoder's picture end: its frame_num bookkeeping, then the
            // field stored and marked into the entry it shares with its pair.
            let recorded = if parity == 0 { [poc, i32::MAX] } else { [i32::MAX, poc] };
            model_store(&mut model, &f.sps, &hdr, &shared, frame_num, parity as u8, poc, recorded, c.encode)?;

            crate::encode::h264_me::prepare_reference(&mut recon);
            census.push((kind, motion.info.mbs.clone()));
            cur.fields[parity] = Some(StoredField { planes: recon, motion });
        }

        cur.col = field_pair_motion(&cur, &g, field_poc);
        cur.frame = interleave_fields(&cur.fields);

        // The frame a decoder outputs: the two fields' rows interleaved,
        // cropped to the displayed size.
        let mut rec = Vec::with_capacity(self.frame_bytes);
        for (i, &(dw, dh)) in self.plane_dims.iter().enumerate() {
            for y in 0..dh as usize {
                let p = &cur.fields[y % 2].as_ref().expect("both fields were coded").planes[i];
                let row = (y / 2 + p.pad) * p.stride + p.pad;
                crate::encode::pack_row(&p.data[row..row + dw as usize], &mut rec);
            }
        }
        Ok(Attempt {
            access: Access { data: out, keyframe: idr, poc: c.poc, encode_index: c.encode, display: c.display },
            rec,
            recon: Vec::new(),
            motion: super::h264_pic::PicMotion::new(0, 0),
            fields: Some(FieldsOut { model, frame: cur, census, frame_coded: false }),
        })
    }

    /// The parameter sets, and at an IDR the HDR10 SEIs, that open every
    /// interlaced access unit — the progressive envelope less the buffer
    /// SEIs, which interlaced coding refuses.
    fn interlaced_prefix(&self, idr: bool) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            3,
            &syn::write_sps(&self.cfg, &self.geom, LOG2_MAX_FRAME_NUM, LOG2_MAX_POC_LSB, self.cpb.as_ref()),
        ));
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, 3, &syn::write_pps(&self.cfg, self.pps_qp)));
        if idr {
            if let Some(m) = self.cfg.mastering_display.as_ref() {
                out.extend_from_slice(&syn::annexb(syn::NAL_SEI, 0, &syn::write_mastering_display_sei(m)));
            }
            if let Some(cl) = self.cfg.content_light.as_ref() {
                out.extend_from_slice(&syn::annexb(syn::NAL_SEI, 0, &syn::write_content_light_level_sei(cl)));
            }
        }
        out
    }

    /// Code one interlaced frame as one frame picture (`field_pic_flag` 0),
    /// keeping nothing. The field order survives in the POCs: the field
    /// first in time takes the frame's POC and the other the next one, the
    /// bottom field's carried as `delta_pic_order_cnt_bottom` (+1 top field
    /// first, −1 bottom field first). The references are *frames* — index 0
    /// of the frame lists the decoder builds, whatever each frame was coded
    /// as — and a B frame reads colocated motion through the decoder's own
    /// frame-over-frame or frame-over-field-pair mapping.
    fn code_attempt_ilace_frame(&self, c: &Coded, src: &[S], qp: u8) -> Result<Attempt<S>> {
        let f = self.fields.as_ref().expect("an interlaced encoder carries its field state");
        let g = self.geom;
        let idr = c.kind == Kind::Idr;
        let cabac = self.cfg.entropy == Entropy::Cabac;
        let mut model = f.model.fork();
        let mut out = self.interlaced_prefix(idr);
        let frame_num = if idr { 0 } else { self.frame_num };
        let id = model.next_id;
        model.next_id += 1;
        let shared = Arc::new(SharedFrame::new(Frame::<u8>::empty(), id, true));
        let kind = c.kind;
        let (top_offset, delta) = match f.order {
            FieldOrder::TopFirst => (0, 1),
            FieldOrder::BottomFirst => (1, -1),
        };
        let nal_type = if idr { syn::NAL_IDR } else { syn::NAL_SLICE };
        let nal_ref_idc = if c.reference { 3 } else { 0 };
        let header = syn::SliceHeader {
            kind,
            frame_num,
            idr_pic_id: self.idr_pic_id,
            poc_lsb: ((c.poc + top_offset) as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
            qp,
            log2_max_frame_num: LOG2_MAX_FRAME_NUM,
            log2_max_poc_lsb: LOG2_MAX_POC_LSB,
            reference: c.reference,
            deblock: true,
            cabac,
            direct_spatial: kind == Kind::B,
            pred_weights: None,
            interlaced: true,
            bottom_field: None,
            delta_poc_bottom: delta,
        };
        let hdr = f.read_back(&header, self.pps_qp, nal_type, nal_ref_idc)?;
        if hdr.is_idr() || model.dpb.pics.is_empty() {
            model.dpb.configure(&f.sps);
        }
        let (top, bottom) = crate::h264::dpb::compute_poc(&f.sps, &hdr, &mut model.poc);
        let poc = top.min(bottom);
        let mut ref0: [Option<u64>; 2] = [None, None];
        if !kind.is_intra() {
            let rl = crate::h264::dpb::build_ref_lists(&mut model.dpb, &f.sps, &hdr, poc, PARITY_FRAME)?;
            for (l, r) in ref0.iter_mut().enumerate().take(if kind == Kind::B { 2 } else { 1 }) {
                let (i, _) = rl.lists[l].first().copied().unwrap_or(crate::h264::dpb::MISSING_REF);
                if i >= model.dpb.pics.len() {
                    return Err(Error::bitstream(format!(
                        "H.264 encode: frame picture {} has no reference at index 0 of list {l}",
                        c.poc
                    )));
                }
                *r = Some(model.dpb.pics[i].frame.id);
            }
        }
        let frame_of = |r: Option<u64>| -> Result<&StoredFrame<S>> {
            let rid = r.expect("an inter frame has its list 0 (and a B frame its list 1)");
            f.stored.iter().find(|s| s.id == rid).ok_or_else(|| {
                Error::bitstream("H.264 encode: the reference model names a frame that was never reconstructed")
            })
        };

        let mut planes = Vec::with_capacity(self.plane_dims.len());
        let mut off = 0usize;
        for &(pw, ph) in &self.plane_dims {
            let n = (pw * ph) as usize;
            planes.push(syn::Plane { data: &src[off..off + n], stride: pw as usize, width: pw, height: ph });
            off += n;
        }
        let (cw, ch) = g.chroma_mb();
        let mut recon: Vec<syn::Recon<S>> =
            vec![syn::recon_plane(g.coded_width, g.coded_height, crate::h264::frame::LUMA_PAD)];
        if cw != 0 {
            let pad = if self.cfg.chroma == crate::ChromaFormat::Yuv444 {
                crate::h264::frame::LUMA_PAD
            } else {
                crate::h264::frame::CHROMA_PAD
            };
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
        }

        let mut w = BitWriter::with_capacity(self.frame_bytes + 256);
        syn::write_slice_header(&header, self.pps_qp, &mut w);
        let motion = match kind {
            Kind::Idr | Kind::I => {
                if cabac {
                    super::h264_cabac_mb::write_intra_picture_cabac(&mut w, &g, &self.tools, qp, &planes, &mut recon)
                } else {
                    let m = super::h264_cavlc_mb::write_intra_picture(&mut w, &g, &self.tools, qp, &planes, &mut recon);
                    w.rbsp_trailing_bits();
                    m
                }
            }
            Kind::P => {
                let r = frame_of(ref0[0])?;
                if cabac {
                    super::h264_cabac_mb::write_p_picture_cabac(&mut w, &g, &self.tools, qp, &planes, &mut recon, &r.frame, None)
                } else {
                    let m = super::h264_cavlc_mb::write_p_picture(&mut w, &g, &self.tools, qp, &planes, &mut recon, &r.frame, None);
                    w.rbsp_trailing_bits();
                    m
                }
            }
            Kind::B => {
                let (r0, r1) = (frame_of(ref0[0])?, frame_of(ref0[1])?);
                let refs2 = [&r0.frame[..], &r1.frame[..]];
                let col = super::h264_pic::Colocated {
                    frame: &r1.col,
                    map: crate::h264::recon::ColMap {
                        cur_parity: PARITY_FRAME,
                        col_parity: PARITY_FRAME,
                        cur_poc: poc,
                        cur_mbaff: false,
                        mb_width: g.mbs_wide as usize,
                    },
                };
                if cabac {
                    super::h264_cabac_mb::write_b_picture_cabac(&mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, &col)
                } else {
                    let m = super::h264_cavlc_mb::write_b_picture(&mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, &col);
                    w.rbsp_trailing_bits();
                    m
                }
            }
        };
        out.extend_from_slice(&syn::annexb(nal_type, nal_ref_idc, &w.into_nal()));
        model_store(&mut model, &f.sps, &hdr, &shared, frame_num, PARITY_FRAME, poc, [top, bottom], c.encode)?;

        let mut rec = Vec::with_capacity(self.frame_bytes);
        syn::crop_into(&recon[0], g.width, g.height, &mut rec);
        for (i, p) in recon.iter().enumerate().skip(1) {
            let (dw, dh) = self.plane_dims[i];
            syn::crop_into(p, dw, dh, &mut rec);
        }
        crate::encode::h264_me::prepare_reference(&mut recon);
        // The frame's motion as the decoder lays out a frame picture's: its
        // own rows, no field macroblock, the frame not field-coded.
        let n = (g.mbs_wide * g.mbs_high) as usize;
        let mut col = Frame::<u8>::empty();
        col.mb_width = g.mbs_wide as usize;
        col.mb_height = g.mbs_high as usize;
        col.motion = motion.frame.motion.clone();
        col.mb_intra = motion.frame.mb_intra.clone();
        col.mb_field = vec![false; n];
        col.field_poc = [top, bottom];
        let fields = [0usize, 1].map(|p| {
            Some(StoredField { planes: extract_field(&recon, p), motion: super::h264_pic::PicMotion::new(0, 0) })
        });
        let census = vec![(kind, motion.info.mbs.clone())];
        Ok(Attempt {
            access: Access { data: out, keyframe: idr, poc: c.poc, encode_index: c.encode, display: c.display },
            rec,
            recon: Vec::new(),
            motion: super::h264_pic::PicMotion::new(0, 0),
            fields: Some(FieldsOut { model, frame: StoredFrame { id, fields, frame: recon, col }, census, frame_coded: true }),
        })
    }

    /// Picture-adaptive frame/field coding: the frame coded both ways and
    /// the cheaper kept, by the Lagrangian every H.264 decision here is
    /// framed as — squared error of the displayed reconstruction against
    /// the source, plus `lambda` (`h264_intra::lambda`) times the bits.
    /// Measured, not estimated: both candidates are fully coded, so each
    /// side of the comparison is what that candidate actually costs.
    ///
    /// The multiplier is scaled by `4^(BitDepth - 8)`, the growth of a
    /// squared error with the sample range, because here both terms are
    /// real — a squared error against real bits — which is exactly the
    /// case the unscaled placeholder multipliers of the mode decisions are
    /// not (`satd_lambda` records why those stay unscaled).
    ///
    /// Ties go to the frame picture, whose header is the smaller.
    fn code_attempt_paff(&self, c: &Coded, src: &[S], qp: u8) -> Result<Attempt<S>> {
        let field = self.code_attempt_fields(c, src, qp)?;
        let frame = self.code_attempt_ilace_frame(c, src, qp)?;
        let scale = f64::from(1u32 << (2 * (self.cfg.bit_depth - 8)));
        let lam = f64::from(super::h264_intra::lambda(i32::from(qp))) * scale;
        let cost = |a: &Attempt<S>| ssd_packed(src, &a.rec) as f64 + lam * (a.access.data.len() * 8) as f64;
        Ok(if cost(&field) < cost(&frame) { field } else { frame })
    }

    /// Indices into `refs` of the nearest reference before and after `poc`.
    ///
    /// Nearest rather than first: a decoder's default list order is by
    /// distance from the current picture, and an encoder that assumes a
    /// different order writes vectors against the wrong picture.
    fn lists_for(&self, poc: i32) -> (Option<usize>, Option<usize>) {
        let past = self
            .refs
            .iter()
            .enumerate()
            .filter(|(_, (p, _, _))| *p < poc)
            .max_by_key(|(_, (p, _, _))| *p)
            .map(|(i, _)| i);
        let future = self
            .refs
            .iter()
            .enumerate()
            .filter(|(_, (p, _, _))| *p > poc)
            .min_by_key(|(_, (p, _, _))| *p)
            .map(|(i, _)| i);
        (past, future)
    }

    /// See [`H264Encoder::rate_report`].
    fn rate_report(&self) -> Option<(f64, f64)> {
        let rc = self.rc.as_ref()?;
        let target = match self.cfg.rate {
            RateControl::Bitrate { bps } => bps as f64,
            _ => return None,
        };
        Some((rc.achieved_bps(self.cfg.fps), target))
    }

    /// See [`H264Encoder::picture_qp`].
    fn picture_qp(&self, kind: Kind) -> u8 {
        match self.cfg.rate {
            // Under a bitrate target the quantiser is not a function of
            // the configuration at all — it is chosen per picture from what
            // the pictures before it actually cost, and `code_picture` asks
            // the controller rather than asking here. This arm reports the
            // neutral quantiser because there is no configured answer to
            // give; a caller wanting the real one has to look at the
            // stream, picture by picture.
            RateControl::Bitrate { .. } => 26,
            RateControl::Lossless => 0,
            RateControl::ConstantQp(q) => match kind {
                // The usual offsets: anchors are coded better than the
                // pictures that reference them, and B pictures that nothing
                // references can afford to be worse.
                Kind::Idr | Kind::I => q.saturating_sub(3),
                Kind::P => q,
                Kind::B => q.saturating_add(2).min(51),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaFormat;
    use crate::encode::Config;

    fn cfg(w: u32, h: u32, chroma: ChromaFormat, depth: u32) -> Config {
        Config { width: w, height: h, chroma, bit_depth: depth, ..Config::default() }
    }

    #[test]
    fn frame_size_matches_every_chroma_format() {
        for (chroma, per_px) in [
            (ChromaFormat::Monochrome, 1.0),
            (ChromaFormat::Yuv420, 1.5),
            (ChromaFormat::Yuv422, 2.0),
            (ChromaFormat::Yuv444, 3.0),
        ] {
            let e = H264Encoder::new(cfg(64, 64, chroma, 8)).unwrap();
            let want = (64.0 * 64.0 * per_px) as usize;
            assert_eq!(e.frame_bytes(), want, "{chroma:?}");
            // Two bytes per sample deeper, whatever the depth.
            for depth in [10u32, 12, 14] {
                let e = H264Encoder::new(cfg(64, 64, chroma, depth)).unwrap();
                assert_eq!(e.frame_bytes(), 2 * want, "{chroma:?} at {depth} bits");
            }
        }
    }

    /// Odd dimensions are the case cropping exists for, and chroma rounds up.
    #[test]
    fn odd_dimensions_round_chroma_up() {
        let e = H264Encoder::new(cfg(50, 34, ChromaFormat::Yuv420, 8)).unwrap();
        assert_eq!(e.frame_bytes(), 50 * 34 + 2 * 25 * 17);
    }

    #[test]
    fn a_bad_configuration_fails_before_anything_is_written() {
        assert!(H264Encoder::new(cfg(0, 64, ChromaFormat::Yuv420, 8)).is_err());
        assert!(H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 7)).is_err());
        assert!(H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 16)).is_err());
    }

    /// Fifteen bits and up stay refused: `Config::validate` bounds the
    /// depth to what the decoder admits, and the refusal names the bound.
    #[test]
    fn deeper_than_fourteen_bits_refuses() {
        let Err(err) = H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 15)) else {
            panic!("15-bit was accepted")
        };
        assert!(format!("{err}").contains("bit depth outside 8..=14"), "{err}");
    }

    #[test]
    fn a_wrong_sized_picture_is_rejected_by_size_not_by_luck() {
        let mut e = H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 8)).unwrap();
        let err = e.push(&vec![0u8; 100]).unwrap_err();
        assert!(format!("{err}").contains("expected"), "{err}");
    }

    /// A source sample above the declared depth is refused by name, not
    /// coded: nothing downstream checks the range, and a wrapped sample
    /// would be a desync far from its cause.
    #[test]
    fn a_sample_above_the_declared_depth_refuses() {
        let mut e = H264Encoder::new(Config { gop: 0, ..cfg(64, 64, ChromaFormat::Yuv420, 10) }).unwrap();
        let mut frame = vec![0u8; e.frame_bytes()];
        frame[..2].copy_from_slice(&1024u16.to_le_bytes());
        let err = e.push(&frame).expect_err("1024 does not fit 10 bits");
        assert!(format!("{err}").contains("H.264 encode: source sample 1024 exceeds the declared 10-bit depth"), "{err}");
    }

    /// Both entropy coders code whole GOPs now — all-intra and IP alike —
    /// so what this pins is that no configuration in that envelope errors,
    /// and that the pictures come out typed as expected.
    #[test]
    fn both_entropy_coders_code_intra_and_inter_gops() {
        let frame = vec![0u8; 64 * 64 * 3 / 2];
        for entropy in [Entropy::Cabac, Entropy::Cavlc] {
            let mut e = H264Encoder::new(Config {
                gop: 0,
                entropy,
                ..cfg(64, 64, ChromaFormat::Yuv420, 8)
            })
            .unwrap();
            let out = e.push(&frame).unwrap();
            assert_eq!(out.len(), 1, "{entropy:?}: an all-intra picture should code");
            assert!(out[0].keyframe, "{entropy:?}: the first picture is an IDR");

            let mut e = H264Encoder::new(Config {
                gop: 8,
                entropy,
                ..cfg(64, 64, ChromaFormat::Yuv420, 8)
            })
            .unwrap();
            let mut coded = 0;
            for i in 0..4 {
                let out = e
                    .push(&frame)
                    .unwrap_or_else(|err| panic!("{entropy:?} inter picture {i}: {err}"));
                coded += out.len();
            }
            assert_eq!(coded, 4, "{entropy:?}: every pushed picture codes");
        }
    }

    /// Every picture offered must be codable exactly once, whatever the GOP
    /// structure. Deriving the display index from the pictures still held
    /// fails this the moment the scheduler releases them as fast as they
    /// arrive, which is what `gop = 0` does.
    #[test]
    fn every_pushed_picture_is_looked_up_by_its_own_index() {
        for (gop, bframes) in [(0, 0), (1, 0), (8, 0), (8, 2)] {
            let mut e = H264Encoder::new(Config {
                gop,
                bframes,
                ..cfg(64, 64, ChromaFormat::Yuv420, 8)
            })
            .unwrap();
            let frame = vec![0u8; 64 * 64 * 3 / 2];
            let mut units = Vec::new();
            for i in 0..6 {
                // Coding, holding, and refusing an unbuilt tool are all
                // legitimate. "scheduler released an absent picture" is not:
                // it means a picture was released under an index the map
                // never held, which is the bookkeeping fault this exists to
                // catch, and it would otherwise look like a coding bug.
                match e.push(&frame) {
                    Ok(u) => units.extend(u),
                    Err(err) => panic!("gop={gop} b={bframes} picture {i}: {err}"),
                }
            }
            // Flushing releases whatever is still held, and must not find a
            // picture missing either.
            match e.flush() {
                Ok(u) => units.extend(u),
                Err(err) => {
                    let s = format!("{err}");
                    assert!(!s.contains("absent picture"), "gop={gop} b={bframes} flush: {s}");
                }
            }
            // Every access unit names the picture it codes by stream-wide
            // display index: the six pushed, each exactly once, and — with
            // B pictures — not in the order they were coded.
            let mut displays: Vec<u64> = units.iter().map(|u| u.display).collect();
            displays.sort_unstable();
            assert_eq!(displays, (0..6).collect::<Vec<u64>>(), "gop={gop} b={bframes}");
            if bframes > 0 {
                assert!(
                    units.iter().any(|u| u.display != u.encode_index),
                    "gop={gop} b={bframes}: no picture was coded out of display order"
                );
            }
        }
    }

    #[test]
    fn lossless_has_no_quantiser_and_b_pictures_are_coded_worse_than_anchors() {
        let e = H264Encoder::new(Config {
            rate: RateControl::Lossless,
            ..cfg(64, 64, ChromaFormat::Yuv420, 8)
        })
        .unwrap();
        assert_eq!(e.picture_qp(Kind::Idr), 0);

        let e = H264Encoder::new(Config {
            rate: RateControl::ConstantQp(26),
            ..cfg(64, 64, ChromaFormat::Yuv420, 8)
        })
        .unwrap();
        assert!(e.picture_qp(Kind::Idr) < e.picture_qp(Kind::P));
        assert!(e.picture_qp(Kind::P) < e.picture_qp(Kind::B));
    }

    /// `count` pictures of `w` by `h` at `bit_depth` bits, packed as
    /// little-endian `u16`, using the whole sample range: a ramp over the
    /// range plus a few low bits of noise, translating a little per
    /// picture so the inter pictures have motion to find. The H.265 side's
    /// own generator, for the same reason it exists there.
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

    /// Deep pictures code and decode: for 10, 12 and 14 bits, every
    /// chroma format, both entropy coders, intra / P / B with the 8x8
    /// transform and the sub-partitions on, lossy and lossless — the
    /// production decoder rebuilds each picture byte for byte from the
    /// stream (SELF, in process), a lossless stream reproduces the source
    /// exactly, the decoded depth is the declared one, and the pictures
    /// really are deep.
    ///
    /// The last clause is the vacuity guard: a 10-bit source whose every
    /// sample fitted 8 bits would round-trip through an encoder that
    /// silently narrowed, so the source is built to use the whole range
    /// and the test asserts the reconstruction does too.
    #[test]
    fn deep_pictures_round_trip_through_the_decoder() {
        for bit_depth in [10u32, 12, 14] {
            for chroma in [ChromaFormat::Monochrome, ChromaFormat::Yuv420, ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
                let frames = deep_frames(64, 64, chroma, bit_depth, 5);
                let max = (1u32 << bit_depth) - 1;
                let deep = |bytes: &[u8]| bytes.chunks_exact(2).any(|p| u32::from(u16::from_le_bytes([p[0], p[1]])) > 255);
                assert!(deep(&frames[0]), "{bit_depth}-bit {chroma:?}: the source never leaves 8 bits");
                assert!(frames[0].chunks_exact(2).all(|p| u32::from(u16::from_le_bytes([p[0], p[1]])) <= max));

                // Lossless is all-IDR: PCM has no inter spelling, so the
                // inter pictures of a lossless stream are all-skip copies
                // of their reference, exact only over still content — the
                // gate's `lossless-intra` row is `--gop 0` for that reason.
                for (rate, gop, bframes, entropy, t8x8) in [
                    (RateControl::ConstantQp(26), 8u32, 0u32, Entropy::Cabac, false),
                    (RateControl::ConstantQp(40), 8, 2, Entropy::Cavlc, true),
                    (RateControl::ConstantQp(20), 8, 2, Entropy::Cabac, true),
                    (RateControl::Lossless, 0, 0, Entropy::Cavlc, false),
                    (RateControl::Lossless, 0, 0, Entropy::Cabac, false),
                ] {
                    let tag = format!("{bit_depth}-bit {chroma:?} {rate:?} gop={gop} bframes={bframes} {entropy:?} t8x8={t8x8}");
                    let mut e = H264Encoder::new(Config {
                        rate,
                        gop,
                        bframes,
                        entropy,
                        transform_8x8: t8x8,
                        subparts: t8x8,
                        ..cfg(64, 64, chroma, bit_depth)
                    })
                    .unwrap_or_else(|err| panic!("{tag}: {err}"));
                    assert_eq!(e.frame_bytes(), frames[0].len(), "{tag}: two bytes per sample");
                    let mut units = Vec::new();
                    for f in &frames {
                        units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                    }
                    units.extend(e.flush().unwrap());
                    assert_eq!(units.len(), frames.len(), "{tag}: one access unit per picture");
                    if gop != 0 {
                        assert!(units[1..].iter().any(|u| !u.keyframe), "{tag}: no inter picture was coded");
                    }
                    if bframes > 0 {
                        assert!(units.iter().any(|u| u.encode_index as usize != (u.poc / 2) as usize), "{tag}: no B picture was held back");
                    }
                    let census = e.shape_census();
                    if rate == RateControl::Lossless {
                        assert_eq!(census.counts[0][3], (64 / 16 * 64 / 16) * frames.len() as u64, "{tag}: lossless is all PCM");
                    } else {
                        // The transform paths were taken, not PCM.
                        assert_eq!(census.counts[0][3], 0, "{tag}: a lossy intra picture coded PCM");
                        assert!(census.counts[1].iter().sum::<u64>() > 0, "{tag}: no P macroblock");
                        if bframes > 0 {
                            assert!(census.counts[2].iter().sum::<u64>() > 0, "{tag}: no B macroblock");
                        }
                    }

                    // SELF, through the production decoder. It emits
                    // display order; the reconstructions are in coding
                    // order, so each decoded picture is matched to the
                    // reconstruction whose access unit carries its POC
                    // (display index `poc / 2`, as `gop.rs` counts it) —
                    // except that every picture of an all-IDR stream
                    // restarts its POC at zero, where coding order *is*
                    // display order.
                    let mut dec = crate::h264::H264Decoder::new();
                    for u in &units {
                        dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: decoder rejected the stream: {err}"));
                    }
                    dec.flush().unwrap_or_else(|err| panic!("{tag}: decoder failed to flush: {err}"));
                    let mut by_display = vec![None; units.len()];
                    for u in &units {
                        let display = if gop == 0 { u.encode_index as usize } else { (u.poc / 2) as usize };
                        by_display[display] = Some(u.encode_index as usize);
                    }
                    for (i, coded) in by_display.iter().enumerate() {
                        let want = &e.reconstructions()[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
                        let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing"));
                        assert_eq!(got.bit_depth, bit_depth, "{tag}: decoded depth");
                        let got = got.into_packed();
                        if got != *want {
                            let at = got.iter().zip(want.iter()).position(|(a, b)| a != b);
                            panic!(
                                "{tag}: picture {i} decoded differently than the encoder reconstructed it ({} vs {} bytes, first difference at byte {at:?}: {:?} vs {:?})",
                                got.len(),
                                want.len(),
                                at.map(|k| &got[k..(k + 8).min(got.len())]),
                                at.map(|k| &want[k..(k + 8).min(want.len())]),
                            );
                        }
                    }
                    assert!(deep(&e.reconstructions()[0]), "{tag}: the reconstruction never leaves 8 bits");
                    if rate == RateControl::Lossless {
                        for (display, coded) in by_display.iter().enumerate() {
                            assert!(
                                e.reconstructions()[coded.unwrap()] == frames[display],
                                "{tag}: picture {display} is not lossless"
                            );
                        }
                    }
                }
            }
        }
    }
    /// Pictures whose four quadrants differ sharply in variance — flat, a
    /// ramp, noise, a checkerboard — so adaptive quantisation has something
    /// to move: the H.265 side's generator, whose quadrants are four
    /// macroblocks each here. `count` frames, drifting a little each so the
    /// inter pictures carry residual.
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

    /// Encode `frames` under `config` and hold the stream to the encoder's
    /// reconstructions with [`self_check`], returning the access units and
    /// the census.
    fn encode_and_self_check(tag: &str, config: Config, frames: &[Vec<u8>]) -> (Vec<Access>, ShapeCensus) {
        let gop = config.gop;
        let mut e = H264Encoder::new(config).unwrap_or_else(|err| panic!("{tag}: {err}"));
        let mut units = Vec::new();
        for f in frames {
            units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
        }
        units.extend(e.flush().unwrap_or_else(|err| panic!("{tag}: {err}")));
        assert_eq!(units.len(), frames.len(), "{tag}: one access unit per picture");
        self_check(tag, gop, &units, e.reconstructions());
        (units, e.shape_census().clone())
    }

    /// Decode `units` with the production decoder and hold every picture to
    /// the reconstruction the encoder kept for it (SELF, in process) —
    /// matched through each access unit's POC, as
    /// `deep_pictures_round_trip_through_the_decoder` matches them.
    fn self_check(tag: &str, gop: u32, units: &[Access], recons: &[Vec<u8>]) {
        let mut dec = crate::h264::H264Decoder::new();
        for u in units {
            dec.push_annexb(&u.data).unwrap_or_else(|err| panic!("{tag}: decoder rejected the stream: {err}"));
        }
        dec.flush().unwrap_or_else(|err| panic!("{tag}: decoder failed to flush: {err}"));
        let mut by_display = vec![None; units.len()];
        for u in units {
            let display = if gop == 0 { u.encode_index as usize } else { (u.poc / 2) as usize };
            by_display[display] = Some(u.encode_index as usize);
        }
        for (i, coded) in by_display.iter().enumerate() {
            let want = &recons[coded.unwrap_or_else(|| panic!("{tag}: display index {i} never coded"))];
            let got = dec.next_picture().unwrap_or_else(|| panic!("{tag}: picture {i} missing")).into_packed();
            assert!(got == *want, "{tag}: picture {i} decoded differently than the encoder reconstructed it");
        }
    }

    /// Adaptive quantisation — a quantiser per macroblock, carried by
    /// `mb_qp_delta` — round-trips through the production decoder for
    /// intra, P and B pictures, both entropy coders, every chroma format,
    /// the 8x8 transform and the sub-partitions, at 8, 10 and 14 bits and
    /// on both sides of the chroma QP table's identity region; and the
    /// census proves it moved something: a stream whose every delta was
    /// zero, or whose every macroblock kept the picture quantiser, would
    /// pass SELF while proving only the syntax.
    ///
    /// SELF is the check that matters: a residual coded at one quantiser
    /// and scaled at another desyncs the reconstruction, and a
    /// residual-free macroblock filtered at a quantiser a decoder does not
    /// hold for it moves the deblocked samples.
    #[test]
    fn adaptive_quantisation_round_trips_and_moves_the_quantiser() {
        for (chroma, bit_depth, bframes, entropy, tools) in [
            (ChromaFormat::Yuv420, 8u32, 0u32, Entropy::Cabac, false),
            (ChromaFormat::Yuv420, 8, 2, Entropy::Cavlc, false),
            (ChromaFormat::Yuv420, 8, 2, Entropy::Cabac, true),
            (ChromaFormat::Yuv422, 8, 0, Entropy::Cavlc, true),
            (ChromaFormat::Yuv444, 8, 2, Entropy::Cabac, false),
            (ChromaFormat::Yuv444, 8, 0, Entropy::Cavlc, true),
            (ChromaFormat::Monochrome, 8, 0, Entropy::Cabac, false),
            (ChromaFormat::Yuv420, 10, 2, Entropy::Cabac, true),
            (ChromaFormat::Yuv420, 10, 0, Entropy::Cavlc, false),
            (ChromaFormat::Yuv422, 14, 2, Entropy::Cavlc, true),
        ] {
            let frames = aq_frames(chroma, bit_depth, 6);
            for qp in [22u8, 40] {
                let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes} {entropy:?} t8x8+subparts={tools} qp {qp}");
                let (_, census) = encode_and_self_check(
                    &tag,
                    Config {
                        gop: 8,
                        bframes,
                        entropy,
                        transform_8x8: tools,
                        subparts: tools,
                        rate: RateControl::ConstantQp(qp),
                        aq_strength: 2.0,
                        ..cfg(64, 64, chroma, bit_depth)
                    },
                    &frames,
                );
                for (pic, name) in [(0usize, "intra"), (1, "P")] {
                    assert!(census.qp_moved[pic] > 0, "{tag}: no {name} macroblock left the picture quantiser: {census:?}");
                    assert!(census.qp_delta[pic] > 0, "{tag}: no {name} macroblock coded a non-zero mb_qp_delta: {census:?}");
                }
                if bframes > 0 {
                    assert!(census.pictures[2] > 0, "{tag}: no B picture was coded");
                }
            }
        }

        // Strength 0 is off: the stream is what the encoder writes with the
        // switch absent, and the census says nothing moved.
        let frames = aq_frames(ChromaFormat::Yuv420, 8, 3);
        for entropy in [Entropy::Cabac, Entropy::Cavlc] {
            let encode = |strength: f32| -> (Vec<u8>, ShapeCensus) {
                let mut e = H264Encoder::new(Config { gop: 8, entropy, aq_strength: strength, ..cfg(64, 64, ChromaFormat::Yuv420, 8) }).unwrap();
                let mut out = Vec::new();
                for f in &frames {
                    for u in e.push(f).unwrap() {
                        out.extend_from_slice(&u.data);
                    }
                }
                for u in e.flush().unwrap() {
                    out.extend_from_slice(&u.data);
                }
                (out, e.shape_census().clone())
            };
            let (off, census) = encode(0.0);
            assert_eq!(off, encode(Config::default().aq_strength).0, "{entropy:?}");
            assert_eq!(census.qp_moved, [0; 3], "{entropy:?}: nothing moves with the switch off");
            assert_eq!(census.qp_delta, [0; 3], "{entropy:?}");
            assert_ne!(off, encode(1.0).0, "{entropy:?}: strength 1 must change the stream");
        }

        // Lossless has no quantiser to adapt: refused by name.
        let err = H264Encoder::new(Config { rate: RateControl::Lossless, aq_strength: 1.0, ..cfg(64, 64, ChromaFormat::Yuv420, 8) })
            .err()
            .expect("adaptive quantisation on a lossless stream must refuse");
        assert!(format!("{err}").contains("adaptive quantisation"), "{err}");
    }
    /// A textured picture fading a step per frame — a gain and an offset
    /// together, `base * (1 - i/16) - 2i`, chroma untouched — in `chroma` at
    /// `bit_depth`, `count` frames: a fitted weighting has both halves of
    /// its table to carry here, which a pure gain would not give it.
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
                        let base = 60 + ((x * 3 + y * 5 + (x * y) / 7) % 150) as i32 + ((seed >> 28) as i32 - 8);
                        let v = (f64::from(base) * gain - 2.0 * i as f64).round().clamp(0.0, 255.0) as u32;
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

    /// Explicit weighted prediction on a fade: the P pictures choose a
    /// weighting (census `wp_on`), it holds macroblock by macroblock
    /// (`wp_won` above `wp_lost`, the model check), the stream is markedly
    /// smaller than the same encode without it, and it round-trips through
    /// the production decoder — both entropy coders, every chroma format,
    /// with B pictures (which stay default-weighted) and sub-partitions and
    /// the 8x8 transform, at 8, 10 and 12 bits. On a held clip nothing is
    /// chosen and the table of defaults costs its flags. Lossless refuses.
    #[test]
    fn weighted_prediction_pays_on_a_fade_and_round_trips() {
        for (chroma, bit_depth, bframes, entropy, tools) in [
            (ChromaFormat::Yuv420, 8u32, 0u32, Entropy::Cabac, false),
            (ChromaFormat::Yuv420, 8, 2, Entropy::Cavlc, false),
            (ChromaFormat::Yuv422, 8, 0, Entropy::Cavlc, true),
            (ChromaFormat::Yuv444, 8, 0, Entropy::Cabac, true),
            (ChromaFormat::Monochrome, 8, 0, Entropy::Cabac, false),
            (ChromaFormat::Yuv420, 10, 2, Entropy::Cabac, true),
            (ChromaFormat::Yuv444, 12, 0, Entropy::Cavlc, false),
        ] {
            let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes} {entropy:?} t8x8+subparts={tools}");
            let frames = fade_frames(chroma, bit_depth, 8);
            let run = |weighted_pred: bool| -> (Vec<Access>, Vec<Vec<u8>>, ShapeCensus) {
                let mut e = H264Encoder::new(Config {
                    gop: 8,
                    bframes,
                    entropy,
                    transform_8x8: tools,
                    subparts: tools,
                    weighted_pred,
                    ..cfg(64, 64, chroma, bit_depth)
                })
                .unwrap_or_else(|err| panic!("{tag}: {err}"));
                let mut units = Vec::new();
                for f in &frames {
                    units.extend(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                }
                units.extend(e.flush().unwrap_or_else(|err| panic!("{tag}: {err}")));
                (units, e.reconstructions().to_vec(), e.shape_census().clone())
            };
            let (with, recon, census) = run(true);
            let (without, _, plain) = run(false);
            let bytes = |u: &[Access]| u.iter().map(|a| a.data.len()).sum::<usize>();
            assert!(census.wp_on[1] > 0, "{tag}: no P picture chose a weighting: {census:?}");
            assert!(census.wp_won[1] > census.wp_lost[1], "{tag}: the fit lost more macroblocks than it won: {census:?}");
            assert_eq!(census.wp_on[2], 0, "{tag}: a B picture carries no table");
            assert_eq!((plain.wp_on, plain.wp_won, plain.wp_lost), ([0; 3], [0; 3], [0; 3]), "{tag}: the census counts nothing with the switch off");
            assert!(
                (bytes(&with) as f64) < (bytes(&without) as f64) * 0.9,
                "{tag}: weighting saved little on a fade: {} against {} bytes",
                bytes(&with),
                bytes(&without)
            );
            self_check(&tag, 8, &with, &recon);
        }

        // A held clip: nothing is chosen, and the table of defaults costs its
        // flags and its denominators, a byte or two per P slice.
        let held: Vec<Vec<u8>> = std::iter::repeat_n(fade_frames(ChromaFormat::Yuv420, 8, 1).remove(0), 6).collect();
        let encode = |weighted_pred: bool| -> (usize, ShapeCensus) {
            let mut e = H264Encoder::new(Config { gop: 8, weighted_pred, ..cfg(64, 64, ChromaFormat::Yuv420, 8) }).unwrap();
            let mut bytes = 0;
            for f in &held {
                bytes += e.push(f).unwrap().iter().map(|a| a.data.len()).sum::<usize>();
            }
            bytes += e.flush().unwrap().iter().map(|a| a.data.len()).sum::<usize>();
            (bytes, e.shape_census().clone())
        };
        let (with, census) = encode(true);
        let (without, _) = encode(false);
        assert_eq!(census.wp_on[1], 0, "a held clip chose a weighting: {census:?}");
        assert!(with >= without && with <= without + 2 * held.len(), "the table of defaults should cost bits, not bytes: {with} against {without}");

        let err = H264Encoder::new(Config { rate: RateControl::Lossless, weighted_pred: true, ..cfg(64, 64, ChromaFormat::Yuv420, 8) })
            .err()
            .expect("weighted prediction on a lossless stream must refuse");
        assert!(format!("{err}").contains("weighted prediction"), "{err}");
    }
    /// `count` interlaced frames of `w` by `h` at `bit_depth`, packed as the
    /// encoder takes them: every row of a frame drawn at its own field's
    /// instant — the top field at time `2f`, the bottom at `2f + 1` — from a
    /// texture moving three samples right and one down per field, so the
    /// two fields of a frame disagree the way captured interlaced video's
    /// do. Chroma rows alternate fields as luma rows do. Deeper than 8 bits
    /// the low bits carry a ramp, so the depth is really used.
    fn interlaced_frames(w: usize, h: usize, chroma: ChromaFormat, bit_depth: u32, count: usize) -> Vec<Vec<u8>> {
        woven_frames(w, h, chroma, bit_depth, count, 1)
    }

    /// [`interlaced_frames`] with the two fields `field_gap` instants apart:
    /// 1 is interlaced capture, 0 a progressive frame merely stored as
    /// fields — the same motion, but each frame one instant.
    fn woven_frames(w: usize, h: usize, chroma: ChromaFormat, bit_depth: u32, count: usize, field_gap: usize) -> Vec<Vec<u8>> {
        let (sw, sh) = match chroma {
            ChromaFormat::Yuv420 => (2usize, 2usize),
            ChromaFormat::Yuv422 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = if chroma == ChromaFormat::Monochrome { (0, 0) } else { (w / sw, h / sh) };
        let shift = bit_depth - 8;
        let at = |x: usize, y: usize, t: usize, c: usize| -> u32 {
            let (x, y) = (x + 3 * t, y + t);
            ((x * 7 + y * 13 + (x * y) / 5 + c * 37) % 200) as u32 + 28
        };
        (0..count)
            .map(|f| {
                let mut s: Vec<u32> = Vec::with_capacity(w * h + 2 * cw * ch);
                for y in 0..h {
                    for x in 0..w {
                        s.push(at(x, y, 2 * f + field_gap * (y % 2), 0));
                    }
                }
                for c in 1..3 {
                    for y in 0..ch {
                        for x in 0..cw {
                            s.push(at(x * sw, y * sh, 2 * f + field_gap * (y % 2), c) / 2 + 64);
                        }
                    }
                }
                if shift == 0 {
                    s.iter().map(|&v| v as u8).collect()
                } else {
                    s.iter()
                        .enumerate()
                        .flat_map(|(i, &v)| (((v << shift) | (i as u32 & ((1 << shift) - 1))) as u16).to_le_bytes())
                        .collect()
                }
            })
            .collect()
    }

    /// Every frame coded as two field pictures round-trips through the
    /// production decoder (SELF, in process — the decoder pairs the fields
    /// and outputs the frame): I, P and B frames, both entropy coders, both
    /// field orders, every chroma format, one and two reference frames,
    /// the 8x8 transform and the sub-partitions, at 8 and 10 bits. The
    /// census proves fields were coded, two per frame, and that the inter
    /// paths ran.
    #[test]
    fn field_pictures_round_trip_through_the_decoder() {
        use crate::encode::{FieldCoding, FieldOrder};
        for (chroma, bit_depth, bframes, entropy, tools, order, max_refs) in [
            (ChromaFormat::Yuv420, 8u32, 0u32, Entropy::Cabac, false, FieldOrder::TopFirst, 1u32),
            (ChromaFormat::Yuv420, 8, 0, Entropy::Cavlc, true, FieldOrder::BottomFirst, 1),
            (ChromaFormat::Yuv420, 8, 2, Entropy::Cabac, true, FieldOrder::TopFirst, 1),
            (ChromaFormat::Yuv420, 8, 2, Entropy::Cavlc, false, FieldOrder::BottomFirst, 2),
            (ChromaFormat::Yuv422, 8, 0, Entropy::Cavlc, true, FieldOrder::TopFirst, 2),
            (ChromaFormat::Yuv444, 8, 2, Entropy::Cabac, false, FieldOrder::BottomFirst, 1),
            (ChromaFormat::Monochrome, 8, 0, Entropy::Cabac, true, FieldOrder::TopFirst, 1),
            (ChromaFormat::Yuv420, 10, 2, Entropy::Cabac, true, FieldOrder::BottomFirst, 1),
        ] {
            let tag = format!("{chroma:?} {bit_depth}-bit bframes={bframes} {entropy:?} tools={tools} {order:?} refs={max_refs}");
            let frames = interlaced_frames(64, 64, chroma, bit_depth, 7);
            let (_, census) = encode_and_self_check(
                &tag,
                Config {
                    gop: 8,
                    bframes,
                    entropy,
                    transform_8x8: tools,
                    subparts: tools,
                    max_refs,
                    interlace: Some(order),
                    field_coding: FieldCoding::Field,
                    ..cfg(64, 64, chroma, bit_depth)
                },
                &frames,
            );
            assert_eq!(census.field_pictures, 2 * frames.len() as u64, "{tag}: two field pictures per frame: {census:?}");
            assert!(census.counts[1].iter().sum::<u64>() > 0, "{tag}: no P field macroblock: {census:?}");
            if bframes > 0 {
                assert!(census.pictures[2] > 0, "{tag}: no B field was coded: {census:?}");
            }
        }
    }

    /// Picture-adaptive frame/field coding round-trips through the
    /// production decoder whatever it chooses — frame pictures over
    /// field-coded references and field pictures over frame-coded ones, in
    /// P and B frames (the B pictures' colocated motion crossing the two
    /// kinds), both entropy coders and field orders — and it chooses both
    /// ways: fields somewhere on interlaced content, whose fields are
    /// instants apart, and frames somewhere on progressive content stored
    /// as fields. A decision that always chose one would pass SELF and
    /// prove nothing about the other.
    #[test]
    fn picture_adaptive_coding_chooses_both_and_round_trips() {
        use crate::encode::{FieldCoding, FieldOrder};
        let mut totals = [[0u64; 2]; 2]; // [content][field, frame]
        for (content, gap) in [(0usize, 1usize), (1, 0)] {
            for (chroma, bit_depth, bframes, entropy, tools, order) in [
                (ChromaFormat::Yuv420, 8u32, 2u32, Entropy::Cabac, false, FieldOrder::TopFirst),
                (ChromaFormat::Yuv420, 8, 2, Entropy::Cavlc, true, FieldOrder::BottomFirst),
                (ChromaFormat::Yuv420, 8, 0, Entropy::Cabac, true, FieldOrder::BottomFirst),
                (ChromaFormat::Yuv422, 10, 2, Entropy::Cabac, false, FieldOrder::TopFirst),
            ] {
                let tag = format!("gap {gap} {chroma:?} {bit_depth}-bit bframes={bframes} {entropy:?} tools={tools} {order:?}");
                let frames = woven_frames(64, 64, chroma, bit_depth, 7, gap);
                let (_, census) = encode_and_self_check(
                    &tag,
                    Config {
                        gop: 8,
                        bframes,
                        entropy,
                        transform_8x8: tools,
                        subparts: tools,
                        interlace: Some(order),
                        field_coding: FieldCoding::Paff,
                        ..cfg(64, 64, chroma, bit_depth)
                    },
                    &frames,
                );
                assert_eq!(census.field_pictures + 2 * census.frame_pictures, 2 * frames.len() as u64, "{tag}: every frame coded once: {census:?}");
                totals[content][0] += census.field_pictures;
                totals[content][1] += census.frame_pictures;
            }
        }
        assert!(totals[0][0] > 0, "no field picture chosen on interlaced content: {totals:?}");
        assert!(totals[1][1] > 0, "no frame picture chosen on progressive content: {totals:?}");
    }

    /// Interlaced coding refuses by name what it cannot deliver: a height
    /// an interlaced crop cannot reach (four rows at a time in 4:2:0, two
    /// otherwise), and on H.265 — which has no interlaced coding tools —
    /// the switch itself. A progressive configuration is untouched by the
    /// new fields' defaults.
    #[test]
    fn interlaced_configurations_refuse_by_name() {
        use crate::encode::{FieldCoding, FieldOrder};
        for (chroma, height, ok) in [
            (ChromaFormat::Yuv420, 62u32, false),
            (ChromaFormat::Yuv420, 63, false),
            (ChromaFormat::Yuv422, 63, false),
            (ChromaFormat::Monochrome, 33, false),
            (ChromaFormat::Yuv420, 60, true),
            (ChromaFormat::Yuv444, 62, true),
        ] {
            let c = Config { interlace: Some(FieldOrder::TopFirst), field_coding: FieldCoding::Field, ..cfg(64, height, chroma, 8) };
            let err = H264Encoder::new(c).err().map(|e| format!("{e}")).unwrap_or_default();
            assert_eq!(!err.contains("crops in units of"), ok, "{chroma:?} height {height}: {err}");
        }
        let err = crate::encode::h265::H265Encoder::new(Config { interlace: Some(FieldOrder::BottomFirst), ..cfg(64, 64, ChromaFormat::Yuv420, 8) })
            .err()
            .expect("H.265 has no interlaced tools");
        assert!(format!("{err}").contains("H.265 encode: interlaced coding"), "{err}");
        assert_eq!(Config::default().interlace, None, "progressive by default");
    }

    /// A lookahead is refused by name on H.264, with a bitrate target (where
    /// it would otherwise mean something) as without one (where the
    /// configuration's own check names the missing target first).
    #[test]
    fn a_lookahead_is_refused_by_name() {
        let err = H264Encoder::new(Config {
            lookahead: 8,
            rate: RateControl::Bitrate { bps: 64_000 },
            ..cfg(64, 64, ChromaFormat::Yuv420, 8)
        })
        .err()
        .expect("an H.264 lookahead must refuse");
        assert!(format!("{err}").contains("rate lookahead is not calibrated for H.264"), "{err}");
        let err = H264Encoder::new(Config { lookahead: 4, ..cfg(64, 64, ChromaFormat::Yuv420, 8) })
            .err()
            .expect("a lookahead at a constant quantiser must refuse");
        assert!(format!("{err}").contains("lookahead"), "{err}");
    }
}
