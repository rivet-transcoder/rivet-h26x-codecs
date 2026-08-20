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
//! and real compression on both picture types through both entropy coders:
//! intra and P pictures go through prediction, transform, quantisation and
//! the loop filter, decided once in the shared walks of
//! [`super::h264_pic`] and spelled by the CAVLC
//! ([`super::h264_cavlc_mb`]) or CABAC ([`super::h264_cabac_mb`]) writers.
//! What still codes as `I_PCM` does so for a stated reason each —
//! lossless, because PCM *is* the exact mode and the transform path is
//! lossy; and 4:4:4, which the intra coder does not cover yet. B pictures
//! are all-skip — and so are the P pictures of a stream that *has* B
//! pictures, because an all-skip B assumes zero motion in its colocated
//! picture (see `transform_p` below). `tools/verify_encode.sh` reports
//! each hole as what it is, which is the honest state: the plumbing is
//! proven and every hole has a name.

use super::gop::{Coded, Kind, Scheduler};
use super::h264_syntax as syn;
use super::{Access, Config, Entropy, RateControl};
use crate::bitwriter::BitWriter;
use crate::{Error, Result};

/// H.264 encoder. See the module documentation for what is and is not built.
pub struct H264Encoder {
    cfg: Config,
    sched: Scheduler,
    /// Source pictures held in display order, indexed by display position, so
    /// that a B picture held back by the scheduler still has its samples when
    /// its anchor arrives.
    held: std::collections::BTreeMap<u64, Vec<u8>>,
    /// Reconstructions, in coding order, for the SELF check.
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
    /// Beside each picture's planes, its per-macroblock motion record (the
    /// walk's [`super::h264_deblock::FilterMb`]s): what a later B picture's
    /// spatial direct derivation reads as colocated motion.
    /// INVARIANT(16x16-only): one record per macroblock is a faithful
    /// motion history only while every partition is 16x16 — see the type.
    refs: Vec<(i32, Vec<syn::Recon>, Vec<super::h264_deblock::FilterMb>)>,
    /// `frame_num`, which counts *reference* pictures and wraps.
    frame_num: u32,
    idr_pic_id: u32,
    /// Plane sizes of one source picture, derived once.
    plane_dims: Vec<(u32, u32)>,
    /// Kernels and derived tables for the transform intra path, built once.
    tools: super::h264_pic::IntraTools,
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
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        if cfg.bit_depth > 8 {
            // The reconstruction planes are u8, so anything deeper would be
            // silently truncated. The decoder handles 8 to 14 and this must
            // too, but a narrowed stream that looks legal is worse than a
            // refusal that names itself.
            return Err(Error::unsupported(
                "H.264 encode: bit depth above 8 (encoder in progress)",
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
        let bps = if cfg.bit_depth > 8 { 2 } else { 1 };
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
        let geom = syn::Geometry::new(&cfg);
        let mut plane_dims = vec![(cfg.width, cfg.height)];
        if cfg.chroma != crate::ChromaFormat::Monochrome {
            let cw = (cfg.width as usize).div_ceil(sw as usize) as u32;
            let chh = (cfg.height as usize).div_ceil(sh as usize) as u32;
            plane_dims.push((cw, chh));
            plane_dims.push((cw, chh));
        }
        Ok(Self {
            cfg,
            sched,
            held: std::collections::BTreeMap::new(),
            recon: Vec::new(),
            frame_bytes: (luma + chroma) * bps,
            next_display: 0,
            geom,
            refs: Vec::new(),
            frame_num: 0,
            idr_pic_id: 0,
            plane_dims,
            tools: super::h264_pic::IntraTools::new(),
        })
    }

    /// How many bytes one source picture must be.
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// The reconstructions produced so far, in coding order. The SELF property
    /// compares these against decoding the bitstream.
    pub fn reconstructions(&self) -> &[Vec<u8>] {
        &self.recon
    }

    /// Offer the next picture in display order. Returns whatever became
    /// codable — nothing when the picture is a B held for its anchor, several
    /// when an anchor releases the Bs behind it.
    pub fn push(&mut self, picture: &[u8]) -> Result<Vec<Access>> {
        if picture.len() != self.frame_bytes {
            return Err(Error::bitstream(format!(
                "H.264 encode: picture is {} bytes, expected {}",
                picture.len(),
                self.frame_bytes
            )));
        }
        // Insert before scheduling: the scheduler may release this very
        // picture (every picture is an IDR when `gop` is 0), and `code` looks
        // the samples up by display index.
        let display = self.next_display;
        self.next_display += 1;
        self.held.insert(display, picture.to_vec());
        let ready = self.sched.push();
        self.code(ready)
    }

    /// Code everything still held back.
    pub fn flush(&mut self) -> Result<Vec<Access>> {
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
            out.push(self.code_picture(c, &src)?);
        }
        Ok(out)
    }

    /// Code one picture: parameter sets, slice header, then the slice data
    /// of whichever path the configuration selects — the transform writers
    /// (either entropy coder), PCM where exactness or 4:4:4 demands it, or
    /// the all-skip inter fallback.
    fn code_picture(&mut self, c: Coded, src: &[u8]) -> Result<Access> {
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
        let mut recon: Vec<syn::Recon> =
            vec![syn::recon_plane(g.coded_width, g.coded_height, crate::h264::frame::LUMA_PAD)];
        if cw != 0 {
            let pad = crate::h264::frame::CHROMA_PAD;
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
            recon.push(syn::recon_plane(g.mbs_wide * cw, g.mbs_high * ch, pad));
        }

        // An IDR restarts the count, and the header below carries the reset
        // value. Doing this after writing would send the *previous* GOP's
        // frame_num in the one picture that must not carry it.
        if idr {
            self.frame_num = 0;
        }
        let qp = self.picture_qp(c.kind);
        // Whether this picture takes the transform intra path rather than
        // I_PCM. Lossless stays PCM because PCM is the exactly-lossless mode
        // and the transform path quantises; 4:4:4 stays PCM because
        // `code_macroblock` has no ChromaArrayType 3 path (chroma coded like
        // luma, a fourth residual layout); CABAC intra stays PCM until its
        // macroblock writer exists.
        let transform_intra = idr
            && matches!(self.cfg.rate, RateControl::ConstantQp(_))
            && self.cfg.chroma != crate::ChromaFormat::Yuv444;
        // Whether a P picture takes the motion-search path rather than
        // all-skip. The same envelope as the intra transform path, plus
        // `bframes == 0`: a stream with B pictures must keep its P motion
        // zero, because the all-skip B reconstruction below assumes the
        // colocated picture's motion is zero — temporal direct reads the
        // future reference's vectors, and a B_Skip over a P with real
        // motion reconstructs *from those vectors* in a decoder. Coding
        // real B pictures (or replicating direct derivation) lifts this.
        let transform_p = !idr
            && c.kind == Kind::P
            && matches!(self.cfg.rate, RateControl::ConstantQp(_))
            && self.cfg.chroma != crate::ChromaFormat::Yuv444;
        // B pictures share the envelope: inside it every picture type is
        // transform-coded, so a colocated picture's motion is always the
        // real record the direct derivation needs; outside it everything
        // is PCM or all-skip, where the zero-colocated assumption of the
        // temporal-direct fallback below still holds. The old bframes==0
        // hold-back on P is gone for exactly this reason — its ceiling was
        // the all-skip B reconstruction assuming zero colocated motion,
        // and real B pictures model colocated motion instead.
        let transform_b = !idr
            && c.kind == Kind::B
            && matches!(self.cfg.rate, RateControl::ConstantQp(_))
            && self.cfg.chroma != crate::ChromaFormat::Yuv444;
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            3,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_FRAME_NUM, LOG2_MAX_POC_LSB),
        ));
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, 3, &syn::write_pps(&self.cfg, qp)));

        let cabac = self.cfg.entropy == Entropy::Cabac;
        let mut w = BitWriter::with_capacity(self.frame_bytes + 256);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                frame_num: self.frame_num,
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
            },
            qp,
            &mut w,
        );
        // Every coded picture leaves a per-macroblock motion record: the
        // transform walks return the real one; the PCM and all-skip paths
        // synthesize what a decoder would store for them (intra; skip at
        // reference 0 with zero vectors — both lists for a B skip).
        let total_mbs = (g.mbs_wide * g.mbs_high) as usize;
        let synth = |kind: crate::h264::mb::MbKind, l0: bool, l1: bool| {
            vec![
                super::h264_deblock::FilterMb {
                    kind,
                    nz_mask: if kind == crate::h264::mb::MbKind::IPcm { 0xffff } else { 0 },
                    l0: l0.then_some(crate::h264::frame::Mv::ZERO),
                    l1: l1.then_some(crate::h264::frame::Mv::ZERO),
                };
                total_mbs
            ]
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
            let col = &self.refs[p1].2;
            if cabac {
                motion = super::h264_cabac_mb::write_b_picture_cabac(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, col,
                );
            } else {
                motion = super::h264_cavlc_mb::write_b_picture(
                    &mut w, &g, &self.tools, qp, &planes, &mut recon, refs2, col,
                );
                w.rbsp_trailing_bits();
            }
        } else {
            // Every macroblock skipped — still what B pictures do, and what
            // P pictures fall back to outside the transform envelope.
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
                            *d = ((x as u16 + y as u16 + 1) >> 1) as u8;
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
        self.recon.push(cropped);

        if c.reference {
            self.frame_num = (self.frame_num + 1) & ((1 << LOG2_MAX_FRAME_NUM) - 1);
        }
        if idr {
            self.idr_pic_id ^= 1;
        }
        if c.reference {
            // Replicate the borders motion compensation reads — once, here,
            // so every stored reference is search-ready and the P path never
            // has to wonder whether a plane's border is stale.
            crate::encode::h264_me::prepare_reference(&mut recon);
            self.refs.push((c.poc, recon, motion));
            // The DPB the SPS declares. Dropping the oldest keeps the encoder
            // inside what it told the decoder to allocate.
            let cap = (self.cfg.max_refs.max(1) as usize) + 1;
            while self.refs.len() > cap {
                self.refs.remove(0);
            }
        }
        Ok(Access {
            data: out,
            keyframe: idr,
            poc: c.poc,
            encode_index: c.encode,
        })
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

    /// The quantiser this picture is coded at, before any adaptive
    /// adjustment. Lossless is signalled separately, so it has no QP of its
    /// own and reports the lowest.
    pub fn picture_qp(&self, kind: Kind) -> u8 {
        match self.cfg.rate {
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
        }
    }

    /// The decoder handles 8 to 14 bits and this encoder does not yet, because
    /// its reconstruction planes are u8 and anything deeper would be silently
    /// narrowed. It must refuse by name rather than emit a stream whose
    /// samples were truncated on the way through.
    #[test]
    fn deeper_than_eight_bits_refuses_rather_than_truncating() {
        for depth in [10u32, 12, 14] {
            // A match, not unwrap_err: the Ok side is the encoder itself,
            // and making it Debug would print whole reconstruction planes.
            match H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, depth)) {
                Ok(_) => panic!("{depth}-bit was accepted; samples would be truncated"),
                Err(err) => {
                    let s = format!("{err}");
                    assert!(s.contains("bit depth above 8"), "{depth}-bit: {s}");
                }
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

    #[test]
    fn a_wrong_sized_picture_is_rejected_by_size_not_by_luck() {
        let mut e = H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 8)).unwrap();
        let err = e.push(&vec![0u8; 100]).unwrap_err();
        assert!(format!("{err}").contains("expected"), "{err}");
    }

    /// Both entropy coders code whole GOPs now — all-intra and IP alike —
    /// so what this pins is that no configuration in that envelope errors,
    /// and that the pictures come out typed as expected. The named holes
    /// that remain (4:4:4 transform coding, B pictures with real motion)
    /// are stated in the module docs rather than asserted here, because a
    /// hole is verified by the encode gate reporting it, not by a unit
    /// test guarding its error string.
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
            for i in 0..6 {
                // Coding, holding, and refusing an unbuilt tool are all
                // legitimate. "scheduler released an absent picture" is not:
                // it means a picture was released under an index the map
                // never held, which is the bookkeeping fault this exists to
                // catch, and it would otherwise look like a coding bug.
                if let Err(err) = e.push(&frame) {
                    panic!("gop={gop} b={bframes} picture {i}: {err}");
                }
            }
            // Flushing releases whatever is still held, and must not find a
            // picture missing either.
            if let Err(err) = e.flush() {
                let s = format!("{err}");
                assert!(!s.contains("absent picture"), "gop={gop} b={bframes} flush: {s}");
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
}
