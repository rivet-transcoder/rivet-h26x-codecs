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
//! Everything above entropy coding is here: configuration, picture typing and
//! coding order, and the access-unit envelope. What is missing is the part
//! that writes bits, and it is missing deliberately rather than by oversight —
//! the bitstream writer and the CABAC encoder are being built alongside this,
//! and wiring a half-written entropy coder in early would mean the gate could
//! not tell "no encoder yet" from "encoder is wrong".
//!
//! So [`H264Encoder::push`] refuses with `Unsupported` at exactly the point
//! where a coded slice would be written. `tools/verify_encode.sh` reports that
//! as ENCODE-FAIL with the reason, which is the honest state: the plumbing is
//! proven and the hole has a name.

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
    /// `frame_num`, which counts *reference* pictures and wraps.
    frame_num: u32,
    idr_pic_id: u32,
    /// Plane sizes of one source picture, derived once.
    plane_dims: Vec<(u32, u32)>,
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
        if cfg.bit_depth > 14 {
            return Err(Error::unsupported("H.264 encode: bit depth above 14"));
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
            frame_num: 0,
            idr_pic_id: 0,
            plane_dims,
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

    /// Code one picture.
    ///
    /// Every macroblock is `I_PCM` for now: samples carried raw, no
    /// prediction, no transform, no residual. That is a legal H.264 stream
    /// and an exactly lossless one, which is what makes it the right first
    /// output — it proves the parameter sets, the slice header, the
    /// macroblock layer, NAL framing and emulation prevention against a real
    /// decoder before any question of quality exists.
    fn code_picture(&mut self, c: Coded, src: &[u8]) -> Result<Access> {
        if self.cfg.entropy == Entropy::Cabac {
            // I_PCM through CABAC needs the mb_type binarisation and an
            // engine re-initialisation after each PCM block. Worth doing, and
            // not yet done; saying so beats emitting a stream that is subtly
            // wrong.
            return Err(Error::unsupported(
                "H.264 encode: CABAC slice writing (encoder in progress; --cavlc works)",
            ));
        }
        let g = self.geom;
        let idr = c.kind == Kind::Idr;
        if !idr {
            // Inter prediction is the next thing built. Until then only the
            // all-intra configurations produce a stream.
            return Err(Error::unsupported(
                "H.264 encode: inter prediction (encoder in progress; --gop 0 works)",
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
        let mut recon: Vec<syn::PaddedPlane> = vec![syn::PaddedPlane::new(g.coded_width, g.coded_height)];
        if cw != 0 {
            recon.push(syn::PaddedPlane::new(g.mbs_wide * cw, g.mbs_high * ch));
            recon.push(syn::PaddedPlane::new(g.mbs_wide * cw, g.mbs_high * ch));
        }

        let qp = self.picture_qp(c.kind);
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            3,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_FRAME_NUM, LOG2_MAX_POC_LSB),
        ));
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, 3, &syn::write_pps(&self.cfg, qp)));

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
            },
            qp,
            &mut w,
        );
        for mb_y in 0..g.mbs_high {
            for mb_x in 0..g.mbs_wide {
                syn::write_pcm_macroblock(&mut w, &g, mb_x, mb_y, &planes, &mut recon);
            }
        }
        w.rbsp_trailing_bits();
        out.extend_from_slice(&syn::annexb(
            if idr { syn::NAL_IDR } else { syn::NAL_SLICE },
            if c.reference { 3 } else { 0 },
            &w.into_nal(),
        ));

        let mut cropped = Vec::with_capacity(self.frame_bytes);
        recon[0].crop_into(g.width, g.height, &mut cropped);
        for (i, p) in recon.iter().enumerate().skip(1) {
            let (dw, dh) = self.plane_dims[i];
            p.crop_into(dw, dh, &mut cropped);
        }
        self.recon.push(cropped);

        if c.reference {
            self.frame_num = (self.frame_num + 1) & ((1 << LOG2_MAX_FRAME_NUM) - 1);
        }
        if idr {
            self.frame_num = 0;
            self.idr_pic_id ^= 1;
        }
        Ok(Access {
            data: out,
            keyframe: idr,
            poc: c.poc,
            encode_index: c.encode,
        })
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
    fn frame_size_matches_every_chroma_format_and_depth() {
        for (chroma, per_px) in [
            (ChromaFormat::Monochrome, 1.0),
            (ChromaFormat::Yuv420, 1.5),
            (ChromaFormat::Yuv422, 2.0),
            (ChromaFormat::Yuv444, 3.0),
        ] {
            for depth in [8u32, 10] {
                let e = H264Encoder::new(cfg(64, 64, chroma, depth)).unwrap();
                let want = (64.0 * 64.0 * per_px) as usize * if depth > 8 { 2 } else { 1 };
                assert_eq!(e.frame_bytes(), want, "{chroma:?} {depth}-bit");
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

    /// The hole is where it should be: the plumbing above it runs, and what
    /// fails names itself.
    #[test]
    fn coding_refuses_at_the_entropy_stage_and_says_so() {
        let mut e = H264Encoder::new(cfg(64, 64, ChromaFormat::Yuv420, 8)).unwrap();
        let err = e.push(&vec![0u8; 64 * 64 * 3 / 2]).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("slice writing"), "{s}");
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
                // Two outcomes are legitimate: the picture was held for its
                // anchor (Ok, nothing released), or it reached the named hole
                // in entropy coding. "scheduler released an absent picture"
                // is neither, and means the index bookkeeping is wrong.
                match e.push(&frame) {
                    Ok(released) => assert!(
                        released.is_empty(),
                        "gop={gop} b={bframes} picture {i}: coded something with no entropy coder"
                    ),
                    Err(err) => {
                        let s = format!("{err}");
                        assert!(
                            s.contains("slice writing"),
                            "gop={gop} b={bframes} picture {i}: {s}"
                        );
                    }
                }
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
