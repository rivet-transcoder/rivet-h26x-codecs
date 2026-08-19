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

use super::{Access, Config, Entropy, RateControl};
use super::gop::{Coded, Kind, Scheduler};
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
}

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
        Ok(Self {
            cfg,
            sched,
            held: std::collections::BTreeMap::new(),
            recon: Vec::new(),
            frame_bytes: (luma + chroma) * bps,
            next_display: 0,
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

    /// Where the bits would be written.
    fn code_picture(&mut self, c: Coded, _src: &[u8]) -> Result<Access> {
        let _ = (&self.cfg.rate, &self.cfg.entropy, c.kind, c.reference);
        // The hole, named. Everything above this line is exercised by
        // verify_encode.sh already: configuration, geometry, picture typing,
        // coding order, and the access-unit envelope.
        Err(Error::unsupported(match self.cfg.entropy {
            Entropy::Cabac => "H.264 encode: CABAC slice writing (encoder in progress)",
            Entropy::Cavlc => "H.264 encode: CAVLC slice writing (encoder in progress)",
        }))
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
