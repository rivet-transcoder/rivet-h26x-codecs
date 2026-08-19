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

use super::gop::{Coded, Kind, Scheduler};
use super::h265_syntax as syn;
use super::{Access, Config, RateControl};
use crate::{Error, Result};

/// H.265 encoder. See the module documentation for what is and is not built.
pub struct H265Encoder {
    cfg: Config,
    sched: Scheduler,
    /// Source pictures held in display order, so a B picture kept back by the
    /// scheduler still has its samples when its anchor arrives.
    held: std::collections::BTreeMap<u64, Vec<u8>>,
    /// Reconstructions in coding order, for the SELF check.
    recon: Vec<Vec<u8>>,
    frame_bytes: usize,
    /// Display index of the next picture offered — an explicit counter, for
    /// the reason recorded on the H.264 encoder: `held` empties as pictures
    /// code, so inferring the index from it fails the moment the scheduler
    /// releases pictures as fast as they arrive.
    next_display: u64,
    geom: syn::Geometry,
}

/// The POC LSB width the SPS declares. Fixed and generous, as on the H.264
/// side: a wrap that never happens is a class of bug that never happens.
const LOG2_MAX_POC_LSB: u32 = 16;

impl H265Encoder {
    /// Fails rather than starting if the configuration cannot produce a legal
    /// stream — an encoder that fails late has usually already emitted a
    /// header describing something it then cannot deliver.
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        if cfg.bit_depth > 8 {
            // Same refusal, same reason as H.264: the reconstruction planes
            // this encoder will share with the decision side are u8, and a
            // silently narrowed stream that looks legal is worse than a
            // refusal that names itself.
            return Err(Error::unsupported(
                "H.265 encode: bit depth above 8 (encoder in progress)",
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
        Ok(Self {
            geom: syn::Geometry::new(&cfg),
            sched: Scheduler::new(cfg.gop, cfg.bframes),
            cfg,
            held: std::collections::BTreeMap::new(),
            recon: Vec::new(),
            frame_bytes: luma + chroma,
            next_display: 0,
        })
    }

    /// How many bytes one source picture must be.
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// The reconstructions produced so far, in coding order.
    pub fn reconstructions(&self) -> &[Vec<u8>] {
        &self.recon
    }

    /// Offer the next picture in display order.
    pub fn push(&mut self, picture: &[u8]) -> Result<Vec<Access>> {
        if picture.len() != self.frame_bytes {
            return Err(Error::bitstream(format!(
                "H.265 encode: picture is {} bytes, expected {}",
                picture.len(),
                self.frame_bytes
            )));
        }
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
            let src = self.held.remove(&c.display).ok_or_else(|| {
                Error::bitstream("H.265 encode: scheduler released an absent picture")
            })?;
            out.push(self.code_picture(c, &src)?);
        }
        Ok(out)
    }

    /// Code one picture — or rather, everything up to the point where the
    /// coding tree would be written, which is where this encoder currently
    /// refuses by name.
    fn code_picture(&mut self, c: Coded, _src: &[u8]) -> Result<Access> {
        let g = self.geom;
        // The parameter sets are real and proven against the crate's own
        // parsers; building them here keeps this path exercised while the
        // slice payload is still missing.
        let qp = match self.cfg.rate {
            RateControl::Lossless => 0,
            RateControl::ConstantQp(q) => q.min(51),
        };
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(syn::NAL_VPS, &syn::write_vps(&g)));
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB),
        ));
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, &syn::write_pps(qp)));
        let _ = (c.kind == Kind::Idr, c.poc, c.reference, out);
        // The hole, named. H.265 has no CAVLC, so even an all-PCM picture
        // needs the CABAC coding-tree writer; nothing simpler is legal.
        Err(Error::unsupported(
            "H.265 encode: coding-tree serialisation (encoder in progress)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaFormat;

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

    /// The hole is where it should be: everything above the coding tree runs,
    /// and the refusal names the missing piece rather than the codec.
    #[test]
    fn coding_refuses_at_the_coding_tree_and_says_so() {
        for (gop, bframes) in [(0u32, 0u32), (8, 0), (8, 2)] {
            let mut e = H265Encoder::new(Config {
                gop,
                bframes,
                ..cfg(64, 64, ChromaFormat::Yuv420)
            })
            .unwrap();
            let frame = vec![0u8; 64 * 64 * 3 / 2];
            let mut named = false;
            for i in 0..4 {
                match e.push(&frame) {
                    Ok(released) => assert!(
                        released.is_empty(),
                        "gop={gop} b={bframes} picture {i}: coded something with no coding tree"
                    ),
                    Err(err) => {
                        let s = format!("{err}");
                        assert!(s.contains("coding-tree"), "gop={gop} b={bframes}: {s}");
                        named = true;
                        break;
                    }
                }
            }
            assert!(named, "gop={gop} b={bframes}: never reached the named hole");
        }
    }

    #[test]
    fn deeper_than_eight_bits_refuses_rather_than_truncating() {
        for depth in [10u32, 12, 14] {
            match H265Encoder::new(Config {
                bit_depth: depth,
                ..cfg(64, 64, ChromaFormat::Yuv420)
            }) {
                Ok(_) => panic!("{depth}-bit was accepted; samples would be truncated"),
                Err(err) => {
                    let s = format!("{err}");
                    assert!(s.contains("bit depth above 8"), "{depth}-bit: {s}");
                }
            }
        }
    }
}
