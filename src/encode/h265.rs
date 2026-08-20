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
use super::h265_deblock::deblock_picture;
use super::h265_intra::{CuDecision, IntraCtx, IntraPicture};
use super::h265_me::{InterCuDecision, InterCuKind, InterPicture, MAX_MERGE_CAND};
use super::h265_syntax as syn;
use super::{Access, Config, RateControl};
use crate::bitwriter::BitWriter;
use crate::cabac_enc::CabacEncoder;
use crate::dsp::distortion::DistortionDsp;
use crate::dsp::hevc::{HevcDsp, install_simd_u8};
use crate::dsp::hevc_enc::HevcEncDsp;
use crate::dsp::Cpu;
use crate::hevc::ctx::Contexts;
use crate::hevc::ctu::{
    SplitCuNb, write_cbf_chroma, write_cbf_luma, write_cu_skip_flag,
    write_cu_transquant_bypass_flag, write_merge_flag, write_merge_idx, write_mvd,
    write_mvp_flag, write_part_mode_inter, write_pred_mode_flag, write_rqt_root_cbf,
    write_intra_chroma_pred_mode, write_mpm_idx, write_prev_intra_luma_pred_flag,
    write_rem_intra_luma_pred_mode, write_split_cu_flag, write_split_transform_flag,
};
use crate::hevc::residual::{ResidualParams, residual_scan_idx, write_residual};
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
    /// Reference pictures as the decoder holds them — full `Frame`s with
    /// their motion grids and extended borders, not cropped bytes: the
    /// inter decision predicts through the decoder's own MC, which reads
    /// padded planes, and derives candidates from stored motion.
    refs: Vec<crate::hevc::frame::Frame<u8>>,
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
            refs: Vec::new(),
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

    /// Code one picture.
    ///
    /// The real path is all-intra 4:2:0 at a constant QP: one CU per CTU
    /// (the geometry guarantees whole CTUs of 16 or 32), decided by the
    /// intra machinery and serialised through the coding-tree writers that
    /// live beside their readers. Everything else refuses by name.
    fn code_picture(&mut self, c: Coded, src: &[u8]) -> Result<Access> {
        let g = self.geom;
        // Lossless is transquant bypass: the PPS enables it, every CU says
        // it, and the residuals travel raw. The QP still appears in the
        // headers because the syntax demands one, and it still matters to
        // exactly one thing — the CABAC context initialisation, which both
        // sides derive from the slice QP — while scaling never runs: the
        // decoder's residual path skips dequantisation and the transform for
        // a bypassed CU, so any legal value would decode identically. 26 is
        // the middle of the road and needs no explanation in a debugger.
        let bypass = matches!(self.cfg.rate, RateControl::Lossless);
        let qp = match self.cfg.rate {
            RateControl::Lossless => 26,
            RateControl::ConstantQp(q) => q.min(51),
        };
        if c.kind != Kind::Idr {
            if bypass {
                // Transquant bypass on an inter CU is legal, but the inter
                // decision does not model it and a lossy stream sold as
                // lossless is the worst possible answer.
                return Err(Error::unsupported(
                    "H.265 encode: lossless inter pictures (encoder in progress)",
                ));
            }
            if self.cfg.chroma != crate::ChromaFormat::Yuv420 {
                return Err(Error::unsupported(
                    "H.265 encode: inter pictures outside 4:2:0 (encoder in progress)",
                ));
            }
            if c.kind == Kind::B {
                return Err(Error::unsupported(
                    "H.265 encode: B pictures (encoder in progress; P works)",
                ));
            }
            return self.code_p_picture(c, src, qp);
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
        let pad = |src: &[u8], sw: usize, sh: usize, tw: usize, th: usize| -> Vec<u8> {
            let mut out = vec![0u8; tw * th];
            for y in 0..th {
                let sy = y.min(sh - 1);
                for x in 0..tw {
                    out[y * tw + x] = src[sy * sw + x.min(sw - 1)];
                }
            }
            out
        };
        let py = pad(&src[..dw * dh], dw, dh, cw, ch);
        let (pcb, pcr) = if cat != 0 {
            (
                pad(&src[dw * dh..dw * dh + cdw * cdh], cdw, cdh, ccw, cch),
                pad(&src[dw * dh + cdw * cdh..], cdw, cdh, ccw, cch),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // The decision machinery, on the decoder's own kernels.
        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let ictx = IntraCtx {
            dsp: &dsp,
            enc: &enc,
            dist: &dist,
            qp: qp as i32,
            bit_depth: 8,
            strong_smoothing: false,
            bypass,
        };
        let mut pic = IntraPicture::<u8>::new_with_chroma(cw, ch, g.log2_ctb, 8, chroma);
        // The transform-split search is on: a CU may carry four quarter-size
        // TUs where that wins the decision module's cost comparison, and the
        // writer below spells both shapes.
        pic.split_depth = 1;

        // Parameter sets, then the one slice.
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(syn::NAL_VPS, &syn::write_vps(&g)));
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB),
        ));
        // Inter pictures are not filtered yet and the PPS flag is
        // picture-wide, so only an all-intra stream may declare the
        // deblocking filter it actually applies.
        let deblock = self.cfg.gop == 0;
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, &syn::write_pps(qp, bypass, deblock)));

        let mut w = BitWriter::with_capacity(cw * ch / 2);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp,
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                // An IDR references nothing, so its reference picture set
                // is empty.
                ref_deltas: Vec::new(),
            },
            qp,
            syn::NAL_IDR_N_LP,
            deblock,
            &mut w,
        );
        // byte_alignment(): one, then zeros to the byte.
        w.flag(true);
        w.align_zero();

        let (wc, hc) = (g.ctbs_wide as usize, g.ctbs_high as usize);
        let mut cx = Contexts::new(0, qp as i32);
        // The decisions outlive the loop: the deblocker derives its
        // boundary strengths from them, exactly as a decoder derives them
        // from what it just parsed.
        let mut decisions = Vec::with_capacity(wc * hc);
        {
            let mut e = CabacEncoder::new(&mut w);
            for cy in 0..hc {
                for cxu in 0..wc {
                    let d = pic.code_ctu(&ictx, cxu, cy, &py, cw, &pcb, &pcr, ccw);
                    write_ctu_intra(&mut e, &mut cx, &d, cxu, cy, bypass, cat);
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                    decisions.push(d);
                }
            }
        }
        if deblock {
            // After the whole picture reconstructs — intra prediction
            // reads unfiltered neighbours — and before the crop, because
            // the filtered planes are what a decoder emits and therefore
            // what SELF compares against. Bypass CUs are exempt sample
            // for sample, so lossless stays exact with the filter on.
            deblock_picture(&ictx, &mut pic, &decisions);
        }
        w.align_zero();
        out.extend_from_slice(&syn::annexb(syn::NAL_IDR_N_LP, &w.into_nal()));

        // The reconstruction, cropped to display size — what a decoder
        // emits, and therefore what SELF compares against.
        let mut rec = Vec::with_capacity(self.frame_bytes);
        let crop = |p: &crate::hevc::frame::Plane16<u8>, tw: usize, th: usize, out: &mut Vec<u8>| {
            let o = p.origin();
            for y in 0..th {
                let row = o + y * p.stride;
                out.extend_from_slice(&p.data[row..row + tw]);
            }
        };
        crop(&pic.recon.y, dw, dh, &mut rec);
        if cat != 0 {
            crop(&pic.recon.cb, cdw, cdh, &mut rec);
            crop(&pic.recon.cr, cdw, cdh, &mut rec);
        }
        self.recon.push(rec);

        // Keep the picture as a reference the way the decoder keeps it:
        // borders extended for motion compensation, POC set, motion grid
        // intact. An IDR empties the buffer first - everything before it
        // is discarded, which is what makes it a random access point.
        let mut frame = pic.recon;
        frame.poc = c.poc as i32;
        frame.extend_rows(0, frame.height);
        self.refs.clear();
        self.refs.push(frame);

        Ok(Access { data: out, keyframe: true, poc: c.poc, encode_index: c.encode })
    }

    /// Code one P picture: every CTU one inter CU, decided against the
    /// single stored reference and serialised through the coding-tree
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
    fn code_p_picture(&mut self, c: Coded, src: &[u8], qp: u8) -> Result<Access> {
        let g = self.geom;
        let (dw, dh) = (self.cfg.width as usize, self.cfg.height as usize);
        let (cw, ch) = (g.coded_width as usize, g.coded_height as usize);
        let (cdw, cdh) = (dw.div_ceil(2), dh.div_ceil(2));
        let (ccw, cch) = (cw / 2, ch / 2);
        let pad = |src: &[u8], sw: usize, sh: usize, tw: usize, th: usize| -> Vec<u8> {
            let mut out = vec![0u8; tw * th];
            for y in 0..th {
                let sy = y.min(sh - 1);
                for x in 0..tw {
                    out[y * tw + x] = src[sy * sw + x.min(sw - 1)];
                }
            }
            out
        };
        let py = pad(&src[..dw * dh], dw, dh, cw, ch);
        let pcb = pad(&src[dw * dh..dw * dh + cdw * cdh], cdw, cdh, ccw, cch);
        let pcr = pad(&src[dw * dh + cdw * cdh..], cdw, cdh, ccw, cch);

        // The parameter sets this picture is coded against, parsed back
        // through the decoder's own parsers: the candidate derivation the
        // decision module calls reads decoder structures, and building
        // them from the very bytes the stream carries is what keeps the
        // encoder's idea of the geometry and the decoder's identical.
        let sps_rbsp = syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB);
        let pps_rbsp = syn::write_pps(qp, false, false);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps_rbsp))?;
        let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps_rbsp))?;
        pps.resolve_tiles(&sps)?;

        let refp = self
            .refs
            .last()
            .ok_or_else(|| Error::bitstream("H.265 encode: a P picture with no reference"))?;
        let ref_poc = refp.poc;

        let cpu = Cpu::detect_honouring_env();
        let mut dsp = HevcDsp::<u8>::SCALAR;
        install_simd_u8(&mut dsp, cpu);
        let enc = HevcEncDsp::new(cpu);
        let dist = DistortionDsp::<u8>::new(cpu);
        let mctx = IntraCtx {
            dsp: &dsp,
            enc: &enc,
            dist: &dist,
            qp: qp as i32,
            bit_depth: 8,
            strong_smoothing: false,
            bypass: false,
        };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, c.poc as i32);

        let mut w = BitWriter::with_capacity(cw * ch / 4);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp,
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                // One reference, in the past: the inline short term set
                // this becomes is what builds RefPicList0.
                ref_deltas: vec![ref_poc - c.poc as i32],
            },
            qp,
            syn::NAL_TRAIL_R,
            // Inter pictures are not filtered by this encoder yet, so the
            // PPS this stream carries has the filter off and the header
            // must agree.
            false,
            &mut w,
        );
        w.flag(true); // byte_alignment()
        w.align_zero();

        let (wc, hc) = (g.ctbs_wide as usize, g.ctbs_high as usize);
        // Initialisation type 1 is what the decoder derives for a P slice
        // when cabac_init_flag is absent, which the PPS guarantees.
        let mut cx = Contexts::new(1, qp as i32);
        // cu_skip_flag's context counts *skipped* available neighbours, so
        // the walk carries what it decided, one entry per CTU.
        let mut skipped = vec![false; wc * hc];
        {
            let mut e = CabacEncoder::new(&mut w);
            for cy in 0..hc {
                for cxu in 0..wc {
                    let d = pic.code_ctu(&mctx, refp, cxu, cy, &py, cw, &pcb, &pcr, ccw);
                    if matches!(d.kind, InterCuKind::UseIntra) {
                        // Legal, and the decision module offers it, but
                        // coding an intra CU inside a P slice needs the
                        // intra decision working on this picture's own
                        // reconstruction - one picture, two decision
                        // modules - which is its own piece of wiring.
                        return Err(Error::unsupported(
                            "H.265 encode: intra coding units inside a P slice (encoder in progress)",
                        ));
                    }
                    let left = (cxu > 0).then(|| skipped[cy * wc + cxu - 1]);
                    let above = (cy > 0).then(|| skipped[(cy - 1) * wc + cxu]);
                    write_cu_inter(&mut e, &mut cx, &d, left, above);
                    skipped[cy * wc + cxu] = matches!(d.kind, InterCuKind::Skip { .. });
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                }
            }
        }
        w.align_zero();
        let mut out = Vec::new();
        out.extend_from_slice(&syn::annexb(syn::NAL_TRAIL_R, &w.into_nal()));

        let mut rec = Vec::with_capacity(self.frame_bytes);
        let crop = |p: &crate::hevc::frame::Plane16<u8>, tw: usize, th: usize, out: &mut Vec<u8>| {
            let o = p.origin();
            for y in 0..th {
                let row = o + y * p.stride;
                out.extend_from_slice(&p.data[row..row + tw]);
            }
        };
        crop(&pic.recon.y, dw, dh, &mut rec);
        crop(&pic.recon.cb, cdw, cdh, &mut rec);
        crop(&pic.recon.cr, cdw, cdh, &mut rec);
        self.recon.push(rec);

        // This picture becomes the reference for the next one. One slot:
        // the decision searches exactly one reference, and the slice
        // header it just wrote declares exactly one.
        let mut frame = pic.recon;
        frame.poc = c.poc as i32;
        frame.extend_rows(0, frame.height);
        self.refs.clear();
        self.refs.push(frame);

        Ok(Access { data: out, keyframe: false, poc: c.poc, encode_index: c.encode })
    }
}

/// Serialise one inter coding unit - one whole-CTU `PART_2Nx2N` CU, in the
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
fn write_cu_inter(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    d: &InterCuDecision,
    left_skip: Option<bool>,
    above_skip: Option<bool>,
) {
    let log2 = d.log2_cu;
    // One CU per CTU, so the coding quadtree never splits and the flag is
    // coded exactly once, false - the same shape the intra writer spells.
    let nb = SplitCuNb {
        left_depth: left_skip.map(|_| 0),
        above_depth: above_skip.map(|_| 0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);

    let skip = matches!(d.kind, InterCuKind::Skip { .. });
    write_cu_skip_flag(e, cx, left_skip, above_skip, skip);
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
            write_mvd(e, cx, mvd);
            write_mvp_flag(e, cx, mvp_flag != 0);
            write_rqt_root_cbf(e, cx, d.rqt_root_cbf);
            if !d.rqt_root_cbf {
                return;
            }
        }
        InterCuKind::Skip { .. } | InterCuKind::UseIntra => unreachable!("handled above"),
    }

    // The transform tree: one CU-sized TU, no split. cbf_luma is coded
    // only because a chroma cbf is set or the depth is nonzero - for an
    // inter leaf at depth 0 with both chroma cbfs clear the reader infers
    // cbf_luma 1 and reads no bin, so writing one would desync.
    write_split_transform_flag(e, cx, log2, false);
    write_cbf_chroma(e, cx, 0, d.cbf_chroma[0]);
    write_cbf_chroma(e, cx, 0, d.cbf_chroma[1]);
    if d.cbf_chroma[0] || d.cbf_chroma[1] {
        write_cbf_luma(e, cx, 0, d.cbf_luma);
    } else {
        debug_assert!(d.cbf_luma, "an inter leaf with no chroma cbf has cbf_luma inferred 1");
    }

    let n = 1usize << log2;
    // Inter blocks always scan diagonally: the mode-dependent scans are an
    // intra rule (7.4.9.11).
    let params = |log2_size: u32, c_idx: usize| ResidualParams {
        log2_size,
        c_idx,
        scan_idx: 0,
        bypass: false,
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
    let nc = n / 2;
    for comp in 0..2 {
        if d.cbf_chroma[comp] {
            write_residual(e, cx, &params(log2 - 1, comp + 1), &d.chroma[comp][..nc * nc]);
        }
    }
}

/// Serialise one all-intra CTU holding exactly one `PART_2Nx2N` CU whose
/// transform tree is either a single CU-sized TU or one level of splitting
/// into four quarter TUs — the two shapes the decision machinery produces
/// at CTB 16 or 32, and the geometry guarantees no partial CTUs.
///
/// The walk is the reader's, specialised to those shapes: `coding_quadtree`
/// reads one `split_cu_flag` (the CTB is above the minimum CU size, so the
/// flag is coded, false); `coding_unit` reads the luma mode syntax for one
/// prediction block and the chroma mode (no `part_mode` — that is read only
/// at the minimum CU size, which a 16/32 CTB never is); `transform_tree`
/// reads one coded `split_transform_flag` (the SPS makes the maximum
/// transform equal the CTB precisely so the unsplit shape is expressible,
/// and declares hierarchy depth 2 so the split one is too), the chroma cbfs
/// at depth 0, and then per leaf `cbf_luma` (always coded for intra) and
/// the residual blocks — see the split branch below for the per-child
/// ordering. Anything that stops matching the reader here desyncs the
/// arithmetic coder and fails SELF wholesale, which is exactly the property
/// the encode gate checks.
///
/// `pps_bypass` mirrors the PPS's `transquant_bypass_enabled_flag`: when
/// set, `coding_unit` reads a `cu_transquant_bypass_flag` as its very first
/// bin, so this writer spells one — the CU's own choice, `d.bypass` — and
/// when clear, nothing is written and the CU must not claim bypass.
fn write_ctu_intra(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, ctu_x: usize, ctu_y: usize, pps_bypass: bool, cat: u32) {
    let log2 = d.log2_cu;
    debug_assert!((4..=5).contains(&log2), "one CU per CTU wants CTB 16 or 32");
    debug_assert!(!d.nxn, "PART_NxN exists only at the minimum CU size");
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // Every coded neighbour has depth 0 (one CU per CTU), and in a single
    // slice availability is picture geometry.
    let nb = SplitCuNb {
        left_depth: (ctu_x > 0).then_some(0),
        above_depth: (ctu_y > 0).then_some(0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }

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

        // P pictures code: a GOP of 4:2:0 pictures produces one access
        // unit each, the first a keyframe and the rest not.
        let mut e = H265Encoder::new(Config { gop: 8, ..cfg(64, 64, ChromaFormat::Yuv420) }).unwrap();
        let frame = vec![64u8; 64 * 64 * 3 / 2];
        let mut units = Vec::new();
        for _ in 0..3 {
            units.extend(e.push(&frame).expect("a P picture should code"));
        }
        units.extend(e.flush().unwrap());
        assert_eq!(units.len(), 3, "one access unit per picture");
        assert!(units[0].keyframe, "the first is an IDR");
        assert!(!units[1].keyframe && !units[2].keyframe, "the rest are P");
        assert!(units.iter().all(|u| !u.data.is_empty()));

        // The named holes that remain, each reached by the configuration
        // that asks for it.
        let holes: [(Config, usize, &str); 3] = [
            (
                Config { gop: 8, bframes: 2, ..cfg(64, 64, ChromaFormat::Yuv420) },
                64 * 64 * 3 / 2,
                "B pictures",
            ),
            (Config { gop: 8, ..cfg(64, 64, ChromaFormat::Yuv422) }, 64 * 64 * 2, "outside 4:2:0"),
            (
                Config {
                    gop: 8,
                    rate: super::super::RateControl::Lossless,
                    ..cfg(64, 64, ChromaFormat::Yuv420)
                },
                64 * 64 * 3 / 2,
                "lossless inter",
            ),
        ];
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
            bypass: false,
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
