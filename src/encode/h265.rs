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
use super::h265_deblock::{deblock_inter_picture, deblock_picture};
use super::h265_intra::{CuDecision, IntraCtx, IntraPicture};
use super::h265_me::{InterCuDecision, InterCuKind, InterPicture, PCuDecision, MAX_MERGE_CAND};
use super::h265_rc::{PicKind, RateController};
use super::h265_sao::{SaoPlan, sao_picture};
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
    SaoCtx, SaoMergeNb, SplitCuNb, write_cbf_chroma, write_cbf_luma, write_cu_skip_flag, write_sao,
    write_cu_transquant_bypass_flag, write_merge_flag, write_merge_idx, write_mvd,
    write_inter_pred_idc, write_mvp_flag, write_part_mode_inter, write_pred_mode_flag,
    write_rqt_root_cbf,
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
    pps_qp: u8,
    /// Bytes emitted so far, to hold the controller's ledger to.
    emitted: u64,
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
        let (sw, sh) = cfg.chroma.subsampling();
        let luma = cfg.width as usize * cfg.height as usize;
        let chroma = if cfg.chroma == crate::ChromaFormat::Monochrome {
            0
        } else {
            2 * (cfg.width as usize).div_ceil(sw as usize)
                * (cfg.height as usize).div_ceil(sh as usize)
        };
        let g = syn::Geometry::new(&cfg);
        // The PPS quantiser: the constant one where there is one, and the
        // middle of the road where the controller will vary it per picture.
        // Its only effect on the stream is the size of each
        // `slice_qp_delta`, since both sides derive everything else from
        // the slice quantiser.
        let pps_qp = match cfg.rate {
            RateControl::ConstantQp(q) => q.min(51),
            RateControl::Lossless => 26,
            RateControl::Bitrate { .. } => 26,
        };
        let rc = match cfg.rate {
            RateControl::Bitrate { bps } => Some(RateController::new(bps, cfg.fps, cfg.width, cfg.height, cfg.gop)),
            _ => None,
        };
        Ok(Self {
            geom: g,
            sched: Scheduler::new(cfg.gop, cfg.bframes),
            rc,
            pps_qp,
            emitted: 0,
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
        let rc = self.rc.as_ref()?;
        let target = match self.cfg.rate {
            RateControl::Bitrate { bps } => bps as f64,
            _ => return None,
        };
        Some((rc.achieved_bps(self.cfg.fps), target))
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
            let access = self.code_picture(c, &src)?;
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

    /// Code one picture.
    ///
    /// The real path is all-intra 4:2:0 at a constant QP: one CU per CTU
    /// (the geometry guarantees whole CTUs of 16 or 32), decided by the
    /// intra machinery and serialised through the coding-tree writers that
    /// live beside their readers. Everything else refuses by name.
    fn code_picture(&mut self, c: Coded, src: &[u8]) -> Result<Access> {
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
        let bypass = matches!(self.cfg.rate, RateControl::Lossless);
        let qp = match self.cfg.rate {
            RateControl::Lossless => 26,
            RateControl::ConstantQp(q) => q.min(51),
            // The controller chooses, once per picture, in coding order.
            // `account` below closes the loop with what it actually cost.
            RateControl::Bitrate { .. } => {
                let kind = if c.kind == Kind::Idr { PicKind::Intra } else { PicKind::Inter };
                self.rc.as_mut().expect("a bitrate configuration builds a controller").pick_qp(kind)
            }
        };
        if c.kind != Kind::Idr {
            return self.code_inter_picture(c, src, qp, bypass);
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
        out.extend_from_slice(&syn::annexb(syn::NAL_VPS, &syn::write_vps(&self.cfg, &g)));
        out.extend_from_slice(&syn::annexb(
            syn::NAL_SPS,
            &syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB),
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
        out.extend_from_slice(&syn::annexb(syn::NAL_PPS, &syn::write_pps(self.pps_qp, bypass, deblock)));

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
                // Present exactly when the SPS enabled SAO, and the chroma
                // flag only outside monochrome - the reader's two gates.
                sao: sao_flags(self.cfg.sao, cat),
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
        let mut decisions = Vec::with_capacity(wc * hc);
        for cy in 0..hc {
            for cxu in 0..wc {
                decisions.push(pic.code_ctu(&ictx, cxu, cy, &py, cw, &pcb, &pcr, ccw));
            }
        }
        // After the whole picture reconstructs — intra prediction reads
        // unfiltered neighbours — and before the crop, because the
        // filtered planes are what a decoder emits and therefore what SELF
        // compares against. Bypass CUs are exempt sample for sample, so
        // lossless stays exact with the filter on.
        let mut info = deblock_picture(&ictx, &mut pic, &decisions);
        // Then SAO, over the deblocked samples, which is the order 8.7
        // fixes and the order `decoder.rs` applies them in.
        let plan = self.cfg.sao.then(|| {
            let (sps, pps) = parsed_sets(&self.cfg, &g, qp, bypass, deblock);
            sao_picture(&ictx, &mut pic.recon, &mut info, &sps, &pps, &py, cw, &pcb, &pcr, ccw)
        });
        {
            let mut e = CabacEncoder::new(&mut w);
            for cy in 0..hc {
                for cxu in 0..wc {
                    let addr = cy * wc + cxu;
                    write_sao_for(&mut e, &mut cx, plan.as_ref(), addr, cxu, cy, qp, cat);
                    write_ctu_intra(&mut e, &mut cx, &decisions[addr], cxu, cy, bypass, cat);
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                }
            }
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
        // An IDR empties the buffer: everything before it is discarded,
        // which is what makes it a random access point.
        self.refs.clear();
        self.retain_reference(&c, pic.recon);

        Ok(Access { data: out, keyframe: true, poc: c.poc, encode_index: c.encode })
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
    fn retain_reference(&mut self, c: &Coded, mut frame: crate::hevc::frame::Frame<u8>) {
        if !c.reference {
            return;
        }
        frame.poc = c.poc as i32;
        frame.extend_rows(0, frame.height);
        self.refs.push(frame);
        while self.refs.len() > 2 {
            self.refs.remove(0);
        }
    }

    /// Code one inter picture — P or B: every CTU one inter CU, decided against the
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
    fn code_inter_picture(&mut self, c: Coded, src: &[u8], qp: u8, bypass: bool) -> Result<Access> {
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

        // The parameter sets this picture is coded against, parsed back
        // through the decoder's own parsers: the candidate derivation the
        // decision module calls reads decoder structures, and building
        // them from the very bytes the stream carries is what keeps the
        // encoder's idea of the geometry and the decoder's identical.
        let sps_rbsp = syn::write_sps(&self.cfg, &g, LOG2_MAX_POC_LSB);
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
        let pps_rbsp = syn::write_pps(self.pps_qp, bypass, true);
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&sps_rbsp))?;
        let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&pps_rbsp))?;
        pps.resolve_tiles(&sps)?;

        // List 0 is the nearest reference in the past; a B picture's list 1
        // is the nearest in the future. The scheduler codes both anchors
        // before releasing the B pictures between them, so both are here by
        // the time one of those codes — and the retention below keeps them
        // until the last picture that can reference them has been coded.
        let cur = c.poc as i32;
        let past = self
            .refs
            .iter()
            .filter(|f| f.poc < cur)
            .max_by_key(|f| f.poc)
            .ok_or_else(|| Error::bitstream("H.265 encode: an inter picture with no past reference"))?;
        let future = self.refs.iter().filter(|f| f.poc > cur).min_by_key(|f| f.poc);
        if c.kind == Kind::B && future.is_none() {
            return Err(Error::bitstream(
                "H.265 encode: a B picture with no future reference",
            ));
        }
        let ref_poc = past.poc;
        let future_poc = future.map(|f| f.poc);

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
            bypass,
        };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, c.poc as i32);

        let mut w = BitWriter::with_capacity(cw * ch / 4);
        syn::write_slice_header(
            &syn::SliceHeader {
                kind: c.kind,
                poc_lsb: (c.poc as u32) & ((1 << LOG2_MAX_POC_LSB) - 1),
                qp,
                log2_max_poc_lsb: LOG2_MAX_POC_LSB,
                // The inline short term reference picture set: one past
                // entry, which becomes RefPicList0, and for a B picture one
                // future entry, which becomes RefPicList1.
                ref_deltas: match future_poc {
                    Some(f) if c.kind == Kind::B => vec![ref_poc - cur, f - cur],
                    _ => vec![ref_poc - cur],
                },
                // As in `code_picture`, and from the same switch.
                sao: sao_flags(self.cfg.sao, cat),
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
        // cu_skip_flag's context counts *skipped* available neighbours, so
        // the walk carries what it decided, one entry per CTU.
        let mut skipped = vec![false; wc * hc];
        // The decisions outlive the loop: the deblocker derives its
        // boundary strengths from them, as a decoder does from what it
        // has just parsed.
        let mut decisions = Vec::with_capacity(wc * hc);
        // Decide and reconstruct first; serialise below. See the same
        // split in `code_picture` for why SAO forces it.
        for cy in 0..hc {
            for cxu in 0..wc {
                let d = match future {
                    Some(r1) if c.kind == Kind::B => {
                        pic.code_ctu_b(&mctx, past, r1, cxu, cy, &py, cw, &pcb, &pcr, ccw)
                    }
                    _ => pic.code_ctu(&mctx, past, cxu, cy, &py, cw, &pcb, &pcr, ccw),
                };
                // The decision module answers `UseIntra` when its
                // flatness proxy says inter has lost. The CU is then
                // coded by the intra decision over *this* picture's
                // reconstruction - the same `code_cu_2nx2n_intra` an
                // I slice runs, reading the inter neighbours already
                // reconstructed beside it, which the PPS's
                // `constrained_intra_pred_flag` 0 makes references.
                let coded = if matches!(d.kind, InterCuKind::UseIntra) {
                    PCuDecision::Intra(Box::new(pic.code_ctu_intra(&mctx, cxu, cy, &py, cw, &pcb, &pcr, ccw)))
                } else {
                    PCuDecision::Inter(d)
                };
                skipped[cy * wc + cxu] = matches!(&coded, PCuDecision::Inter(d) if matches!(d.kind, InterCuKind::Skip { .. }));
                decisions.push(coded);
            }
        }
        // After the whole picture reconstructs and before the crop, for
        // the same reasons the intra path gives: intra prediction — which
        // a P slice now also performs — reads unfiltered neighbours, and
        // the filtered planes are what a decoder emits and therefore what
        // SELF compares against. This picture becomes the next one's
        // reference filtered, which is what a decoder's DPB holds.
        deblock_inter_picture(&mctx, &mut pic, &decisions);
        // Then SAO, over the deblocked samples. `InterPicture` already
        // holds the decoder-grade state both filters read, so unlike the
        // intra path there is nothing to hand across.
        let plan = self.cfg.sao.then(|| {
            let InterPicture { info, recon, .. } = &mut pic;
            sao_picture(&mctx, recon, info, &sps, &pps, &py, cw, &pcb, &pcr, ccw)
        });
        {
            let mut e = CabacEncoder::new(&mut w);
            for cy in 0..hc {
                for cxu in 0..wc {
                    let addr = cy * wc + cxu;
                    let left = (cxu > 0).then(|| skipped[addr - 1]);
                    let above = (cy > 0).then(|| skipped[addr - wc]);
                    write_sao_for(&mut e, &mut cx, plan.as_ref(), addr, cxu, cy, qp, cat);
                    match &decisions[addr] {
                        PCuDecision::Inter(d) => write_cu_inter(&mut e, &mut cx, d, left, above, cat, bypass),
                        PCuDecision::Intra(d) => write_cu_intra_in_p(&mut e, &mut cx, d, left, above, cat, bypass),
                    }
                    e.encode_terminate(u32::from(cy == hc - 1 && cxu == wc - 1));
                }
            }
        }
        w.align_zero();
        let mut out = Vec::new();
        // A picture nothing will reference is a sub-layer non-reference
        // picture, and saying so lets a decoder discard it.
        let nal = if c.reference { syn::NAL_TRAIL_R } else { syn::NAL_TRAIL_N };
        out.extend_from_slice(&syn::annexb(nal, &w.into_nal()));

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

        self.retain_reference(&c, pic.recon);

        Ok(Access { data: out, keyframe: false, poc: c.poc, encode_index: c.encode })
    }
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
fn parsed_sets(cfg: &Config, g: &syn::Geometry, qp: u8, bypass: bool, deblock: bool) -> (crate::hevc::sps::Sps, crate::hevc::pps::Pps) {
    let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(cfg, g, LOG2_MAX_POC_LSB)))
        .expect("the encoder's own SPS parses");
    let mut pps = crate::hevc::pps::Pps::parse(&crate::nal::unescape_rbsp(&syn::write_pps(qp, bypass, deblock)))
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
fn write_sao_for(e: &mut CabacEncoder, cx: &mut Contexts, plan: Option<&SaoPlan>, addr: usize, cxu: usize, cy: usize, qp: u8, cat: u32) {
    let Some(plan) = plan else { return };
    let _ = qp;
    let sctx = SaoCtx {
        sao_luma: true,
        sao_chroma: cat != 0,
        cat,
        // Eight bits everywhere this encoder writes; `H265Encoder::new`
        // refuses anything deeper by name.
        cmax: (1u32 << (8u32.min(10) - 5)) - 1,
        // No PPS range extension, so no offset scaling.
        shift: (0, 0),
    };
    // One slice, one tile: the reader's availability test for the merge
    // flags is exactly the picture edge.
    let nb = SaoMergeNb { left: cxu > 0, up: cy > 0 };
    write_sao(e, cx, &sctx, &nb, plan.merges[addr], &plan.params[addr]);
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
///
/// `cat` is `ChromaArrayType`, and the transform tree below is the only
/// part of this walk that depends on it - see the comments there for the
/// per-format cbf and residual shapes, which are the inter mirror of what
/// `write_ctu_intra`'s unsplit branch spells.
fn write_cu_inter(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    d: &InterCuDecision,
    left_skip: Option<bool>,
    above_skip: Option<bool>,
    cat: u32,
    pps_bypass: bool,
) {
    let log2 = d.log2_cu;
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // One CU per CTU, so the coding quadtree never splits and the flag is
    // coded exactly once, false - the same shape the intra writer spells.
    let nb = SplitCuNb {
        left_depth: left_skip.map(|_| 0),
        above_depth: above_skip.map(|_| 0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);
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
        InterCuKind::BAmvp { idc, mvd, mvp_flag } => {
            write_merge_flag(e, cx, false);
            // inter_pred_idc, then per list -- L0's mvd and mvp_flag, then
            // L1's, interleaved as `prediction_unit` reads them rather
            // than grouped by element. No ref_idx in either list: each
            // declares exactly one active reference. See
            // `write_inter_pred_idc`'s docblock for the `w + h != 12`
            // reading; a whole-CTU CU is never 12 and is always CtDepth 0.
            let n = 1i32 << log2;
            write_inter_pred_idc(e, cx, n, n, 0, u32::from(idc));
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
pub(crate) fn write_ctu_intra(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, ctu_x: usize, ctu_y: usize, pps_bypass: bool, cat: u32) {
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    // Every coded neighbour has depth 0 (one CU per CTU), and in a single
    // slice availability is picture geometry.
    let nb = SplitCuNb {
        left_depth: (ctu_x > 0).then_some(0),
        above_depth: (ctu_y > 0).then_some(0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);
    // An I slice reads no `cu_skip_flag` and no `pred_mode_flag` — both
    // are gated on `slice_type != I` (ctu.rs:405, ctu.rs:434) — so the
    // CU starts at the bypass flag.
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }
    write_cu_intra_body(e, cx, d, cat);
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
fn write_cu_intra_in_p(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    d: &CuDecision,
    left_skip: Option<bool>,
    above_skip: Option<bool>,
    cat: u32,
    pps_bypass: bool,
) {
    debug_assert!(pps_bypass || !d.bypass, "a bypass CU is unspellable unless the PPS enables the flag");
    let nb = SplitCuNb {
        left_depth: left_skip.map(|_| 0),
        above_depth: above_skip.map(|_| 0),
    };
    write_split_cu_flag(e, cx, &nb, 0, false);
    // `coding_unit` reads cu_transquant_bypass_flag BEFORE cu_skip_flag,
    // so it comes first here too.
    if pps_bypass {
        write_cu_transquant_bypass_flag(e, cx, d.bypass);
    }
    write_cu_skip_flag(e, cx, left_skip, above_skip, false);
    write_pred_mode_flag(e, cx, true);
    write_cu_intra_body(e, cx, d, cat);
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
fn write_cu_intra_body(e: &mut CabacEncoder, cx: &mut Contexts, d: &CuDecision, cat: u32) {
    let log2 = d.log2_cu;
    debug_assert!((4..=5).contains(&log2), "one CU per CTU wants CTB 16 or 32");
    debug_assert!(!d.nxn, "PART_NxN exists only at the minimum CU size");

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
            write_cu_inter(&mut e, &mut cx, d, None, None, 1, false);
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
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB)))
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
            bypass: true,
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
                let d = pic.code_ctu(&ctx, &refp, cx, cy, &py, w, &pcb, &pcr, cw);
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
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(
            &config,
            &g,
            LOG2_MAX_POC_LSB,
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
            IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 30, bit_depth: 8, strong_smoothing: false, bypass: false };
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
                let d = pic.code_ctu(&ctx, &refp, cx, cy, &py, w, &pcb, &pcr, cw);
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
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB))).unwrap();
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
        let ctx = IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 26, bit_depth: 8, strong_smoothing: false, bypass: false };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, 1);
        let (py, pc) = flat.split_at(w * h);
        let (pcb, pcr) = pc.split_at(w * h / 4);
        let mut intra_cus = 0usize;
        for cy in 0..2 {
            for cx in 0..2 {
                let d = pic.code_ctu(&ctx, &refp, cx, cy, py, w, pcb, pcr, w / 2);
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
        let sps = crate::hevc::sps::Sps::parse(&crate::nal::unescape_rbsp(&syn::write_sps(&config, &g, LOG2_MAX_POC_LSB))).unwrap();
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
        let ctx = IntraCtx { dsp: &dsp, enc: &enc_dsp, dist: &dist, qp: 30, bit_depth: 8, strong_smoothing: false, bypass: false };
        let mut pic = InterPicture::<u8>::new(&sps, &pps, 1);
        let (py, pc) = split.split_at(w * h);
        let (pcb, pcr) = pc.split_at(w * h / 4);
        let (mut intra_cus, mut split_cus) = (0usize, 0usize);
        for cy in 0..2 {
            for cx in 0..2 {
                let d = pic.code_ctu(&ctx, &refp, cx, cy, py, w, pcb, pcr, w / 2);
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
