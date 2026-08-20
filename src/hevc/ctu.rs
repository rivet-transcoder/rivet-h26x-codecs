//! Coding tree unit decoding (H.265 7.3.8): SAO syntax, the coding
//! quadtree, coding units, prediction units (motion derivation and
//! compensation), transform trees and units (residual + reconstruction),
//! and the per-CU bookkeeping the loop filters and later CUs read.

use crate::cabac::Cabac;
use crate::cabac_enc::CabacEncoder;
use crate::picture::ChromaFormat;
use crate::{Error, Result};

use super::ctx::*;
use super::frame::{Frame, MotionInfo, Mv, SharedFrame, Sample, fill_motion};
use super::inter::{McScratch, Weighting, predict_block};
use super::intra::{IntraScratch, predict as intra_predict};
use super::mvpred::{Cand, PuPos, RefCtx, amvp, merge_candidate};
use super::pic::{AvailCtx, PicInfo, SaoParams};
use super::pps::Pps;
use super::residual::{ResidualParams, ScalingSource, parse_residual, rdpcm_residual, residual_scan_idx, rotate_residual4, scale_coefficients, transform_skip_residual};
use crate::dsp::hevc::HevcDsp;
use super::slice::{SliceHeader, SliceType};
use super::sps::{ScalingList, Sps};

/// Partition modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    /// 2Nx2N.
    P2Nx2N,
    /// 2NxN.
    P2NxN,
    /// Nx2N.
    PNx2N,
    /// NxN.
    PNxN,
    /// 2NxnU.
    P2NxnU,
    /// 2NxnD.
    P2NxnD,
    /// nLx2N.
    PnLx2N,
    /// nRx2N.
    PnRx2N,
}

impl PartMode {
    /// The prediction blocks `(x, y, w, h)` relative to the CB of size `n`
    /// (the first `count` entries are valid).
    fn pus(self, n: i32) -> Pus {
        let z = (0, 0, 0, 0);
        match self {
            PartMode::P2Nx2N => Pus { list: [(0, 0, n, n), z, z, z], count: 1 },
            PartMode::P2NxN => Pus { list: [(0, 0, n, n / 2), (0, n / 2, n, n / 2), z, z], count: 2 },
            PartMode::PNx2N => Pus { list: [(0, 0, n / 2, n), (n / 2, 0, n / 2, n), z, z], count: 2 },
            PartMode::PNxN => Pus { list: [(0, 0, n / 2, n / 2), (n / 2, 0, n / 2, n / 2), (0, n / 2, n / 2, n / 2), (n / 2, n / 2, n / 2, n / 2)], count: 4 },
            PartMode::P2NxnU => Pus { list: [(0, 0, n, n / 4), (0, n / 4, n, n * 3 / 4), z, z], count: 2 },
            PartMode::P2NxnD => Pus { list: [(0, 0, n, n * 3 / 4), (0, n * 3 / 4, n, n / 4), z, z], count: 2 },
            PartMode::PnLx2N => Pus { list: [(0, 0, n / 4, n), (n / 4, 0, n * 3 / 4, n), z, z], count: 2 },
            PartMode::PnRx2N => Pus { list: [(0, 0, n * 3 / 4, n), (n * 3 / 4, 0, n / 4, n), z, z], count: 2 },
        }
    }
}

/// The prediction blocks of a partitioning, without a heap allocation.
struct Pus {
    list: [(i32, i32, i32, i32); 4],
    count: usize,
}

impl Pus {
    #[inline(always)]
    fn iter(&self) -> impl Iterator<Item = &(i32, i32, i32, i32)> {
        self.list[..self.count].iter()
    }
}

/// Everything one slice segment's CTUs need.
pub struct SliceDec<'a, S: Sample = u16> {
    /// SPS.
    pub sps: &'a Sps,
    /// PPS.
    pub pps: &'a Pps,
    /// Slice header (of the independent segment).
    pub hdr: &'a SliceHeader,
    /// The picture being decoded.
    pub frame: &'a mut Frame<S>,
    /// Per-picture side data.
    pub info: &'a mut PicInfo,
    /// The arithmetic decoder over the current substream.
    pub cabac: Cabac<'a>,
    /// Context variables.
    pub cx: Contexts,
    /// Reference picture context.
    pub refs: RefCtx<'a, S>,
    /// Reference frames per list.
    pub ref_frames: [Vec<&'a Frame<S>>; 2],
    /// The same references with their progress (for waiting on rows still
    /// being decoded by another thread).
    pub ref_shared: [Vec<&'a SharedFrame<S>>; 2],
    /// The collocated picture's progress.
    pub col_shared: Option<&'a SharedFrame<S>>,
    /// Slice index (into `info.slices`).
    pub slice_idx: u16,
    /// `SliceAddrRs`.
    pub slice_addr: u32,
    /// Resolved scaling lists (None = flat).
    pub scaling: Option<ScalingList>,
    /// `QpY` of the current CU (running).
    pub qp_y: i32,
    /// `qPY_PREV` for the next quantisation group.
    pub qp_y_prev: i32,
    /// `CuQpDeltaVal`.
    pub cu_qp_delta_val: i32,
    /// `IsCuQpDeltaCoded`.
    pub is_cu_qp_delta_coded: bool,
    /// `IsCuChromaQpOffsetCoded`.
    pub is_cu_chroma_qp_offset_coded: bool,
    /// `CuQpOffsetCb`, `CuQpOffsetCr`.
    pub cu_qp_offset_c: [i32; 2],
    /// The current quantisation group's top-left.
    pub qg: (i32, i32),
    /// `qPY_PREV` resolved for the current QG.
    pub qg_qp_prev: i32,
    /// Whether the next QG is the first in the slice / tile / CTB row (WPP).
    pub first_qg: bool,
    /// The last PU parsed used merge (for `rqt_root_cbf` presence).
    pub last_pu_merged: bool,
    /// Current CTB address (raster).
    pub ctb_addr_rs: usize,
    /// Current CTB address (tile scan).
    pub ctb_addr_ts: usize,
    /// Scratch coefficient buffer.
    pub coeffs: Vec<i16>,
    /// The current TU's luma residual (cross-component prediction), and
    /// whether it is the current TU's.
    pub luma_res: Vec<i16>,
    /// See `luma_res`.
    pub luma_res_valid: bool,
    /// The kernels.
    pub dsp: HevcDsp<S>,
    /// Motion compensation scratch.
    pub mc: McScratch<S>,
    /// Intra reference-sample scratch.
    pub intra: IntraScratch,
    /// Non-fatal problems seen.
    pub warnings: u64,
    /// Debug tracing (from the `H26X_TRACE_*` environment variables).
    pub trace: TraceCfg,
}

/// What to print while decoding (debugging aid; all off by default).
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceCfg {
    /// `H26X_TRACE_CU`: one line per coding unit.
    pub cu: bool,
    /// `H26X_TRACE_PU=x,y`: the prediction units covering luma (x, y).
    pub pu: Option<(i32, i32)>,
    /// `H26X_TRACE_TB=c,x,y`: the transform blocks of component c covering (x, y).
    pub tb: Option<(usize, usize, usize)>,
    /// `H26X_TRACE_CTB`: a checksum per CTB after reconstruction.
    pub ctb: bool,
}

impl TraceCfg {
    /// Read the environment once.
    pub fn from_env() -> Self {
        let pair = |name: &str| -> Option<Vec<i64>> {
            let v = std::env::var(name).ok()?;
            Some(v.split(',').filter_map(|t| t.parse().ok()).collect())
        };
        TraceCfg {
            cu: std::env::var_os("H26X_TRACE_CU").is_some(),
            ctb: std::env::var_os("H26X_TRACE_CTB").is_some(),
            pu: pair("H26X_TRACE_PU").filter(|p| p.len() == 2).map(|p| (p[0] as i32, p[1] as i32)),
            tb: pair("H26X_TRACE_TB").filter(|p| p.len() == 3).map(|p| (p[0] as usize, p[1] as usize, p[2] as usize)),
        }
    }
    fn tb_hit(&self, c_idx: usize, x: usize, y: usize, n: usize) -> bool {
        matches!(self.tb, Some((c, tx, ty)) if c == c_idx && tx >= x && tx < x + n && ty >= y && ty < y + n)
    }
}

#[inline(always)]
fn bin(c: &mut Cabac, cx: &mut Contexts, ctx: usize) -> u32 {
    c.decision(&mut cx.c[ctx])
}

impl<'a, S: Sample> SliceDec<'a, S> {
    fn bit_depth(&self) -> u32 {
        self.sps.bit_depth_luma
    }

    /// The current block's side of the z-scan availability test (6.4.1),
    /// for a block about to ask after several neighbours.
    fn avail_ctx(&self, xc: i32, yc: i32) -> AvailCtx {
        self.info.avail_ctx(xc, yc, self.frame.width as i32, self.frame.height as i32)
    }

    // ------------------------------------------------------------------
    // SAO syntax (7.3.8.3)
    // ------------------------------------------------------------------
    fn parse_sao(&mut self, rx: usize, ry: usize) -> Result<()> {
        let ctb = self.ctb_addr_rs;
        let wc = self.info.wc;
        let mut merge_left = false;
        let mut merge_up = false;
        if rx > 0 {
            let left_in_slice = ctb as u32 > self.slice_addr;
            let left_in_tile = self.info.ctb_tile[ctb] == self.info.ctb_tile[ctb - 1];
            if left_in_slice && left_in_tile {
                merge_left = bin(&mut self.cabac, &mut self.cx, SAO_MERGE_FLAG_OFFSET) != 0;
            }
        }
        if ry > 0 && !merge_left {
            let up_in_slice = (ctb - wc) as u32 >= self.slice_addr;
            let up_in_tile = self.info.ctb_tile[ctb] == self.info.ctb_tile[ctb - wc];
            if up_in_slice && up_in_tile {
                merge_up = bin(&mut self.cabac, &mut self.cx, SAO_MERGE_FLAG_OFFSET) != 0;
            }
        }
        if merge_left {
            self.info.sao[ctb] = self.info.sao[ctb - 1];
            return Ok(());
        }
        if merge_up {
            self.info.sao[ctb] = self.info.sao[ctb - wc];
            return Ok(());
        }
        let mut params = [SaoParams::default(); 3];
        let ncomp = if self.sps.chroma_format_idc != 0 { 3 } else { 1 };
        let bd = self.bit_depth();
        let cmax = (1u32 << (bd.min(10) - 5)) - 1;
        for c_idx in 0..ncomp {
            if !((self.hdr.sao_luma && c_idx == 0) || (self.hdr.sao_chroma && c_idx > 0)) {
                continue;
            }
            if c_idx == 0 || c_idx == 1 {
                // sao_type_idx: TR cMax 2, first bin ctx, second bypass.
                let t = if bin(&mut self.cabac, &mut self.cx, SAO_TYPE_IDX_OFFSET) == 0 {
                    0
                } else if self.cabac.bypass() == 0 {
                    1
                } else {
                    2
                };
                params[c_idx].type_idx = t;
                if c_idx == 1 {
                    params[2].type_idx = t;
                }
            }
            if params[c_idx].type_idx == 0 {
                continue;
            }
            // SaoOffsetVal = sign * abs << log2OffsetScale (7-72); the scale
            // comes from the PPS range extension (0 without it).
            let shift = if c_idx == 0 { self.pps.log2_sao_offset_scale.0 } else { self.pps.log2_sao_offset_scale.1 };
            let mut abs = [0i32; 4];
            for a in abs.iter_mut() {
                let mut v = 0u32;
                while v < cmax && self.cabac.bypass() != 0 {
                    v += 1;
                }
                *a = v as i32;
            }
            if params[c_idx].type_idx == 1 {
                for a in abs.iter_mut() {
                    if *a != 0 && self.cabac.bypass() != 0 {
                        *a = -*a;
                    }
                }
                params[c_idx].band_or_class = self.cabac.bypass_bits(5) as u8;
                for i in 0..4 {
                    params[c_idx].offsets[i] = (abs[i] << shift) as i16;
                }
            } else {
                // Edge: offsets 0,1 positive, 2,3 negative.
                params[c_idx].offsets = [(abs[0] << shift) as i16, (abs[1] << shift) as i16, (-(abs[2] << shift)) as i16, (-(abs[3] << shift)) as i16];
                if c_idx == 0 {
                    params[0].band_or_class = self.cabac.bypass_bits(2) as u8;
                } else if c_idx == 1 {
                    let cls = self.cabac.bypass_bits(2) as u8;
                    params[1].band_or_class = cls;
                    params[2].band_or_class = cls;
                } else {
                    // Cr shares the class with Cb (already set).
                }
            }
        }
        self.info.sao[ctb] = params;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Coding tree
    // ------------------------------------------------------------------

    /// Decode one CTU (SAO + coding quadtree). Returns Ok(()) after
    /// `end_of_slice_segment_flag` is left for the caller.
    pub fn decode_ctu(&mut self, ctb_addr_rs: usize, ctb_addr_ts: usize) -> Result<()> {
        self.ctb_addr_rs = ctb_addr_rs;
        self.ctb_addr_ts = ctb_addr_ts;
        let wc = self.info.wc;
        let rx = ctb_addr_rs % wc;
        let ry = ctb_addr_rs / wc;
        self.info.ctb_slice[ctb_addr_rs] = self.slice_idx;
        self.info.ctb_slice_addr[ctb_addr_rs] = self.slice_addr;
        if self.hdr.sao_luma || self.hdr.sao_chroma {
            self.parse_sao(rx, ry)?;
        }
        let log2 = self.sps.log2_ctb_size;
        let x0 = (rx << log2) as i32;
        let y0 = (ry << log2) as i32;
        self.coding_quadtree(x0, y0, log2, 0)
    }

    fn coding_quadtree(&mut self, x0: i32, y0: i32, log2_cb: u32, depth: u32) -> Result<()> {
        let size = 1i32 << log2_cb;
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let split = if x0 + size <= pw && y0 + size <= ph && log2_cb > self.sps.log2_min_cb_size {
            // split_cu_flag with context from the neighbours' depth.
            let mut inc = 0usize;
            let ac = self.avail_ctx(x0, y0);
            if self.info.available_at(&ac, x0 - 1, y0) && self.info.ct_depth[self.info.idx4((x0 - 1) as usize, y0 as usize)] as u32 > depth {
                inc += 1;
            }
            if self.info.available_at(&ac, x0, y0 - 1) && self.info.ct_depth[self.info.idx4(x0 as usize, (y0 - 1) as usize)] as u32 > depth {
                inc += 1;
            }
            bin(&mut self.cabac, &mut self.cx, SPLIT_CODING_UNIT_FLAG_OFFSET + inc) != 0
        } else {
            log2_cb > self.sps.log2_min_cb_size
        };
        if self.hdr.cu_chroma_qp_offset_enabled && log2_cb >= self.sps.log2_ctb_size - self.pps.diff_cu_chroma_qp_offset_depth {
            self.is_cu_chroma_qp_offset_coded = false;
        }
        if self.pps.cu_qp_delta_enabled && log2_cb >= self.sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta_val = 0;
            self.qg = (x0, y0);
            // qPY_PREV: SliceQpY for the first QG of a slice / tile / WPP row,
            // else the QpY of the last CU of the previous QG.
            self.qg_qp_prev = if self.first_qg { self.hdr.slice_qp } else { self.qp_y_prev };
            self.first_qg = false;
        }
        if split {
            let half = size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            self.coding_quadtree(x0, y0, log2_cb - 1, depth + 1)?;
            if x1 < pw {
                self.coding_quadtree(x1, y0, log2_cb - 1, depth + 1)?;
            }
            if y1 < ph {
                self.coding_quadtree(x0, y1, log2_cb - 1, depth + 1)?;
            }
            if x1 < pw && y1 < ph {
                self.coding_quadtree(x1, y1, log2_cb - 1, depth + 1)?;
            }
        } else {
            self.coding_unit(x0, y0, log2_cb, depth)?;
        }
        // End of a quantisation group: remember QpY for qPY_PREV.
        if self.pps.cu_qp_delta_enabled {
            let mask = (1i32 << (self.sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth)) - 1;
            if ((x0 + size) & mask) == 0 && ((y0 + size) & mask) == 0 {
                self.qp_y_prev = self.qp_y;
            }
        }
        Ok(())
    }

    /// `qPY_PRED` for the CU at `(x_cb, y_cb)` in the current QG (8.6.1).
    fn qp_y_pred(&mut self, x_cb: i32, y_cb: i32) -> i32 {
        let (xq, yq) = self.qg;
        let prev = self.qg_qp_prev;
        let ctb_cur = self.info.ctb_of(x_cb as usize, y_cb as usize);
        let ac = self.avail_ctx(x_cb, y_cb);
        let qa = if self.info.available_at(&ac, xq - 1, yq) && self.info.ctb_of((xq - 1) as usize, yq as usize) == ctb_cur {
            self.info.qp_y[self.info.idx4((xq - 1) as usize, yq as usize)] as i32
        } else {
            prev
        };
        let qb = if self.info.available_at(&ac, xq, yq - 1) && self.info.ctb_of(xq as usize, (yq - 1) as usize) == ctb_cur {
            self.info.qp_y[self.info.idx4(xq as usize, (yq - 1) as usize)] as i32
        } else {
            prev
        };
        (qa + qb + 1) >> 1
    }

    fn set_qp(&mut self, x_cb: i32, y_cb: i32) {
        let pred = self.qp_y_pred(x_cb, y_cb);
        let bd_off = 6 * (self.sps.bit_depth_luma as i32 - 8);
        self.qp_y = ((pred + self.cu_qp_delta_val + 52 + 2 * bd_off) % (52 + bd_off)) - bd_off;
    }

    fn coding_unit(&mut self, x0: i32, y0: i32, log2_cb: u32, depth: u32) -> Result<()> {
        let n = 1i32 << log2_cb;
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let w4 = self.info.w4;
        let cw = (n.min(pw - x0)) as usize;
        let ch = (n.min(ph - y0)) as usize;
        let mut bypass = false;
        if self.pps.transquant_bypass_enabled {
            bypass = bin(&mut self.cabac, &mut self.cx, CU_TRANSQUANT_BYPASS_FLAG_OFFSET) != 0;
        }
        let mut skip = false;
        if self.hdr.slice_type != SliceType::I {
            let mut inc = 0usize;
            let ac = self.avail_ctx(x0, y0);
            if self.info.available_at(&ac, x0 - 1, y0) && self.info.skip[self.info.idx4((x0 - 1) as usize, y0 as usize)] != 0 {
                inc += 1;
            }
            if self.info.available_at(&ac, x0, y0 - 1) && self.info.skip[self.info.idx4(x0 as usize, (y0 - 1) as usize)] != 0 {
                inc += 1;
            }
            skip = bin(&mut self.cabac, &mut self.cx, SKIP_FLAG_OFFSET + inc) != 0;
        }
        // Record depth / skip / pred mode for the CU area now (neighbours in
        // this CU need them; availability uses pred_mode != 2).
        PicInfo::fill4(&mut self.info.ct_depth, w4, x0 as usize, y0 as usize, cw, ch, depth as u8);
        PicInfo::fill4(&mut self.info.skip, w4, x0 as usize, y0 as usize, cw, ch, skip as u8);

        // QP for this CU (delta may still arrive in a TU; recomputed then).
        self.set_qp(x0, y0);

        let intra;
        let mut part_mode = PartMode::P2Nx2N;
        let mut pcm = false;
        if skip {
            intra = false;
        } else {
            intra = if self.hdr.slice_type != SliceType::I {
                bin(&mut self.cabac, &mut self.cx, PRED_MODE_FLAG_OFFSET) != 0
            } else {
                true
            };
            if !intra || log2_cb == self.sps.log2_min_cb_size {
                part_mode = self.parse_part_mode(intra, log2_cb)?;
            }
        }
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0 as usize, y0 as usize, cw, ch, intra as u8);
        // Motion of an intra CU: none (TMVP treats it as unavailable). An
        // inter CU's prediction units cover it entirely and write their own.
        if intra {
            let w4 = self.frame.w4;
            fill_motion(&mut self.frame.motion, w4, x0 as usize, y0 as usize, cw, ch, MotionInfo::INTRA);
        }

        let mut intra_modes = [1u32; 4];
        let mut chroma_mode_syntax = [0u32; 4];
        let cat = self.sps.chroma_array_type();
        if intra {
            if part_mode == PartMode::P2Nx2N
                && self.sps.pcm_enabled
                && log2_cb >= self.sps.pcm.2
                && log2_cb <= self.sps.pcm.3
            {
                pcm = self.cabac.terminate() != 0;
            }
            if pcm {
                self.decode_pcm(x0, y0, log2_cb)?;
            } else {
                let npu = if part_mode == PartMode::PNxN { 4 } else { 1 };
                let pb = if part_mode == PartMode::PNxN { n / 2 } else { n };
                let mut prev_flags = [false; 4];
                for i in 0..npu {
                    prev_flags[i] = bin(&mut self.cabac, &mut self.cx, PREV_INTRA_LUMA_PRED_FLAG_OFFSET) != 0;
                }
                for i in 0..npu {
                    let xp = x0 + (i as i32 % 2) * pb;
                    let yp = y0 + (i as i32 / 2) * pb;
                    let mode = if prev_flags[i] {
                        // mpm_idx: TR cMax 2 bypass.
                        let mut idx = 0usize;
                        while idx < 2 && self.cabac.bypass() != 0 {
                            idx += 1;
                        }
                        let cands = self.mpm_candidates(xp, yp);
                        cands[idx]
                    } else {
                        let rem = self.cabac.bypass_bits(5);
                        let mut cands = self.mpm_candidates(xp, yp);
                        cands.sort_unstable();
                        let mut m = rem;
                        for c in cands {
                            if m >= c {
                                m += 1;
                            }
                        }
                        m
                    };
                    intra_modes[i] = mode;
                    PicInfo::fill4(&mut self.info.intra_mode, w4, xp as usize, yp as usize, pb as usize, pb as usize, mode as u8);
                }
                if cat != 0 {
                    // intra_chroma_pred_mode: bin0 ctx; 0 -> 4, else 2 bypass
                    // bits. One per CU, or one per PB in 4:4:4 NxN.
                    let nc = if cat == 3 { npu } else { 1 };
                    for cm in chroma_mode_syntax.iter_mut().take(nc) {
                        *cm = if bin(&mut self.cabac, &mut self.cx, INTRA_CHROMA_PRED_MODE_OFFSET) == 0 { 4 } else { self.cabac.bypass_bits(2) };
                    }
                }
            }
        } else {
            // Inter: prediction units.
            let pus = part_mode.pus(n);
            for (part_idx, &(px, py, pwid, phei)) in pus.iter().enumerate() {
                self.prediction_unit(x0, y0, n, x0 + px, y0 + py, pwid, phei, part_idx as u32, skip)?;
            }
        }

        // Residual.
        let mut rqt_root_cbf = true;
        if !pcm {
            if !intra && !(part_mode == PartMode::P2Nx2N && self.last_pu_merged) && !skip {
                rqt_root_cbf = bin(&mut self.cabac, &mut self.cx, NO_RESIDUAL_DATA_FLAG_OFFSET) != 0;
            } else if skip {
                rqt_root_cbf = false;
            }
            // Chroma intra mode (IntraPredModeC, 8.4.3) per PB (one for the
            // CU unless 4:4:4 NxN), with the 4:2:2 mode mapping (Table 8-3).
            let mut chroma_modes = [0u32; 4];
            if intra {
                for i in 0..4 {
                    let luma = intra_modes[if cat == 3 { i } else { 0 }];
                    let syn = chroma_mode_syntax[if cat == 3 { i } else { 0 }];
                    chroma_modes[i] = intra_chroma_mode(cat, luma, syn);
                }
            }
            if rqt_root_cbf {
                let intra_split = intra && part_mode == PartMode::PNxN;
                let max_depth = if intra {
                    self.sps.max_th_depth_intra + intra_split as u32
                } else {
                    self.sps.max_th_depth_inter
                };
                let cu = CuCtx { x0, y0, log2_cb, intra, part_mode, intra_split, max_depth, chroma_modes, chroma_syntax: chroma_mode_syntax, bypass };
                self.transform_tree(&cu, x0, y0, x0, y0, log2_cb, 0, 0, [[true; 2]; 2])?;
            } else if intra {
                // Intra CU with no residual still needs its prediction.
                // (rqt_root_cbf is not present for intra CUs: always 1.)
                unreachable!("intra CUs always carry a transform tree");
            }
        }

        if self.trace.cu {
            eprintln!("cu poc={} x={} y={} n={} intra={} skip={} pcm={} bypass={} qp={} part={:?} modes={:?} csyn={:?}", self.refs.cur_poc, x0, y0, n, intra, skip, pcm, bypass, self.qp_y, part_mode, intra_modes, chroma_mode_syntax);
        }
        // Bookkeeping: QP over the CU, filter exemption, PU edges.
        PicInfo::fill4(&mut self.info.qp_y, w4, x0 as usize, y0 as usize, cw, ch, self.qp_y as i8);
        let exempt = ((pcm && self.sps.pcm.4) as u8) | ((bypass as u8) << 1) | (bypass as u8);
        if exempt != 0 {
            PicInfo::fill4(&mut self.info.filter_exempt, w4, x0 as usize, y0 as usize, cw, ch, exempt);
        }
        // Prediction block edges (for deblocking): left/top edge of every PU.
        let pus = part_mode.pus(n);
        for &(px, py, pwid, phei) in pus.iter() {
            let (ax, ay) = ((x0 + px) as usize, (y0 + py) as usize);
            for yy in (ay..(ay + phei as usize).min(ph as usize)).step_by(4) {
                let i = self.info.idx4(ax, yy);
                self.info.edges[i] |= 2;
            }
            for xx in (ax..(ax + pwid as usize).min(pw as usize)).step_by(4) {
                let i = self.info.idx4(xx, ay);
                self.info.edges[i] |= 8;
            }
        }
        // Coding block edges are transform block edges too (marked by the
        // transform tree); a skipped CU has no transform tree, so mark here.
        for yy in (y0 as usize..(y0 as usize + ch)).step_by(4) {
            let i = self.info.idx4(x0 as usize, yy);
            self.info.edges[i] |= 1;
        }
        for xx in (x0 as usize..(x0 as usize + cw)).step_by(4) {
            let i = self.info.idx4(xx, y0 as usize);
            self.info.edges[i] |= 4;
        }
        Ok(())
    }

    fn parse_part_mode(&mut self, intra: bool, log2_cb: u32) -> Result<PartMode> {
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET) != 0 {
            return Ok(PartMode::P2Nx2N);
        }
        if intra {
            return Ok(PartMode::PNxN);
        }
        let min = log2_cb == self.sps.log2_min_cb_size;
        if min {
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
                return Ok(PartMode::P2NxN);
            }
            if log2_cb == 3 {
                return Ok(PartMode::PNx2N);
            }
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 2) != 0 {
                return Ok(PartMode::PNx2N);
            }
            return Ok(PartMode::PNxN);
        }
        if !self.sps.amp_enabled {
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
                return Ok(PartMode::P2NxN);
            }
            return Ok(PartMode::PNx2N);
        }
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
            // Horizontal.
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 3) != 0 {
                return Ok(PartMode::P2NxN);
            }
            return Ok(if self.cabac.bypass() != 0 { PartMode::P2NxnD } else { PartMode::P2NxnU });
        }
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 3) != 0 {
            return Ok(PartMode::PNx2N);
        }
        Ok(if self.cabac.bypass() != 0 { PartMode::PnRx2N } else { PartMode::PnLx2N })
    }

    /// The three MPM candidates (8.4.2) for the PU at `(xp, yp)`.
    fn mpm_candidates(&self, xp: i32, yp: i32) -> [u32; 3] {
        let ac = self.avail_ctx(xp, yp);
        let cand = |xn: i32, yn: i32, is_above: bool| -> u32 {
            if !self.info.available_at(&ac, xn, yn) {
                return 1;
            }
            let i = self.info.idx4(xn as usize, yn as usize);
            if self.info.pred_mode[i] != 1 {
                return 1; // not intra (or pcm counts as intra with mode DC: pcm CUs store DC)
            }
            if is_above && yn < ((yp >> self.sps.log2_ctb_size) << self.sps.log2_ctb_size) {
                return 1;
            }
            self.info.intra_mode[i] as u32
        };
        let a = cand(xp - 1, yp, false);
        let b = cand(xp, yp - 1, true);
        if a == b {
            if a < 2 {
                [0, 1, 26]
            } else {
                [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
            }
        } else {
            let c = if a != 0 && b != 0 {
                0
            } else if a != 1 && b != 1 {
                1
            } else {
                26
            };
            [a, b, c]
        }
    }

    fn decode_pcm(&mut self, x0: i32, y0: i32, log2_cb: u32) -> Result<()> {
        let n = 1usize << log2_cb;
        let r = self.cabac.reader();
        r.align();
        let (bdy, bdc) = (self.sps.pcm.0, self.sps.pcm.1);
        let shift_y = self.sps.bit_depth_luma - bdy;
        let shift_c = self.sps.bit_depth_chroma - bdc;
        let stride = self.frame.y.stride;
        let off = self.frame.y.offset(x0 as isize, y0 as isize);
        for y in 0..n {
            for x in 0..n {
                let v = r.bits(bdy) << shift_y;
                self.frame.y.data[off + y * stride + x] = S::from_i32(v as i32);
            }
        }
        if self.frame.chroma != ChromaFormat::Monochrome {
            let (sw, sh) = self.sps.sub_wh();
            let cs = self.frame.cb.stride;
            let coff = self.frame.cb.offset((x0 as usize / sw) as isize, (y0 as usize / sh) as isize);
            for y in 0..n / sh {
                for x in 0..n / sw {
                    let v = r.bits(bdc) << shift_c;
                    self.frame.cb.data[coff + y * cs + x] = S::from_i32(v as i32);
                }
            }
            for y in 0..n / sh {
                for x in 0..n / sw {
                    let v = r.bits(bdc) << shift_c;
                    self.frame.cr.data[coff + y * cs + x] = S::from_i32(v as i32);
                }
            }
        }
        if r.overrun() {
            return Err(Error::bitstream("PCM samples truncated"));
        }
        self.cabac.reinit();
        // PCM CUs report intra mode DC for their neighbours.
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.intra_mode, w4, x0 as usize, y0 as usize, n, n, 1);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prediction units
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn prediction_unit(&mut self, x_cb: i32, y_cb: i32, n_cb: i32, x_pb: i32, y_pb: i32, w: i32, h: i32, part_idx: u32, skip: bool) -> Result<()> {
        let pu = PuPos { x_cb, y_cb, n_cb, x_pb, y_pb, w, h, part_idx };
        // TMVP reads the collocated picture's motion within this CTB row.
        if let Some(col) = self.col_shared {
            let row_end = ((y_cb >> self.sps.log2_ctb_size) + 1) << self.sps.log2_ctb_size;
            col.progress.wait_decoded(row_end.min(self.frame.height as i32));
        }
        let cand: Cand;
        let mut merged = false;
        if skip {
            let idx = self.parse_merge_idx();
            cand = merge_candidate(self.info, self.frame, &self.refs, &pu, idx);
            merged = true;
        } else {
            let merge_flag = bin(&mut self.cabac, &mut self.cx, MERGE_FLAG_OFFSET) != 0;
            if merge_flag {
                let idx = self.parse_merge_idx();
                cand = merge_candidate(self.info, self.frame, &self.refs, &pu, idx);
                merged = true;
            } else {
                // inter_pred_idc
                let mut pred_idc = 0u32; // 0 L0, 1 L1, 2 BI
                if self.hdr.slice_type == SliceType::B {
                    if w + h != 12 {
                        let depth = self.info.ct_depth[self.info.idx4(x_cb as usize, y_cb as usize)] as usize;
                        if bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + depth) != 0 {
                            pred_idc = 2;
                        } else {
                            pred_idc = bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + 4);
                        }
                    } else {
                        pred_idc = bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + 4);
                    }
                }
                let mut c = Cand { mv: [Mv::ZERO; 2], ref_idx: [-1; 2] };
                let mut mvds = [Mv::ZERO; 2];
                let mut mvp_flags = [0u32; 2];
                for list in 0..2 {
                    let uses = match pred_idc {
                        0 => list == 0,
                        1 => list == 1,
                        _ => true,
                    };
                    if !uses {
                        continue;
                    }
                    let nref = self.hdr.num_ref_idx[list];
                    let ri = if nref > 1 { self.parse_ref_idx(nref) } else { 0 };
                    c.ref_idx[list] = ri as i8;
                    if list == 1 && self.hdr.mvd_l1_zero && pred_idc == 2 {
                        mvds[1] = Mv::ZERO;
                    } else {
                        mvds[list] = self.parse_mvd()?;
                    }
                    mvp_flags[list] = bin(&mut self.cabac, &mut self.cx, MVP_LX_FLAG_OFFSET);
                }
                for list in 0..2 {
                    if c.ref_idx[list] < 0 {
                        continue;
                    }
                    if c.ref_idx[list] as usize >= self.ref_frames[list].len() {
                        return Err(Error::bitstream("ref_idx beyond the reference list"));
                    }
                    let mvp = amvp(self.info, self.frame, &self.refs, &pu, list, c.ref_idx[list], mvp_flags[list]);
                    // uLX = (mvpLX + mvdLX + 2^16) % 2^16 -> wrapping i16 add.
                    c.mv[list] = Mv::new(mvp.x.wrapping_add(mvds[list].x), mvp.y.wrapping_add(mvds[list].y));
                }
                cand = c;
            }
        }
        self.last_pu_merged = merged;
        // Store motion.
        let mut mi = MotionInfo { mv: cand.mv, ref_delta: [0; 2], ref_idx: cand.ref_idx, flags: 0, pad: 0 };
        for list in 0..2 {
            if cand.ref_idx[list] >= 0 {
                let ri = cand.ref_idx[list] as usize;
                if ri >= self.refs.pocs[list].len() {
                    return Err(Error::bitstream("merge candidate references beyond the list"));
                }
                mi.ref_delta[list] = (self.refs.cur_poc - self.refs.pocs[list][ri]).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                mi.flags |= (self.refs.long_term[list][ri] as u8) << list;
            }
        }
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let cw = w.min(pw - x_pb) as usize;
        let ch = h.min(ph - y_pb) as usize;
        {
            let w4 = self.frame.w4;
            fill_motion(&mut self.frame.motion, w4, x_pb as usize, y_pb as usize, cw, ch, mi);
        }
        // Motion compensation: wait for the reference rows the filters reach
        // (8-tap luma: 3 above / 4 below; 4-tap chroma: 1 / 2, in luma rows).
        let pic_h = self.frame.height as i32;
        for list in 0..2 {
            if cand.ref_idx[list] < 0 {
                continue;
            }
            let mv = cand.mv[list];
            let yi = y_pb + (mv.y as i32 >> 2);
            let need_l = yi + h + 4;
            // Chroma: eighth-sample vertical vector in chroma rows.
            let sh = self.sps.sub_wh().1 as i32;
            let mvcy = if sh == 2 { mv.y as i32 } else { mv.y as i32 * 2 };
            let yci = (y_pb / sh) + (mvcy >> 3);
            let need_c = sh * (yci + h / sh + 2);
            // Reads above the picture clamp to (or pad from) row 0, which is
            // only ready once row 0 is published; reads below need it all.
            let need = need_l.max(need_c).clamp(1, pic_h);
            self.ref_shared[list][cand.ref_idx[list] as usize].progress.wait_done(need);
        }
        let weighting = self.weighting_for(cand.ref_idx);
        if let Some((tx, ty)) = self.trace.pu {
            if tx >= x_pb && tx < x_pb + w && ty >= y_pb && ty < y_pb + h {
                eprintln!(
                    "pu poc={} x={x_pb} y={y_pb} w={w} h={h} merged={merged} mv={:?} ref_idx={:?} ref_delta={:?} weighting={:?}",
                    self.refs.cur_poc, cand.mv, cand.ref_idx, mi.ref_delta, weighting
                );
            }
        }
        let f0 = if cand.ref_idx[0] >= 0 { Some((self.ref_frames[0][cand.ref_idx[0] as usize], cand.mv[0])) } else { None };
        let f1 = if cand.ref_idx[1] >= 0 { Some((self.ref_frames[1][cand.ref_idx[1] as usize], cand.mv[1])) } else { None };
        // Blocks may extend past the picture edge (the last CTB row/col);
        // predict the whole PB — the border absorbs it.
        predict_block(&self.dsp, &mut self.mc, self.frame, x_pb as usize, y_pb as usize, w as usize, h as usize, f0, f1, weighting);
        Ok(())
    }

    fn weighting_for(&self, ref_idx: [i8; 2]) -> [Weighting; 3] {
        let explicit = match self.hdr.slice_type {
            SliceType::P => self.pps.weighted_pred,
            SliceType::B => self.pps.weighted_bipred,
            SliceType::I => false,
        };
        let Some(t) = (if explicit { self.hdr.pred_weights.as_ref() } else { None }) else {
            return [Weighting::Default; 3];
        };
        let shift1 = 14 - self.bit_depth() as i32;
        let mut out = [Weighting::Default; 3];
        for c in 0..3 {
            let log2_wd = if c == 0 { t.luma_log2_denom as i32 } else { t.chroma_log2_denom as i32 } + shift1;
            let mut w = [1i32; 2];
            let mut o = [0i32; 2];
            for list in 0..2 {
                if ref_idx[list] < 0 {
                    continue;
                }
                let e = &t.lists[list][ref_idx[list] as usize];
                if c == 0 {
                    w[list] = e.luma.0;
                    o[list] = e.luma.1;
                } else {
                    w[list] = e.chroma[c - 1].0;
                    o[list] = e.chroma[c - 1].1;
                }
            }
            out[c] = Weighting::Explicit { log2_wd, w, o };
        }
        out
    }

    fn parse_merge_idx(&mut self) -> usize {
        let max = self.hdr.max_num_merge_cand as usize;
        if max <= 1 {
            return 0;
        }
        let mut idx = 0usize;
        if bin(&mut self.cabac, &mut self.cx, MERGE_IDX_OFFSET) != 0 {
            idx = 1;
            while idx < max - 1 && self.cabac.bypass() != 0 {
                idx += 1;
            }
        }
        idx
    }

    fn parse_ref_idx(&mut self, nref: u32) -> u32 {
        let cmax = nref - 1;
        let mut v = 0u32;
        while v < cmax {
            let b = if v < 2 { bin(&mut self.cabac, &mut self.cx, REF_IDX_L0_OFFSET + v as usize) } else { self.cabac.bypass() };
            if b == 0 {
                break;
            }
            v += 1;
        }
        v
    }

    fn parse_mvd(&mut self) -> Result<Mv> {
        let g0x = bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER0_FLAG_OFFSET) != 0;
        let g0y = bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER0_FLAG_OFFSET) != 0;
        // The generated table (FFmpeg element order) keeps greater1 at slot +1.
        let g1x = if g0x { bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER1_FLAG_OFFSET + 1) != 0 } else { false };
        let g1y = if g0y { bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER1_FLAG_OFFSET + 1) != 0 } else { false };
        let mut out = [0i32; 2];
        for (i, (g0, g1)) in [(g0x, g1x), (g0y, g1y)].iter().enumerate() {
            if !g0 {
                continue;
            }
            let mut abs = 1i32;
            if *g1 {
                // abs_mvd_minus2: EG1 bypass.
                let mut k = 1u32;
                let mut v = 0i32;
                loop {
                    if self.cabac.bypass() != 0 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Error::bitstream("mvd runaway"));
                        }
                    } else {
                        break;
                    }
                }
                v += self.cabac.bypass_bits(k) as i32;
                abs = v + 2;
            }
            let sign = self.cabac.bypass();
            out[i] = if sign != 0 { -abs } else { abs };
            if !(-32768..=32767).contains(&out[i]) {
                return Err(Error::bitstream("mvd out of range"));
            }
        }
        Ok(Mv::new(out[0] as i16, out[1] as i16))
    }

    // ------------------------------------------------------------------
    // Transform tree
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        cu: &CuCtx,
        x0: i32,
        y0: i32,
        x_base: i32,
        y_base: i32,
        log2: u32,
        depth: u32,
        blk_idx: u32,
        parent_cbf_c: [[bool; 2]; 2],
    ) -> Result<()> {
        let split = if log2 <= self.sps.log2_max_tb_size
            && log2 > self.sps.log2_min_tb_size
            && depth < cu.max_depth
            && !(cu.intra_split && depth == 0)
        {
            bin(&mut self.cabac, &mut self.cx, SPLIT_TRANSFORM_FLAG_OFFSET + (5 - log2) as usize) != 0
        } else {
            let inter_split = self.sps.max_th_depth_inter == 0 && !cu.intra && cu.part_mode != PartMode::P2Nx2N && depth == 0;
            log2 > self.sps.log2_max_tb_size || (cu.intra_split && depth == 0) || inter_split
        };
        // cbf_cb / cbf_cr, per component and (4:2:2) per vertical half.
        let cat = self.sps.chroma_array_type();
        let mut cbf_c = [[false; 2]; 2];
        if cat != 0 && (log2 > 2 || cat == 3) {
            for c in 0..2 {
                if depth == 0 || parent_cbf_c[c][0] {
                    cbf_c[c][0] = bin(&mut self.cabac, &mut self.cx, CBF_CB_CR_OFFSET + depth as usize) != 0;
                    if cat == 2 && (!split || log2 == 3) {
                        cbf_c[c][1] = bin(&mut self.cabac, &mut self.cx, CBF_CB_CR_OFFSET + depth as usize) != 0;
                    }
                }
            }
        } else if cat != 0 && log2 == 2 {
            // Inferred from the parent (chroma is coded at the parent size).
            cbf_c = parent_cbf_c;
        }
        if split {
            let half = 1i32 << (log2 - 1);
            self.transform_tree(cu, x0, y0, x0, y0, log2 - 1, depth + 1, 0, cbf_c)?;
            self.transform_tree(cu, x0 + half, y0, x0, y0, log2 - 1, depth + 1, 1, cbf_c)?;
            self.transform_tree(cu, x0, y0 + half, x0, y0, log2 - 1, depth + 1, 2, cbf_c)?;
            self.transform_tree(cu, x0 + half, y0 + half, x0, y0, log2 - 1, depth + 1, 3, cbf_c)?;
            return Ok(());
        }
        let cbf_luma = if cu.intra || depth != 0 || cbf_c[0][0] || cbf_c[1][0] || cbf_c[0][1] || cbf_c[1][1] {
            bin(&mut self.cabac, &mut self.cx, CBF_LUMA_OFFSET + (depth == 0) as usize) != 0
        } else {
            true
        };
        self.transform_unit(cu, x0, y0, x_base, y_base, log2, depth, blk_idx, cbf_luma, cbf_c)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        cu: &CuCtx,
        x0: i32,
        y0: i32,
        x_base: i32,
        y_base: i32,
        log2: u32,
        depth: u32,
        blk_idx: u32,
        cbf_luma: bool,
        cbf_c: [[bool; 2]; 2],
    ) -> Result<()> {
        let n = 1usize << log2;
        let (pw, ph) = (self.frame.width as usize, self.frame.height as usize);
        // Transform block edges + cbf for the deblocking filter.
        {
            let w4 = self.info.w4;
            let _ = w4;
            for yy in (y0 as usize..(y0 as usize + n).min(ph)).step_by(4) {
                let i = self.info.idx4(x0 as usize, yy);
                self.info.edges[i] |= 1;
            }
            for xx in (x0 as usize..(x0 as usize + n).min(pw)).step_by(4) {
                let i = self.info.idx4(xx, y0 as usize);
                self.info.edges[i] |= 4;
            }
            if cbf_luma {
                let w4 = self.info.w4;
                PicInfo::fill4(&mut self.info.cbf_luma, w4, x0 as usize, y0 as usize, n.min(pw - x0 as usize), n.min(ph - y0 as usize), 1);
            }
        }
        let cbf_chroma = cbf_c[0][0] || cbf_c[1][0] || cbf_c[0][1] || cbf_c[1][1];
        if (cbf_luma || cbf_chroma) && self.pps.cu_qp_delta_enabled && !self.is_cu_qp_delta_coded {
            // cu_qp_delta_abs / sign.
            let mut prefix = 0u32;
            let mut ctx = CU_QP_DELTA_OFFSET;
            while prefix < 5 && bin(&mut self.cabac, &mut self.cx, ctx) != 0 {
                prefix += 1;
                ctx = CU_QP_DELTA_OFFSET + 1;
            }
            let mut val = prefix as i32;
            if prefix >= 5 {
                let mut k = 0u32;
                let mut v = 0i32;
                loop {
                    if self.cabac.bypass() != 0 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Error::bitstream("cu_qp_delta runaway"));
                        }
                    } else {
                        break;
                    }
                }
                v += self.cabac.bypass_bits(k) as i32;
                val += v;
            }
            if val != 0 && self.cabac.bypass() != 0 {
                val = -val;
            }
            self.is_cu_qp_delta_coded = true;
            self.cu_qp_delta_val = val;
            let bd_off = 6 * (self.sps.bit_depth_luma as i32 - 8);
            if val < -(26 + bd_off / 2) || val > 25 + bd_off / 2 {
                return Err(Error::bitstream("CuQpDeltaVal out of range"));
            }
            self.set_qp(cu.x0, cu.y0);
        }
        // cu_chroma_qp_offset_flag / _idx (range extension chroma QP offset lists).
        if self.hdr.cu_chroma_qp_offset_enabled && cbf_chroma && !cu.bypass && !self.is_cu_chroma_qp_offset_coded {
            let flag = bin(&mut self.cabac, &mut self.cx, CU_CHROMA_QP_OFFSET_FLAG_OFFSET) != 0;
            let mut idx = 0usize;
            let len = self.pps.chroma_qp_offset_lists.len();
            if flag && len > 1 {
                // TR, cMax = len - 1, all bins one context.
                while idx + 1 < len && bin(&mut self.cabac, &mut self.cx, CU_CHROMA_QP_OFFSET_IDX_OFFSET) != 0 {
                    idx += 1;
                }
            }
            self.cu_qp_offset_c = if flag { [self.pps.chroma_qp_offset_lists[idx].0, self.pps.chroma_qp_offset_lists[idx].1] } else { [0, 0] };
            self.is_cu_chroma_qp_offset_coded = true;
        }

        // Luma: intra prediction (per TB) then residual.
        if cu.intra {
            let mode = self.info.intra_mode[self.info.idx4(x0 as usize, y0 as usize)] as u32;
            self.intra_predict_block(0, x0 as usize, y0 as usize, n, mode, cu.bypass);
        }
        // Cross-component prediction (4:4:4): chroma residuals borrow a scaled
        // copy of this TU's luma residual.
        let ccp = self.pps.cross_component_prediction && cbf_luma;
        self.luma_res_valid = false;
        if cbf_luma {
            let mode = if cu.intra { self.info.intra_mode[self.info.idx4(x0 as usize, y0 as usize)] as u32 } else { 0 };
            self.residual_block(cu, x0 as usize, y0 as usize, log2, 0, mode, ccp, 0)?;
        }
        // Chroma: at this TU when the chroma block is at least 4x4 (always
        // in 4:4:4), else once for the parent at its fourth 4x4 luma block;
        // 4:2:2 chroma blocks are two squares stacked vertically.
        let cat = self.sps.chroma_array_type();
        if cat != 0 {
            let (sw, sh) = self.sps.sub_wh();
            let here = if log2 > 2 || cat == 3 {
                Some((x0 as usize / sw, y0 as usize / sh, if cat == 3 { log2 } else { log2 - 1 }))
            } else if blk_idx == 3 {
                Some((x_base as usize / sw, y_base as usize / sh, 2))
            } else {
                None
            };
            if let Some((xc, yc, log2c)) = here {
                let nc = 1usize << log2c;
                // The prediction block this TU lies in (4:4:4 NxN has four
                // chroma modes).
                let half = 1i32 << (cu.log2_cb - 1);
                let pb = if cu.intra_split && cat == 3 { ((y0 - cu.y0 >= half) as usize) * 2 + (x0 - cu.x0 >= half) as usize } else { 0 };
                let mode = cu.chroma_modes[pb];
                for c in 0..2usize {
                    // cross_comp_pred(): the residual scale for this component.
                    let mut res_scale = 0i32;
                    if ccp && (!cu.intra || cu.chroma_syntax[pb] == 4) {
                        // log2_res_scale_abs_plus1: TR cMax 4, one context per bin.
                        let mut v = 0usize;
                        while v < 4 && bin(&mut self.cabac, &mut self.cx, LOG2_RES_SCALE_ABS_OFFSET + 4 * c + v) != 0 {
                            v += 1;
                        }
                        if v > 0 {
                            let sign = bin(&mut self.cabac, &mut self.cx, RES_SCALE_SIGN_FLAG_OFFSET + c) != 0;
                            res_scale = (1 << (v - 1)) * if sign { -1 } else { 1 };
                        }
                    }
                    for t in 0..(if cat == 2 { 2 } else { 1 }) {
                        let yct = yc + t * nc;
                        if cu.intra {
                            self.intra_predict_block(1 + c, xc, yct, nc, mode, cu.bypass);
                        }
                        if cbf_c[c][t] {
                            self.residual_block(cu, xc, yct, log2c, 1 + c, mode, false, res_scale)?;
                        } else if res_scale != 0 {
                            self.add_scaled_luma_residual(xc, yct, log2c, 1 + c, res_scale);
                        }
                    }
                }
            }
        }
        let _ = depth;
        Ok(())
    }

    /// Intra prediction of one transform block of component `c_idx` at
    /// component coordinates `(x, y)`, size `n`.
    fn intra_predict_block(&mut self, c_idx: usize, x: usize, y: usize, n: usize, mode: u32, bypass: bool) {
        // Availability of the neighbouring samples in luma coordinates.
        let (sw, sh) = if c_idx == 0 { (1, 1) } else { self.sps.sub_wh() };
        let xl = (x * sw) as i32;
        let yl = (y * sh) as i32;
        let cip = self.pps.constrained_intra_pred;
        // Every reference sample is a neighbour of this one transform block.
        let ac = self.avail_ctx(xl, yl);
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let bd = if c_idx == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
        let strong = self.sps.strong_intra_smoothing;
        let filter = (c_idx == 0 || self.sps.chroma_array_type() == 3) && !self.sps.intra_smoothing_disabled();
        let boundary_filter = c_idx == 0 && !(self.sps.implicit_rdpcm() && bypass);
        // Borrow the side data on its own so the scratch, which lives beside
        // it, stays reachable.
        let info: &PicInfo = self.info;
        let check = |xn: i32, yn: i32| -> bool {
            if !info.available_at(&ac, xn, yn) {
                return false;
            }
            !cip || info.pred_mode[info.idx4(xn as usize, yn as usize)] == 1
        };
        // Left samples y = 0..2n and top samples x = 0..2n, in units of 4x4
        // luma blocks. Without constrained intra prediction, availability is
        // uniform over each aligned n-block of neighbours (the left n-block,
        // the below-left n-block: z-scan order decides for the whole block)
        // as long as it lies inside the picture — one check per half.
        let side = |vertical: bool, av_side: &mut [bool; 64]| {
            // Along this edge: the subsampling and the luma span of the block.
            let scale = if vertical { sh } else { sw };
            let unit = 4 / scale; // samples per 4x4 luma block along the edge
            let span = (n * scale) as i32;
            for half in 0..2 {
                let start = half * n; // in component samples
                let (nx, ny) = if vertical { (xl - 1, yl + half as i32 * span) } else { (xl + half as i32 * span, yl - 1) };
                let inside = if vertical { ny + span <= ph } else { nx + span <= pw };
                if !cip && inside {
                    let a = check(nx, ny);
                    for k in 0..n {
                        if start + k < 64 {
                            av_side[start + k] = a;
                        }
                    }
                } else {
                    let mut i = 0;
                    while i < n {
                        let (cx, cy) = if vertical { (nx, ny + (i * scale) as i32) } else { (nx + (i * scale) as i32, ny) };
                        let a = check(cx, cy);
                        for k in 0..unit {
                            if start + i + k < 64 {
                                av_side[start + i + k] = a;
                            }
                        }
                        i += unit;
                    }
                }
            }
        };
        let corner = check(xl - 1, yl - 1);
        let sc = &mut self.intra;
        sc.avail.corner = corner;
        side(true, &mut sc.avail.left);
        side(false, &mut sc.avail.top);
        let plane = match c_idx {
            0 => &mut self.frame.y,
            1 => &mut self.frame.cb,
            _ => &mut self.frame.cr,
        };
        intra_predict(plane, sc, x, y, n, mode, c_idx, filter, boundary_filter, bd, strong);
    }

    /// Cross-component prediction for a chroma block without a residual of
    /// its own: add the scaled luma residual of the TU.
    fn add_scaled_luma_residual(&mut self, x: usize, y: usize, log2: u32, c_idx: usize, res_scale: i32) {
        let n = 1usize << log2;
        if !self.luma_res_valid || self.luma_res.len() != n * n {
            return;
        }
        let mut coeffs = std::mem::take(&mut self.coeffs);
        if coeffs.len() < n * n {
            coeffs.resize(1024, 0);
        }
        for (r, &l) in coeffs[..n * n].iter_mut().zip(&self.luma_res) {
            *r = ((res_scale * l as i32) >> 3).clamp(-32768, 32767) as i16;
        }
        let bd = self.sps.bit_depth_chroma;
        let max = (1i32 << bd) - 1;
        let plane = if c_idx == 1 { &mut self.frame.cb } else { &mut self.frame.cr };
        let stride = plane.stride;
        let off = plane.offset(x as isize, y as isize);
        (self.dsp.add_residual)(&mut plane.data[off..], stride, &coeffs, n, max);
        self.coeffs = coeffs;
    }

    /// Parse and add the residual of one transform block of component
    /// `c_idx` at component coordinates `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    fn residual_block(&mut self, cu: &CuCtx, x: usize, y: usize, log2: u32, c_idx: usize, pred_mode: u32, keep_luma: bool, res_scale: i32) -> Result<()> {
        let n = 1usize << log2;
        // scanIdx (7.4.9.11).
        let scan_idx = residual_scan_idx(cu.intra, log2, c_idx, self.sps.chroma_array_type(), pred_mode);
        let params = ResidualParams {
            log2_size: log2,
            c_idx,
            scan_idx,
            bypass: cu.bypass,
            transform_skip_allowed: self.pps.transform_skip_enabled && log2 <= self.pps.log2_max_transform_skip_size,
            sign_hiding: self.pps.sign_data_hiding,
            intra: cu.intra,
            pred_mode_intra: pred_mode,
            ts_context: self.sps.ts_context(),
            implicit_rdpcm: self.sps.implicit_rdpcm(),
            explicit_rdpcm: self.sps.explicit_rdpcm(),
            persistent_rice: self.sps.persistent_rice(),
            trace: self.trace.tb_hit(c_idx, x, y, n),
        };
        let mut coeffs = std::mem::take(&mut self.coeffs);
        if coeffs.len() < n * n {
            coeffs.resize(1024, 0);
        }
        let ri = parse_residual(&mut self.cabac, &mut self.cx, &params, &mut coeffs)?;
        let ts = ri.transform_skip;
        // QP for the component.
        let qp = if c_idx == 0 {
            self.qp_y + 6 * (self.sps.bit_depth_luma as i32 - 8)
        } else {
            let bd_off_c = 6 * (self.sps.bit_depth_chroma as i32 - 8);
            let off = if c_idx == 1 { self.pps.cb_qp_offset + self.hdr.cb_qp_offset } else { self.pps.cr_qp_offset + self.hdr.cr_qp_offset };
            let qpi = (self.qp_y + off + self.cu_qp_offset_c[c_idx - 1]).clamp(-bd_off_c, 57);
            chroma_qp(self.sps.chroma_array_type(), qpi) + bd_off_c
        };
        let bd = if c_idx == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
        // 4x4 intra transform-skipped / bypassed blocks may be coded rotated.
        let rotate = self.sps.ts_rotation() && log2 == 2 && cu.intra && (ts || cu.bypass);
        if cu.bypass {
            if rotate {
                rotate_residual4(&mut coeffs);
            }
            if let Some(vertical) = ri.rdpcm {
                rdpcm_residual(&mut coeffs, log2, vertical);
            }
        }
        if !cu.bypass {
            let scaling = match &self.scaling {
                None => ScalingSource::Flat,
                Some(sl) => {
                    let size_id = (log2 - 2) as usize;
                    let matrix_id = if cu.intra { c_idx } else { 3 + c_idx };
                    let list = &sl.lists[size_id][matrix_id];
                    let dc = if size_id >= 2 { sl.dc[size_id - 2][matrix_id] } else { 16 };
                    ScalingSource::List(&list[..if size_id == 0 { 16 } else { 64 }], dc)
                }
            };
            scale_coefficients(&mut coeffs, log2, qp, bd, scaling, ts, ri.max_x, ri.max_y);
            let bd_shift = 20 - bd as i32;
            if ts {
                if rotate {
                    rotate_residual4(&mut coeffs);
                }
                transform_skip_residual(&mut coeffs, log2, bd);
                if let Some(vertical) = ri.rdpcm {
                    rdpcm_residual(&mut coeffs, log2, vertical);
                }
            } else if cu.intra && log2 == 2 && c_idx == 0 {
                (self.dsp.idst4)(&mut coeffs, bd_shift, ri.max_x, ri.max_y);
            } else {
                (self.dsp.idct[(log2 - 2) as usize])(&mut coeffs, bd_shift, ri.max_x, ri.max_y);
            }
        }
        if keep_luma {
            self.luma_res.clear();
            self.luma_res.extend_from_slice(&coeffs[..n * n]);
            self.luma_res_valid = true;
        }
        if res_scale != 0 && self.luma_res_valid && self.luma_res.len() == n * n {
            // Cross-component prediction (7.3.8.12 / 8.6.6): equal bit depths here.
            for (r, &l) in coeffs[..n * n].iter_mut().zip(&self.luma_res) {
                *r = (*r as i32 + ((res_scale * l as i32) >> 3)).clamp(-32768, 32767) as i16;
            }
        }
        // Add to the prediction.
        let max = (1i32 << bd) - 1;
        let plane = match c_idx {
            0 => &mut self.frame.y,
            1 => &mut self.frame.cb,
            _ => &mut self.frame.cr,
        };
        let stride = plane.stride;
        let off = plane.offset(x as isize, y as isize);
        if params.trace {
            eprintln!("tb c={c_idx} x={x} y={y} n={n} bypass={} ts={ts} qp={qp} scan={scan_idx}", cu.bypass);
            for yy in 0..n {
                let pred: Vec<i32> = (0..n).map(|xx| plane.data[off + yy * stride + xx].to_i32()).collect();
                let res: Vec<i16> = (0..n).map(|xx| coeffs[yy * n + xx]).collect();
                eprintln!("  pred {pred:?} res {res:?}");
            }
        }
        (self.dsp.add_residual)(&mut plane.data[off..], stride, &coeffs, n, max);
        self.coeffs = coeffs;
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Writers: the exact inverses of the intra coding-quadtree readers above,
// kept in this file beside them because inverse pairs drift when they are
// edited apart. Free functions rather than methods, because an encoder has
// no `SliceDec`: everything the reader digs out of its own state arrives
// here as a documented argument, already derived by the caller. What the
// reader *infers* rather than reads — `split_cu_flag` at a picture edge,
// `pred_mode_flag` in an I slice, `rqt_root_cbf` for intra CUs, the NxN
// restriction of `part_mode` to the minimum CB size — the caller must
// infer identically, by walking the same conditions; no writer exists for
// an uncoded bin. Inter-slice syntax has no writers yet: they arrive with
// inter prediction.
//
// Nothing outside the round-trip tests calls these yet — the H.265 encoder
// that will is being built alongside them; drop the allows when it lands.
// ----------------------------------------------------------------------

/// The neighbour facts `split_cu_flag`'s context derivation needs, already
/// resolved by the caller: `CtDepth` of the 4x4 block left of the coding
/// block's top-left corner and of the one above it, or `None` where that
/// neighbour is unavailable in the z-scan sense of 6.4.1 (outside the
/// picture, in another slice or tile, or not yet coded).
///
/// This is the same seam the reader uses at the top of `coding_quadtree`:
/// it consults `PicInfo::available_at` plus the `ct_depth` array, and an
/// encoder consults whatever bookkeeping it maintains — the flag's context
/// counts the available neighbours coded deeper than the current depth.
#[allow(dead_code)]
pub(crate) struct SplitCuNb {
    /// `CtDepth` of the block at `(x0 - 1, y0)`, if available.
    pub left_depth: Option<u8>,
    /// `CtDepth` of the block at `(x0, y0 - 1)`, if available.
    pub above_depth: Option<u8>,
}

/// Write `split_cu_flag`: the inverse of the read at the top of
/// `coding_quadtree`. Only for a flag the reader will actually read — when
/// the coding block lies fully inside the picture and is larger than the
/// minimum CB — otherwise the split is inferred and writing anything here
/// desyncs the stream. `depth` is the current coding-tree depth the
/// neighbour depths are compared against.
#[allow(dead_code)]
pub(crate) fn write_split_cu_flag(e: &mut CabacEncoder, cx: &mut Contexts, nb: &SplitCuNb, depth: u32, split: bool) {
    let mut inc = 0usize;
    if nb.left_depth.is_some_and(|d| d as u32 > depth) {
        inc += 1;
    }
    if nb.above_depth.is_some_and(|d| d as u32 > depth) {
        inc += 1;
    }
    e.encode_decision(&mut cx.c[SPLIT_CODING_UNIT_FLAG_OFFSET + inc], split as u32);
}

/// Write `cu_transquant_bypass_flag`: the inverse of the read at the top of
/// `coding_unit` — the first bin of every CU, before the skip flag and the
/// intra syntax, coded only when the PPS sets
/// `transquant_bypass_enabled_flag` (write nothing otherwise; the reader
/// infers false). One context, no neighbours. Under bypass the CU's
/// residuals are raw spatial differences — see `write_residual` for what
/// that does and does not change in the spelling.
#[allow(dead_code)]
pub(crate) fn write_cu_transquant_bypass_flag(e: &mut CabacEncoder, cx: &mut Contexts, bypass: bool) {
    e.encode_decision(&mut cx.c[CU_TRANSQUANT_BYPASS_FLAG_OFFSET], bypass as u32);
}

/// Write `part_mode` for an intra CU: the inverse of `parse_part_mode`
/// with `intra` true — a single context-coded bin, 1 for 2Nx2N, 0 for NxN.
/// Coded only when `log2_cb == log2_min_cb_size` (the reader does not read
/// it otherwise, and infers 2Nx2N); NxN exists only there.
///
/// There is no `pred_mode_flag` writer: in an I slice the reader never
/// reads one — intra is inferred — and inter slices are future work.
#[allow(dead_code)]
pub(crate) fn write_part_mode_intra(e: &mut CabacEncoder, cx: &mut Contexts, nxn: bool) {
    e.encode_decision(&mut cx.c[PART_MODE_OFFSET], !nxn as u32);
}

/// Write `prev_intra_luma_pred_flag` for one prediction unit. The reader
/// reads all of a CU's flags (one per PU: one, or four for NxN) *before*
/// any `mpm_idx` / `rem_intra_luma_pred_mode`, so a caller with NxN must
/// call this four times and only then write the four payloads.
///
/// Whether the PU's mode is one of its three MPM candidates — and which
/// index, or which remainder — is the *decision* side's problem, the same
/// seam as H.264's `PredMode`: the candidate derivation (8.4.2) belongs to
/// the encoder's mode decision, which mirrors `mpm_candidates` over its own
/// reconstruction state. The writers take the already-derived values.
#[allow(dead_code)]
pub(crate) fn write_prev_intra_luma_pred_flag(e: &mut CabacEncoder, cx: &mut Contexts, prev: bool) {
    e.encode_decision(&mut cx.c[PREV_INTRA_LUMA_PRED_FLAG_OFFSET], prev as u32);
}

/// Write `mpm_idx` (0..=2): truncated-unary bypass with cMax 2, the inverse
/// of the reader's `while idx < 2 && bypass() != 0` — `idx` ones, then a
/// terminating zero unless the cap was reached.
#[allow(dead_code)]
pub(crate) fn write_mpm_idx(e: &mut CabacEncoder, idx: u32) {
    debug_assert!(idx <= 2);
    for _ in 0..idx {
        e.encode_bypass(1);
    }
    if idx < 2 {
        e.encode_bypass(0);
    }
}

/// Write `rem_intra_luma_pred_mode` (0..=31): five bypass bits. The
/// remainder counts modes with the three MPM candidates removed: the
/// decision side takes its target mode and subtracts one for every
/// candidate smaller than it — the inverse of the reader's re-insertion
/// over the sorted candidates.
#[allow(dead_code)]
pub(crate) fn write_rem_intra_luma_pred_mode(e: &mut CabacEncoder, rem: u32) {
    debug_assert!(rem < 32);
    e.encode_bypass_bits(5, rem);
}

/// Write `intra_chroma_pred_mode`'s binarisation: `syntax` is the coded
/// value 0..=4, where 4 means "same as luma" (one context-coded 0) and
/// 0..=3 select from Table 8-2 (a context-coded 1, then two bypass bits).
/// One per CU, or one per PB in 4:4:4 NxN — the caller follows the
/// reader's loop. [`intra_chroma_mode`] maps what this spells to the
/// resulting `IntraPredModeC`; an encoder chooses the syntax and derives
/// the mode through that one copy of the mapping.
#[allow(dead_code)]
pub(crate) fn write_intra_chroma_pred_mode(e: &mut CabacEncoder, cx: &mut Contexts, syntax: u32) {
    debug_assert!(syntax <= 4);
    if syntax == 4 {
        e.encode_decision(&mut cx.c[INTRA_CHROMA_PRED_MODE_OFFSET], 0);
    } else {
        e.encode_decision(&mut cx.c[INTRA_CHROMA_PRED_MODE_OFFSET], 1);
        e.encode_bypass_bits(2, syntax);
    }
}

/// Write `split_transform_flag`: the inverse of the read at the top of
/// `transform_tree`. Coded only when the reader would read it — `log2` at
/// most the maximum TB size, above the minimum, depth below
/// `MaxTrafoDepth`, and not the forced split of an NxN intra CU's first
/// level — otherwise the split is inferred from those same conditions.
/// The context depends only on the TB size.
#[allow(dead_code)]
pub(crate) fn write_split_transform_flag(e: &mut CabacEncoder, cx: &mut Contexts, log2: u32, split: bool) {
    e.encode_decision(&mut cx.c[SPLIT_TRANSFORM_FLAG_OFFSET + (5 - log2) as usize], split as u32);
}

/// Write `cbf_cb` or `cbf_cr` (they share contexts): the context is the
/// transform-tree depth of the node coding the flag. Coded at a node when
/// the chroma block is at least 4x4 (`log2 > 2` outside 4:4:4) and either
/// `trafo_depth == 0` or the parent's corresponding flag was 1; at
/// `log2 == 2` the reader inherits the parent's flags instead — nothing
/// may be written there. In 4:2:2 the second (lower) chroma square has its
/// own flag at the same context; the caller writes it right after the
/// first, mirroring the reader's order (cb both halves, then cr).
#[allow(dead_code)]
pub(crate) fn write_cbf_chroma(e: &mut CabacEncoder, cx: &mut Contexts, trafo_depth: u32, cbf: bool) {
    e.encode_decision(&mut cx.c[CBF_CB_CR_OFFSET + trafo_depth as usize], cbf as u32);
}

/// Write `cbf_luma`: context 1 at transform-tree depth 0, else 0. Read by
/// the reader at every leaf of an intra CU's transform tree (for inter it
/// is inferred 1 when nothing else in the leaf is coded — future work).
#[allow(dead_code)]
pub(crate) fn write_cbf_luma(e: &mut CabacEncoder, cx: &mut Contexts, trafo_depth: u32, cbf: bool) {
    e.encode_decision(&mut cx.c[CBF_LUMA_OFFSET + (trafo_depth == 0) as usize], cbf as u32);
}

/// `IntraPredModeC` (8.4.3): the mode `intra_chroma_pred_mode` syntax
/// `syn` selects given the PB's luma mode — Table 8-2's substitution of 34
/// where the selection collides with luma, and Table 8-3's 4:2:2 mapping.
/// One copy, used by the parser above and by the decision side of the
/// encoder (which needs it for the chroma scan order, among others).
pub(crate) fn intra_chroma_mode(chroma_array_type: u32, luma: u32, syn: u32) -> u32 {
    let m = match syn {
        0 => 0,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => luma,
    };
    let m = if syn < 4 && m == luma { 34 } else { m };
    if chroma_array_type == 2 { MODE_422[m as usize] } else { m }
}

/// `QpC` as a function of `qPi` (8.6.1): Table 8-10 for 4:2:0, otherwise
/// `Min(qPi, 51)`.
pub fn chroma_qp(chroma_array_type: u32, qpi: i32) -> i32 {
    if chroma_array_type != 1 {
        return qpi.min(51);
    }
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        T[(qpi - 30) as usize]
    }
}

/// The 4:2:2 chroma intra mode mapping (Table 8-3), by `modeIdc`.
/// `pub(crate)` for the encoder's intra decision module, whose mirrored
/// derivation must use THIS table rather than a retyped copy — a derived
/// table can drift, a shared one cannot. The decision module still carries
/// its copy from when this file was frozen; the allow goes away with it.
#[allow(dead_code)]
pub(crate) const MODE_422: [u32; 35] = [0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 29, 30, 31];

#[cfg(test)]
mod write_round_trip {
    use super::*;
    use crate::bitwriter::BitWriter;
    use crate::encode::Config;
    use crate::encode::gop::Kind;
    use crate::encode::h265_syntax::{Geometry as EncGeometry, NAL_IDR_N_LP, SliceHeader as EncSliceHeader, write_pps, write_slice_header, write_sps};
    use crate::hevc::pic::Geometry as PicGeometry;
    use crate::hevc::residual::write_residual;
    use crate::nal::{HevcNalHeader, unescape_rbsp};
    use std::sync::Arc;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
        fn chance(&mut self, pct: u32) -> bool {
            self.below(100) < pct
        }
    }

    /// The writer side of the round trip: what an encoder's slice writer
    /// will be, over synthetic decisions. It mirrors the reader's *walk* —
    /// the same recursion, the same inference conditions, the same fill
    /// order for the side-data arrays the context derivations and the MPM
    /// lists read — and calls the production writers for every coded bin.
    /// Its bookkeeping arrays stand in for `PicInfo`: `coded` plays the
    /// z-scan availability test's role (a 4x4 is available exactly when an
    /// earlier CU or PU wrote it — one slice, one tile, raster CTBs), and
    /// `ct_depth` / `intra_mode` mirror the arrays of the same names.
    struct CtuWriter {
        pw: i32,
        ph: i32,
        w4: usize,
        log2_ctb: u32,
        log2_min_cb: u32,
        log2_min_tb: u32,
        log2_max_tb: u32,
        max_depth_intra: u32,
        coded: Vec<bool>,
        ct_depth: Vec<u8>,
        intra_mode: Vec<u8>,
        /// Expected `PicInfo::cbf_luma` after the decode.
        exp_cbf_luma: Vec<u8>,
        /// Expected `PicInfo::filter_exempt`: 3 over every bypass CU.
        exp_exempt: Vec<u8>,
        /// The PPS's `transquant_bypass_enabled_flag`: when set, every CU
        /// writes a `cu_transquant_bypass_flag` and may carry raw residuals.
        pps_bypass: bool,
        cx: Contexts,
        rng: Lcg,
    }

    impl CtuWriter {
        fn new(sps: &Sps, qp: i32, seed: u64, pps_bypass: bool) -> Self {
            let w4 = (sps.width as usize).div_ceil(4);
            let h4 = (sps.height as usize).div_ceil(4);
            CtuWriter {
                pw: sps.width as i32,
                ph: sps.height as i32,
                w4,
                log2_ctb: sps.log2_ctb_size,
                log2_min_cb: sps.log2_min_cb_size,
                log2_min_tb: sps.log2_min_tb_size,
                log2_max_tb: sps.log2_max_tb_size,
                max_depth_intra: sps.max_th_depth_intra,
                coded: vec![false; w4 * h4],
                ct_depth: vec![0; w4 * h4],
                // PicInfo::new starts intra_mode at 1 (DC), and so must the
                // expectation.
                intra_mode: vec![1; w4 * h4],
                exp_cbf_luma: vec![0; w4 * h4],
                exp_exempt: vec![0; w4 * h4],
                pps_bypass,
                cx: Contexts::new(0, qp),
                rng: Lcg(seed),
            }
        }

        fn idx4(&self, x: i32, y: i32) -> usize {
            (y as usize >> 2) * self.w4 + (x as usize >> 2)
        }

        /// `CtDepth` of the neighbour at `(xn, yn)` if it is available in
        /// the z-scan sense — for a raster walk over one slice and one
        /// tile, exactly "inside the picture and already written".
        fn neighbour_depth(&self, xn: i32, yn: i32) -> Option<u8> {
            if xn < 0 || yn < 0 || xn >= self.pw || yn >= self.ph {
                return None;
            }
            let i = self.idx4(xn, yn);
            if self.coded[i] { Some(self.ct_depth[i]) } else { None }
        }

        fn write_ctu(&mut self, e: &mut CabacEncoder, ctb_addr_rs: usize, wc: usize) {
            let rx = ctb_addr_rs % wc;
            let ry = ctb_addr_rs / wc;
            let x0 = (rx << self.log2_ctb) as i32;
            let y0 = (ry << self.log2_ctb) as i32;
            self.quadtree(e, x0, y0, self.log2_ctb, 0);
        }

        /// The writer's `coding_quadtree`: the flag is coded only where the
        /// reader reads one, and inferred by the same conditions elsewhere.
        fn quadtree(&mut self, e: &mut CabacEncoder, x0: i32, y0: i32, log2_cb: u32, depth: u32) {
            let size = 1i32 << log2_cb;
            let split = if x0 + size <= self.pw && y0 + size <= self.ph && log2_cb > self.log2_min_cb {
                let split = self.rng.chance(if log2_cb >= 6 { 75 } else { 45 });
                let nb = SplitCuNb { left_depth: self.neighbour_depth(x0 - 1, y0), above_depth: self.neighbour_depth(x0, y0 - 1) };
                write_split_cu_flag(e, &mut self.cx, &nb, depth, split);
                split
            } else {
                log2_cb > self.log2_min_cb
            };
            if split {
                let half = size / 2;
                let x1 = x0 + half;
                let y1 = y0 + half;
                self.quadtree(e, x0, y0, log2_cb - 1, depth + 1);
                if x1 < self.pw {
                    self.quadtree(e, x1, y0, log2_cb - 1, depth + 1);
                }
                if y1 < self.ph {
                    self.quadtree(e, x0, y1, log2_cb - 1, depth + 1);
                }
                if x1 < self.pw && y1 < self.ph {
                    self.quadtree(e, x1, y1, log2_cb - 1, depth + 1);
                }
            } else {
                self.cu(e, x0, y0, log2_cb, depth);
            }
        }

        /// The three MPM candidates (8.4.2), mirroring `mpm_candidates`
        /// over the writer's own arrays: this derivation belongs to the
        /// decision side, and this is what the encoder's copy will be.
        fn mpm(&self, xp: i32, yp: i32) -> [u32; 3] {
            let cand = |xn: i32, yn: i32, is_above: bool| -> u32 {
                if xn < 0 || yn < 0 || xn >= self.pw || yn >= self.ph {
                    return 1;
                }
                let i = self.idx4(xn, yn);
                if !self.coded[i] {
                    return 1;
                }
                if is_above && yn < ((yp >> self.log2_ctb) << self.log2_ctb) {
                    return 1;
                }
                self.intra_mode[i] as u32
            };
            let a = cand(xp - 1, yp, false);
            let b = cand(xp, yp - 1, true);
            if a == b {
                if a < 2 {
                    [0, 1, 26]
                } else {
                    [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
                }
            } else {
                let c = if a != 0 && b != 0 {
                    0
                } else if a != 1 && b != 1 {
                    1
                } else {
                    26
                };
                [a, b, c]
            }
        }

        fn cu(&mut self, e: &mut CabacEncoder, x0: i32, y0: i32, log2_cb: u32, depth: u32) {
            let n = 1i32 << log2_cb;
            debug_assert!(x0 + n <= self.pw && y0 + n <= self.ph, "leaf CUs lie inside the coded picture");
            // cu_transquant_bypass_flag is the first bin of the CU when the
            // PPS enables it — before part_mode, mirroring the reader.
            let by = self.pps_bypass && self.rng.chance(65);
            if self.pps_bypass {
                write_cu_transquant_bypass_flag(e, &mut self.cx, by);
            }
            if by {
                // The decoder records bypass CUs as filter-exempt (bit 0)
                // and transquant-bypassed (bit 1).
                PicInfo::fill4(&mut self.exp_exempt, self.w4, x0 as usize, y0 as usize, n as usize, n as usize, 3u8);
            }
            // The reader records CtDepth for the whole CU before parsing
            // inside it; later siblings' split_cu_flag contexts read it.
            PicInfo::fill4(&mut self.ct_depth, self.w4, x0 as usize, y0 as usize, n as usize, n as usize, depth as u8);
            let nxn = log2_cb == self.log2_min_cb && self.rng.chance(40);
            if log2_cb == self.log2_min_cb {
                write_part_mode_intra(e, &mut self.cx, nxn);
            }
            let npu = if nxn { 4 } else { 1 };
            let pb = if nxn { n / 2 } else { n };
            // Choose a target mode per PU and derive its spelling against
            // the MPM list *at that PU's turn* — each PU's list reads the
            // modes of the PUs before it, so derivation and fill interleave
            // even though the flags are all written first.
            let mut flags = [false; 4];
            let mut payload = [0u32; 4];
            let mut modes = [0u32; 4];
            for i in 0..npu {
                let xp = x0 + (i as i32 % 2) * pb;
                let yp = y0 + (i as i32 / 2) * pb;
                let target = self.rng.below(35);
                let cands = self.mpm(xp, yp);
                if let Some(idx) = cands.iter().position(|&c| c == target) {
                    flags[i] = true;
                    payload[i] = idx as u32;
                } else {
                    let mut rem = target;
                    for &c in &cands {
                        if c < target {
                            rem -= 1;
                        }
                    }
                    flags[i] = false;
                    payload[i] = rem;
                }
                modes[i] = target;
                PicInfo::fill4(&mut self.intra_mode, self.w4, xp as usize, yp as usize, pb as usize, pb as usize, target as u8);
                PicInfo::fill4(&mut self.coded, self.w4, xp as usize, yp as usize, pb as usize, pb as usize, true);
            }
            for &f in flags.iter().take(npu) {
                write_prev_intra_luma_pred_flag(e, &mut self.cx, f);
            }
            for i in 0..npu {
                if flags[i] {
                    write_mpm_idx(e, payload[i]);
                } else {
                    write_rem_intra_luma_pred_mode(e, payload[i]);
                }
            }
            // intra_chroma_pred_mode: one per CU in 4:2:0.
            let csyn = self.rng.below(5);
            write_intra_chroma_pred_mode(e, &mut self.cx, csyn);
            let chroma_mode = intra_chroma_mode(1, modes[0], csyn);
            // rqt_root_cbf is not coded for intra CUs; the tree follows.
            let intra_split = nxn;
            let max_depth = self.max_depth_intra + intra_split as u32;
            self.tt(e, x0, y0, log2_cb, 0, 0, [true; 2], intra_split, max_depth, chroma_mode, by);
        }

        /// The writer's `transform_tree`, over the same inference
        /// conditions as the reader's.
        #[allow(clippy::too_many_arguments)]
        fn tt(&mut self, e: &mut CabacEncoder, x0: i32, y0: i32, log2: u32, depth: u32, blk_idx: u32, parent_cbf: [bool; 2], intra_split: bool, max_depth: u32, chroma_mode: u32, bypass: bool) {
            let split = if log2 <= self.log2_max_tb && log2 > self.log2_min_tb && depth < max_depth && !(intra_split && depth == 0) {
                let s = self.rng.chance(40);
                write_split_transform_flag(e, &mut self.cx, log2, s);
                s
            } else {
                log2 > self.log2_max_tb || (intra_split && depth == 0)
            };
            let mut cbf_c = [false; 2];
            if log2 > 2 {
                for c in 0..2 {
                    if depth == 0 || parent_cbf[c] {
                        cbf_c[c] = self.rng.chance(45);
                        write_cbf_chroma(e, &mut self.cx, depth, cbf_c[c]);
                    }
                }
            } else {
                // 4x4: chroma is coded at the parent's size; the flags are
                // inherited, nothing is written.
                cbf_c = parent_cbf;
            }
            if split {
                let half = 1i32 << (log2 - 1);
                self.tt(e, x0, y0, log2 - 1, depth + 1, 0, cbf_c, intra_split, max_depth, chroma_mode, bypass);
                self.tt(e, x0 + half, y0, log2 - 1, depth + 1, 1, cbf_c, intra_split, max_depth, chroma_mode, bypass);
                self.tt(e, x0, y0 + half, log2 - 1, depth + 1, 2, cbf_c, intra_split, max_depth, chroma_mode, bypass);
                self.tt(e, x0 + half, y0 + half, log2 - 1, depth + 1, 3, cbf_c, intra_split, max_depth, chroma_mode, bypass);
                return;
            }
            // A leaf: cbf_luma is always coded for intra CUs. All-zero-cbf
            // leaves are legal and must round-trip too.
            let cbf_luma = self.rng.chance(70);
            write_cbf_luma(e, &mut self.cx, depth, cbf_luma);
            if cbf_luma {
                let mode = self.intra_mode[self.idx4(x0, y0)] as u32;
                let coeffs = self.gen_coeffs(log2, bypass);
                let p = res_params(log2, 0, crate::hevc::residual::residual_scan_idx(true, log2, 0, 1, mode), bypass);
                write_residual(e, &mut self.cx, &p, &coeffs);
                let nn = 1usize << log2;
                PicInfo::fill4(&mut self.exp_cbf_luma, self.w4, x0 as usize, y0 as usize, nn, nn, 1);
            }
            // Chroma: at this TU when its blocks are at least 4x4, else once
            // at the fourth 4x4 luma block, at the parent's size — with the
            // *inherited* flags. Coordinates carry no syntax; only the size
            // and the flags do.
            let here = if log2 > 2 {
                Some(log2 - 1)
            } else if blk_idx == 3 {
                Some(2)
            } else {
                None
            };
            if let Some(log2c) = here {
                for c in 0..2usize {
                    if cbf_c[c] {
                        let p = res_params(log2c, 1 + c, crate::hevc::residual::residual_scan_idx(true, log2c, 1 + c, 1, chroma_mode), bypass);
                        let coeffs = self.gen_coeffs(log2c, bypass);
                        write_residual(e, &mut self.cx, &p, &coeffs);
                    }
                }
            }
        }

        /// Coefficient blocks shaped like the ones that matter: runs of ±1
        /// (quantised residual is mostly that), single outliers that drive
        /// the Rice escape, only-DC sub-blocks (the inference path), dense
        /// blocks, and sparse noise. Under `bypass` the levels are raw
        /// spatial residuals, so a slice of them is overwritten with
        /// full-range 8-bit values (±128..±255) — the magnitudes a real
        /// lossless block carries and quantised blocks rarely do.
        fn gen_coeffs(&mut self, log2: u32, bypass: bool) -> Vec<i16> {
            let n = 1usize << log2;
            let mut c = vec![0i16; n * n];
            loop {
                match self.rng.below(6) {
                    0 => {
                        const MAGS: [i16; 9] = [1, 2, 3, 4, 5, 9, 100, 5000, 32767];
                        let m = MAGS[self.rng.below(9) as usize];
                        let p = self.rng.below((n * n) as u32) as usize;
                        c[p] = if self.rng.chance(50) { -m } else { m };
                    }
                    1 => {
                        let len = 1 + self.rng.below((n * n) as u32) as usize;
                        for (i, v) in c.iter_mut().enumerate().take(len) {
                            *v = if i % 3 == 0 { -1 } else { 1 };
                        }
                    }
                    2 => {
                        let len = (2 + self.rng.below(14) as usize).min(n * n);
                        for v in c.iter_mut().take(len) {
                            *v = 1;
                        }
                        if self.rng.chance(50) {
                            c[0] = 900;
                        } else {
                            c[len - 1] = -900;
                        }
                    }
                    3 => {
                        for v in c.iter_mut() {
                            *v = [0, 1, -1, 2, -2, 3, -3, 4][self.rng.below(8) as usize];
                        }
                    }
                    4 => {
                        c[0] = self.rng.below(3) as i16 + 1;
                    }
                    _ => {
                        c[n * n - 1] = 1;
                        if n >= 8 {
                            for sy in 0..n / 4 {
                                for sx in 0..n / 4 {
                                    if (sx == 0 && sy == 0) || (sx == n / 4 - 1 && sy == n / 4 - 1) {
                                        continue;
                                    }
                                    if self.rng.chance(40) {
                                        c[(sy * 4) * n + sx * 4] = [1, -1, 2, 700][self.rng.below(4) as usize];
                                    }
                                }
                            }
                        }
                    }
                }
                if bypass && self.rng.chance(60) {
                    for v in c.iter_mut() {
                        if self.rng.chance(25) {
                            let m = 128 + self.rng.below(128) as i16;
                            *v = if self.rng.chance(50) { -m } else { m };
                        }
                    }
                }
                if c.iter().any(|&v| v != 0) {
                    return c;
                }
            }
        }
    }

    fn res_params(log2: u32, c_idx: usize, scan_idx: u32, bypass: bool) -> ResidualParams {
        ResidualParams {
            log2_size: log2,
            c_idx,
            scan_idx,
            bypass,
            transform_skip_allowed: false,
            sign_hiding: false,
            intra: true,
            pred_mode_intra: 0,
            ts_context: false,
            implicit_rdpcm: false,
            explicit_rdpcm: false,
            persistent_rice: false,
            trace: false,
        }
    }

    /// Write a complete IDR slice NAL — parameter sets from the encoder's
    /// own writers, header, byte alignment, then every CTU's coding
    /// quadtree — and decode it with the *production* slice decoder.
    /// Compare everything the decoder reconstructs that the writer decided:
    /// the coding-tree depths, the luma intra modes (through the MPM
    /// spelling), the luma cbfs, the QP map, and — the desync detector —
    /// the entire CABAC context state after the last CTU.
    fn round_trip(width: u32, height: u32, qp: i32, seed: u64) {
        round_trip_cfg(width, height, qp, seed, false);
    }

    fn round_trip_cfg(width: u32, height: u32, qp: i32, seed: u64, bypass: bool) {
        // The configuration under test is the one the encoder actually
        // writes: parse the written SPS/PPS with the production parsers and
        // drive both sides from the result.
        let cfg = Config { width, height, chroma: ChromaFormat::Yuv420, bit_depth: 8, ..Config::default() };
        let g = EncGeometry::new(&cfg);
        let sps = Sps::parse(&unescape_rbsp(&write_sps(&cfg, &g, 8))).expect("the encoder's SPS must parse");
        let mut pps = Pps::parse(&unescape_rbsp(&write_pps(26, bypass))).expect("the encoder's PPS must parse");
        pps.resolve_tiles(&sps).expect("one tile covering the picture");
        assert!(!pps.sign_data_hiding && !pps.transform_skip_enabled && !pps.cu_qp_delta_enabled);
        assert_eq!(pps.transquant_bypass_enabled, bypass, "the PPS must carry the bypass switch");

        // The slice NAL: two header bytes, the slice segment header, byte
        // alignment, CABAC slice data (a terminate after every CTU).
        let wc = sps.pic_width_in_ctbs() as usize;
        let hc = sps.pic_height_in_ctbs() as usize;
        let mut w = BitWriter::new();
        w.bits(8, ((NAL_IDR_N_LP as u32) & 0x3f) << 1);
        w.bits(8, 1); // nuh_layer_id 0, nuh_temporal_id_plus1 1
        let eh = EncSliceHeader { kind: Kind::Idr, poc_lsb: 0, qp: qp as u8, log2_max_poc_lsb: 8 };
        write_slice_header(&eh, 26, NAL_IDR_N_LP, &mut w);
        w.flag(true); // byte_alignment(): alignment_bit_equal_to_one
        w.align_zero();
        let mut wr = CtuWriter::new(&sps, qp, seed, bypass);
        {
            let mut e = CabacEncoder::new(&mut w);
            for ctb in 0..wc * hc {
                wr.write_ctu(&mut e, ctb, wc);
                e.encode_terminate((ctb == wc * hc - 1) as u32);
            }
        }
        w.align_zero();
        let rbsp = w.into_rbsp();

        // The production header parser reads the header back (this is what
        // caught the spurious SAO bit the header writer used to emit).
        let nal = HevcNalHeader::parse(&rbsp).expect("NAL header");
        let psc = pps.clone();
        let ssc = sps.clone();
        let (hdr, pps, sps) = SliceHeader::parse(&rbsp, nal, &|_| Some(psc.clone()), &|_| Some(ssc.clone()), None).expect("the encoder's slice header must parse");
        assert_eq!(hdr.slice_type, SliceType::I);
        assert_eq!(hdr.slice_qp, qp, "the header must carry the QP the contexts initialise from");
        assert_eq!(hdr.data_bit_offset % 8, 0);

        // The production slice decoder, assembled the way `decoder.rs`
        // assembles it for a one-slice I picture.
        let mut frame = Frame::<u8>::new(sps.width as usize, sps.height as usize, ChromaFormat::Yuv420, 8);
        let geo = Arc::new(PicGeometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let cabac = Cabac::new(&rbsp[(hdr.data_bit_offset / 8) as usize..]);
        let mut dec = SliceDec {
            sps: &sps,
            pps: &pps,
            hdr: &hdr,
            frame: &mut frame,
            info: &mut info,
            cabac,
            cx: Contexts::new(0, hdr.slice_qp),
            refs: RefCtx {
                pocs: [Vec::new(), Vec::new()],
                long_term: [Vec::new(), Vec::new()],
                col: None,
                cur_poc: 0,
                no_backward_pred: true,
                tmvp: false,
                max_merge_cand: hdr.max_num_merge_cand as usize,
                log2_par_mrg_level: pps.log2_parallel_merge_level,
                is_b: false,
                num_ref_idx: [0, 0],
                col_from_l0: true,
            },
            ref_frames: [Vec::new(), Vec::new()],
            ref_shared: [Vec::new(), Vec::new()],
            col_shared: None,
            slice_idx: 0,
            slice_addr: 0,
            scaling: None,
            qp_y: hdr.slice_qp,
            qp_y_prev: hdr.slice_qp,
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            is_cu_chroma_qp_offset_coded: false,
            cu_qp_offset_c: [0, 0],
            qg: (0, 0),
            qg_qp_prev: hdr.slice_qp,
            first_qg: true,
            last_pu_merged: false,
            ctb_addr_rs: 0,
            ctb_addr_ts: 0,
            coeffs: vec![0; 1024],
            luma_res: Vec::new(),
            luma_res_valid: false,
            dsp: HevcDsp::<u8>::SCALAR,
            mc: McScratch::new(),
            intra: IntraScratch::default(),
            warnings: 0,
            trace: TraceCfg::default(),
        };
        for ctb in 0..wc * hc {
            dec.decode_ctu(ctb, ctb).unwrap_or_else(|e| panic!("{width}x{height} qp={qp} seed={seed}: CTU {ctb} did not decode: {e}"));
            let end = dec.cabac.terminate();
            assert_eq!(end != 0, ctb == wc * hc - 1, "end_of_slice_segment_flag at CTU {ctb}");
        }
        assert!(!dec.cabac.overrun(), "the decoder ran past what the writer wrote");
        assert_eq!(dec.warnings, 0);

        // Decoded fields against the writer's decisions…
        let tag = format!("{width}x{height} qp={qp} seed={seed}");
        assert!(wr.coded.iter().all(|&c| c), "{tag}: the writer's walk must tile the picture");
        assert_eq!(dec.info.ct_depth, wr.ct_depth, "{tag}: CtDepth differs");
        assert_eq!(dec.info.intra_mode, wr.intra_mode, "{tag}: intra modes differ");
        assert_eq!(dec.info.cbf_luma, wr.exp_cbf_luma, "{tag}: cbf_luma differs");
        assert_eq!(dec.info.filter_exempt, wr.exp_exempt, "{tag}: the bypass CUs (filter-exempt map) differ");
        assert!(dec.info.pred_mode.iter().all(|&p| p == 1), "{tag}: every CU is intra");
        assert!(dec.info.qp_y.iter().all(|&q| q as i32 == qp), "{tag}: QP map differs");
        // …and the whole context state: bins can agree while states
        // diverge, and then the *next* block falls apart far from the
        // cause. This is the assertion that makes the round trip a proof.
        assert_eq!(dec.cx.c, wr.cx.c, "{tag}: CABAC context states diverged");
        assert_eq!(dec.cx.stat_coeff, wr.cx.stat_coeff, "{tag}");
    }

    /// One aligned 64x64 CTU, several seeds and QPs (the initial context
    /// states depend on the QP, so each QP is a different starting point).
    #[test]
    fn round_trips_a_single_ctu() {
        for (qp, seed) in [(26, 1u64), (26, 2), (12, 3), (39, 4), (51, 5), (0, 6)] {
            round_trip(64, 64, qp, seed);
        }
    }

    /// Several CTUs: the neighbour contexts (split_cu_flag's depth
    /// comparison, the MPM lists' above-CTB-row rule) cross CTB borders.
    #[test]
    fn round_trips_multiple_ctus() {
        for seed in 1..=4u64 {
            round_trip(128, 64, 26, seed);
        }
        for seed in 5..=8u64 {
            round_trip(96, 96, 33, seed); // 32x32 CTUs, 3x3 of them
        }
    }

    /// A picture that is not a whole number of CTUs: the boundary CTBs
    /// force splits the reader *infers* — no flag is coded — and skip the
    /// quadtree children that fall outside; the writer must walk the same
    /// shape without writing a bin for any of it.
    #[test]
    fn round_trips_partial_ctus() {
        for seed in 1..=4u64 {
            round_trip(40, 40, 26, seed);
        }
        for seed in 5..=7u64 {
            round_trip(24, 16, 45, seed); // 16x16 CTUs, right column partial
        }
        round_trip(72, 48, 18, 8); // 64x64 CTUs, both edges partial
    }

    /// Transquant bypass: the PPS enables the switch, every CU spells a
    /// `cu_transquant_bypass_flag` (about a third decline it, so both
    /// values of the flag and the mix of bypassed and quantised residual
    /// share one context state), and bypass CUs carry full-range raw
    /// residuals (±255) — the magnitudes lossless coding actually
    /// produces. The decoded filter-exempt map proves the flag's *value*
    /// arrived, not merely that a bin did.
    #[test]
    fn round_trips_transquant_bypass() {
        for (qp, seed) in [(26, 21u64), (26, 22), (12, 23), (51, 24)] {
            round_trip_cfg(64, 64, qp, seed, true);
        }
        round_trip_cfg(96, 96, 33, 25, true); // 32x32 CTUs, 3x3
        round_trip_cfg(40, 40, 26, 26, true); // partial CTBs under bypass
    }
}

/// The coding unit facts the transform tree needs.
pub struct CuCtx {
    /// CB position.
    pub x0: i32,
    /// See `x0`.
    pub y0: i32,
    /// `log2CbSize`.
    pub log2_cb: u32,
    /// Intra?
    pub intra: bool,
    /// Partition mode.
    pub part_mode: PartMode,
    /// `IntraSplitFlag`.
    pub intra_split: bool,
    /// `MaxTrafoDepth`.
    pub max_depth: u32,
    /// `IntraPredModeC` per prediction block (all equal unless 4:4:4 NxN).
    pub chroma_modes: [u32; 4],
    /// `intra_chroma_pred_mode` syntax per prediction block (4 = from luma).
    pub chroma_syntax: [u32; 4],
    /// `cu_transquant_bypass_flag`.
    pub bypass: bool,
}
