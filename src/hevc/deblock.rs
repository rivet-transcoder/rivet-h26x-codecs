//! Deblocking filter (H.265 8.7.2): boundary strengths from the per-4x4
//! side data, then all vertical edges of the picture followed by all
//! horizontal edges, luma and chroma (4:2:0).

use crate::dsp::hevc::HevcDsp;

use super::ctu::chroma_qp;
use super::frame::{Frame, MotionInfo, Mv, Sample};
use super::pic::{PicInfo, SliceFilterParams};
use super::pps::Pps;
use super::tables_gen::{BETA_TABLE, TC_TABLE};

/// Boundary strength from the motion of the two sides (8.7.2.4, the
/// inter-vs-inter cases).
fn motion_bs(a: &MotionInfo, b: &MotionInfo) -> u8 {
    let na = a.uses(0) as u8 + a.uses(1) as u8;
    let nb = b.uses(0) as u8 + b.uses(1) as u8;
    if na != nb {
        return 1;
    }
    let far = |p: Mv, q: Mv| -> bool { (p.x as i32 - q.x as i32).abs() >= 4 || (p.y as i32 - q.y as i32).abs() >= 4 };
    if na == 1 {
        let la = if a.uses(0) { 0 } else { 1 };
        let lb = if b.uses(0) { 0 } else { 1 };
        // Same picture: the same POC distance from this one (and long-term-ness).
        if a.ref_delta[la] != b.ref_delta[lb] || a.long_term(la) != b.long_term(lb) {
            return 1;
        }
        return far(a.mv[la], b.mv[lb]) as u8;
    }
    // Two vectors each.
    let (pa0, pa1) = ((a.ref_delta[0], a.long_term(0)), (a.ref_delta[1], a.long_term(1)));
    let (pb0, pb1) = ((b.ref_delta[0], b.long_term(0)), (b.ref_delta[1], b.long_term(1)));
    let same_set = (pa0 == pb0 && pa1 == pb1) || (pa0 == pb1 && pa1 == pb0);
    if !same_set {
        return 1;
    }
    if pa0 != pa1 {
        // Two different pictures: match by picture.
        if pa0 == pb0 {
            (far(a.mv[0], b.mv[0]) || far(a.mv[1], b.mv[1])) as u8
        } else {
            (far(a.mv[0], b.mv[1]) || far(a.mv[1], b.mv[0])) as u8
        }
    } else {
        // The same picture twice on both sides.
        let c1 = far(a.mv[0], b.mv[0]) || far(a.mv[1], b.mv[1]);
        let c2 = far(a.mv[0], b.mv[1]) || far(a.mv[1], b.mv[0]);
        (c1 && c2) as u8
    }
}

/// Compute the boundary strengths of the vertical (`ver`) and horizontal
/// (`hor`) edges at 4x4 granularity (index = 4x4 block whose left / top
/// side the edge is; only the 8x8 luma grid gets nonzero values).
fn boundary_strengths<S: Sample>(frame: &Frame<S>, info: &PicInfo, pps: &Pps, by0: usize, by1: usize, ver: &mut Vec<u8>, hor: &mut Vec<u8>) -> [u8; 2] {
    let w4 = info.w4;
    ver.clear();
    ver.resize(w4 * (by1 - by0), 0);
    hor.clear();
    hor.resize(w4 * (by1 - by0), 0);
    let ctb_mask = (1usize << info.log2_ctb) - 1;
    let wc = info.wc;
    let mut or = [0u8; 2];
    for by in by0..by1 {
        let y = by * 4;
        // Horizontal edges lie on the 8-sample grid: odd 4x4 rows have none,
        // and vertical edges only at even columns — visit only those.
        let hor_row = y > 0 && y % 8 == 0;
        let (bx_start, bx_step) = if hor_row { (0, 1) } else { (2, 2) };
        // The row stays in one CTB row, so the slice only changes when the
        // walk crosses a CTB column: look it up once per run.
        let mut last_ctb = usize::MAX;
        let mut sl = &info.slices[0];
        let mut filtering = false;
        let row = by * w4;
        let out = (by - by0) * w4;
        let mut bx = bx_start;
        while bx < w4 {
            let x = bx * 4;
            let q = row + bx;
            let edges = info.edges[q];
            let want_v = x > 0 && x % 8 == 0 && (edges & 3) != 0;
            let want_h = hor_row && (edges & 12) != 0;
            if !want_v && !want_h {
                bx += bx_step;
                continue;
            }
            let cq = info.ctb_of(x, y);
            if cq != last_ctb {
                last_ctb = cq;
                let sl_idx = info.ctb_slice[cq];
                filtering = sl_idx != u16::MAX && !info.slices[sl_idx as usize].deblocking_disabled;
                if filtering {
                    sl = &info.slices[sl_idx as usize];
                }
            }
            if !filtering {
                bx += bx_step;
                continue;
            }
            let mq = &frame.motion[q];
            let intra_q = info.pred_mode[q] == 1;
            // Vertical edge on the left side of this block.
            if want_v {
                let p = q - 1;
                let mut ok = true;
                // At a CTB boundary the neighbour is the CTB to the left.
                if x & ctb_mask == 0 {
                    let cp = cq - 1;
                    if info.ctb_tile[cq] != info.ctb_tile[cp] && !pps.loop_filter_across_tiles {
                        ok = false;
                    }
                    if info.ctb_slice_addr[cq] != info.ctb_slice_addr[cp] && !sl.loop_filter_across_slices {
                        ok = false;
                    }
                }
                if ok {
                    let b = if intra_q || info.pred_mode[p] == 1 {
                        2
                    } else if (edges & 1) != 0 && (info.cbf_luma[p] != 0 || info.cbf_luma[q] != 0) {
                        1
                    } else {
                        motion_bs(&frame.motion[p], mq)
                    };
                    ver[out + bx] = b;
                    or[0] |= b;
                }
            }
            // Horizontal edge on the top side.
            if want_h {
                let p = q - w4;
                let mut ok = true;
                // Likewise the CTB above.
                if y & ctb_mask == 0 {
                    let cp = cq - wc;
                    if info.ctb_tile[cq] != info.ctb_tile[cp] && !pps.loop_filter_across_tiles {
                        ok = false;
                    }
                    if info.ctb_slice_addr[cq] != info.ctb_slice_addr[cp] && !sl.loop_filter_across_slices {
                        ok = false;
                    }
                }
                if ok {
                    let b = if intra_q || info.pred_mode[p] == 1 {
                        2
                    } else if (edges & 4) != 0 && (info.cbf_luma[p] != 0 || info.cbf_luma[q] != 0) {
                        1
                    } else {
                        motion_bs(&frame.motion[p], mq)
                    };
                    hor[out + bx] = b;
                    or[1] |= b;
                }
            }
            bx += bx_step;
        }
    }
    or
}

/// The slice filter parameters of the CTB a luma sample belongs to,
/// remembering the last CTB looked up: a walk along an edge stays inside
/// one for a long run, so the lookup is a compare on all but the first
/// block of each run.
struct SliceCache {
    ctb: usize,
    idx: usize,
}

impl SliceCache {
    #[inline(always)]
    fn get<'a>(&mut self, info: &'a PicInfo, x: usize, y: usize) -> &'a SliceFilterParams {
        let c = info.ctb_of(x, y);
        if c != self.ctb {
            self.ctb = c;
            self.idx = info.ctb_slice[c] as usize;
        }
        &info.slices[self.idx]
    }
}

/// Reusable per-row boundary-strength buffers.
#[derive(Default)]
pub struct DeblockScratch {
    ver: Vec<u8>,
    hor: Vec<u8>,
}


/// Deblock the 4x4-block rows `by0..by1` in place: all their vertical edges,
/// then all their horizontal edges (including the top edge of row `by0`,
/// which reaches three samples up). Row-by-row application in order is
/// equivalent to the picture-level order the standard describes, because
/// the edges of one row and the next never touch the same samples.
///
/// Segments are filtered two (luma) or four (chroma) at a time along an
/// edge — eight lines per kernel call — with per-segment parameters; a
/// segment with bS 0 gets tc = beta = 0, which the kernels leave alone.
#[allow(clippy::too_many_arguments)]
pub fn deblock_rows<S: Sample>(dsp: &HevcDsp<S>, scratch: &mut DeblockScratch, frame: &mut Frame<S>, info: &PicInfo, pps: &Pps, bit_depth_luma: u32, bit_depth_chroma: u32, by0: usize, by1: usize) {
    if by0 >= by1 {
        return;
    }
    let w4 = info.w4;
    let or = boundary_strengths(frame, info, pps, by0, by1, &mut scratch.ver, &mut scratch.hor);
    let max_l = (1i32 << bit_depth_luma) - 1;
    let max_c = (1i32 << bit_depth_chroma) - 1;
    let sh_l = bit_depth_luma as i32 - 8;
    let sh_c = bit_depth_chroma as i32 - 8;
    let has_chroma = frame.chroma != crate::picture::ChromaFormat::Monochrome;
    let (sw, sh) = frame.chroma.subsampling();
    let (sw, sh) = (sw as usize, sh as usize);
    let cat = if has_chroma { if sw == 2 && sh == 2 { 1 } else if sw == 2 { 2 } else { 3 } } else { 0 };

    // Luma parameters of one 4x4 edge segment (bS > 0), given its slice.
    let luma_params = |sl: &SliceFilterParams, b: u8, p: usize, q: usize| -> (i32, i32, bool, bool) {
        let qp = (info.qp_y[p] as i32 + info.qp_y[q] as i32 + 1) >> 1;
        let beta = BETA_TABLE[(qp + sl.beta_offset).clamp(0, 51) as usize] as i32 * (1 << sh_l);
        let tc = TC_TABLE[(qp + 2 * (b as i32 - 1) + sl.tc_offset).clamp(0, 53) as usize] as i32 * (1 << sh_l);
        (beta, tc, info.filter_exempt[p] & 1 != 0, info.filter_exempt[q] & 1 != 0)
    };
    // Chroma tc of one segment (bS == 2) for both components: they share the
    // averaged luma QP and differ only in the PPS offset.
    let chroma_tc = |sl: &SliceFilterParams, p: usize, q: usize| -> [i32; 2] {
        let qp_avg = (info.qp_y[p] as i32 + info.qp_y[q] as i32 + 1) >> 1;
        let mut out = [0i32; 2];
        for (c, o) in out.iter_mut().enumerate() {
            let qpi = qp_avg + if c == 0 { sl.cb_qp_offset } else { sl.cr_qp_offset };
            let qpc = if qpi < 0 { qpi } else { chroma_qp(cat, qpi) };
            *o = TC_TABLE[(qpc + 2 + sl.tc_offset).clamp(0, 53) as usize] as i32 * (1 << sh_c);
        }
        out
    };
    let mut slices = SliceCache { ctb: usize::MAX, idx: 0 };

    for pass in 0..2 {
        // Nothing to filter on this pass: no edge of these rows has a
        // boundary strength at all.
        if or[pass] == 0 {
            continue;
        }
        let bs = if pass == 0 { &scratch.ver } else { &scratch.hor };
        // Luma: pairs of segments along the edge (two rows for a vertical
        // edge, two columns for a horizontal one).
        {
            let stride = frame.y.stride;
            if pass == 0 {
                // Vertical edges sit on the 8-sample luma grid: only even 4x4
                // columns, never the first.
                let mut by = by0;
                while by < by1 {
                    let two = by + 1 < by1;
                    let r0 = (by - by0) * w4;
                    let mut bx = 2;
                    while bx < w4 {
                        let b0 = bs[r0 + bx];
                        let b1 = if two { bs[r0 + w4 + bx] } else { 0 };
                        if b0 | b1 == 0 {
                            bx += 2;
                            continue;
                        }
                        let (x, y) = (bx * 4, by * 4);
                        let mut beta = [0i32; 2];
                        let mut tc = [0i32; 2];
                        let mut np = [false; 2];
                        let mut nq = [false; 2];
                        for (seg, b) in [(0usize, b0), (1, b1)] {
                            if b != 0 {
                                let q = (by + seg) * w4 + bx;
                                let sl = slices.get(info, x, y + 4 * seg);
                                let (bt, t, p_, q_) = luma_params(sl, b, q - 1, q);
                                beta[seg] = bt;
                                tc[seg] = t;
                                np[seg] = p_;
                                nq[seg] = q_;
                            }
                        }
                        let pos = frame.y.offset(x as isize, y as isize);
                        (dsp.deblock_luma_v)(&mut frame.y.data, pos, stride, beta, tc, np, nq, max_l);
                        bx += 2;
                    }
                    by += 2;
                }
            } else {
                // Horizontal edges likewise: only even 4x4 rows, never the
                // first two (y = 0 and y = 4 carry none).
                let mut by = by0.max(2);
                by += by & 1;
                while by < by1 {
                    let r = (by - by0) * w4;
                    let mut bx = 0;
                    while bx < w4 {
                        let b0 = bs[r + bx];
                        let b1 = if bx + 1 < w4 { bs[r + bx + 1] } else { 0 };
                        if b0 | b1 == 0 {
                            bx += 2;
                            continue;
                        }
                        let (x, y) = (bx * 4, by * 4);
                        let mut beta = [0i32; 2];
                        let mut tc = [0i32; 2];
                        let mut np = [false; 2];
                        let mut nq = [false; 2];
                        for (seg, b) in [(0usize, b0), (1, b1)] {
                            if b != 0 {
                                let q = by * w4 + bx + seg;
                                let sl = slices.get(info, x + 4 * seg, y);
                                let (bt, t, p_, q_) = luma_params(sl, b, q - w4, q);
                                beta[seg] = bt;
                                tc[seg] = t;
                                np[seg] = p_;
                                nq[seg] = q_;
                            }
                        }
                        let pos = frame.y.offset(x as isize, y as isize);
                        (dsp.deblock_luma_h)(&mut frame.y.data, pos, stride, beta, tc, np, nq, max_l);
                        bx += 2;
                    }
                    by += 2;
                }
            }
        }
        // Chroma: bS == 2 edges on the 8x8 chroma-sample grid. A kernel call
        // covers eight chroma lines as four 2-line segments; a luma 4x4
        // segment is 4 / SubHeightC (vertical edges) or 4 / SubWidthC
        // (horizontal) chroma lines, so it feeds one kernel segment in 4:2:0
        // and two (same parameters) where chroma is not subsampled that way.
        if has_chroma && or[pass] & 2 != 0 {
            let stride = frame.cb.stride;
            if pass == 0 {
                // Vertical edges: chroma x on the 8-grid = luma x on the 8·SubWidthC grid.
                let per_seg = 2 / sh; // kernel segments per luma segment
                let luma_segs = 4 / per_seg; // luma segments per call
                let mut by = by0;
                while by < by1 {
                    let cnt = (by1 - by).min(luma_segs);
                    for bx in (0..w4).step_by(2 * sw) {
                        let mut any = false;
                        let mut tcs = [[0i32; 4]; 2];
                        let mut np = [false; 4];
                        let mut nq = [false; 4];
                        for seg in 0..cnt {
                            let q = (by + seg) * w4 + bx;
                            if bs[q - by0 * w4] != 2 {
                                continue;
                            }
                            any = true;
                            let (x, y) = (bx * 4, (by + seg) * 4);
                            let t = chroma_tc(slices.get(info, x, y), q - 1, q);
                            for k in 0..per_seg {
                                let ks = seg * per_seg + k;
                                tcs[0][ks] = t[0];
                                tcs[1][ks] = t[1];
                                np[ks] = info.filter_exempt[q - 1] & 1 != 0;
                                nq[ks] = info.filter_exempt[q] & 1 != 0;
                            }
                        }
                        if !any {
                            continue;
                        }
                        let (x, y) = (bx * 4, by * 4);
                        for (c, plane) in [(0usize, &mut frame.cb), (1, &mut frame.cr)] {
                            let pos = plane.offset((x / sw) as isize, (y / sh) as isize);
                            (dsp.deblock_chroma_v)(&mut plane.data, pos, stride, tcs[c], np, nq, max_c);
                        }
                    }
                    by += luma_segs;
                }
            } else {
                // Horizontal edges: chroma y on the 8-grid = luma y on the 8·SubHeightC grid.
                let per_seg = 2 / sw;
                let luma_segs = 4 / per_seg;
                for by in by0..by1 {
                    if (by * 4) % (8 * sh) != 0 {
                        continue;
                    }
                    let mut bx = 0;
                    while bx < w4 {
                        let cnt = (w4 - bx).min(luma_segs);
                        let mut any = false;
                        let mut tcs = [[0i32; 4]; 2];
                        let mut np = [false; 4];
                        let mut nq = [false; 4];
                        for seg in 0..cnt {
                            let q = by * w4 + bx + seg;
                            if bs[q - by0 * w4] != 2 {
                                continue;
                            }
                            any = true;
                            let (x, y) = ((bx + seg) * 4, by * 4);
                            let t = chroma_tc(slices.get(info, x, y), q - w4, q);
                            for k in 0..per_seg {
                                let ks = seg * per_seg + k;
                                tcs[0][ks] = t[0];
                                tcs[1][ks] = t[1];
                                np[ks] = info.filter_exempt[q - w4] & 1 != 0;
                                nq[ks] = info.filter_exempt[q] & 1 != 0;
                            }
                        }
                        if any {
                            let (x, y) = (bx * 4, by * 4);
                            for (c, plane) in [(0usize, &mut frame.cb), (1, &mut frame.cr)] {
                                let pos = plane.offset((x / sw) as isize, (y / sh) as isize);
                                (dsp.deblock_chroma_h)(&mut plane.data, pos, stride, tcs[c], np, nq, max_c);
                            }
                        }
                        bx += luma_segs;
                    }
                }
            }
        }
    }
}
