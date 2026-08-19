//! The deblocking filter (H.264 clause 8.7), run over the whole picture
//! once every slice is decoded: boundary strength per edge segment, then
//! the luma and chroma edge filters, macroblock by macroblock in raster
//! order — vertical edges first, then horizontal, luma then chroma.

//! Where the time in here goes, measured by pricing each part at zero — the
//! kernels made no-ops, the dispatch skipped with the derived thresholds fed
//! through `black_box`, and the whole stage skipped — against a same-rung
//! control reading 1.000. Two CABAC clips at the AVX2 rung, as a share of
//! whole-decode time:
//!
//! | | cabac3 | bbb_720p_cabac |
//! |---|---|---|
//! | boundary strengths and thresholds | 5.9% | 6.7% |
//! | the filter kernels | 4.4% | 4.6% |
//! | dispatch: `tc4` on the stack, indirect call | 0.7% | 0.1% |
//! | **the stage** | **11.0%** | **11.4%** |
//!
//! The reason to write that down is that the obvious optimisation is the
//! wrong one. Collapsing the six-to-eight per-macroblock calls into a single
//! "filter every internal edge" entry point targets the bottom row, which is
//! under one percent and below what this machine can measure — and it would
//! cost a `H264Dsp` signature change across four architecture files. The room
//! is in the top row: deriving boundary strengths costs more than all the
//! filtering, and more than eight times the dispatch it is usually blamed on.

use crate::dsp::h264::{H264Dsp, LumaDeblockFn, LumaDeblockIntraFn};
use crate::sample::Sample;

use super::frame::Frame;
use super::mb::PicInfo;
use super::tables::{ALPHA, BETA, TC0};

/// The per-slice deblocking parameters the filter needs at each macroblock.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeblockParams {
    /// `disable_deblocking_filter_idc`.
    pub disable_idc: u32,
    /// `FilterOffsetA`.
    pub offset_a: i32,
    /// `FilterOffsetB`.
    pub offset_b: i32,
}

/// The 16-bit "has coefficients" mask of a macroblock's 4x4 luma blocks
/// (raster), with an 8x8 transform's four blocks all set when any is
/// (kept by reconstruction in [`super::mb::MbInfo::nz_mask`]).
#[inline]
fn nz_mask(info: &PicInfo, addr: usize) -> u16 {
    info.mbs[addr].nz_mask
}

/// bS 0 or 1 from motion alone (8.7.2.1, the last three rules), for two
/// blocks of inter macroblocks with no coefficients. `mvy_limit` is 4 for
/// frame macroblocks and 2 for field macroblocks (their vertical vectors
/// are in quarter field samples; NOTE 3 of 8.7.2.1).
#[inline]
fn motion_bs<S: Sample>(
    frame: &Frame<S>,
    pa: usize,
    p_blk: usize,
    qa: usize,
    q_blk: usize,
    mvy_limit: i16,
) -> u8 {
    use super::frame::{BlockMotion, Mv};
    let p0 = &frame.motion[0][pa * 16 + p_blk];
    let p1 = &frame.motion[1][pa * 16 + p_blk];
    let q0 = &frame.motion[0][qa * 16 + q_blk];
    let q1 = &frame.motion[1][qa * 16 + q_blk];
    let (pn, qn) = (
        (p0.ref_idx >= 0) as u32 + (p1.ref_idx >= 0) as u32,
        (q0.ref_idx >= 0) as u32 + (q1.ref_idx >= 0) as u32,
    );
    if pn != qn {
        return 1;
    }
    let mv_far =
        |a: Mv, b: Mv| -> bool { (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= mvy_limit };
    #[inline(always)]
    fn same_pic(a: &BlockMotion, b: &BlockMotion) -> bool {
        a.same_ref(b)
    }
    if pn == 1 {
        let pp = if p0.ref_idx >= 0 { p0 } else { p1 };
        let qq = if q0.ref_idx >= 0 { q0 } else { q1 };
        if !same_pic(pp, qq) {
            return 1;
        }
        return mv_far(pp.mv, qq.mv) as u8;
    }
    let straight = same_pic(p0, q0) && same_pic(p1, q1);
    let crossed = same_pic(p0, q1) && same_pic(p1, q0);
    if !straight && !crossed {
        return 1;
    }
    if same_pic(p0, p1) {
        let pair_a = mv_far(p0.mv, q0.mv) || mv_far(p1.mv, q1.mv);
        let pair_b = mv_far(p0.mv, q1.mv) || mv_far(p1.mv, q0.mv);
        return (pair_a && pair_b) as u8;
    }
    if straight {
        return (mv_far(p0.mv, q0.mv) || mv_far(p1.mv, q1.mv)) as u8;
    }
    (mv_far(p0.mv, q1.mv) || mv_far(p1.mv, q0.mv)) as u8
}

#[inline(always)]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

/// The four tC0 values a bS byte can select at one indexA, indexed by the
/// strength itself (0 = leave the segment alone). One of these serves a
/// whole group of edges that share a QP average, instead of a table walk
/// per segment.
#[inline]
fn tc_lut(index_a: i32, bd_shift: u32) -> [i16; 4] {
    let t = &TC0[index_a as usize];
    [
        -1,
        (t[0] as i16) << bd_shift,
        (t[1] as i16) << bd_shift,
        (t[2] as i16) << bd_shift,
    ]
}

/// The four segments' tC0 of a packed edge (byte `k` holds segment `k`'s
/// strength, which is 0..3 whenever this is reached — bS 4 has its own
/// kernel).
#[inline(always)]
fn tc4(bs: u32, lut: &[i16; 4]) -> [i16; 4] {
    let b = bs.to_le_bytes();
    [
        lut[(b[0] & 3) as usize],
        lut[(b[1] & 3) as usize],
        lut[(b[2] & 3) as usize],
        lut[(b[3] & 3) as usize],
    ]
}

/// One packed edge through the bS 4 kernel or the bS < 4 one. Within an
/// edge the strengths are either all 4 or all below it (only a macroblock
/// edge against an intra macroblock is 4, and then for its whole length),
/// so byte 0 decides for the edge.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn filter_edge<S: Sample>(
    f: LumaDeblockFn<S>,
    f_intra: LumaDeblockIntraFn<S>,
    data: &mut [S],
    off: usize,
    stride: usize,
    bs: u32,
    alpha: i32,
    beta: i32,
    lut: &[i16; 4],
    max: i32,
) {
    if bs as u8 == 4 {
        f_intra(data, off, stride, alpha, beta, max);
    } else {
        f(data, off, stride, alpha, beta, &tc4(bs, lut), max);
    }
}

/// A four-segment "these have coefficients" mask spread to bS 2 in one
/// byte per segment.
#[rustfmt::skip]
static BS2_SPREAD: [u32; 16] = [
    0x0000_0000, 0x0000_0002, 0x0000_0200, 0x0000_0202,
    0x0002_0000, 0x0002_0002, 0x0002_0200, 0x0002_0202,
    0x0200_0000, 0x0200_0002, 0x0200_0200, 0x0200_0202,
    0x0202_0000, 0x0202_0002, 0x0202_0200, 0x0202_0202,
];

/// Bits `s`, `s + 4`, `s + 8` and `s + 12` of a 4x4 block mask gathered
/// into bits 0..3 — one column of it, which is what a vertical edge
/// reads. (A horizontal edge reads a row, already a nibble.)
#[inline(always)]
fn gather_col(x: u16, s: u32) -> u32 {
    let x = ((x >> s) as u32) & 0x1111;
    (x | (x >> 3) | (x >> 6) | (x >> 9)) & 0xF
}

/// One internal edge's strengths, packed a byte per segment. An internal
/// edge has the same macroblock on both sides, so none of the frame /
/// field geometry of a macroblock edge reaches it and both drivers derive
/// it the same way: a coefficient on either side settles a segment at
/// bS 2, and otherwise motion can differ only where a partition boundary
/// runs, which `MbInfo::part_edges` already records. `coef` is the
/// nonzero mask ORed with itself shifted to the neighbour in the edge's
/// direction, `part` the partition edges for that direction.
#[inline]
fn internal_edge<S: Sample>(
    frame: &Frame<S>,
    addr: usize,
    coef: u16,
    part: u16,
    e: usize,
    vertical: bool,
    mvy_limit: i16,
) -> u32 {
    let c = if vertical {
        gather_col(coef, e as u32 - 1)
    } else {
        (coef >> ((e - 1) * 4)) as u32 & 0xF
    };
    let mut bs = BS2_SPREAD[c as usize];
    // Motion only where a partition boundary runs and no coefficient
    // decided the segment already.
    let mut rest = ((part >> (e * 4)) as u32 & 0xF) & !c;
    while rest != 0 {
        let k = rest.trailing_zeros() as usize;
        let (pb, qb) = if vertical {
            (k * 4 + e - 1, k * 4 + e)
        } else {
            ((e - 1) * 4 + k, e * 4 + k)
        };
        bs |= (motion_bs(frame, addr, pb, addr, qb, mvy_limit) as u32) << (k * 8);
        rest &= rest - 1;
    }
    bs
}

/// The motion-derived strengths of the segments in `rest`, packed one
/// byte per segment. `chg` marks the segments (1..3) whose two sides are
/// not the same pair of partitions as the segment before them: within a
/// run without such a boundary the comparison of 8.7.2.1 reads the same
/// motion on both sides and is made once.
#[inline(always)]
fn edge_motion(rest: u32, chg: u32, mut bs_of: impl FnMut(usize) -> u8) -> u32 {
    if rest == 0 {
        return 0;
    }
    let mut packed = 0u32;
    let mut held = 0u32;
    let mut known = false;
    for k in 0..4 {
        if chg >> k & 1 != 0 {
            known = false;
        }
        if rest >> k & 1 == 0 {
            continue;
        }
        if !known {
            held = bs_of(k) as u32;
            known = true;
        }
        packed |= held << (k * 8);
    }
    packed
}

/// The thresholds one QP average yields under 8.7.2.2: alpha, beta, and
/// the four tC0 values a boundary strength can select (index 0, bS 0,
/// meaning "leave the segment alone").
#[derive(Clone, Copy)]
struct Thr {
    alpha: i32,
    beta: i32,
    lut: [i16; 4],
}

/// Every QP average a picture can present, for one slice's filter
/// offsets. 8.7.2.2 is a pure function of the average and of those
/// offsets, and a macroblock asks for up to nine of them across its
/// planes and edge groups, so they are derived once per slice and looked
/// up. QP_Y runs from -QpBdOffset_Y to 51 and so does the average, which
/// the table is biased by; a 14-bit picture, the deepest H.264 allows,
/// needs all 88 entries.
struct ThrTable {
    t: [Thr; 88],
    bias: i32,
}

impl ThrTable {
    fn new(par: &DeblockParams, bd_shift: u32) -> ThrTable {
        let bias = 6 * bd_shift as i32;
        let mut t = [Thr {
            alpha: 0,
            beta: 0,
            lut: [0; 4],
        }; 88];
        let n = (52 + bias as usize).min(88);
        for (i, e) in t.iter_mut().enumerate().take(n) {
            let qp_av = i as i32 - bias;
            let index_a = clip3(0, 51, qp_av + par.offset_a) as usize;
            let index_b = clip3(0, 51, qp_av + par.offset_b) as usize;
            *e = Thr {
                alpha: (ALPHA[index_a] as i32) << bd_shift,
                beta: (BETA[index_b] as i32) << bd_shift,
                lut: tc_lut(index_a as i32, bd_shift),
            };
        }
        ThrTable { t, bias }
    }

    /// The entry for one edge's two QPs. Only averages the table was
    /// built for are reachable, so the clamp is on the array bound.
    #[inline(always)]
    fn get(&self, qp_p: i32, qp_q: i32) -> &Thr {
        &self.t[clip3(0, 87, ((qp_p + qp_q + 1) >> 1) + self.bias) as usize]
    }
}

/// The luma edge set of one plane: the four vertical edges left to right,
/// then the four horizontal ones top to bottom, at this plane's QP. Also
/// the chroma planes of a 4:4:4 picture, whose chromaStyleFilteringFlag
/// is 0.
#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_luma_style<S: Sample>(
    dsp: &H264Dsp<S>,
    plane: &mut super::frame::PaddedPlane<S>,
    bs_v: &[u32; 4],
    bs_h: &[u32; 4],
    qp_cur: i32,
    qp_left: i32,
    qp_above: i32,
    internal_odd: bool,
    x0: usize,
    y0: usize,
    thr: &ThrTable,
    max: i32,
) {
    // The odd internal edges are no transform edges under the 8x8
    // transform, whatever strength they carry for 4:2:2 chroma.
    let step = if internal_odd { 1usize } else { 2 };
    let (iv, ih) = if internal_odd {
        (bs_v[1] | bs_v[2] | bs_v[3], bs_h[1] | bs_h[2] | bs_h[3])
    } else {
        (bs_v[2], bs_h[2])
    };
    if bs_v[0] | bs_h[0] | iv | ih == 0 {
        return;
    }
    let stride = plane.stride;
    let base = plane.offset(x0 as isize, y0 as isize);
    // Every internal edge shares the macroblock's own QP average.
    let ti = thr.get(qp_cur, qp_cur);
    let (f, fi) = (dsp.deblock_luma_v, dsp.deblock_luma_v_intra);
    if bs_v[0] != 0 {
        let t = thr.get(qp_left, qp_cur);
        filter_edge(
            f,
            fi,
            &mut plane.data,
            base,
            stride,
            bs_v[0],
            t.alpha,
            t.beta,
            &t.lut,
            max,
        );
    }
    let mut e = step;
    while e < 4 {
        if bs_v[e] != 0 {
            filter_edge(
                f,
                fi,
                &mut plane.data,
                base + e * 4,
                stride,
                bs_v[e],
                ti.alpha,
                ti.beta,
                &ti.lut,
                max,
            );
        }
        e += step;
    }
    let (f, fi) = (dsp.deblock_luma_h, dsp.deblock_luma_h_intra);
    if bs_h[0] != 0 {
        let t = thr.get(qp_above, qp_cur);
        filter_edge(
            f,
            fi,
            &mut plane.data,
            base,
            stride,
            bs_h[0],
            t.alpha,
            t.beta,
            &t.lut,
            max,
        );
    }
    let mut e = step;
    while e < 4 {
        if bs_h[e] != 0 {
            filter_edge(
                f,
                fi,
                &mut plane.data,
                base + e * 4 * stride,
                stride,
                bs_h[e],
                ti.alpha,
                ti.beta,
                &ti.lut,
                max,
            );
        }
        e += step;
    }
}

/// The chroma edge set of both components of a 4:2:0 or 4:2:2 picture:
/// vertical edges at chroma x = 0 and 4 (luma edges 0 and 2), horizontal
/// ones at chroma y = 0 and 4 for 4:2:0 (luma edges 0 and 2) and at 0, 4,
/// 8, 12 for 4:2:2 (all four luma edge rows), each taking the strengths
/// of the luma edge it lies on. Cb and Cr run together: they share the
/// edge set and the geometry, and differ only in the QP each is filtered
/// at.
#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_chroma_style<S: Sample>(
    dsp: &H264Dsp<S>,
    cb: &mut super::frame::PaddedPlane<S>,
    cr: &mut super::frame::PaddedPlane<S>,
    bs_v: &[u32; 4],
    bs_h: &[u32; 4],
    qp_cur: [i32; 2],
    qp_left: [i32; 2],
    qp_above: [i32; 2],
    c422: bool,
    xc0: usize,
    yc0: usize,
    thr: &ThrTable,
    max: i32,
) {
    let ih = if c422 {
        bs_h[1] | bs_h[2] | bs_h[3]
    } else {
        bs_h[2]
    };
    if bs_v[0] | bs_h[0] | bs_v[2] | ih == 0 {
        return;
    }
    let stride = cb.stride;
    let base = cb.offset(xc0 as isize, yc0 as isize);
    let (tib, tir) = (thr.get(qp_cur[0], qp_cur[0]), thr.get(qp_cur[1], qp_cur[1]));
    if bs_v[0] != 0 {
        let tb = thr.get(qp_left[0], qp_cur[0]);
        let tr = thr.get(qp_left[1], qp_cur[1]);
        chroma_v_edge(dsp, cb, base, stride, c422, bs_v[0], tb, max);
        chroma_v_edge(dsp, cr, base, stride, c422, bs_v[0], tr, max);
    }
    if bs_v[2] != 0 {
        chroma_v_edge(dsp, cb, base + 4, stride, c422, bs_v[2], tib, max);
        chroma_v_edge(dsp, cr, base + 4, stride, c422, bs_v[2], tir, max);
    }
    let (f, fi) = (dsp.deblock_chroma_h, dsp.deblock_chroma_h_intra);
    if bs_h[0] != 0 {
        let tb = thr.get(qp_above[0], qp_cur[0]);
        let tr = thr.get(qp_above[1], qp_cur[1]);
        filter_edge(
            f,
            fi,
            &mut cb.data,
            base,
            stride,
            bs_h[0],
            tb.alpha,
            tb.beta,
            &tb.lut,
            max,
        );
        filter_edge(
            f,
            fi,
            &mut cr.data,
            base,
            stride,
            bs_h[0],
            tr.alpha,
            tr.beta,
            &tr.lut,
            max,
        );
    }
    if ih != 0 {
        // Chroma row of the edge: 4:2:0 only e = 2 (row 4), 4:2:2 e * 4.
        let (step, scale) = if c422 { (1usize, 4) } else { (2usize, 2) };
        let mut e = step;
        while e < 4 {
            if bs_h[e] != 0 {
                let off = base + e * scale * stride;
                filter_edge(
                    f,
                    fi,
                    &mut cb.data,
                    off,
                    stride,
                    bs_h[e],
                    tib.alpha,
                    tib.beta,
                    &tib.lut,
                    max,
                );
                filter_edge(
                    f,
                    fi,
                    &mut cr.data,
                    off,
                    stride,
                    bs_h[e],
                    tir.alpha,
                    tir.beta,
                    &tir.lut,
                    max,
                );
            }
            e += step;
        }
    }
}

/// Deblock macroblock rows `r0..r1` in raster order (each row's top edges
/// reach three lines into the row above). Rows must be filtered in order,
/// and a row only after the row below it is decoded (intra prediction
/// reads unfiltered neighbours), which is how the picture-level filter
/// order is preserved when rows are filtered as decoding proceeds.
pub fn deblock_mb_rows<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    params: &[DeblockParams],
    r0: usize,
    r1: usize,
) {
    if frame.mbaff {
        // Pair by pair, in the macroblocks' own frame / field geometry.
        deblock_mbaff_pairs(dsp, frame, info, params, r0 / 2, r1.div_ceil(2));
        return;
    }
    let mbw = info.mb_width;
    let mbh = info.mb_height;
    // Thresholds scale with the bit depth (8.7.2.2): alpha, beta and tC0 are
    // multiplied by 2^(BitDepth - 8).
    let bd_shift = frame.bit_depth - 8;
    let max = (1i32 << frame.bit_depth) - 1;
    // A field picture: every macroblock is a field macroblock — intra
    // horizontal edges get bS 3 (only vertical ones 4), and vertical vector
    // differences count in field units.
    let field_pic = frame.field_coded;
    let mvy_limit: i16 = if field_pic { 2 } else { 4 };
    // Chroma geometry, fixed for the picture.
    let c422 = frame.chroma == crate::picture::ChromaFormat::Yuv422;
    let c420 = frame.chroma == crate::picture::ChromaFormat::Yuv420;
    // 4:4:4 filters its chroma planes with the luma filters and edge set
    // (chromaStyleFilteringFlag is 0 there), at the chroma QP.
    let c444 = frame.chroma == crate::picture::ChromaFormat::Yuv444;
    let mbh_c = if c422 { 16 } else { 8 };
    // An intra macroblock's edges, packed one byte per segment: 4 on its
    // macroblock edges (3 on a horizontal one in a field picture), 3 on
    // its internal ones.
    let intra_top: u32 = if field_pic { 0x0303_0303 } else { 0x0404_0404 };
    // The filter thresholds of the slice being filtered, rebuilt when
    // the macroblocks cross into another one.
    let mut thr_slice = usize::MAX;
    let mut thr = ThrTable::new(&DeblockParams::default(), bd_shift);

    for mby in r0..r1.min(mbh) {
        for mbx in 0..mbw {
            let addr = mby * mbw + mbx;
            let m = &info.mbs[addr];
            if !m.decoded {
                continue;
            }
            let par = params[m.slice as usize];
            if par.disable_idc == 1 {
                continue;
            }
            if m.slice as usize != thr_slice {
                thr = ThrTable::new(&par, bd_shift);
                thr_slice = m.slice as usize;
            }
            // Unavailable neighbours address the macroblock itself: their
            // QPs are then never read, `filter_left` / `filter_top` being
            // what decides whether the edge exists at all.
            let left = if mbx > 0 { addr - 1 } else { addr };
            let above = if mby > 0 { addr - mbw } else { addr };
            let across_slices = par.disable_idc != 2;
            let filter_left = mbx > 0
                && info.mbs[left].decoded
                && (across_slices || info.mbs[left].slice == m.slice);
            let filter_top = mby > 0
                && info.mbs[above].decoded
                && (across_slices || info.mbs[above].slice == m.slice);

            // Boundary strengths of the four vertical and the four
            // horizontal edges, each edge's four segments packed one per
            // byte: an edge is entirely bS 0 — the common case — exactly
            // when its word is zero.
            let mut bs_v = [0u32; 4];
            let mut bs_h = [0u32; 4];
            // The odd internal edges are no luma transform edges under the
            // 8x8 transform (luma skips them below), but 4:2:2 chroma still
            // filters its edges there with the strength they would have.
            let internal_odd = !m.transform_8x8;
            if m.kind.is_intra() {
                if filter_left {
                    bs_v[0] = 0x0404_0404;
                }
                if filter_top {
                    bs_h[0] = intra_top;
                }
                bs_v[1] = 0x0303_0303;
                bs_v[2] = 0x0303_0303;
                bs_v[3] = 0x0303_0303;
                bs_h[1] = 0x0303_0303;
                bs_h[2] = 0x0303_0303;
                bs_h[3] = 0x0303_0303;
            } else {
                let nz = m.nz_mask;
                let [pe_v, pe_h] = m.part_edges;
                // The odd edges are not luma transform edges under the 8x8
                // transform; only 4:2:2 chroma still needs their strengths.
                let step = if internal_odd || c422 { 1usize } else { 2 };
                // Internal edges: coefficients, else motion — which only
                // differs across a partition boundary. Nothing to do at all
                // for the common inter macroblock with one motion and no
                // coefficients (a skip): every internal edge is bS 0.
                if nz != 0 || pe_v != 0 {
                    // A block or its right neighbour has coefficients.
                    let coef_v = nz | (nz >> 1);
                    let mut e = step;
                    while e < 4 {
                        bs_v[e] = internal_edge(frame, addr, coef_v, pe_v, e, true, mvy_limit);
                        e += step;
                    }
                }
                if nz != 0 || pe_h != 0 {
                    // A block or its lower neighbour has coefficients.
                    let coef_h = nz | (nz >> 4);
                    let mut e = step;
                    while e < 4 {
                        bs_h[e] = internal_edge(frame, addr, coef_h, pe_h, e, false, mvy_limit);
                        e += step;
                    }
                }
                // A macroblock edge between two inter macroblocks is where
                // this filter still reads the picture-wide motion arrays, and
                // that read is what `motion_bs` costs — two Vecs 128 bytes
                // apart per macroblock, 0.42% of a whole decode on the line
                // that first touches them.
                //
                // Caching a single-partition macroblock's motion in `PicInfo`
                // and comparing one word instead was built and measured, and
                // is not here on purpose. It settles 52% of these edges and is
                // worth 0.958x / 0.977x of this function on cabac3 / cavlc3 —
                // real, and reproducibly separated from noise. But the filter
                // is ~9% of a decode, so that is 0.3% end to end, under the
                // floor of the machine measuring it, in exchange for an
                // invariant spanning derivation and this file. It lives on
                // `perf/deblock-motion-key`, green, one commit, with the
                // numbers and the invariant analysis in its message — worth
                // reviving only if deblocking's share of a decode grows.
                if filter_left {
                    let ml = &info.mbs[left];
                    if ml.kind.is_intra() {
                        bs_v[0] = 0x0404_0404;
                    } else {
                        let c = gather_col(ml.nz_mask, 3) | gather_col(nz, 0);
                        let rest = !c & 0xF;
                        // Segment k faces the same two partitions as segment
                        // k - 1 unless a partition boundary crosses one side
                        // there, so one comparison of 8.7.2.1 answers for a
                        // whole run of segments.
                        let chg = gather_col(ml.part_edges[1], 3) | gather_col(pe_h, 0);
                        bs_v[0] = BS2_SPREAD[c as usize]
                            | edge_motion(rest, chg, |k| {
                                motion_bs(frame, left, k * 4 + 3, addr, k * 4, mvy_limit)
                            });
                    }
                }
                if filter_top {
                    let ma = &info.mbs[above];
                    if ma.kind.is_intra() {
                        bs_h[0] = intra_top;
                    } else {
                        let c = ((ma.nz_mask >> 12) | nz) as u32 & 0xF;
                        let rest = !c & 0xF;
                        let chg = gather_col(ma.part_edges[0], 3) | gather_col(pe_v, 0);
                        bs_h[0] = BS2_SPREAD[c as usize]
                            | edge_motion(rest, chg, |k| {
                                motion_bs(frame, above, 12 + k, addr, k, mvy_limit)
                            });
                    }
                }
                // An inter macroblock whose every edge came out bS 0 has
                // nothing filtered in any plane.
                if bs_v[0] | bs_v[1] | bs_v[2] | bs_v[3] | bs_h[0] | bs_h[1] | bs_h[2] | bs_h[3]
                    == 0
                {
                    continue;
                }
            }

            let (x0, y0) = (mbx * 16, mby * 16);
            let (ml, ma) = (&info.mbs[left], &info.mbs[above]);
            filter_luma_style(
                dsp,
                &mut frame.y,
                &bs_v,
                &bs_h,
                m.qp as i32,
                ml.qp as i32,
                ma.qp as i32,
                internal_odd,
                x0,
                y0,
                &thr,
                max,
            );
            if c444 {
                for comp in 0..2 {
                    let plane = if comp == 0 {
                        &mut frame.cb
                    } else {
                        &mut frame.cr
                    };
                    filter_luma_style(
                        dsp,
                        plane,
                        &bs_v,
                        &bs_h,
                        m.qpc[comp] as i32,
                        ml.qpc[comp] as i32,
                        ma.qpc[comp] as i32,
                        internal_odd,
                        x0,
                        y0,
                        &thr,
                        max,
                    );
                }
            } else if c420 || c422 {
                filter_chroma_style(
                    dsp,
                    &mut frame.cb,
                    &mut frame.cr,
                    &bs_v,
                    &bs_h,
                    [m.qpc[0] as i32, m.qpc[1] as i32],
                    [ml.qpc[0] as i32, ml.qpc[1] as i32],
                    [ma.qpc[0] as i32, ma.qpc[1] as i32],
                    c422,
                    mbx * 8,
                    mby * mbh_c,
                    &thr,
                    max,
                );
            }
        }
    }
}

// ----------------------------------------------------------------------
// MBAFF frames (8.7 with MbaffFrameFlag = 1)
// ----------------------------------------------------------------------

/// One side of an edge: the macroblock and the 4x4 block (raster) whose
/// samples sit on that side of a given line.
#[derive(Clone, Copy)]
struct Side {
    addr: usize,
    blk: usize,
}

/// bS for one set of samples across an edge (8.7.2.1) in an MBAFF frame:
/// `mixed` is mixedModeEdgeFlag, `vertical` verticalEdgeFlag, and both
/// sides carry the macroblock and 4x4 block holding p0 / q0.
#[allow(clippy::too_many_arguments)]
fn bs_mbaff<S: Sample>(
    frame: &Frame<S>,
    info: &PicInfo,
    p: Side,
    q: Side,
    mixed: bool,
    vertical: bool,
    nz_p: u16,
    nz_q: u16,
) -> u8 {
    let mp = &info.mbs[p.addr];
    let mq = &info.mbs[q.addr];
    let intra = mp.kind.is_intra() || mq.kind.is_intra();
    if intra {
        // Macroblock edges: 4 on vertical ones, and on horizontal ones
        // between two frame macroblocks; 3 otherwise (mixed, fields) and on
        // internal edges.
        let mb_edge = p.addr != q.addr;
        return if mb_edge && (vertical || (!mixed && !mp.field && !mq.field)) {
            4
        } else {
            3
        };
    }
    if (nz_p >> p.blk) & 1 != 0 || (nz_q >> q.blk) & 1 != 0 {
        return 2;
    }
    if mixed {
        return 1;
    }
    let mvy_limit: i16 = if mq.field { 2 } else { 4 };
    motion_bs(frame, p.addr, p.blk, q.addr, q.blk, mvy_limit)
}

/// tC0 for one line of a bS < 4 edge (bS 0: no filtering, `None`).
#[inline]
fn tc0_line(bs: u8, lut: &[i16; 4]) -> Option<Option<i32>> {
    match bs {
        0 => None,
        4 => Some(None),
        _ => Some(Some(lut[(bs & 3) as usize] as i32)),
    }
}

/// Filter one line of luma or chroma samples across an edge at `pos`
/// (the q0 sample), `across` being the sample step across the edge.
#[inline]
fn filter_line<S: Sample>(
    data: &mut [S],
    pos: usize,
    across: usize,
    tc0: Option<i32>,
    alpha: i32,
    beta: i32,
    chroma: bool,
    max: i32,
) {
    let mut p = [0i32; 4];
    let mut q = [0i32; 4];
    let taps = if chroma { 2 } else { 4 };
    for k in 0..taps {
        p[k] = data[pos - (k + 1) * across].to_i32();
        q[k] = data[pos + k * across].to_i32();
    }
    crate::dsp::h264::deblock_line(&mut p, &mut q, tc0, alpha, beta, chroma, max);
    let writes = if chroma { 1 } else { 3 };
    for k in 0..writes {
        data[pos - (k + 1) * across] = S::from_i32(p[k]);
        data[pos + k * across] = S::from_i32(q[k]);
    }
}

/// Deblock the macroblock pairs of pair rows `pr0..pr1` of an MBAFF frame,
/// pair by pair (top macroblock then bottom), each macroblock's edges in
/// its own geometry — a field macroblock's on its own field lines — with
/// the mixed frame / field edges of 8.7 filtered line by line.
fn deblock_mbaff_pairs<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    params: &[DeblockParams],
    pr0: usize,
    pr1: usize,
) {
    let mbw = info.mb_width;
    let pairs_h = info.mb_height / 2;
    let bd_shift = frame.bit_depth - 8;
    let max = (1i32 << frame.bit_depth) - 1;
    let chroma420 = frame.chroma == crate::picture::ChromaFormat::Yuv420;
    let chroma422 = frame.chroma == crate::picture::ChromaFormat::Yuv422;
    // 4:2:0 and 4:2:2 chroma; 4:4:4 MBAFF (luma-style chroma) is not
    // handled here (no such streams are in circulation) — luma only.
    let do_chroma = chroma420 || chroma422;
    let mbh_c = if chroma422 { 16 } else { 8 };
    // The filter thresholds of the slice being filtered, rebuilt when the
    // macroblocks cross into another one.
    let mut cur_slice = usize::MAX;
    let mut par = DeblockParams::default();
    let mut thr = ThrTable::new(&par, bd_shift);
    for pr in pr0..pr1.min(pairs_h) {
        for mbx in 0..mbw {
            let top_addr = (2 * pr) * mbw + mbx;
            for bottom in 0..2usize {
                let sa = top_addr + bottom * mbw;
                let m = &info.mbs[sa];
                if !m.decoded {
                    continue;
                }
                if m.slice as usize != cur_slice {
                    cur_slice = m.slice as usize;
                    par = params[cur_slice];
                    thr = ThrTable::new(&par, bd_shift);
                }
                if par.disable_idc == 1 {
                    continue;
                }
                let mf = m.field;
                let dy = if mf { 2 } else { 1 };
                let across_slices = par.disable_idc != 2;
                // Neighbouring pairs (top macroblock storage addresses).
                let left_top = if mbx > 0 { Some(top_addr - 1) } else { None };
                let above_top = if pr > 0 {
                    Some(top_addr - 2 * mbw)
                } else {
                    None
                };
                let avail = |a: Option<usize>| {
                    a.is_some_and(|t| {
                        info.mbs[t].decoded && (across_slices || info.mbs[t].slice == m.slice)
                    })
                };
                let filter_left = avail(left_top);
                let filter_top = if mf {
                    avail(above_top)
                } else if bottom == 1 {
                    true
                } else {
                    avail(above_top)
                };
                // Luma geometry.
                let (x0, y_p) = (mbx * 16, pr * 32 + if mf { bottom } else { 16 * bottom });
                let ystride = frame.y.stride;
                let cstride = frame.cb.stride;
                let (xc0, yc_p) = (
                    mbx * 8,
                    pr * 2 * mbh_c + if mf { bottom } else { mbh_c * bottom },
                );
                let nz_q = nz_mask(info, sa);
                let internal_odd = !m.transform_8x8;
                // An internal edge compares the macroblock against itself,
                // so its vectors are in its own units (NOTE 3 of 8.7.2.1).
                let mvy_int: i16 = if m.field { 2 } else { 4 };

                // ---------- vertical edges ----------
                // Left MB edge.
                if filter_left {
                    let lt = left_top.unwrap();
                    let left_field = info.mbs[lt].field;
                    if left_field == mf {
                        // Same kind: one macroblock on the p side, sixteen
                        // lines in the current geometry.
                        let pa = if bottom == 1 { lt + mbw } else { lt };
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: k * 4 + 3,
                                },
                                Side {
                                    addr: sa,
                                    blk: k * 4,
                                },
                                false,
                                true,
                                nz_p,
                                nz_q,
                            );
                        }
                        let t = thr.get(info.mbs[pa].qp as i32, m.qp as i32);
                        let (alpha, beta) = (t.alpha, t.beta);
                        let off = frame.y.offset(x0 as isize, y_p as isize);
                        let any = u32::from_le_bytes(bs) != 0;
                        if any {
                            if bs[0] == 4 {
                                (dsp.deblock_luma_v_intra)(
                                    &mut frame.y.data,
                                    off,
                                    ystride * dy,
                                    alpha,
                                    beta,
                                    max,
                                );
                            } else {
                                (dsp.deblock_luma_v)(
                                    &mut frame.y.data,
                                    off,
                                    ystride * dy,
                                    alpha,
                                    beta,
                                    &tc4(u32::from_le_bytes(bs), &t.lut),
                                    max,
                                );
                            }
                        }
                        if do_chroma && any {
                            for comp in 0..2 {
                                let t = thr.get(info.mbs[pa].qpc[comp] as i32, m.qpc[comp] as i32);
                                let plane = if comp == 0 {
                                    &mut frame.cb
                                } else {
                                    &mut frame.cr
                                };
                                let base = plane.offset(xc0 as isize, yc_p as isize);
                                let stride = plane.stride * dy;
                                chroma_v_edge(
                                    dsp,
                                    plane,
                                    base,
                                    stride,
                                    chroma422,
                                    u32::from_le_bytes(bs),
                                    &t,
                                    max,
                                );
                            }
                        }
                    } else {
                        // Mixed: line by line — each line's p samples belong
                        // to one of the two left macroblocks.
                        let mut line_bs = [0u8; 16];
                        let mut line_pa = [0usize; 16];
                        for k in 0..16 {
                            // The pair line of this current line and the p
                            // macroblock / block row holding it.
                            let (pa, prow) = if !mf {
                                // Frame MB, field left pair: pair line 16*bottom+k
                                // is field line (16*bottom+k)/2 of the field MB of
                                // that parity.
                                let pl = 16 * bottom + k;
                                (if pl % 2 == 0 { lt } else { lt + mbw }, pl / 2)
                            } else {
                                // Field MB, frame left pair: pair line 2k+bottom
                                // is line (2k+bottom) % 16 of frame MB (2k+bottom)/16.
                                let pl = 2 * k + bottom;
                                (if pl < 16 { lt } else { lt + mbw }, pl % 16)
                            };
                            let nz_p = nz_mask(info, pa);
                            line_bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: (prow / 4) * 4 + 3,
                                },
                                Side {
                                    addr: sa,
                                    blk: (k / 4) * 4,
                                },
                                true,
                                true,
                                nz_p,
                                nz_q,
                            );
                            line_pa[k] = pa;
                        }
                        // The sixteen lines are two halves of eight, each
                        // belonging to one macroblock of the left pair and
                        // lying two rows apart, so each is one kernel call:
                        // one QP, one pair of thresholds, and a strength
                        // that changes every two lines (`LumaDeblock8Fn`).
                        for g in 0..2usize {
                            let (first, span) = if !mf { (g, 2) } else { (g * 8, 1) };
                            let mut bs = 0u32;
                            for j in 0..4 {
                                bs |= (line_bs[first + span * 2 * j] as u32) << (j * 8);
                            }
                            if bs == 0 {
                                continue;
                            }
                            let t = thr.get(info.mbs[line_pa[first]].qp as i32, m.qp as i32);
                            let off = frame.y.offset(x0 as isize, (y_p + dy * first) as isize);
                            filter_edge(
                                dsp.deblock_luma8_v,
                                dsp.deblock_luma8_v_intra,
                                &mut frame.y.data,
                                off,
                                ystride * 2,
                                bs,
                                t.alpha,
                                t.beta,
                                &t.lut,
                                max,
                            );
                        }
                        if do_chroma {
                            // Chroma line k takes the bS of the luma line at
                            // its position in the same field: 2k for a field
                            // MB, and for a frame MB 2k on even lines (top
                            // field) but 2k - 1 on odd ones (bottom field) —
                            // and that line's p macroblock.
                            //
                            // This is the last of the filter that still runs a
                            // line at a time, and the ceiling for giving it a
                            // kernel the way luma got one is written down here
                            // so nobody has to measure it again. Replacing
                            // this loop with nothing at all — which is the
                            // most any kernel could win — takes the MBAFF
                            // filter to 0.814x on CVMA1_TOSHIBA_B and 0.942x
                            // on CAMP_MOT_MBAFF_L30. So the whole prize is
                            // ~19% of MBAFF deblocking on a stream dense with
                            // mixed edges and ~6% on one that is not, of a
                            // filter that is itself single-digit percent of a
                            // decode, on a path only MBAFF streams reach.
                            //
                            // Two things make the achievable share much less
                            // than that, and they are why this was left alone
                            // rather than done. The halves here are four lines
                            // for 4:2:0, not eight: the chroma lines alternate
                            // between the two p macroblocks just as the luma
                            // ones do, but there are half as many of them. Four
                            // lines of two taps is sixteen bytes of work behind
                            // a transpose, where luma's eight lines of four
                            // taps mapped exactly onto `load_transposed_8x8`
                            // and cost nothing to write. And it would need its
                            // own signature and its own contract checked across
                            // the suite, for a kernel whose gain would sit near
                            // the noise floor of the machine measuring it.
                            //
                            // Worth correcting one guess that was made about
                            // this before it was measured: the call count is
                            // not half of luma's, it is equal — eight chroma
                            // lines times two components against luma's
                            // sixteen. The per-line work is smaller; the number
                            // of calls is not.
                            for k in 0..mbh_c {
                                let l = if chroma422 {
                                    k
                                } else if !mf && k % 2 == 1 {
                                    2 * k - 1
                                } else {
                                    2 * k
                                };
                                if line_bs[l] == 0 {
                                    continue;
                                }
                                let pa = line_pa[l];
                                for comp in 0..2 {
                                    let t =
                                        thr.get(info.mbs[pa].qpc[comp] as i32, m.qpc[comp] as i32);
                                    let (alpha, beta) = (t.alpha, t.beta);
                                    let Some(tc) = tc0_line(line_bs[l], &t.lut) else {
                                        continue;
                                    };
                                    let plane = if comp == 0 {
                                        &mut frame.cb
                                    } else {
                                        &mut frame.cr
                                    };
                                    let pos = plane.offset(xc0 as isize, (yc_p + dy * k) as isize);
                                    filter_line(
                                        &mut plane.data,
                                        pos,
                                        1,
                                        tc,
                                        alpha,
                                        beta,
                                        true,
                                        max,
                                    );
                                }
                            }
                        }
                    }
                }
                // Internal vertical edges.
                for e in 1..4 {
                    if e % 2 == 1 && !internal_odd {
                        // Chroma still has its edge at e == 2 only (4:2:0).
                        continue;
                    }
                    let bs = if m.kind.is_intra() {
                        0x0303_0303
                    } else {
                        let coef = nz_q | (nz_q >> 1);
                        internal_edge(frame, sa, coef, m.part_edges[0], e, true, mvy_int)
                    };
                    if bs == 0 {
                        continue;
                    }
                    let t = thr.get(m.qp as i32, m.qp as i32);
                    let (alpha, beta) = (t.alpha, t.beta);
                    let off = frame.y.offset((x0 + e * 4) as isize, y_p as isize);
                    if bs as u8 == 4 {
                        (dsp.deblock_luma_v_intra)(
                            &mut frame.y.data,
                            off,
                            ystride * dy,
                            alpha,
                            beta,
                            max,
                        );
                    } else {
                        (dsp.deblock_luma_v)(
                            &mut frame.y.data,
                            off,
                            ystride * dy,
                            alpha,
                            beta,
                            &tc4(bs, &t.lut),
                            max,
                        );
                    }
                    if do_chroma && e == 2 {
                        for comp in 0..2 {
                            let t = thr.get(m.qpc[comp] as i32, m.qpc[comp] as i32);
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            let base = plane.offset((xc0 + 4) as isize, yc_p as isize);
                            let stride = plane.stride * dy;
                            chroma_v_edge(dsp, plane, base, stride, chroma422, bs, &t, max);
                        }
                    }
                }

                // ---------- horizontal edges ----------
                // Top MB edge.
                if filter_top {
                    if !mf && bottom == 1 {
                        // Bottom frame MB: the edge with the top frame MB of
                        // the pair — two frame macroblocks.
                        let pa = sa - mbw;
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: 12 + k,
                                },
                                Side { addr: sa, blk: k },
                                false,
                                false,
                                nz_p,
                                nz_q,
                            );
                        }
                        top_edge_plain(
                            dsp, frame, info, &thr, sa, pa, &bs, x0, y_p, xc0, yc_p, 1, do_chroma,
                            max,
                        );
                    } else if !mf {
                        // Top frame MB below a pair.
                        let at = above_top.unwrap();
                        if info.mbs[at + mbw].field {
                            // Above pair is field: the edge is filtered as two
                            // field edges, the current MB's even lines against
                            // the above top field MB, its odd lines against the
                            // bottom one (fieldModeInFrameFilteringFlag = 1).
                            for pass in 0..2usize {
                                let pa = at + pass * mbw;
                                let nz_p = nz_mask(info, pa);
                                let mut bs = [0u8; 4];
                                for k in 0..4 {
                                    bs[k] = bs_mbaff(
                                        frame,
                                        info,
                                        Side {
                                            addr: pa,
                                            blk: 12 + k,
                                        },
                                        Side { addr: sa, blk: k },
                                        true,
                                        false,
                                        nz_p,
                                        nz_q,
                                    );
                                }
                                if u32::from_le_bytes(bs) == 0 {
                                    continue;
                                }
                                let t = thr.get(info.mbs[pa].qp as i32, m.qp as i32);
                                let (alpha, beta) = (t.alpha, t.beta);
                                // q0 at row y_p + pass, step 2 across the edge.
                                let off = frame.y.offset(x0 as isize, (y_p + pass) as isize);
                                if bs[0] == 4 {
                                    (dsp.deblock_luma_h_intra)(
                                        &mut frame.y.data,
                                        off,
                                        ystride * 2,
                                        alpha,
                                        beta,
                                        max,
                                    );
                                } else {
                                    (dsp.deblock_luma_h)(
                                        &mut frame.y.data,
                                        off,
                                        ystride * 2,
                                        alpha,
                                        beta,
                                        &tc4(u32::from_le_bytes(bs), &t.lut),
                                        max,
                                    );
                                }
                                if do_chroma {
                                    for comp in 0..2 {
                                        let t = thr
                                            .get(info.mbs[pa].qpc[comp] as i32, m.qpc[comp] as i32);
                                        let (alpha, beta) = (t.alpha, t.beta);
                                        let plane = if comp == 0 {
                                            &mut frame.cb
                                        } else {
                                            &mut frame.cr
                                        };
                                        let off =
                                            plane.offset(xc0 as isize, (yc_p + pass) as isize);
                                        if bs[0] == 4 {
                                            (dsp.deblock_chroma_h_intra)(
                                                &mut plane.data,
                                                off,
                                                cstride * 2,
                                                alpha,
                                                beta,
                                                max,
                                            );
                                        } else {
                                            (dsp.deblock_chroma_h)(
                                                &mut plane.data,
                                                off,
                                                cstride * 2,
                                                alpha,
                                                beta,
                                                &tc4(u32::from_le_bytes(bs), &t.lut),
                                                max,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            // Above pair is frame: a plain frame edge with its
                            // bottom macroblock.
                            let pa = at + mbw;
                            let nz_p = nz_mask(info, pa);
                            let mut bs = [0u8; 4];
                            for k in 0..4 {
                                bs[k] = bs_mbaff(
                                    frame,
                                    info,
                                    Side {
                                        addr: pa,
                                        blk: 12 + k,
                                    },
                                    Side { addr: sa, blk: k },
                                    false,
                                    false,
                                    nz_p,
                                    nz_q,
                                );
                            }
                            top_edge_plain(
                                dsp, frame, info, &thr, sa, pa, &bs, x0, y_p, xc0, yc_p, 1,
                                do_chroma, max,
                            );
                        }
                    } else {
                        // Field MB: its previous field lines are two frame
                        // rows up — in the same-parity field MB of a field
                        // pair above, or the bottom frame MB of a frame pair
                        // (a mixed edge).
                        let at = above_top.unwrap();
                        let above_field = info.mbs[at + mbw].field;
                        let pa = if above_field {
                            at + bottom * mbw
                        } else {
                            at + mbw
                        };
                        let mixed = !above_field;
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: 12 + k,
                                },
                                Side { addr: sa, blk: k },
                                mixed,
                                false,
                                nz_p,
                                nz_q,
                            );
                        }
                        top_edge_plain(
                            dsp, frame, info, &thr, sa, pa, &bs, x0, y_p, xc0, yc_p, 2, do_chroma,
                            max,
                        );
                    }
                }
                // Internal horizontal edges (luma skips the odd ones under the
                // 8x8 transform; 4:2:2 chroma has all three, 4:2:0 only e = 2).
                for e in 1..4 {
                    let luma_edge = e % 2 == 0 || internal_odd;
                    let chroma_edge = do_chroma && (e == 2 || chroma422);
                    if !luma_edge && !chroma_edge {
                        continue;
                    }
                    let bs = if m.kind.is_intra() {
                        0x0303_0303
                    } else {
                        let coef = nz_q | (nz_q >> 4);
                        internal_edge(frame, sa, coef, m.part_edges[1], e, false, mvy_int)
                    };
                    if bs == 0 {
                        continue;
                    }
                    if luma_edge {
                        let t = thr.get(m.qp as i32, m.qp as i32);
                        let (alpha, beta) = (t.alpha, t.beta);
                        let off = frame.y.offset(x0 as isize, (y_p + dy * e * 4) as isize);
                        if bs as u8 == 4 {
                            (dsp.deblock_luma_h_intra)(
                                &mut frame.y.data,
                                off,
                                ystride * dy,
                                alpha,
                                beta,
                                max,
                            );
                        } else {
                            (dsp.deblock_luma_h)(
                                &mut frame.y.data,
                                off,
                                ystride * dy,
                                alpha,
                                beta,
                                &tc4(bs, &t.lut),
                                max,
                            );
                        }
                    }
                    if chroma_edge {
                        // Chroma row of the edge: 4:2:0 only e = 2 (row 4), 4:2:2 e * 4.
                        let cy = if chroma422 { e * 4 } else { 4 };
                        for comp in 0..2 {
                            let t = thr.get(m.qpc[comp] as i32, m.qpc[comp] as i32);
                            let (alpha, beta) = (t.alpha, t.beta);
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            let off = plane.offset(xc0 as isize, (yc_p + dy * cy) as isize);
                            if bs as u8 == 4 {
                                (dsp.deblock_chroma_h_intra)(
                                    &mut plane.data,
                                    off,
                                    cstride * dy,
                                    alpha,
                                    beta,
                                    max,
                                );
                            } else {
                                (dsp.deblock_chroma_h)(
                                    &mut plane.data,
                                    off,
                                    cstride * dy,
                                    alpha,
                                    beta,
                                    &tc4(bs, &t.lut),
                                    max,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A vertical chroma edge of one macroblock at chroma column `xc`, rows
/// `yc, yc + dy, ...`: its eight lines for 4:2:0; sixteen for 4:2:2 in
/// two eight-line halves, each line taking the bS of the luma line at
/// its position.
fn chroma_v_edge<S: Sample>(
    dsp: &H264Dsp<S>,
    plane: &mut super::frame::PaddedPlane<S>,
    base: usize,
    stride: usize,
    c422: bool,
    bs: u32,
    t: &Thr,
    max: i32,
) {
    if !c422 {
        filter_edge(
            dsp.deblock_chroma_v,
            dsp.deblock_chroma_v_intra,
            &mut plane.data,
            base,
            stride,
            bs,
            t.alpha,
            t.beta,
            &t.lut,
            max,
        );
        return;
    }
    let b = bs.to_le_bytes();
    for half in 0..2 {
        // Chroma lines 8h..8h+7 sit against luma segments 2h and 2h + 1.
        let (b0, b1) = (b[2 * half], b[2 * half + 1]);
        if b0 == 0 && b1 == 0 {
            continue;
        }
        let off = base + 8 * half * stride;
        if b0 == 4 {
            (dsp.deblock_chroma_v_intra)(&mut plane.data, off, stride, t.alpha, t.beta, max);
        } else {
            let (t0, t1) = (t.lut[(b0 & 3) as usize], t.lut[(b1 & 3) as usize]);
            (dsp.deblock_chroma_v)(
                &mut plane.data,
                off,
                stride,
                t.alpha,
                t.beta,
                &[t0, t0, t1, t1],
                max,
            );
        }
    }
}

/// A top macroblock edge (luma and chroma) against one p macroblock, with
/// the current macroblock's row step `dy` (its lines above the edge being
/// the p macroblock's, `dy` rows apart).
#[allow(clippy::too_many_arguments)]
fn top_edge_plain<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    thr: &ThrTable,
    sa: usize,
    pa: usize,
    bs: &[u8; 4],
    x0: usize,
    y_p: usize,
    xc0: usize,
    yc_p: usize,
    dy: usize,
    do_chroma: bool,
    max: i32,
) {
    let bsw = u32::from_le_bytes(*bs);
    if bsw == 0 {
        return;
    }
    let m = &info.mbs[sa];
    let t = thr.get(info.mbs[pa].qp as i32, m.qp as i32);
    let (alpha, beta) = (t.alpha, t.beta);
    let ystride = frame.y.stride;
    let off = frame.y.offset(x0 as isize, y_p as isize);
    if bs[0] == 4 {
        (dsp.deblock_luma_h_intra)(&mut frame.y.data, off, ystride * dy, alpha, beta, max);
    } else {
        (dsp.deblock_luma_h)(
            &mut frame.y.data,
            off,
            ystride * dy,
            alpha,
            beta,
            &tc4(bsw, &t.lut),
            max,
        );
    }
    if do_chroma {
        let cstride = frame.cb.stride;
        for comp in 0..2 {
            let t = thr.get(info.mbs[pa].qpc[comp] as i32, m.qpc[comp] as i32);
            let (alpha, beta) = (t.alpha, t.beta);
            let plane = if comp == 0 {
                &mut frame.cb
            } else {
                &mut frame.cr
            };
            let off = plane.offset(xc0 as isize, yc_p as isize);
            if bs[0] == 4 {
                (dsp.deblock_chroma_h_intra)(&mut plane.data, off, cstride * dy, alpha, beta, max);
            } else {
                (dsp.deblock_chroma_h)(
                    &mut plane.data,
                    off,
                    cstride * dy,
                    alpha,
                    beta,
                    &tc4(bsw, &t.lut),
                    max,
                );
            }
        }
    }
}
