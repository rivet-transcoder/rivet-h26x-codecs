//! Residual coding (H.265 7.3.8.11 / 9.3.4.2.3–7): parsing the transform
//! coefficient levels of one transform block, then scaling (8.6.3); the
//! inverse transforms live in [`crate::dsp::hevc`]. The *writer* for the
//! same syntax ([`write_residual`]) lives here too, beside the parser it
//! inverts, and the two share every scan table and context derivation —
//! an inverse pair drifts when its halves are edited apart.

use crate::cabac::Cabac;
use crate::cabac_enc::CabacEncoder;
use crate::{Error, Result};

use super::ctx::*;
use super::tables::{DIAG_SCAN4X4_X, DIAG_SCAN4X4_Y, DIAG_SCAN8X8_X, DIAG_SCAN8X8_Y};

/// The scan of positions inside a 4x4 sub-block, and of sub-blocks inside
/// the transform block, for `scan_idx` (0 diagonal, 1 horizontal, 2 vertical).
/// Returns `(x, y)` for scan position `i` in a `size x size` grid.
///
/// `pub(crate)` for the encoder's rate-distortion quantisation, which has
/// to know which coefficient is *last in scan order* before it can ask
/// what dropping it would save. That is the one thing RDOQ cannot get
/// from the writer by calling it, and the alternative — a second copy of
/// these tables on the encode side — is the drift this crate keeps
/// deleting. Visibility only; the derivation is untouched.
#[inline]
pub(crate) fn scan_pos(scan_idx: u32, log2_size: u32, i: usize) -> (usize, usize) {
    match (scan_idx, log2_size) {
        (0, 0) => (0, 0),
        (0, 1) => {
            const X: [usize; 4] = [0, 0, 1, 1];
            const Y: [usize; 4] = [0, 1, 0, 1];
            (X[i], Y[i])
        }
        (0, 2) => (DIAG_SCAN4X4_X[i] as usize, DIAG_SCAN4X4_Y[i] as usize),
        (0, 3) => (DIAG_SCAN8X8_X[i] as usize, DIAG_SCAN8X8_Y[i] as usize),
        (1, l) => {
            let n = 1usize << l;
            (i % n, i / n)
        }
        (2, l) => {
            let n = 1usize << l;
            (i / n, i % n)
        }
        _ => (0, 0),
    }
}

/// `ctxIdxMap` for 4x4 significance (Table 9-50).
const CTX_IDX_MAP_4X4: [u8; 15] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8];

/// Per scan: the `(xP, yP)` of the sixteen positions of a 4x4 sub-block, and
/// the position-dependent part of `sigCtx` (9.3.4.2.5) for each `prevCsbf`.
struct SubBlockTables {
    /// `[scan_idx][n] = (xP, yP)`.
    pos: [[(u8, u8); 16]; 3],
    /// `[scan_idx][prev_csbf][n]` = 0, 1 or 2.
    sig: [[[u8; 16]; 4]; 3],
    /// Inverse 4x4 scan: `[scan_idx][yP * 4 + xP]` = scan position.
    inv4: [[u8; 16]; 3],
    /// `ctxIdxMap` of a 4x4 block by scan position: `[scan_idx][n]`. Scan
    /// position 15 is always the last significant coefficient of the block,
    /// so it has no entry and is never asked for.
    ctx4: [[u8; 16]; 3],
    /// Inverse sub-block scan: `[scan_idx][log2_sb][yS * w + xS]` = index.
    inv_sb: [[[u8; 64]; 4]; 3],
}

fn sub_block_tables() -> &'static SubBlockTables {
    static T: std::sync::OnceLock<SubBlockTables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = SubBlockTables { pos: [[(0, 0); 16]; 3], sig: [[[0; 16]; 4]; 3], inv4: [[0; 16]; 3], ctx4: [[0; 16]; 3], inv_sb: [[[0; 64]; 4]; 3] };
        for scan in 0..3 {
            for log2_sb in 0..4 {
                let w = 1usize << log2_sb;
                for i in 0..w * w {
                    let (x, y) = scan_pos(scan as u32, log2_sb as u32, i);
                    t.inv_sb[scan][log2_sb][y * w + x] = i as u8;
                }
            }
            for n in 0..16 {
                let (xp, yp) = scan_pos(scan as u32, 2, n);
                t.inv4[scan][yp * 4 + xp] = n as u8;
                if n < 15 {
                    t.ctx4[scan][n] = CTX_IDX_MAP_4X4[yp * 4 + xp];
                }
            }
            for n in 0..16 {
                let (xp, yp) = scan_pos(scan as u32, 2, n);
                t.pos[scan][n] = (xp as u8, yp as u8);
                for prev in 0..4 {
                    t.sig[scan][prev][n] = match prev {
                        0 => {
                            if xp + yp == 0 {
                                2
                            } else if xp + yp < 3 {
                                1
                            } else {
                                0
                            }
                        }
                        1 => {
                            if yp == 0 {
                                2
                            } else if yp == 1 {
                                1
                            } else {
                                0
                            }
                        }
                        2 => {
                            if xp == 0 {
                                2
                            } else if xp == 1 {
                                1
                            } else {
                                0
                            }
                        }
                        _ => 2,
                    };
                }
            }
        }
        t
    })
}

/// The context assignment of `last_sig_coeff_{x,y}_prefix` (9.3.4.2.3) as
/// `(ctxOffset, ctxShift)`: bin `binIdx` of either prefix uses context
/// `OFFSET + ctxOffset + (binIdx >> ctxShift)` of its element's range.
///
/// One copy of the formula, read by [`parse_residual`] and
/// [`write_residual`] both, so the two directions cannot disagree about it.
#[inline]
fn last_sig_ctx(c_idx: usize, log2: u32) -> (u32, u32) {
    if c_idx == 0 {
        (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2)
    } else {
        (15, log2 - 2)
    }
}

/// The `sig_coeff_flag` context index for every scan position of one 4x4
/// sub-block (9.3.4.2.5), written into `ctx_of[..fill]` — absolute indices
/// into [`Contexts::c`]. `prev_csbf` is the neighbour pattern (bit 0 =
/// right sub-block coded, bit 1 = below), `(xs, ys)` the sub-block's
/// position in sub-block units, `ts_sig_ctx` the single-context override
/// for transform-skipped / bypassed blocks under
/// `transform_skip_context_enabled_flag`.
///
/// Shared between [`parse_residual`] and [`write_residual`]: the
/// significance context derivation is the most intricate table in residual
/// coding, and the conformance suites pin the parser's use of it — sharing
/// it means the writer inherits that proof rather than re-deriving the
/// tables and drifting.
#[allow(clippy::too_many_arguments)]
#[inline]
fn sig_ctx_of(
    tabs: &SubBlockTables,
    scan_idx: u32,
    log2: u32,
    c_idx: usize,
    xs: usize,
    ys: usize,
    prev_csbf: usize,
    ts_sig_ctx: Option<usize>,
    fill: usize,
    ctx_of: &mut [u16; 16],
) {
    let sig_tab = &tabs.sig[scan_idx as usize][prev_csbf];
    // The sub-block-level part of sigCtx (everything but the position).
    let sig_base: u32 = if log2 == 2 {
        0
    } else if c_idx == 0 {
        (if xs + ys > 0 { 3 } else { 0 }) + if log2 == 3 { if scan_idx == 0 { 9 } else { 15 } } else { 21 }
    } else if log2 == 3 {
        9
    } else {
        12
    };
    let sig_ctx_off = SIGNIFICANT_COEFF_FLAG_OFFSET + if c_idx == 0 { 0 } else { 27 };
    if let Some(c) = ts_sig_ctx {
        // Transform-skipped / bypassed: one context for every position.
        ctx_of[..fill].fill((sig_ctx_off + c - if c_idx == 0 { 0 } else { 27 }) as u16);
    } else if log2 == 2 {
        for (t, &m) in ctx_of[..fill].iter_mut().zip(&tabs.ctx4[scan_idx as usize]) {
            *t = (sig_ctx_off + m as usize) as u16;
        }
    } else {
        for (t, &s) in ctx_of[..fill].iter_mut().zip(sig_tab.iter()) {
            *t = (sig_ctx_off + sig_base as usize + s as usize) as u16;
        }
        if xs + ys == 0 && fill > 0 {
            // (xC, yC) == (0, 0): the DC of the block.
            ctx_of[0] = sig_ctx_off as u16;
        }
    }
}

/// `scanIdx` for a transform block (7.4.9.11): mode-dependent for small
/// intra blocks — near-horizontal modes (6..=14) scan vertically,
/// near-vertical modes (22..=30) horizontally, everything else diagonally.
/// `pred_mode` is the intra prediction mode of the block's own component
/// (the chroma mode for a chroma block).
///
/// This is decision-side configuration for [`ResidualParams::scan_idx`]:
/// the parser derives it from the reconstructed modes, and an encoder must
/// derive it from the modes it chose, through this one copy.
pub(crate) fn residual_scan_idx(intra: bool, log2: u32, c_idx: usize, chroma_array_type: u32, pred_mode: u32) -> u32 {
    if intra && (log2 == 2 || (log2 == 3 && (c_idx == 0 || chroma_array_type == 3))) {
        if (6..=14).contains(&pred_mode) {
            2
        } else if (22..=30).contains(&pred_mode) {
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// A coefficient / residual word. `i16` is the 8–12-bit pipeline: the
/// standard clips coefficients to 16 bits there and the transforms' output
/// fits. The range extensions widen that — `extended_precision_processing_flag`
/// lifts the coefficient range to `Max(15, BitDepth + 6)` bits, and above
/// 12 bits the residual itself outgrows `i16` — so the same parser fills
/// `i32` for them, and only them; every kernel of the 8–12-bit path keeps
/// its `i16`.
pub(crate) trait Coeff: Copy + Default + PartialEq + std::fmt::Debug + 'static {
    /// A parsed level, clipped to the coefficient range `CoeffMin..=CoeffMax`
    /// of `1 << log2_range` (15 for `i16`, which ignores the argument).
    fn from_level(v: i32, log2_range: u32) -> Self;
    /// Modular add (residual DPCM accumulates without clipping).
    fn wrapping_add(self, o: Self) -> Self;
}

impl Coeff for i16 {
    #[inline(always)]
    fn from_level(v: i32, _log2_range: u32) -> Self {
        v.clamp(-32768, 32767) as i16
    }
    #[inline(always)]
    fn wrapping_add(self, o: Self) -> Self {
        i16::wrapping_add(self, o)
    }
}

impl Coeff for i32 {
    #[inline(always)]
    fn from_level(v: i32, log2_range: u32) -> Self {
        let m = 1i32 << log2_range;
        v.clamp(-m, m - 1)
    }
    #[inline(always)]
    fn wrapping_add(self, o: Self) -> Self {
        i32::wrapping_add(self, o)
    }
}

/// What the transform block needs from its surroundings.
pub struct ResidualParams {
    /// `log2TrafoSize`.
    pub log2_size: u32,
    /// Colour component 0..=2.
    pub c_idx: usize,
    /// `scanIdx` (7.4.9.11).
    pub scan_idx: u32,
    /// `cu_transquant_bypass_flag`.
    pub bypass: bool,
    /// `transform_skip_enabled_flag` (and the size allows it: log2 <= 2).
    pub transform_skip_allowed: bool,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_hiding: bool,
    /// The CU is intra (`CuPredMode == MODE_INTRA`).
    pub intra: bool,
    /// The intra prediction mode of the block's component (for implicit RDPCM).
    pub pred_mode_intra: u32,
    /// `transform_skip_context_enabled_flag`.
    pub ts_context: bool,
    /// `implicit_rdpcm_enabled_flag`.
    pub implicit_rdpcm: bool,
    /// `explicit_rdpcm_enabled_flag`.
    pub explicit_rdpcm: bool,
    /// `persistent_rice_adaptation_enabled_flag`.
    pub persistent_rice: bool,
    /// Print the parse (debugging).
    pub trace: bool,
}

/// The range-extension knobs of `residual_coding()` that change how far
/// the bins reach rather than which bins exist — kept apart from
/// [`ResidualParams`] because the encoder spells that struct and never
/// these (its writer is 8–12-bit, plain precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualRange {
    /// `log2TransformRange` of the component: 15, or `Max(15, BitDepth + 6)`
    /// under `extended_precision_processing_flag` (7.4.3.2.2). Levels are
    /// clipped to it; with `extended_precision` it also sizes the escape.
    pub log2_range: u32,
    /// `extended_precision_processing_flag`: `coeff_abs_level_remaining`
    /// uses the limited-prefix escape (9.3.3.12).
    pub extended_precision: bool,
}

impl ResidualRange {
    /// Version 1 and the 8–12-bit range extensions: 16-bit coefficients,
    /// the plain escape.
    pub const PLAIN: ResidualRange = ResidualRange { log2_range: 15, extended_precision: false };
}

/// What [`parse_residual`] found.
#[derive(Debug, Clone, Copy)]
pub struct ResidualInfo {
    /// `transform_skip_flag`.
    pub transform_skip: bool,
    /// Largest column with a nonzero coefficient.
    pub max_x: usize,
    /// Largest row with a nonzero coefficient.
    pub max_y: usize,
    /// Residual DPCM to apply (range extension): `Some(vertical)`.
    pub rdpcm: Option<bool>,
}

/// Parse `residual_coding()` for one transform block into `coeffs`
/// (raster order, `1 << (2 * log2_size)` entries used, all set — zeros
/// included).
///
/// `ALIGN` is `cabac_bypass_alignment_enabled_flag`: the arithmetic
/// decoder is aligned before a sub-block's sign / remaining-level bypass
/// bins when any remaining level is coded (7.3.8.11 `escapeDataPresent`,
/// 9.3.4.3.6). A const so the instantiation every 8–12-bit stream runs is
/// the one without it, bin for bin; the flag is an SPS constant.
pub(crate) fn parse_residual<C: Coeff, const ALIGN: bool>(cabac: &mut Cabac, cx: &mut Contexts, p: &ResidualParams, range: &ResidualRange, coeffs: &mut [C]) -> Result<ResidualInfo> {
    let log2 = p.log2_size;
    let n = 1usize << log2;
    // Every coefficient is written here, zeros included, because the inverse
    // transform runs in place and left the whole block dirty. A memset call
    // for the 32 bytes of a 4x4 block costs several times the stores, so give
    // the small sizes a constant length to clear inline.
    match log2 {
        2 => coeffs[..16].fill(C::default()),
        3 => coeffs[..64].fill(C::default()),
        4 => coeffs[..256].fill(C::default()),
        _ => coeffs[..n * n].fill(C::default()),
    }
    let c_idx = p.c_idx;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    let mut transform_skip = false;
    if p.transform_skip_allowed && !p.bypass {
        transform_skip = cabac.decision(&mut cx.c[TRANSFORM_SKIP_FLAG_OFFSET + (c_idx > 0) as usize]) != 0;
    }
    let ts_or_bypass = transform_skip || p.bypass;
    // Residual DPCM: explicit for inter blocks, implicit for intra blocks
    // predicted purely horizontally / vertically (range extension).
    let mut rdpcm = None;
    if !p.intra && p.explicit_rdpcm && ts_or_bypass {
        let cinc = (c_idx > 0) as usize;
        if cabac.decision(&mut cx.c[EXPLICIT_RDPCM_FLAG_OFFSET + cinc]) != 0 {
            rdpcm = Some(cabac.decision(&mut cx.c[EXPLICIT_RDPCM_DIR_FLAG_OFFSET + cinc]) != 0);
        }
    }
    let implicit = p.intra && p.implicit_rdpcm && ts_or_bypass && (p.pred_mode_intra == 10 || p.pred_mode_intra == 26);
    if implicit {
        rdpcm = Some(p.pred_mode_intra == 26);
    }
    // One significance context for transform-skipped / bypassed blocks.
    let ts_sig_ctx: Option<usize> = if p.ts_context && ts_or_bypass { Some(if c_idx == 0 { 42 } else { 27 + 16 }) } else { None };
    // Persistent Rice adaptation: the sub-block type's running statistic.
    let sb_type = (c_idx == 0) as usize * 2 + ts_or_bypass as usize;

    // last_sig_coeff_{x,y}_prefix / suffix.
    let (ctx_offset, ctx_shift) = last_sig_ctx(c_idx, log2);
    let c_max = (log2 << 1) - 1;
    let mut last_x_prefix = 0u32;
    while last_x_prefix < c_max
        && cabac.decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (ctx_offset + (last_x_prefix >> ctx_shift)) as usize]) != 0
    {
        last_x_prefix += 1;
    }
    let mut last_y_prefix = 0u32;
    while last_y_prefix < c_max
        && cabac.decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (ctx_offset + (last_y_prefix >> ctx_shift)) as usize]) != 0
    {
        last_y_prefix += 1;
    }
    let mut last_x = last_x_prefix;
    if last_x_prefix > 3 {
        let nb = (last_x_prefix >> 1) - 1;
        let suffix = cabac.bypass_bits(nb);
        last_x = (1 << nb) * (2 + (last_x_prefix & 1)) + suffix;
    }
    let mut last_y = last_y_prefix;
    if last_y_prefix > 3 {
        let nb = (last_y_prefix >> 1) - 1;
        let suffix = cabac.bypass_bits(nb);
        last_y = (1 << nb) * (2 + (last_y_prefix & 1)) + suffix;
    }
    if p.scan_idx == 2 {
        std::mem::swap(&mut last_x, &mut last_y);
    }
    if p.trace {
        eprintln!("  residual: log2={log2} c={c_idx} scan={} ts={transform_skip} last=({last_x},{last_y}) cabac_pos={}", p.scan_idx, cabac.position());
    }
    if last_x as usize >= n || last_y as usize >= n {
        return Err(Error::bitstream("last significant coefficient outside the block"));
    }

    // Locate the last sub-block and position within it: the inverse scans.
    let log2_sb = log2 - 2; // sub-block grid is (1 << log2_sb) squared
    let tabs = sub_block_tables();
    let last_sub_block = tabs.inv_sb[p.scan_idx as usize][log2_sb as usize][((last_y as usize >> 2) << log2_sb) + (last_x as usize >> 2)] as usize;
    let last_scan_pos = tabs.inv4[p.scan_idx as usize][((last_y as usize & 3) << 2) + (last_x as usize & 3)] as usize;

    // coded_sub_block_flag storage: (1 << log2_sb) squared, raster.
    let sb_w = 1usize << log2_sb;
    let mut csbf = [0u8; 64];
    let mut greater1_ctx_state: u32 = 1; // greater1Ctx carried across sub-blocks
    let mut first_sb_processed = true;

    for i in (0..=last_sub_block).rev() {
        let (xs, ys) = scan_pos(p.scan_idx, log2_sb, i);
        let mut infer_sb_dc_sig = false;
        let coded: bool;
        if i < last_sub_block && i > 0 {
            let mut csbf_ctx = 0u32;
            if xs < sb_w - 1 {
                csbf_ctx += csbf[ys * sb_w + xs + 1] as u32;
            }
            if ys < sb_w - 1 {
                csbf_ctx += csbf[(ys + 1) * sb_w + xs] as u32;
            }
            let inc = if c_idx == 0 { csbf_ctx.min(1) } else { 2 + csbf_ctx.min(1) };
            coded = cabac.decision(&mut cx.c[SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc as usize]) != 0;
            infer_sb_dc_sig = true;
        } else {
            coded = true; // the first and last sub-blocks are inferred coded
        }
        csbf[ys * sb_w + xs] = coded as u8;

        // Significance flags of this sub-block, in reverse scan; the
        // significant positions are kept as a list (reverse scan order).
        let mut sig_pos = [0u8; 16];
        let mut n_sig = 0usize;
        let start_n = if i == last_sub_block { last_scan_pos as i32 - 1 } else { 15 };
        if i == last_sub_block {
            sig_pos[0] = last_scan_pos as u8;
            n_sig = 1;
        }
        // prevCsbf for sigCtx.
        let mut prev_csbf = 0usize;
        if xs < sb_w - 1 {
            prev_csbf += csbf[ys * sb_w + xs + 1] as usize;
        }
        if ys < sb_w - 1 {
            prev_csbf += (csbf[(ys + 1) * sb_w + xs] as usize) << 1;
        }
        let pos_tab = &tabs.pos[p.scan_idx as usize];
        if coded {
            // Significance context per scan position (9.3.4.2.5): which of
            // the three derivations applies is fixed for the whole transform
            // block, and the rest depends only on the position inside this
            // sub-block. Build the context indices of the positions this
            // sub-block will read, once, instead of deriving one per
            // coefficient inside the loop below.
            let mut ctx_of = [0u16; 16];
            let fill = (start_n + 1) as usize;
            sig_ctx_of(tabs, p.scan_idx, log2, c_idx, xs, ys, prev_csbf, ts_sig_ctx, fill, &mut ctx_of);
            // Scan position 0 is the only one that can be inferred, so it
            // comes after the loop rather than being tested inside it.
            let mut nn = start_n;
            while nn > 0 {
                let npos = nn as usize;
                if cabac.decision(&mut cx.c[ctx_of[npos] as usize]) != 0 {
                    sig_pos[n_sig] = npos as u8;
                    n_sig += 1;
                    infer_sb_dc_sig = false;
                }
                nn -= 1;
            }
            if nn == 0 {
                if !infer_sb_dc_sig {
                    if cabac.decision(&mut cx.c[ctx_of[0] as usize]) != 0 {
                        sig_pos[n_sig] = 0;
                        n_sig += 1;
                    }
                } else {
                    // DC of a coded sub-block with no other significant
                    // coefficient: inferred 1.
                    sig_pos[n_sig] = 0;
                    n_sig += 1;
                }
            }
        }
        if p.trace {
            eprintln!("  sb {i} ({xs},{ys}) coded={coded} sig={:?} n_sig={n_sig}", &sig_pos[..n_sig]);
        }
        if n_sig == 0 {
            continue;
        }
        let sig_pos = &sig_pos[..n_sig];

        // Levels: greater1 (up to 8), greater2 (one), signs, remaining.
        let mut ctx_set: u32 = if i == 0 || c_idx > 0 { 0 } else { 2 };
        // 9.3.4.2.6: lastGreater1Ctx is the previous sub-block's greater1Ctx
        // after its final update (a flag of 1 zeroes it, else it grows); a
        // zero bumps the context set.
        if !first_sb_processed && greater1_ctx_state == 0 {
            ctx_set += 1;
        }
        first_sb_processed = false;
        let mut greater1_ctx: u32 = 1;
        let mut abs_level = [1i32; 16]; // indexed like sig_pos
        let mut last_greater1_idx: i32 = -1;
        let g1_ctx_base = COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + (ctx_set * 4) as usize + if c_idx > 0 { 16 } else { 0 };
        for k in 0..n_sig.min(8) {
            let g1 = cabac.decision(&mut cx.c[g1_ctx_base + greater1_ctx.min(3) as usize]) != 0;
            if greater1_ctx > 0 {
                greater1_ctx = if g1 { 0 } else { greater1_ctx + 1 };
            }
            if g1 {
                abs_level[k] = 2;
                if last_greater1_idx == -1 {
                    last_greater1_idx = k as i32;
                }
            }
        }
        greater1_ctx_state = greater1_ctx;
        let first_sig_scan_pos = sig_pos[n_sig - 1] as i32;
        let last_sig_scan_pos = sig_pos[0] as i32;
        let sign_hidden = !p.bypass && p.sign_hiding && rdpcm.is_none() && (last_sig_scan_pos - first_sig_scan_pos > 3);
        if last_greater1_idx != -1 {
            let inc = ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
            let g2 = cabac.decision(&mut cx.c[COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc]) != 0;
            if g2 {
                abs_level[last_greater1_idx as usize] = 3;
            }
        }
        // Signs: one bypass read for all of them (MSB = first in reverse
        // scan); the hidden sign, if any, is the last position's.
        let n_signs = n_sig - sign_hidden as usize;
        if ALIGN {
            // escapeDataPresent (7.3.8.11): a level of this sub-block will be
            // coded as coeff_abs_level_remaining — a ninth significant
            // coefficient, a second greater1 flag, or the greater2 flag.
            let n_g1 = abs_level[..n_sig.min(8)].iter().filter(|&&a| a >= 2).count();
            let escape = n_sig > 8 || n_g1 > 1 || (last_greater1_idx >= 0 && abs_level[last_greater1_idx as usize] == 3);
            if escape {
                cabac.align_bypass();
            }
        }
        let signs = cabac.bypass_bits(n_signs as u32);
        // Remaining levels.
        let mut sum_abs = 0i32;
        let mut c_last_abs: i32 = 0;
        let mut c_last_rice: u32 = 0;
        let mut first_remaining = true;
        let rice_init: u32 = if p.persistent_rice { (cx.stat_coeff[sb_type] / 4) as u32 } else { 0 };
        for k in 0..n_sig {
            let base_level = abs_level[k];
            let threshold = if k < 8 {
                if k as i32 == last_greater1_idx {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            let mut level = base_level;
            if base_level == threshold {
                let rice = if first_remaining {
                    rice_init
                } else {
                    let up = c_last_rice + (c_last_abs > 3 * (1 << c_last_rice)) as u32;
                    if p.persistent_rice { up } else { up.min(4) }
                };
                let rem = if range.extended_precision { decode_abs_level_remaining_limited(cabac, rice, range.log2_range)? } else { decode_abs_level_remaining(cabac, rice)? };
                if first_remaining && p.persistent_rice {
                    // StatCoeff update on the sub-block's first remaining level.
                    let st = &mut cx.stat_coeff[sb_type];
                    if rem >= (3 << (*st / 4)) {
                        *st += 1;
                    } else if 2 * rem < (1 << (*st / 4)) && *st > 0 {
                        *st -= 1;
                    }
                }
                first_remaining = false;
                level = base_level + rem;
                c_last_abs = level;
                c_last_rice = rice;
            }
            let neg = k < n_signs && (signs >> (n_signs - 1 - k)) & 1 != 0;
            let mut v = if neg { -level } else { level };
            if sign_hidden {
                sum_abs += level;
                if k == n_sig - 1 && sum_abs % 2 == 1 {
                    v = -v;
                }
            }
            let (xp, yp) = pos_tab[sig_pos[k] as usize];
            let xc = (xs << 2) + xp as usize;
            let yc = (ys << 2) + yp as usize;
            coeffs[yc * n + xc] = C::from_level(v, range.log2_range);
            max_x = max_x.max(xc);
            max_y = max_y.max(yc);
            if p.trace {
                eprintln!("    n={} ({xc},{yc}) base={base_level} level={level} v={v} sign_hidden={sign_hidden}", sig_pos[k]);
            }
        }
    }
    if cabac.overrun() {
        return Err(Error::bitstream("slice data exhausted in residual coding"));
    }
    Ok(ResidualInfo { transform_skip, max_x, max_y, rdpcm })
}

/// Write `residual_coding()` for one transform block: the inverse of
/// [`parse_residual`]. `coeffs` is the block in raster order — the layout
/// that function decodes *into* — with `1 << (2 * log2_size)` entries, at
/// least one of them nonzero (an all-zero block cannot be spelled here; the
/// caller says it with a cbf instead).
///
/// It lives beside the reader on purpose, and shares its scan tables and
/// context derivations ([`sig_ctx_of`], [`last_sig_ctx`]): the two are
/// exact inverses over a binarisation full of small rules — a last position
/// coded as coordinates rather than a flag, a DC significance bit that is
/// sometimes inferred rather than coded, flag thresholds that decide
/// whether a remaining level follows, a Rice parameter that adapts on the
/// previous level — and every rule read off different lines twice is a
/// desync waiting to surface hundreds of blocks later.
///
/// Only the configuration this crate's encoder writes is spellable — no
/// transform skip, no sign data hiding, none of the range-extension coding
/// tools — and the writer refuses the rest (debug assertions) rather than
/// half-supporting it: `sign_hiding` especially would change which sign
/// bins exist, not just their values.
///
/// `bypass` (`cu_transquant_bypass_flag`) IS spellable, and changes nothing
/// in the spelling under this configuration. Read off the parser: bypass
/// suppresses the `transform_skip_flag` read (the PPS keeps it off anyway),
/// forces sign hiding off (off anyway), and feeds only range-extension
/// tools otherwise (RDPCM, the single significance context, the persistent
/// Rice statistic — all absent from our SPS). What differs is the *values*:
/// the levels are raw spatial residuals rather than quantised coefficients,
/// up to the full sample range — well inside the binarisation, which the
/// escape covers to i16 either way.
///
/// Nothing outside the tests calls it yet — the H.265 encoder that will is
/// being built alongside it; drop the allow when it lands.
#[allow(dead_code)]
pub(crate) fn write_residual(e: &mut CabacEncoder, cx: &mut Contexts, p: &ResidualParams, coeffs: &[i16]) {
    debug_assert!(!p.transform_skip_allowed, "transform_skip_flag writing is not supported (PPS keeps it off)");
    debug_assert!(!p.sign_hiding, "sign data hiding is not supported (PPS keeps it off)");
    debug_assert!(!p.explicit_rdpcm && !p.persistent_rice && !p.ts_context, "range-extension residual tools are not supported");
    let log2 = p.log2_size;
    let n = 1usize << log2;
    let c_idx = p.c_idx;

    // The last significant coefficient in scan order: highest sub-block
    // first, highest scan position within it second — through the same
    // inverse scans the reader uses to locate it from the coordinates.
    let log2_sb = log2 - 2;
    let sb_w = 1usize << log2_sb;
    let tabs = sub_block_tables();
    let scan = p.scan_idx as usize;
    let mut last_x = 0usize;
    let mut last_y = 0usize;
    let mut best = -1i32;
    for y in 0..n {
        for x in 0..n {
            if coeffs[y * n + x] != 0 {
                let sb = tabs.inv_sb[scan][log2_sb as usize][((y >> 2) << log2_sb) + (x >> 2)] as i32;
                let pos = tabs.inv4[scan][((y & 3) << 2) + (x & 3)] as i32;
                let key = sb * 16 + pos;
                if key > best {
                    best = key;
                    last_x = x;
                    last_y = y;
                }
            }
        }
    }
    debug_assert!(best >= 0, "residual_coding cannot spell an all-zero block");

    // last_sig_coeff_{x,y}: the reader swaps x and y *after* reading for a
    // vertical scan, so the writer swaps before writing. Both prefixes come
    // first, then both suffixes — the reader's read order.
    let (lx, ly) = if p.scan_idx == 2 { (last_y as u32, last_x as u32) } else { (last_x as u32, last_y as u32) };
    let (ctx_offset, ctx_shift) = last_sig_ctx(c_idx, log2);
    let c_max = (log2 << 1) - 1;
    // The prefix groups of 9.3.3.9: values 0..=3 spell themselves; above
    // that, prefix p covers 2^(p/2 - 1) values from (2 + p % 2) << (p/2 - 1)
    // — read off the reader's reconstruction `(1 << nb) * (2 + (prefix & 1))
    // + suffix` with `nb = (prefix >> 1) - 1`.
    let prefix_of = |v: u32| -> u32 {
        if v <= 3 {
            return v;
        }
        let msb = 31 - v.leading_zeros();
        2 * msb + ((v - (1 << msb) >= (1 << (msb - 1))) as u32)
    };
    let (px, py) = (prefix_of(lx), prefix_of(ly));
    for i in 0..px {
        e.encode_decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (ctx_offset + (i >> ctx_shift)) as usize], 1);
    }
    if px < c_max {
        e.encode_decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (ctx_offset + (px >> ctx_shift)) as usize], 0);
    }
    for i in 0..py {
        e.encode_decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (ctx_offset + (i >> ctx_shift)) as usize], 1);
    }
    if py < c_max {
        e.encode_decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (ctx_offset + (py >> ctx_shift)) as usize], 0);
    }
    if px > 3 {
        let nb = (px >> 1) - 1;
        e.encode_bypass_bits(nb, lx - (1 << nb) * (2 + (px & 1)));
    }
    if py > 3 {
        let nb = (py >> 1) - 1;
        e.encode_bypass_bits(nb, ly - (1 << nb) * (2 + (py & 1)));
    }

    let last_sub_block = tabs.inv_sb[scan][log2_sb as usize][((last_y >> 2) << log2_sb) + (last_x >> 2)] as usize;
    let last_scan_pos = tabs.inv4[scan][((last_y & 3) << 2) + (last_x & 3)] as usize;

    let pos_tab = &tabs.pos[scan];
    // The raster index of scan position `sp` of sub-block (xs, ys).
    let coord = |xs: usize, ys: usize, sp: usize| -> usize {
        let (xp, yp) = pos_tab[sp];
        ((ys << 2) + yp as usize) * n + (xs << 2) + xp as usize
    };

    let mut csbf = [0u8; 64];
    let mut greater1_ctx_state: u32 = 1; // greater1Ctx carried across sub-blocks
    let mut first_sb_processed = true;

    for i in (0..=last_sub_block).rev() {
        let (xs, ys) = scan_pos(p.scan_idx, log2_sb, i);
        let mut infer_sb_dc_sig = false;
        let coded: bool;
        if i < last_sub_block && i > 0 {
            let mut csbf_ctx = 0u32;
            if xs < sb_w - 1 {
                csbf_ctx += csbf[ys * sb_w + xs + 1] as u32;
            }
            if ys < sb_w - 1 {
                csbf_ctx += csbf[(ys + 1) * sb_w + xs] as u32;
            }
            let inc = if c_idx == 0 { csbf_ctx.min(1) } else { 2 + csbf_ctx.min(1) };
            coded = (0..16).any(|sp| coeffs[coord(xs, ys, sp)] != 0);
            e.encode_decision(&mut cx.c[SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc as usize], coded as u32);
            infer_sb_dc_sig = true;
        } else {
            coded = true; // the first and last sub-blocks are inferred coded
        }
        csbf[ys * sb_w + xs] = coded as u8;
        if !coded {
            continue;
        }

        // Significance flags in reverse scan; the significant positions are
        // collected the way the reader collects them (reverse scan order,
        // the last coefficient of the block first).
        let mut sig_pos = [0u8; 16];
        let mut n_sig = 0usize;
        let start_n = if i == last_sub_block { last_scan_pos as i32 - 1 } else { 15 };
        if i == last_sub_block {
            // Implied by the last-position coordinates; no flag is coded.
            sig_pos[0] = last_scan_pos as u8;
            n_sig = 1;
        }
        let mut prev_csbf = 0usize;
        if xs < sb_w - 1 {
            prev_csbf += csbf[ys * sb_w + xs + 1] as usize;
        }
        if ys < sb_w - 1 {
            prev_csbf += (csbf[(ys + 1) * sb_w + xs] as usize) << 1;
        }
        let mut ctx_of = [0u16; 16];
        let fill = (start_n + 1) as usize;
        sig_ctx_of(tabs, p.scan_idx, log2, c_idx, xs, ys, prev_csbf, None, fill, &mut ctx_of);
        let mut nn = start_n;
        while nn > 0 {
            let sig = coeffs[coord(xs, ys, nn as usize)] != 0;
            e.encode_decision(&mut cx.c[ctx_of[nn as usize] as usize], sig as u32);
            if sig {
                sig_pos[n_sig] = nn as u8;
                n_sig += 1;
                infer_sb_dc_sig = false;
            }
            nn -= 1;
        }
        if nn == 0 {
            let dc_sig = coeffs[coord(xs, ys, 0)] != 0;
            if !infer_sb_dc_sig {
                e.encode_decision(&mut cx.c[ctx_of[0] as usize], dc_sig as u32);
                if dc_sig {
                    sig_pos[n_sig] = 0;
                    n_sig += 1;
                }
            } else {
                // A coded sub-block with no other significant coefficient:
                // the reader infers the DC significant, so no flag may be
                // written — and the DC really is nonzero, because `coded`
                // was derived from these very coefficients.
                debug_assert!(dc_sig, "coded sub-block with nothing significant: the DC inference would lie");
                sig_pos[n_sig] = 0;
                n_sig += 1;
            }
        }
        if n_sig == 0 {
            continue;
        }
        let sig_pos = &sig_pos[..n_sig];
        let abs_of = |k: usize| coeffs[coord(xs, ys, sig_pos[k] as usize)].unsigned_abs() as i32;

        // Levels: greater1 (up to 8), greater2 (one), signs, remaining —
        // with the context-set selection and greater1Ctx carry of 9.3.4.2.6,
        // advanced exactly as the reader advances them.
        let mut ctx_set: u32 = if i == 0 || c_idx > 0 { 0 } else { 2 };
        if !first_sb_processed && greater1_ctx_state == 0 {
            ctx_set += 1;
        }
        first_sb_processed = false;
        let mut greater1_ctx: u32 = 1;
        let mut last_greater1_idx: i32 = -1;
        let g1_ctx_base = COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + (ctx_set * 4) as usize + if c_idx > 0 { 16 } else { 0 };
        for k in 0..n_sig.min(8) {
            let g1 = abs_of(k) > 1;
            e.encode_decision(&mut cx.c[g1_ctx_base + greater1_ctx.min(3) as usize], g1 as u32);
            if greater1_ctx > 0 {
                greater1_ctx = if g1 { 0 } else { greater1_ctx + 1 };
            }
            if g1 && last_greater1_idx == -1 {
                last_greater1_idx = k as i32;
            }
        }
        greater1_ctx_state = greater1_ctx;
        if last_greater1_idx != -1 {
            let inc = ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
            e.encode_decision(&mut cx.c[COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc], (abs_of(last_greater1_idx as usize) > 2) as u32);
        }
        // Signs: one bypass run, MSB = first in reverse scan order. No sign
        // hiding (asserted above), so every significant coefficient has one.
        let mut signs = 0u32;
        for k in 0..n_sig {
            signs = (signs << 1) | (coeffs[coord(xs, ys, sig_pos[k] as usize)] < 0) as u32;
        }
        e.encode_bypass_bits(n_sig as u32, signs);
        // Remaining levels: coded exactly where the flags saturated. The
        // base level the reader will have reconstructed from the flags is 1,
        // plus the greater1 flag, plus the greater2 flag at its one index —
        // and the remaining is present iff that base equals the threshold
        // the flags could not exceed.
        let mut c_last_abs: i32 = 0;
        let mut c_last_rice: u32 = 0;
        let mut first_remaining = true;
        for k in 0..n_sig {
            let a = abs_of(k);
            let (base_level, threshold) = if k < 8 {
                if k as i32 == last_greater1_idx {
                    (1 + (a > 1) as i32 + (a > 2) as i32, 3)
                } else {
                    (1 + (a > 1) as i32, 2)
                }
            } else {
                (1, 1)
            };
            if base_level == threshold {
                let rice = if first_remaining {
                    0 // StatCoeff initialisation is a range-extension tool
                } else {
                    let up = c_last_rice + (c_last_abs > 3 * (1 << c_last_rice)) as u32;
                    up.min(4)
                };
                write_abs_level_remaining(e, rice, (a - base_level) as u32);
                first_remaining = false;
                c_last_abs = a;
                c_last_rice = rice;
            }
        }
    }
}

/// `coeff_abs_level_remaining`: the inverse of [`decode_abs_level_remaining`].
/// A truncated-Rice prefix of up to four ones — the fourth is not followed
/// by a terminating zero, the reader's loop stops there on its own — then
/// either `rice` suffix bits, or an EGk escape with `k = rice + 1` for what
/// lies beyond the four groups.
#[allow(dead_code)]
#[inline]
fn write_abs_level_remaining(e: &mut CabacEncoder, rice: u32, v: u32) {
    let prefix = v >> rice;
    if prefix < 4 {
        for _ in 0..prefix {
            e.encode_bypass(1);
        }
        e.encode_bypass(0);
        e.encode_bypass_bits(rice, v & ((1u32 << rice) - 1));
    } else {
        for _ in 0..4 {
            e.encode_bypass(1);
        }
        // EGk: ones while the remainder covers the next doubling, a zero,
        // then that many bits of what is left.
        let mut rem = v - (4 << rice);
        let mut k = rice + 1;
        while rem >= (1 << k) {
            e.encode_bypass(1);
            rem -= 1 << k;
            k += 1;
            debug_assert!(k <= 30, "coeff_abs_level_remaining too large to binarise");
        }
        e.encode_bypass(0);
        e.encode_bypass_bits(k, rem);
    }
}

/// Rotate a 4x4 residual by 180 degrees (`transform_skip_rotation_enabled_flag`).
pub fn rotate_residual4<C: Copy>(coeffs: &mut [C]) {
    coeffs[..16].reverse();
}

/// Residual DPCM (8.6.6 / 8.6.8): accumulate the residual along rows
/// (horizontal) or down columns (`vertical`).
pub(crate) fn rdpcm_residual<C: Coeff>(coeffs: &mut [C], log2: u32, vertical: bool) {
    let n = 1usize << log2;
    if vertical {
        for y in 1..n {
            for x in 0..n {
                coeffs[y * n + x] = coeffs[y * n + x].wrapping_add(coeffs[(y - 1) * n + x]);
            }
        }
    } else {
        for y in 0..n {
            for x in 1..n {
                coeffs[y * n + x] = coeffs[y * n + x].wrapping_add(coeffs[y * n + x - 1]);
            }
        }
    }
}

/// `coeff_abs_level_remaining` (9.3.3.11 without extended precision).
#[inline]
fn decode_abs_level_remaining(cabac: &mut Cabac, rice: u32) -> Result<i32> {
    let mut prefix = 0u32;
    while prefix < 4 && cabac.bypass() != 0 {
        prefix += 1;
    }
    if prefix < 4 {
        let suffix = cabac.bypass_bits(rice);
        return Ok(((prefix << rice) + suffix) as i32);
    }
    // EGk with k = rice + 1 for (value - (4 << rice)).
    let mut k = rice + 1;
    let mut v: i32 = 0;
    loop {
        if cabac.bypass() != 0 {
            v += 1 << k;
            k += 1;
            if k > 30 {
                return Err(Error::bitstream("coeff_abs_level_remaining runaway"));
            }
        } else {
            break;
        }
    }
    v += cabac.bypass_bits(k) as i32;
    Ok((4 << rice) as i32 + v)
}

/// `coeff_abs_level_remaining` under `extended_precision_processing_flag`
/// (9.3.3.11 with the limited k-th order Exp-Golomb suffix of 9.3.3.12):
/// the same code as [`decode_abs_level_remaining`] until the escape's
/// unary prefix reaches `32 - log2TransformRange` ones in all, where it
/// stops without a terminating zero and `log2TransformRange` raw bits
/// follow instead of `prefix - 3 + rice`. (In HM's terms:
/// `maximumPrefixLength = 32 - (3 + maxLog2TrDynamicRange)` after the
/// three TR ones; the spec counts four TR ones and `maxPreExtLen = 28 -
/// log2TransformRange` beyond them — the same bins.)
#[inline]
fn decode_abs_level_remaining_limited(cabac: &mut Cabac, rice: u32, log2_range: u32) -> Result<i32> {
    let longest = 32 - log2_range;
    let mut prefix = 0u32;
    while prefix < longest && cabac.bypass() != 0 {
        prefix += 1;
    }
    if prefix < 4 {
        let suffix = cabac.bypass_bits(rice);
        return Ok(((prefix << rice) + suffix) as i32);
    }
    let pl = prefix - 3;
    let bits = if pl == longest - 3 { log2_range } else { pl + rice };
    if bits > 32 {
        return Err(Error::bitstream("coeff_abs_level_remaining runaway"));
    }
    let v = cabac.bypass_bits_wide(bits) as i64;
    let base = (((1i64 << pl) - 1 + 3) << rice) + v;
    if base > i32::MAX as i64 {
        return Err(Error::bitstream("coeff_abs_level_remaining out of range"));
    }
    Ok(base as i32)
}

/// The scaling factor `m[x][y]` source for a transform block.
pub enum ScalingSource<'a> {
    /// Flat 16.
    Flat,
    /// A scaling list at the block's size: 16 (4x4) or 64 (8x8 grid, replicated
    /// for 16x16 / 32x32) raster values, plus the DC for 16/32.
    List(&'a [u8], u8),
}

/// Scale (dequantise) the coefficients of one transform block in place
/// (8.6.3): `d = Clip3(coeffMin, coeffMax, ((c * m * levelScale[qP % 6] << (qP / 6)) + (1 << (bdShift - 1))) >> bdShift)`
/// with `bdShift = BitDepth + Log2(nTbS) - 5`, `m = 16` when scaling lists are off
/// (or the block is transform-skipped and larger than 4x4), else the list.
/// Only the `0..=max_x` × `0..=max_y` region can hold nonzero coefficients.
#[allow(clippy::too_many_arguments)]
pub fn scale_coefficients(
    coeffs: &mut [i16],
    log2: u32,
    qp: i32,
    bit_depth: u32,
    scaling: ScalingSource,
    transform_skip: bool,
    max_x: usize,
    max_y: usize,
) {
    const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let n = 1usize << log2;
    let bd_shift = bit_depth as i32 + log2 as i32 - 5;
    let round = 1i64 << (bd_shift - 1);
    let ls = LEVEL_SCALE[(qp % 6) as usize] as i64;
    let q6 = qp / 6;
    let flat = matches!(scaling, ScalingSource::Flat) || (transform_skip && n > 4);
    // Flat scaling with a factor that keeps `c * factor` in i32: a branchless
    // row loop the compiler vectorises (zeros scale to zero: round < 2^bdShift).
    let factor = (16 * ls) << q6;
    if flat && factor < 65536 {
        let factor = factor as i32;
        let round = round as i32;
        for y in 0..=max_y {
            let row = &mut coeffs[y * n..y * n + max_x + 1];
            for c in row.iter_mut() {
                *c = ((*c as i32 * factor + round) >> bd_shift).clamp(-32768, 32767) as i16;
            }
        }
        return;
    }
    for y in 0..=max_y {
        for x in 0..=max_x {
            let c = coeffs[y * n + x];
            if c == 0 {
                continue;
            }
            let m: i64 = if flat {
                16
            } else {
                match &scaling {
                    ScalingSource::List(list, dc) => {
                        if n == 4 {
                            list[y * 4 + x] as i64
                        } else if n == 8 {
                            list[y * 8 + x] as i64
                        } else if x == 0 && y == 0 {
                            *dc as i64
                        } else {
                            let r = n / 8;
                            list[(y / r) * 8 + x / r] as i64
                        }
                    }
                    ScalingSource::Flat => 16,
                }
            };
            let v = ((c as i64 * m * ls) << q6) + round;
            coeffs[y * n + x] = (v >> bd_shift).clamp(-32768, 32767) as i16;
        }
    }
}

/// [`scale_coefficients`] for the wide pipeline (`i32` coefficients, any
/// depth, with or without extended precision): `bdShift = BitDepth +
/// Log2(nTbS) + 10 - log2TransformRange` and the clip to
/// `±(1 << log2TransformRange)` (8.6.3) — the 8–12-bit formula with
/// `log2TransformRange = 15` put back in.
#[allow(clippy::too_many_arguments)]
pub fn scale_coefficients_i32(
    coeffs: &mut [i32],
    log2: u32,
    qp: i32,
    bit_depth: u32,
    log2_range: u32,
    scaling: ScalingSource,
    transform_skip: bool,
    max_x: usize,
    max_y: usize,
) {
    const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let n = 1usize << log2;
    let bd_shift = bit_depth as i32 + log2 as i32 + 10 - log2_range as i32;
    let round = 1i64 << (bd_shift - 1);
    let (cmin, cmax) = (-(1i64 << log2_range), (1i64 << log2_range) - 1);
    let ls = LEVEL_SCALE[(qp % 6) as usize] as i64;
    let q6 = qp / 6;
    let flat = matches!(scaling, ScalingSource::Flat) || (transform_skip && n > 4);
    for y in 0..=max_y {
        for x in 0..=max_x {
            let c = coeffs[y * n + x];
            if c == 0 {
                continue;
            }
            let m: i64 = match (&scaling, flat) {
                (_, true) | (ScalingSource::Flat, _) => 16,
                (ScalingSource::List(list, dc), false) => {
                    if n == 4 {
                        list[y * 4 + x] as i64
                    } else if n == 8 {
                        list[y * 8 + x] as i64
                    } else if x == 0 && y == 0 {
                        *dc as i64
                    } else {
                        let r = n / 8;
                        list[(y / r) * 8 + x / r] as i64
                    }
                }
            };
            let v = ((c as i64 * m * ls) << q6) + round;
            coeffs[y * n + x] = (v >> bd_shift).clamp(cmin, cmax) as i32;
        }
    }
}

/// [`transform_skip_residual`] for the wide pipeline, at any depth and
/// with `extended_precision_processing_flag` folded in (8.6.4.2):
/// `bdShift = Max(20 - BitDepth, extended ? 11 : 0)`, `tsShift =
/// (extended ? Min(5, bdShift - 2) : 5) + Log2(nTbS)`. At 16 bits without
/// extended precision the net shift is to the *left* — the 8–12-bit
/// kernel's `i16` result would not hold it.
pub fn transform_skip_residual_i32(coeffs: &mut [i32], log2: u32, bit_depth: u32, extended: bool) {
    let n = 1usize << log2;
    let bd_shift = (20 - bit_depth as i32).max(if extended { 11 } else { 0 });
    let ts_shift = (if extended { 5.min(bd_shift - 2) } else { 5 }) + log2 as i32;
    let round = 1i64 << (bd_shift - 1);
    for v in coeffs.iter_mut().take(n * n) {
        *v = ((((*v as i64) << ts_shift) + round) >> bd_shift) as i32;
    }
}

/// Transform-skip residual (8.6.4.2 with `transform_skip_flag`): `r = (d << tsShift + round) >> bdShift`.
pub fn transform_skip_residual(coeffs: &mut [i16], log2: u32, bit_depth: u32) {
    let n = 1usize << log2;
    let ts_shift = 5 + log2 as i32;
    let bd_shift = 20 - bit_depth as i32;
    let round = 1i32 << (bd_shift - 1);
    for v in coeffs.iter_mut().take(n * n) {
        *v = ((((*v as i32) << ts_shift) + round) >> bd_shift).clamp(-32768, 32767) as i16;
    }
}

#[cfg(test)]
mod write_round_trip {
    use super::*;
    use crate::bitwriter::BitWriter;

    fn params(log2: u32, c_idx: usize, scan_idx: u32) -> ResidualParams {
        ResidualParams {
            log2_size: log2,
            c_idx,
            scan_idx,
            bypass: false,
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

    /// Encode a sequence of blocks into one codeword, decode it with the
    /// production parser, and require the coefficients, the nonzero extents
    /// and the *entire* context state to come back.
    ///
    /// The context comparison is the half that catches a desync: two
    /// binarisations can spell the same coefficients while leaving the
    /// probability model in different places, and nothing goes wrong until
    /// a later block reads a bin against a state the writer never had.
    /// Chaining blocks through one state is what makes the carried
    /// `greater1Ctx` set selection and the shared contexts load-bearing.
    fn round_trip(qp: i32, blocks: &[(ResidualParams, Vec<i16>)]) {
        let mut enc_cx = Contexts::new(0, qp);
        let mut w = BitWriter::new();
        {
            let mut e = CabacEncoder::new(&mut w);
            for (p, c) in blocks {
                write_residual(&mut e, &mut enc_cx, p, c);
            }
            e.encode_terminate(1);
        }
        w.align_zero();
        let data = w.into_rbsp();

        let mut dec_cx = Contexts::new(0, qp);
        let mut d = Cabac::new(&data);
        let mut out = vec![0i16; 1024];
        for (k, (p, want)) in blocks.iter().enumerate() {
            let n = 1usize << p.log2_size;
            let ri = parse_residual::<i16, false>(&mut d, &mut dec_cx, p, &ResidualRange::PLAIN, &mut out)
                .unwrap_or_else(|e| panic!("block {k}: the reader rejected what the writer produced: {e}"));
            assert!(!ri.transform_skip, "block {k}");
            assert_eq!(&out[..n * n], &want[..], "block {k}: coefficients differ");
            let (mut mx, mut my) = (0usize, 0usize);
            for y in 0..n {
                for x in 0..n {
                    if want[y * n + x] != 0 {
                        mx = mx.max(x);
                        my = my.max(y);
                    }
                }
            }
            assert_eq!((ri.max_x, ri.max_y), (mx, my), "block {k}: extents differ");
        }
        assert_eq!(d.terminate(), 1, "the closing terminate did not read back as 1");
        assert!(!d.overrun(), "the reader ran past what the writer wrote");
        assert_eq!(enc_cx.c, dec_cx.c, "context states diverged: the sides would desync on the next block");
        assert_eq!(enc_cx.stat_coeff, dec_cx.stat_coeff);
    }

    fn one(qp: i32, p: ResidualParams, coeffs: Vec<i16>) {
        round_trip(qp, &[(p, coeffs)]);
    }

    /// The (size, component, scan) shapes legal under this encoder's SPS:
    /// mode-dependent scans exist only for 4x4 blocks and luma 8x8; chroma
    /// blocks in 4:2:0 stop at 16x16.
    fn shapes() -> Vec<(u32, usize, u32)> {
        let mut v = Vec::new();
        for c_idx in [0usize, 1] {
            for log2 in 2..=(if c_idx == 0 { 5u32 } else { 4 }) {
                let scans: &[u32] = if log2 == 2 || (log2 == 3 && c_idx == 0) { &[0, 1, 2] } else { &[0] };
                for &s in scans {
                    v.push((log2, c_idx, s));
                }
            }
        }
        v
    }

    /// Single coefficients at telling positions, with magnitudes either side
    /// of every rule boundary: 1 (no flags), 2 and 3 (the greater1/greater2
    /// flags), 4..6 (a remaining under the Rice prefix), 7.. (the EGk
    /// escape), up to the full i16 range — including `i16::MIN`, whose
    /// magnitude is the one value that does not fit an i16.
    #[test]
    fn round_trips_single_coefficients() {
        for (log2, c_idx, scan) in shapes() {
            let n = 1usize << log2;
            for pos in [0usize, 1, n - 1, (n - 1) * n, n * n - 1, (n / 2) * n + n / 2] {
                for m in [1i16, 2, 3, 4, 5, 6, 7, 9, 100, 32767] {
                    let mut c = vec![0i16; n * n];
                    c[pos] = m;
                    one(26, params(log2, c_idx, scan), c.clone());
                    c[pos] = -m;
                    one(26, params(log2, c_idx, scan), c);
                }
            }
            let mut c = vec![0i16; n * n];
            c[0] = i16::MIN;
            one(26, params(log2, c_idx, scan), c);
        }
    }

    /// Runs of ±1 — what quantised residual mostly is, and a previously
    /// found blind spot: only a run longer than eight reaches the k >= 8
    /// threshold, where even a magnitude of one codes a remaining level.
    #[test]
    fn round_trips_runs_of_ones() {
        for (log2, c_idx, scan) in shapes() {
            let n = 1usize << log2;
            for len in [2usize, 4, 7, 8, 9, 15, 16, 17, 24, 33] {
                if len > n * n {
                    continue;
                }
                // All ones, all minus ones, alternating.
                for pattern in 0..3 {
                    let mut c = vec![0i16; n * n];
                    for (i, v) in c.iter_mut().enumerate().take(len) {
                        *v = match pattern {
                            0 => 1,
                            1 => -1,
                            _ => {
                                if i % 2 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                        };
                    }
                    one(26, params(log2, c_idx, scan), c);
                }
                // The same run under a single outlier at either end: the
                // Rice parameter must climb after the outlier's escape and
                // the remainders after it still round-trip.
                let mut c = vec![0i16; n * n];
                for v in c.iter_mut().take(len) {
                    *v = 1;
                }
                c[0] = 900;
                one(26, params(log2, c_idx, scan), c.clone());
                c[0] = 1;
                c[len - 1] = -21000;
                one(26, params(log2, c_idx, scan), c);
            }
            // Every magnitude at the boundary of the Rice update rule
            // (cLastAbsLevel > 3 << cLastRiceParam), followed by more
            // remainders that read the updated parameter: 3 vs 4 at rice 0,
            // 6 vs 7 at rice 1, and so on. A mutation of the threshold is
            // invisible until a level sits exactly on it with another
            // remaining level behind it.
            for edge in [3i16, 4, 6, 7, 12, 13, 24, 25] {
                let mut c = vec![0i16; 1usize << (2 * log2)];
                c[0] = edge;
                c[1] = 9;
                c[2] = -9;
                c[3] = edge;
                one(26, params(log2, c_idx, scan), c);
            }
        }
    }

    /// The coded_sub_block_flag machinery of blocks larger than 4x4: holes
    /// (sub-blocks skipped entirely), and coded sub-blocks whose only
    /// nonzero coefficient is their DC — the case where the reader *infers*
    /// the DC significant and the writer must not spell it.
    #[test]
    fn round_trips_sub_block_inference() {
        for (log2, c_idx, scan) in shapes() {
            if log2 == 2 {
                continue;
            }
            let n = 1usize << log2;
            let sbs = n / 4;
            // Far corner significant, then every middle sub-block in turn
            // carrying only its DC.
            for sy in 0..sbs {
                for sx in 0..sbs {
                    let mut c = vec![0i16; n * n];
                    c[n * n - 1] = 1;
                    c[(sy * 4) * n + sx * 4] = -3;
                    one(26, params(log2, c_idx, scan), c);
                }
            }
            // A diagonal band of only-DC sub-blocks at once, magnitudes
            // driving every level rule inside the inference case.
            let mut c = vec![0i16; n * n];
            c[n * n - 1] = 2;
            for s in 0..sbs {
                c[(s * 4) * n + s * 4] = [1i16, -2, 3, 800][s % 4];
            }
            one(26, params(log2, c_idx, scan), c);
        }
    }

    /// Dense blocks: every position significant, which drives the greater1
    /// counter to its cap in every sub-block and exercises the context-set
    /// bump carried between sub-blocks.
    #[test]
    fn round_trips_dense_blocks() {
        for (log2, c_idx, scan) in shapes() {
            let n = 1usize << log2;
            let mut c = vec![0i16; n * n];
            for (i, v) in c.iter_mut().enumerate() {
                *v = if i % 2 == 0 { (i % 37) as i16 + 1 } else { -((i % 11) as i16) - 1 };
            }
            one(26, params(log2, c_idx, scan), c.clone());
            for v in c.iter_mut() {
                *v = 1;
            }
            one(26, params(log2, c_idx, scan), c);
        }
    }

    /// Transquant-bypass blocks: same spelling (the parser's bypass path
    /// only suppresses syntax our configuration already keeps off), raw
    /// full-range levels — dense ±255, alternating extremes, and bypass
    /// blocks chained with quantised ones through one context state, since
    /// that is exactly what a stream with per-CU bypass flags looks like.
    #[test]
    fn round_trips_bypass_blocks() {
        let byp = |log2: u32, c_idx: usize, scan: u32| -> ResidualParams {
            ResidualParams { bypass: true, ..params(log2, c_idx, scan) }
        };
        for (log2, c_idx, scan) in shapes() {
            let n = 1usize << log2;
            // Dense full-range: every position at an 8-bit residual extreme.
            let mut c = vec![0i16; n * n];
            for (i, v) in c.iter_mut().enumerate() {
                *v = match i % 4 {
                    0 => 255,
                    1 => -255,
                    2 => 128,
                    _ => -1,
                };
            }
            one(26, byp(log2, c_idx, scan), c);
            // A flat-ish bypass block: one large residual in a zero field.
            let mut c = vec![0i16; n * n];
            c[n + 1] = -255;
            one(26, byp(log2, c_idx, scan), c);
        }
        // Bypassed and quantised blocks interleaved in one codeword: the
        // per-CU flag makes this mix, and the context state is shared.
        let mut blocks = Vec::new();
        for (k, (log2, c_idx, scan)) in shapes().into_iter().enumerate() {
            let n = 1usize << log2;
            let mut c = vec![0i16; n * n];
            for (i, v) in c.iter_mut().enumerate().take(n) {
                *v = if k % 2 == 0 { [255, -37, 1, -255][i % 4] } else { [1, -1, 2, 0][i % 4] };
            }
            let p = if k % 2 == 0 { byp(log2, c_idx, scan) } else { params(log2, c_idx, scan) };
            blocks.push((p, c));
        }
        round_trip(26, &blocks);
    }

    /// Pseudo-random blocks of every shape chained through one codeword and
    /// one context state, at several QPs (the initial context states depend
    /// on the QP). Mostly-zero blocks, the occasional outlier: the shape of
    /// real residual.
    #[test]
    fn round_trips_random_chains() {
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut lcg = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let shapes = shapes();
        for qp in [0i32, 17, 26, 39, 51] {
            for _ in 0..30 {
                let mut blocks = Vec::new();
                for _ in 0..(1 + lcg() % 8) {
                    let (log2, c_idx, scan) = shapes[lcg() as usize % shapes.len()];
                    // Chroma components share contexts; alternate 1 and 2 to
                    // prove the writer treats them identically.
                    let c_idx = if c_idx == 1 && lcg() % 2 == 0 { 2 } else { c_idx };
                    let n = 1usize << log2;
                    let mut c = vec![0i16; n * n];
                    let mut any = false;
                    for v in c.iter_mut() {
                        if lcg() % 5 == 0 {
                            let m = 1 + (lcg() % 60) as i16;
                            let m = if lcg() % 19 == 0 { m * 500 } else { m };
                            *v = if lcg() % 2 == 0 { m } else { -m };
                            any = true;
                        }
                    }
                    if !any {
                        c[(lcg() as usize) % (n * n)] = 1;
                    }
                    blocks.push((params(log2, c_idx, scan), c));
                }
                round_trip(qp, &blocks);
            }
        }
    }
}
