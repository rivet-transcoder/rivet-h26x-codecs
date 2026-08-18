//! Macroblock-level definitions shared by the CAVLC and CABAC parsers and
//! the reconstruction: macroblock types and partitions, the per-picture
//! macroblock info arrays, neighbour derivation (6.4.11), motion vector
//! prediction (8.4.1) including P_Skip and the B direct modes.

use super::frame::{BlockMotion, Frame, Mv};
use super::slice::SliceType;
use super::tables::{BLK4X4_X, BLK4X4_Y};
use crate::sample::Sample;

/// Macroblock prediction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbKind {
    /// Intra 4x4 (`I_NxN` with `transform_size_8x8_flag` 0).
    I4x4,
    /// Intra 8x8 (`I_NxN` with `transform_size_8x8_flag` 1).
    I8x8,
    /// Intra 16x16.
    I16x16,
    /// Raw samples.
    IPcm,
    /// One 16x16 inter partition (P_L0_16x16 or a B 16x16 with any direction).
    Inter16x16,
    /// Two 16x8 partitions.
    Inter16x8,
    /// Two 8x16 partitions.
    Inter8x16,
    /// Four 8x8 partitions with sub-macroblock types.
    Inter8x8,
    /// P_Skip.
    PSkip,
    /// B_Skip.
    BSkip,
    /// B_Direct_16x16.
    BDirect16x16,
}

impl MbKind {
    /// Intra?
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(
            self,
            MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::IPcm
        )
    }
    /// Skipped (no residual, inferred motion)?
    #[inline]
    pub fn is_skip(self) -> bool {
        matches!(self, MbKind::PSkip | MbKind::BSkip)
    }
    /// Uses whole-macroblock direct prediction (B_Skip / B_Direct_16x16)?
    #[inline]
    pub fn is_direct16x16(self) -> bool {
        matches!(self, MbKind::BSkip | MbKind::BDirect16x16)
    }
}

/// Sub-macroblock partitioning of an 8x8 partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubMbShape {
    /// One 8x8.
    S8x8,
    /// Two 8x4.
    S8x4,
    /// Two 4x8.
    S4x8,
    /// Four 4x4.
    S4x4,
    /// B_Direct_8x8.
    Direct,
}

impl SubMbShape {
    /// Number of sub-partitions.
    pub fn count(self) -> usize {
        match self {
            SubMbShape::S8x8 | SubMbShape::Direct => 1,
            SubMbShape::S8x4 | SubMbShape::S4x8 => 2,
            SubMbShape::S4x4 => 4,
        }
    }
    /// Width and height in samples of each sub-partition.
    pub fn size(self) -> (usize, usize) {
        match self {
            SubMbShape::S8x8 | SubMbShape::Direct => (8, 8),
            SubMbShape::S8x4 => (8, 4),
            SubMbShape::S4x8 => (4, 8),
            SubMbShape::S4x4 => (4, 4),
        }
    }
}

/// Prediction direction flags of a partition: bit 0 = list 0, bit 1 = list 1.
pub const PRED_L0: u8 = 1;
/// See [`PRED_L0`].
pub const PRED_L1: u8 = 2;
/// See [`PRED_L0`].
pub const PRED_BI: u8 = 3;

/// One 4x4 block's worth of parsed motion syntax.
#[derive(Debug, Clone, Copy, Default)]
pub struct MvdEntry {
    /// `mvd_lX` for the block (zero for blocks that did not carry one).
    pub mvd: [Mv; 2],
}

/// The parsed macroblock (`macroblock_layer()`), before reconstruction.
#[derive(Debug, Clone)]
pub struct MbLayer {
    /// The kind.
    pub kind: MbKind,
    /// `transform_size_8x8_flag`.
    pub transform_8x8: bool,
    /// Intra 16x16 prediction mode (0..=3).
    pub intra16_mode: u8,
    /// Intra 4x4 (or 8x8, replicated over its four 4x4s) prediction modes in
    /// **raster** 4x4 order within the macroblock.
    pub intra_modes: [u8; 16],
    /// `intra_chroma_pred_mode`.
    pub chroma_mode: u8,
    /// `coded_block_pattern`: bits 0..3 luma 8x8 blocks, bits 4..5 chroma.
    pub cbp: u8,
    /// Per 8x8 partition: prediction direction flags (`PRED_*`).
    pub pred_dir: [u8; 4],
    /// Per 8x8 partition: sub-macroblock shape (for `Inter8x8`).
    pub sub_shape: [SubMbShape; 4],
    /// Per list, per 8x8 partition: reference index (-1 = none).
    pub ref_idx: [[i8; 4]; 2],
    /// Per 4x4 block (raster): the mvds carried.
    pub mvd: [MvdEntry; 16],
    /// `mb_qp_delta`.
    pub qp_delta: i32,
    /// `QP_Y` of the macroblock (the previous macroblock's for one without
    /// `mb_qp_delta`), set by the parser (or the skip path).
    pub qp: i32,
    /// MBAFF: the pair's `mb_field_decoding_flag` (a field macroblock).
    pub field: bool,
    /// Luma-style coefficients per colour plane (`[0]` luma; `[1]`, `[2]`
    /// Cb and Cr in 4:4:4, where chroma is coded like luma): 16 blocks of 16
    /// (4x4 mode, `blk_raster * 16`, each in raster order within the block)
    /// or 4 blocks of 64 (8x8 mode, `blk8 * 64`, raster within the block).
    /// Intra 16x16 AC blocks keep position 0 free (the DC lives in `dc`).
    pub coef: [[i32; 256]; 3],
    /// Intra 16x16 DC coefficients per plane (raster 4x4 over the
    /// macroblock).
    pub dc: [[i32; 16]; 3],
    /// Chroma DC (Cb, Cr) for 4:2:0 / 4:2:2: 4 each for 4:2:0 (2x2 raster),
    /// 8 for 4:2:2 (4x2 raster, rows of two).
    pub chroma_dc: [[i32; 8]; 2],
    /// Chroma AC for 4:2:0 / 4:2:2: [component][block raster (2 columns; 2
    /// or 4 rows)][raster within block, 0 unused].
    pub chroma_ac: [[[i32; 16]; 8]; 2],
    /// Per plane, which 4x4 blocks (raster) have any nonzero coefficient —
    /// the count for CAVLC nC, `!= 0` for the CABAC coded_block_flag and
    /// (luma) the deblocking bS 2 rule.
    pub nz: [[u8; 16]; 3],
    /// Chroma AC nonzero counts (4:2:0 / 4:2:2): [component][block raster]
    /// (4 or 8 blocks).
    pub chroma_nz: [[u8; 8]; 2],
    /// Coded-block flags of the DC blocks: bit 0 luma DC, bit 1 Cb DC, bit 2 Cr DC (CABAC).
    pub dc_cbf: u8,
    /// PCM samples: 256 luma then the Cb then the Cr samples (64 each for
    /// 4:2:0, 128 for 4:2:2).
    pub pcm: Vec<u16>,
}

impl MbLayer {
    /// A blank macroblock of `kind`.
    pub fn new(kind: MbKind) -> Self {
        MbLayer {
            kind,
            qp: 0,
            transform_8x8: false,
            intra16_mode: 0,
            intra_modes: [2; 16],
            chroma_mode: 0,
            cbp: 0,
            pred_dir: [0; 4],
            sub_shape: [SubMbShape::S8x8; 4],
            ref_idx: [[-1; 4]; 2],
            mvd: [MvdEntry::default(); 16],
            qp_delta: 0,
            field: false,
            coef: [[0; 256]; 3],
            dc: [[0; 16]; 3],
            chroma_dc: [[0; 8]; 2],
            chroma_ac: [[[0; 16]; 8]; 2],
            nz: [[0; 16]; 3],
            chroma_nz: [[0; 8]; 2],
            dc_cbf: 0,
            pcm: Vec::new(),
        }
    }
    /// Make this a blank macroblock of `kind` again, cheaply.
    ///
    /// The layer is ~1.7 KiB and is reused across a slice: zeroing all of it
    /// per macroblock was a measurable share of the decode. The parsers write
    /// only nonzero coefficients into zeroed blocks and record which blocks
    /// they touched (`nz`, `chroma_nz`, the kind), so only those blocks
    /// are cleared here and every untouched block is still zero.
    pub fn reset(&mut self, kind: MbKind, cabac: bool) {
        for p in 0..3 {
            // Nothing coded in this plane (always so for Cb / Cr outside
            // 4:4:4): one word test instead of sixteen.
            if u128::from_ne_bytes(self.nz[p]) == 0 {
                continue;
            }
            if self.transform_8x8 || self.kind == MbKind::I8x8 {
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    if self.nz[p][by * 4 + bx] != 0
                        || self.nz[p][by * 4 + bx + 1] != 0
                        || self.nz[p][(by + 1) * 4 + bx] != 0
                        || self.nz[p][(by + 1) * 4 + bx + 1] != 0
                    {
                        self.coef[p][blk8 * 64..blk8 * 64 + 64].fill(0);
                    }
                }
            } else {
                for r in 0..16 {
                    if self.nz[p][r] != 0 {
                        self.coef[p][r * 16..r * 16 + 16].fill(0);
                    }
                }
            }
        }
        if self.kind == MbKind::I16x16 {
            self.dc = [[0; 16]; 3];
        }
        // Chroma DC / AC were written only with the chroma cbp bits set (or
        // for I_PCM, which fills every count).
        if self.cbp & 0x30 != 0 || self.kind == MbKind::IPcm {
            self.chroma_dc = [[0; 8]; 2];
            for comp in 0..2 {
                if u64::from_ne_bytes(self.chroma_nz[comp]) == 0 {
                    continue;
                }
                for b in 0..8 {
                    if self.chroma_nz[comp][b] != 0 {
                        self.chroma_ac[comp][b] = [0; 16];
                    }
                }
            }
        }
        self.kind = kind;
        self.transform_8x8 = false;
        self.intra16_mode = 0;
        // `intra_modes` is written for every block of the kinds that read
        // it (I4x4: all sixteen; I8x8: all four quads); no reset needed.
        self.chroma_mode = 0;
        self.cbp = 0;
        self.pred_dir = [0; 4];
        self.sub_shape = [SubMbShape::S8x8; 4];
        self.ref_idx = [[-1; 4]; 2];
        // Only CABAC reads a macroblock's mvds (its neighbours' contexts).
        if cabac {
            self.mvd = [MvdEntry::default(); 16];
        }
        self.qp_delta = 0;
        self.nz = [[0; 16]; 3];
        self.chroma_nz = [[0; 8]; 2];
        self.dc_cbf = 0;
    }

    /// Whether the macroblock carries any luma residual (cbp luma bits) — the
    /// condition under which `mb_qp_delta` and residual are present, together
    /// with the chroma bits and I16x16.
    pub fn has_residual(&self) -> bool {
        self.cbp != 0 || self.kind == MbKind::I16x16
    }
}

/// What the picture remembers about each decoded macroblock, for its
/// neighbours and for the deblocking filter.
#[derive(Debug, Clone, Copy)]
pub struct MbInfo {
    /// Kind (`IPcm`, skip and intra matter to neighbours).
    pub kind: MbKind,
    /// The slice the macroblock belongs to (index in decode order).
    pub slice: u16,
    /// Decoded (available as a neighbour)?
    pub decoded: bool,
    /// `QP_Y`.
    pub qp: i8,
    /// `QP_C` for Cb and Cr (from `qp` and the PPS offsets).
    pub qpc: [i8; 2],
    /// `coded_block_pattern`.
    pub cbp: u8,
    /// `transform_size_8x8_flag`.
    pub transform_8x8: bool,
    /// `intra_chroma_pred_mode` (CABAC context).
    pub chroma_mode: u8,
    /// `mb_qp_delta != 0` (CABAC context for the next macroblock).
    pub qp_delta_nonzero: bool,
    /// Coded-block flags of the DC blocks (CABAC): bit 0 luma DC (I16x16),
    /// bit 1 Cb DC, bit 2 Cr DC.
    pub dc_cbf: u8,
    /// For `Inter8x8`: which 8x8 partitions are B_Direct_8x8 (bitmask).
    pub sub_direct: u8,
    /// A field macroblock (of a field picture, or a field pair of an MBAFF
    /// frame): its motion is in field units.
    pub field: bool,
    /// The deblocking filter's "has coefficients" mask of the 4x4 luma
    /// blocks (raster), an 8x8 transform's four blocks all set when any is;
    /// all set for I_PCM.
    pub nz_mask: u16,
    /// Internal 4x4 edges that are partition boundaries (where motion can
    /// differ): bit `e * 4 + k` for vertical edge `e` (1..4) at block row
    /// `k`, and for horizontal edge `e` at block column `k`. Only these
    /// need the motion comparison of 8.7.2.1; the rest of an inter
    /// macroblock's internal edges are bS 0 without coefficients.
    pub part_edges: [u16; 2],
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo {
            kind: MbKind::PSkip,
            slice: 0,
            decoded: false,
            qp: 0,
            qpc: [0; 2],
            cbp: 0,
            transform_8x8: false,
            chroma_mode: 0,
            qp_delta_nonzero: false,
            dc_cbf: 0,
            sub_direct: 0,
            field: false,
            nz_mask: 0,
            part_edges: [0; 2],
        }
    }
}

/// The per-picture arrays of macroblock and block information.
pub struct PicInfo {
    /// Width in macroblocks.
    pub mb_width: usize,
    /// Height in macroblocks.
    pub mb_height: usize,
    /// Per macroblock.
    pub mbs: Vec<MbInfo>,
    /// Per 4x4 luma block (raster within MB): nonzero coefficient count
    /// (`TotalCoeff` for CAVLC nC; `!= 0` is the CABAC coded_block_flag).
    pub luma_nz: Vec<u8>,
    /// Per 4x4 chroma block: `addr * 32 + comp * 16 + blk` — four blocks
    /// per component in 4:2:0, eight in 4:2:2, sixteen (luma-style raster)
    /// in 4:4:4.
    pub chroma_nz: Vec<u8>,
    /// Per 4x4 luma block (raster): intra prediction mode (2 = DC for
    /// macroblocks that are not I4x4/I8x8, which is what neighbours read).
    pub intra_modes: Vec<u8>,
    /// Per 4x4 block, per list: the mvd (CABAC contexts).
    pub mvd: [Vec<Mv>; 2],
}

/// Recycled [`PicInfo`]s: the per-picture arrays are several hundred KiB at
/// HD, and allocating them fresh per picture was page faults on every one.
#[derive(Clone, Default)]
pub struct InfoPool(std::sync::Arc<std::sync::Mutex<Vec<PicInfo>>>);

impl InfoPool {
    /// A reset info block for the geometry, recycled when one is available.
    pub fn take(&self, mb_width: usize, mb_height: usize) -> PicInfo {
        let mut g = self.0.lock().unwrap();
        let mut info = match g
            .iter()
            .position(|i| i.mb_width == mb_width && i.mb_height == mb_height)
        {
            Some(i) => g.swap_remove(i),
            None => {
                drop(g);
                PicInfo::new(mb_width, mb_height)
            }
        };
        info.reset();
        info
    }

    /// Return one.
    pub fn give(&self, info: PicInfo) {
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(info);
        }
    }
}

impl PicInfo {
    /// Fresh info arrays for a picture size.
    pub fn new(mb_width: usize, mb_height: usize) -> Self {
        let n = mb_width * mb_height;
        PicInfo {
            mb_width,
            mb_height,
            mbs: vec![MbInfo::default(); n],
            luma_nz: vec![0; n * 16],
            chroma_nz: vec![0; n * 32],
            intra_modes: vec![2; n * 16],
            mvd: [vec![Mv::ZERO; n * 16], vec![Mv::ZERO; n * 16]],
        }
    }
    /// Reset for a new picture.
    pub fn reset(&mut self) {
        for m in &mut self.mbs {
            *m = MbInfo::default();
        }
        self.luma_nz.fill(0);
        self.chroma_nz.fill(0);
        self.intra_modes.fill(2);
        self.mvd[0].fill(Mv::ZERO);
        self.mvd[1].fill(Mv::ZERO);
    }

    /// The nonzero count of 4x4 block `blk` (raster) of colour plane `p` of
    /// macroblock `addr`: luma, or a 4:4:4 chroma plane coded like luma.
    #[inline]
    pub fn plane_nz(&self, p: usize, addr: usize, blk: usize) -> u8 {
        if p == 0 {
            self.luma_nz[addr * 16 + blk]
        } else {
            self.chroma_nz[addr * 32 + (p - 1) * 16 + blk]
        }
    }
}

/// The macroblock neighbours A (left), B (above), C (above-right), D
/// (above-left) of the current macroblock, as addresses when available
/// (decoded and in the same slice).
///
/// Addresses are storage addresses (`frame_row * mb_width + x`). In an
/// MBAFF frame the neighbours of a macroblock depend on whether it and its
/// neighbouring pairs are frame or field pairs (6.4.12.2, Table 6-4); the
/// pair-level neighbours are kept and every lookup goes through
/// [`Self::locate`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MbNeighbours {
    /// Current macroblock address.
    pub addr: usize,
    /// Left (6.4.11.1: the macroblock holding luma sample (-1, 0)).
    pub a: Option<usize>,
    /// Above (holding sample (0, -1)).
    pub b: Option<usize>,
    /// Above-right (holding sample (16, -1)).
    pub c: Option<usize>,
    /// Above-left (holding sample (-1, -1)).
    pub d: Option<usize>,
    /// MBAFF frame?
    pub mbaff: bool,
    /// MBAFF: the current macroblock is a field macroblock.
    pub cur_field: bool,
    /// MBAFF: the current macroblock is the top one of its pair.
    pub is_top: bool,
    /// MBAFF: the top macroblock (storage address) of the neighbouring pairs
    /// A, B, C, D when available, and whether each is a field pair.
    pub pair: [Option<usize>; 4],
    /// See `pair`.
    pub pair_field: [bool; 4],
    /// Picture width in macroblocks (for pair arithmetic).
    pub mb_width: usize,
    /// Neighbouring nonzero-coefficient counts, gathered once (see
    /// [`Self::gather_nz`]): whether the left / top side is available.
    pub nz_avail: [bool; 2],
    /// Per luma-like plane, the count of the block left of block row `r`
    /// (0 for a skipped neighbour, 16 for I_PCM — the stored values).
    pub nz_left: [[u8; 4]; 3],
    /// Per luma-like plane, the count of the block above block column `c`.
    pub nz_top: [[u8; 4]; 3],
    /// Per chroma component (4:2:0 / 4:2:2), the count of the chroma block
    /// left of chroma block row `r` (two or four rows).
    pub nzc_left: [[u8; 4]; 2],
    /// Per chroma component, the count of the chroma block above column `c`.
    pub nzc_top: [[u8; 2]; 2],
}

impl MbNeighbours {
    /// Derive for `addr` (6.4.11.1) with the availability rule of 6.4.9:
    /// a macroblock is available if it is decoded and in the same slice.
    pub fn derive(info: &PicInfo, addr: usize, slice: u16) -> Self {
        let mut nb = MbNeighbours::default();
        nb.derive_into(info, addr, slice);
        nb
    }

    /// [`Self::derive`] into an existing value (the slice decoder keeps one
    /// and refills it per macroblock; the struct is a few hundred bytes and
    /// building it in a temporary was a copy per macroblock). The nonzero
    /// caches are left for [`Self::gather_nz`].
    pub fn derive_into(&mut self, info: &PicInfo, addr: usize, slice: u16) {
        let w = info.mb_width;
        let avail = |a: usize| -> Option<usize> {
            let m = &info.mbs[a];
            if m.decoded && m.slice == slice {
                Some(a)
            } else {
                None
            }
        };
        let x = addr % w;
        let a = if x > 0 { avail(addr - 1) } else { None };
        let b = if addr >= w { avail(addr - w) } else { None };
        let c = if addr >= w && x + 1 < w {
            avail(addr - w + 1)
        } else {
            None
        };
        let d = if addr >= w && x > 0 {
            avail(addr - w - 1)
        } else {
            None
        };
        self.addr = addr;
        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
        self.mbaff = false;
        self.cur_field = false;
        self.is_top = true;
        self.mb_width = w;
    }

    /// Gather the neighbouring nonzero counts CAVLC's nC (9.2.1) and
    /// CABAC's coded_block_flag contexts (9.3.3.1.1.9) read: the blocks
    /// left of each block row and above each block column, for `planes`
    /// luma-like planes (1, or 3 in 4:4:4) and, when `chroma_rows` is 2 or
    /// 4, the 4:2:0 / 4:2:2 chroma blocks. Skipped and I_PCM neighbours
    /// are already stored as 0 / 16 in the picture's arrays; unavailable
    /// sides are flagged in `nz_avail` (`nz_left` / `nz_top` read 0 then).
    pub fn gather_nz(&mut self, info: &PicInfo, planes: usize, chroma_rows: usize) {
        self.nz_avail = [false; 2];
        if !self.mbaff {
            if let Some(a) = self.a {
                self.nz_avail[0] = true;
                for p in 0..planes {
                    for r in 0..4 {
                        self.nz_left[p][r] = info.plane_nz(p, a, r * 4 + 3);
                    }
                }
                if chroma_rows > 0 {
                    for comp in 0..2 {
                        for r in 0..chroma_rows {
                            self.nzc_left[comp][r] = info.chroma_nz[a * 32 + comp * 16 + r * 2 + 1];
                        }
                    }
                }
            }
            if let Some(b) = self.b {
                self.nz_avail[1] = true;
                for p in 0..planes {
                    for c in 0..4 {
                        self.nz_top[p][c] = info.plane_nz(p, b, 12 + c);
                    }
                }
                if chroma_rows > 0 {
                    for comp in 0..2 {
                        for c in 0..2 {
                            self.nzc_top[comp][c] = info.chroma_nz[b * 32 + comp * 16 + (chroma_rows - 1) * 2 + c];
                        }
                    }
                }
            }
            return;
        }
        // MBAFF: each row / column through Table 6-4 (a frame macroblock
        // next to a field pair reads the two macroblocks alternately).
        for r in 0..4 {
            if let Some((addr, blk)) = self.block(-1, r as i32) {
                self.nz_avail[0] = true;
                for p in 0..planes {
                    self.nz_left[p][r] = info.plane_nz(p, addr, blk);
                }
            }
            if let Some((addr, blk)) = self.block(r as i32, -1) {
                self.nz_avail[1] = true;
                for p in 0..planes {
                    self.nz_top[p][r] = info.plane_nz(p, addr, blk);
                }
            }
        }
        if chroma_rows > 0 {
            for r in 0..chroma_rows {
                if let Some((addr, blk)) = self.block_c(-1, r as i32, chroma_rows as i32) {
                    for comp in 0..2 {
                        self.nzc_left[comp][r] = info.chroma_nz[addr * 32 + comp * 16 + blk];
                    }
                }
            }
            for c in 0..2 {
                if let Some((addr, blk)) = self.block_c(c as i32, -1, chroma_rows as i32) {
                    for comp in 0..2 {
                        self.nzc_top[comp][c] = info.chroma_nz[addr * 32 + comp * 16 + blk];
                    }
                }
            }
        }
    }

    /// Derive for the macroblock at storage address `addr` of an MBAFF frame
    /// (6.4.10 for the pairs, 6.4.12.2 for the macroblock-level neighbours),
    /// the current pair being frame / field per `cur_field`.
    pub fn derive_mbaff(info: &PicInfo, addr: usize, slice: u16, cur_field: bool) -> Self {
        let mut nb = MbNeighbours::default();
        nb.derive_mbaff_into(info, addr, slice, cur_field);
        nb
    }

    /// [`Self::derive_mbaff`] into an existing value.
    pub fn derive_mbaff_into(&mut self, info: &PicInfo, addr: usize, slice: u16, cur_field: bool) {
        let w = info.mb_width;
        let x = addr % w;
        let frow = addr / w;
        let pr = frow / 2;
        let is_top = frow % 2 == 0;
        // A pair is available when its top macroblock is decoded in this
        // slice (both macroblocks of a pair are).
        let pair_at = |px: isize, ppr: isize| -> Option<usize> {
            if px < 0 || px >= w as isize || ppr < 0 {
                return None;
            }
            let top = (2 * ppr as usize) * w + px as usize;
            let m = &info.mbs[top];
            if m.decoded && m.slice == slice {
                Some(top)
            } else {
                None
            }
        };
        let pair = [
            pair_at(x as isize - 1, pr as isize),
            pair_at(x as isize, pr as isize - 1),
            pair_at(x as isize + 1, pr as isize - 1),
            pair_at(x as isize - 1, pr as isize - 1),
        ];
        let pair_field = [0, 1, 2, 3].map(|k| pair[k].is_some_and(|t| info.mbs[t].field));
        self.addr = addr;
        self.mbaff = true;
        self.cur_field = cur_field;
        self.is_top = is_top;
        self.pair = pair;
        self.pair_field = pair_field;
        self.mb_width = w;
        self.a = self.locate(-1, 0, 16, 16).map(|(a, _, _)| a);
        self.b = self.locate(0, -1, 16, 16).map(|(a, _, _)| a);
        self.c = self.locate(16, -1, 16, 16).map(|(a, _, _)| a);
        self.d = self.locate(-1, -1, 16, 16).map(|(a, _, _)| a);
    }

    /// The macroblock holding the neighbouring sample `(xn, yn)` (relative
    /// to the current macroblock's top-left, in a `maxw x maxh` component)
    /// and that sample's position in it: `(addr, xW, yW)` (6.4.12).
    /// Positions inside the current macroblock come back as the current
    /// address.
    pub fn locate(&self, xn: i32, yn: i32, maxw: i32, maxh: i32) -> Option<(usize, i32, i32)> {
        if xn > maxw - 1 && yn >= 0 || yn > maxh - 1 {
            return None;
        }
        if !self.mbaff {
            let w = self.mb_width;
            let addr = if yn < 0 {
                if xn < 0 {
                    self.d?
                } else if xn < maxw {
                    self.b?
                } else {
                    self.c?
                }
            } else if xn < 0 {
                self.a?
            } else {
                self.addr
            };
            let _ = w;
            return Some((addr, (xn + maxw) % maxw, (yn + maxh) % maxh));
        }
        // Table 6-4. `pair[k]` is the top macroblock of pair k; `+ 1`
        // (its bottom macroblock) is one frame row down.
        let w = self.mb_width;
        let cf = !self.cur_field;
        let top = self.is_top;
        let (pa, pb, pc, pd) = (self.pair[0], self.pair[1], self.pair[2], self.pair[3]);
        let (fa, fb, fc, fd) = (
            !self.pair_field[0],
            !self.pair_field[1],
            !self.pair_field[2],
            !self.pair_field[3],
        );
        let bot = |t: usize| t + w;
        let (n, ym): (usize, i32) = if xn < 0 && yn < 0 {
            match (cf, top) {
                (true, true) => (bot(pd?), yn),
                (true, false) => {
                    let a = pa?;
                    if fa {
                        (a, yn)
                    } else {
                        (bot(a), (yn + maxh) >> 1)
                    }
                }
                (false, true) => {
                    let d = pd?;
                    if fd { (bot(d), 2 * yn) } else { (d, yn) }
                }
                (false, false) => (bot(pd?), yn),
            }
        } else if xn < 0 {
            let a = pa?;
            match (cf, top) {
                (true, true) => {
                    if fa {
                        (a, yn)
                    } else if yn % 2 == 0 {
                        (a, yn >> 1)
                    } else {
                        (bot(a), yn >> 1)
                    }
                }
                (true, false) => {
                    if fa {
                        (bot(a), yn)
                    } else if yn % 2 == 0 {
                        (a, (yn + maxh) >> 1)
                    } else {
                        (bot(a), (yn + maxh) >> 1)
                    }
                }
                (false, true) => {
                    if fa {
                        if yn < maxh / 2 {
                            (a, yn << 1)
                        } else {
                            (bot(a), (yn << 1) - maxh)
                        }
                    } else {
                        (a, yn)
                    }
                }
                (false, false) => {
                    if fa {
                        if yn < maxh / 2 {
                            (a, (yn << 1) + 1)
                        } else {
                            (bot(a), (yn << 1) + 1 - maxh)
                        }
                    } else {
                        (bot(a), yn)
                    }
                }
            }
        } else if xn < maxw {
            if yn < 0 {
                match (cf, top) {
                    (true, true) => (bot(pb?), yn),
                    (true, false) => (self.addr - w, yn),
                    (false, true) => {
                        let b = pb?;
                        if fb { (bot(b), 2 * yn) } else { (b, yn) }
                    }
                    (false, false) => (bot(pb?), yn),
                }
            } else {
                (self.addr, yn)
            }
        } else {
            // xn > maxw - 1, yn < 0
            match (cf, top) {
                (true, true) => (bot(pc?), yn),
                (true, false) => return None,
                (false, true) => {
                    let c = pc?;
                    if fc { (bot(c), 2 * yn) } else { (c, yn) }
                }
                (false, false) => (bot(pc?), yn),
            }
        };
        Some((n, (xn + maxw) % maxw, (ym + maxh) % maxh))
    }

    /// The 4x4-block neighbour at 4x4-unit offset `(bx + dx, by + dy)` of a
    /// block in the current macroblock (6.4.12), as `(mb_addr, raster 4x4
    /// index in that macroblock)`. `dx` in `-1..=4`, `dy` in `-1..=3`.
    /// Positions inside the current macroblock come back as the current
    /// address (whether they are decoded yet is the caller's business —
    /// see [`block_available`]).
    #[inline]
    pub fn block(&self, bx: i32, by: i32) -> Option<(usize, usize)> {
        if !self.mbaff {
            return if by < 0 {
                if bx < 0 {
                    self.d.map(|a| (a, 15))
                } else if bx < 4 {
                    self.b.map(|a| (a, (12 + bx) as usize))
                } else {
                    self.c.map(|a| (a, 12))
                }
            } else if by < 4 {
                if bx < 0 {
                    self.a.map(|a| (a, (by * 4 + 3) as usize))
                } else if bx < 4 {
                    Some((self.addr, (by * 4 + bx) as usize))
                } else {
                    None
                }
            } else {
                None
            };
        }
        // The neighbouring sample of the block: left of / above its top-left.
        let xn = if bx < 0 {
            -1
        } else if bx > 3 {
            16
        } else {
            bx * 4
        };
        let yn = if by < 0 { -1 } else { by * 4 };
        let (addr, xw, yw) = self.locate(xn, yn, 16, 16)?;
        Some((addr, ((yw / 4) * 4 + xw / 4) as usize))
    }

    /// The chroma 4x4-block neighbour of block `(bx, by)` of a
    /// two-column, `rows`-row chroma grid (4:2:0 / 4:2:2) at offset
    /// `(dx, dy)`, as `(mb_addr, chroma block index)` (6.4.11.5).
    #[inline]
    pub fn block_c(&self, bx: i32, by: i32, rows: i32) -> Option<(usize, usize)> {
        let (maxw, maxh) = (8, rows * 4);
        let xn = if bx < 0 { -1 } else { bx * 4 };
        let yn = if by < 0 { -1 } else { by * 4 };
        let (addr, xw, yw) = self.locate(xn, yn, maxw, maxh)?;
        Some((addr, ((yw / 4) * 2 + xw / 4) as usize))
    }

    /// The macroblocks the left column of luma rows `y0..y1` of the current
    /// macroblock borders (one, or the two of a neighbouring pair of the
    /// other frame / field kind in an MBAFF frame): for intra availability
    /// of the whole column.
    pub fn left_mbs(&self, y0: i32, y1: i32) -> (Option<usize>, Option<usize>) {
        if !self.mbaff {
            return (self.a, self.a);
        }
        let first = self.locate(-1, y0, 16, 16).map(|(a, _, _)| a);
        let last = self.locate(-1, y1 - 1, 16, 16).map(|(a, _, _)| a);
        // Mixed pairs alternate rows: sample the second row too.
        let second = self
            .locate(-1, (y0 + 1).min(y1 - 1), 16, 16)
            .map(|(a, _, _)| a);
        if first != last {
            (first, last)
        } else {
            (first, second)
        }
    }
}

/// Availability of a 4x4 block at raster `(bx, by)` inside the *current*
/// macroblock given which blocks have already been decoded (`done` is a
/// bitmask over the 16 raster positions).
#[inline]
pub fn block_available(done: u16, bx: i32, by: i32) -> bool {
    (0..4).contains(&bx) && (0..4).contains(&by) && (done >> (by * 4 + bx)) & 1 != 0
}

// ---------------------------------------------------------------------------
// Motion vector prediction (8.4.1.3)
// ---------------------------------------------------------------------------

/// A neighbour's motion for one list, for prediction. Eight aligned bytes,
/// so the caches of them move as whole words.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
pub struct NbMotion {
    /// Motion vector (zero when unavailable).
    pub mv: Mv,
    /// Reference index (-1 when unavailable or intra / not using the list).
    pub ref_idx: i8,
    /// Available (partition exists and is decoded)?
    pub avail: bool,
}

impl NbMotion {
    /// The unavailable neighbour.
    pub const NONE: NbMotion = NbMotion {
        avail: false,
        ref_idx: -1,
        mv: Mv::ZERO,
    };
}


/// The motion of the block holding the neighbouring luma sample `(xn, yn)`
/// (relative to the current macroblock's top-left) for one list, as
/// prediction needs it — the form 6.4.11.7 uses, which matters in an MBAFF
/// frame where the row's parity picks the macroblock of a neighbouring pair
/// of the other kind. The general derivation; [`MotionCache`] gathers its
/// results once per macroblock.
pub fn neighbour_motion_at<S: Sample>(
    nb: &MbNeighbours,
    frame: &Frame<S>,
    info: &PicInfo,
    done: u16,
    list: usize,
    xn: i32,
    yn: i32,
) -> NbMotion {
    let Some((addr, xw, yw)) = nb.locate(xn, yn, 16, 16) else {
        return NbMotion::NONE;
    };
    let blk = ((yw / 4) * 4 + xw / 4) as usize;
    if addr == nb.addr && !block_available(done, xn / 4, yn / 4) {
        return NbMotion::NONE;
    }
    if info.mbs[addr].kind.is_intra() && addr != nb.addr {
        // Intra neighbour: available, but "not used for inter prediction":
        // refIdx -1, mv 0.
        return NbMotion {
            avail: true,
            ref_idx: -1,
            mv: Mv::ZERO,
        };
    }
    let m = frame.motion[list][addr * 16 + blk];
    let mut n = NbMotion {
        avail: true,
        ref_idx: m.ref_idx,
        mv: m.mv,
    };
    // MBAFF: a neighbour of the other kind is brought into the current
    // macroblock's units (8.4.1.3.1) — field from frame: vector halved,
    // index doubled; frame from field: the reverse.
    if nb.mbaff && addr != nb.addr && n.ref_idx >= 0 {
        let nf = info.mbs[addr].field;
        if nb.cur_field && !nf {
            n.mv.y /= 2;
            n.ref_idx *= 2;
        } else if !nb.cur_field && nf {
            n.mv.y = n.mv.y.wrapping_mul(2);
            n.ref_idx >>= 1;
        }
    }
    n
}

/// The neighbouring motion a macroblock's partitions predict from,
/// gathered once per macroblock so the many lookups of 8.4.1.3.2 (A, B, C
/// and D of every partition, in both lists) are array reads: the blocks
/// left of luma rows 0 / 4 / 8 / 12 (A of each partition row) and 3 / 7 /
/// 11 (D of the partitions below the first row — a different macroblock in
/// an MBAFF frame), and above columns 0 / 4 / 8 / 12 (B), 16 (C of the last
/// column) and -1 (D of the first row). Positions inside the current
/// macroblock are read live (they are still being decoded).
#[derive(Debug, Clone, Copy)]
pub struct MotionCache {
    /// Per list: rows 0, 4, 8, 12 at 0..4; rows 3, 7, 11 at 4..7.
    left: [[NbMotion; 8]; 2],
    /// Per list: columns 0, 4, 8, 12 at 0..4; column 16 at 4; column -1 at 5.
    top: [[NbMotion; 6]; 2],
}

impl Default for MotionCache {
    fn default() -> Self {
        MotionCache { left: [[NbMotion::NONE; 8]; 2], top: [[NbMotion::NONE; 6]; 2] }
    }
}

impl MotionCache {
    /// Gather the neighbours of macroblock `nb` from `frame`'s motion into
    /// `self` (in place: the cache lives in the slice decoder's scratch).
    pub fn gather<S: Sample>(&mut self, nb: &MbNeighbours, frame: &Frame<S>, info: &PicInfo) {
        let c = self;
        if nb.mbaff {
            // Table 6-4 neighbours, through the general derivation.
            for list in 0..2 {
                for r in 0..4 {
                    c.left[list][r] = neighbour_motion_at(nb, frame, info, 0, list, -1, r as i32 * 4);
                    c.top[list][r] = neighbour_motion_at(nb, frame, info, 0, list, r as i32 * 4, -1);
                }
                for r in 0..3 {
                    c.left[list][4 + r] = neighbour_motion_at(nb, frame, info, 0, list, -1, r as i32 * 4 + 3);
                }
                c.top[list][4] = neighbour_motion_at(nb, frame, info, 0, list, 16, -1);
                c.top[list][5] = neighbour_motion_at(nb, frame, info, 0, list, -1, -1);
            }
            return;
        }
        // An intra neighbour is available but "not used for inter
        // prediction": refIdx -1, mv 0.
        const INTRA: NbMotion = NbMotion { avail: true, ref_idx: -1, mv: Mv::ZERO };
        let of = |list: usize, addr: usize, blk: usize| -> NbMotion {
            let m = frame.motion[list][addr * 16 + blk];
            NbMotion { avail: true, ref_idx: m.ref_idx, mv: m.mv }
        };
        // Each side is written whole: its entries, or NONE when unavailable.
        match nb.a {
            Some(a) => {
                let intra = info.mbs[a].kind.is_intra();
                for list in 0..2 {
                    for r in 0..4 {
                        c.left[list][r] = if intra { INTRA } else { of(list, a, r * 4 + 3) };
                    }
                    // Rows 3, 7, 11 lie in the same left blocks as rows 0, 4, 8.
                    for r in 0..3 {
                        c.left[list][4 + r] = c.left[list][r];
                    }
                }
            }
            None => c.left = [[NbMotion::NONE; 8]; 2],
        }
        match nb.b {
            Some(b) => {
                let intra = info.mbs[b].kind.is_intra();
                for list in 0..2 {
                    for x in 0..4 {
                        c.top[list][x] = if intra { INTRA } else { of(list, b, 12 + x) };
                    }
                }
            }
            None => {
                for list in 0..2 {
                    for x in 0..4 {
                        c.top[list][x] = NbMotion::NONE;
                    }
                }
            }
        }
        for list in 0..2 {
            c.top[list][4] = match nb.c {
                Some(cc) => {
                    if info.mbs[cc].kind.is_intra() { INTRA } else { of(list, cc, 12) }
                }
                None => NbMotion::NONE,
            };
            c.top[list][5] = match nb.d {
                Some(d) => {
                    if info.mbs[d].kind.is_intra() { INTRA } else { of(list, d, 15) }
                }
                None => NbMotion::NONE,
            };
        }
    }

    /// The motion of the block holding luma sample `(xn, yn)` relative to
    /// the current macroblock (6.4.11.7): a cached neighbour outside it, a
    /// live read inside it (available once `done` has the block).
    #[inline(always)]
    pub fn at<S: Sample>(&self, nb: &MbNeighbours, frame: &Frame<S>, done: u16, list: usize, xn: i32, yn: i32) -> NbMotion {
        if yn < 0 {
            if xn < 0 {
                self.top[list][5]
            } else if xn < 16 {
                self.top[list][(xn >> 2) as usize]
            } else {
                self.top[list][4]
            }
        } else if xn < 0 {
            if yn & 3 == 0 { self.left[list][(yn >> 2) as usize] } else { self.left[list][4 + (yn >> 2) as usize] }
        } else if xn < 16 && block_available(done, xn >> 2, yn >> 2) {
            let m = frame.motion[list][nb.addr * 16 + ((yn >> 2) * 4 + (xn >> 2)) as usize];
            NbMotion { avail: true, ref_idx: m.ref_idx, mv: m.mv }
        } else {
            NbMotion::NONE
        }
    }
}

/// The three neighbours used for a partition's motion vector prediction:
/// A (left of the top-left block), B (above the top-left block), C (above
/// the block right of the partition's top-right block, falling back to D
/// (above-left of the top-left block) when C is unavailable) — 8.4.1.3.2.
/// `(bx, by)` is the partition's top-left 4x4 block, `w4` its width in 4x4
/// units.
pub fn prediction_neighbours<S: Sample>(
    cache: &MotionCache,
    nb: &MbNeighbours,
    frame: &Frame<S>,
    done: u16,
    list: usize,
    bx: i32,
    by: i32,
    w4: i32,
) -> (NbMotion, NbMotion, NbMotion) {
    // The samples (x - 1, y), (x, y - 1), (x + w, y - 1) and (x - 1, y - 1)
    // of the partition at (x, y) = (4 bx, 4 by) (6.4.11.7).
    let (x, y) = (bx * 4, by * 4);
    let a = cache.at(nb, frame, done, list, x - 1, y);
    let b = cache.at(nb, frame, done, list, x, y - 1);
    let mut c = cache.at(nb, frame, done, list, x + w4 * 4, y - 1);
    if !c.avail {
        c = cache.at(nb, frame, done, list, x - 1, y - 1);
    }
    (a, b, c)
}

/// Median motion vector prediction (8.4.1.3.1) given the three neighbours,
/// with the "only A available" and "exactly one matching reference" rules.
pub fn median_mvp(a: NbMotion, mut b: NbMotion, mut c: NbMotion, ref_idx: i8) -> Mv {
    if !b.avail && !c.avail && a.avail {
        b = a;
        c = a;
    }
    let ma = a.ref_idx == ref_idx;
    let mb = b.ref_idx == ref_idx;
    let mc = c.ref_idx == ref_idx;
    match (ma, mb, mc) {
        (true, false, false) => a.mv,
        (false, true, false) => b.mv,
        (false, false, true) => c.mv,
        _ => Mv::new(
            median3(a.mv.x, b.mv.x, c.mv.x),
            median3(a.mv.y, b.mv.y, c.mv.y),
        ),
    }
}

#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// Motion vector prediction for a partition of the given shape (8.4.1.3):
/// directional for 16x8 / 8x16, median otherwise. `part_w`/`part_h` in
/// samples, `(x, y)` the partition's top-left in samples within the MB.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv<S: Sample>(
    cache: &MotionCache,
    nb: &MbNeighbours,
    frame: &Frame<S>,
    done: u16,
    list: usize,
    ref_idx: i8,
    x: usize,
    y: usize,
    part_w: usize,
    part_h: usize,
) -> Mv {
    let bx = (x / 4) as i32;
    let by = (y / 4) as i32;
    let w4 = (part_w / 4) as i32;
    let (a, b, c) = prediction_neighbours(cache, nb, frame, done, list, bx, by, w4);
    if part_w == 16 && part_h == 8 {
        if y == 0 {
            if b.ref_idx == ref_idx {
                return b.mv;
            }
        } else if a.ref_idx == ref_idx {
            return a.mv;
        }
    } else if part_w == 8 && part_h == 16 {
        if x == 0 {
            if a.ref_idx == ref_idx {
                return a.mv;
            }
        } else if c.ref_idx == ref_idx {
            return c.mv;
        }
    }
    median_mvp(a, b, c, ref_idx)
}

/// P_Skip motion (8.4.1.1): reference index 0 and either the zero vector or
/// the 16x16 median prediction.
pub fn p_skip_mv<S: Sample>(cache: &MotionCache, nb: &MbNeighbours, frame: &Frame<S>) -> Mv {
    let a = cache.at(nb, frame, 0, 0, -1, 0);
    let b = cache.at(nb, frame, 0, 0, 0, -1);
    if !a.avail
        || !b.avail
        || (a.ref_idx == 0 && a.mv == Mv::ZERO)
        || (b.ref_idx == 0 && b.mv == Mv::ZERO)
    {
        return Mv::ZERO;
    }
    predict_mv(cache, nb, frame, 0, 0, 0, 0, 0, 16, 16)
}

/// Write `motion` for list `list` into the 4x4 blocks of the rectangle
/// `(x, y, w, h)` (samples within the macroblock) of macroblock `addr`.
pub fn fill_motion<S: Sample>(
    frame: &mut Frame<S>,
    addr: usize,
    list: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    motion: BlockMotion,
) {
    for by in y / 4..(y + h) / 4 {
        for bx in x / 4..(x + w) / 4 {
            frame.motion[list][addr * 16 + by * 4 + bx] = motion;
        }
    }
}

/// The reference index of the neighbour, or -1, treated the way
/// `MinPositive` needs it.
#[inline]
fn min_positive(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 { a.min(b) } else { a.max(b) }
}

/// Reference indices for spatial direct prediction (8.4.1.2.2): the
/// `MinPositive` over the whole-macroblock neighbours A, B, C for each list.
pub fn spatial_direct_ref_idx<S: Sample>(
    cache: &MotionCache,
    nb: &MbNeighbours,
    frame: &Frame<S>,
) -> [i8; 2] {
    let mut out = [0i8; 2];
    for list in 0..2 {
        let (a, b, c) = prediction_neighbours(cache, nb, frame, 0, list, 0, 0, 4);
        out[list] = min_positive(a.ref_idx, min_positive(b.ref_idx, c.ref_idx));
    }
    out
}

/// The 4x4 raster index of the colocated block for direct prediction of
/// 8x8 partition `part` / sub-partition `sub` (8.4.1.2.1): with
/// `direct_8x8_inference` the corner 4x4 of the 8x8 (0, 3, 12, 15 in raster),
/// else the actual sub-block.
pub fn colocated_block(direct_8x8_inference: bool, part: usize, sub: usize) -> usize {
    let px = (part & 1) * 2;
    let py = (part >> 1) * 2;
    if direct_8x8_inference {
        // Corner: for part 0 the top-left 4x4 (0,0); part 1 top-right (3,0);
        // part 2 bottom-left (0,3); part 3 bottom-right (3,3).
        let cx = if part & 1 == 0 { 0 } else { 3 };
        let cy = if part >> 1 == 0 { 0 } else { 3 };
        cy * 4 + cx
    } else {
        let sx = px + (sub & 1);
        let sy = py + (sub >> 1);
        sy * 4 + sx
    }
}

/// The colocated motion for direct prediction: `(mvCol, refIdxCol, the
/// referenced picture's (id, parity))`; intra colocated → refIdx -1.
pub fn colocated_motion<S: Sample>(col: &Frame<S>, addr: usize, blk: usize) -> (Mv, i8, u16, u8) {
    if col.mb_intra[addr] {
        return (Mv::ZERO, -1, 0, super::frame::PARITY_NONE);
    }
    let m0 = col.motion[0][addr * 16 + blk];
    if m0.ref_idx >= 0 {
        return (m0.mv, m0.ref_idx, m0.ref_id, m0.ref_parity);
    }
    let m1 = col.motion[1][addr * 16 + blk];
    (m1.mv, m1.ref_idx, m1.ref_id, m1.ref_parity)
}

/// The raster 4x4 index from the standard's `luma4x4BlkIdx`.
#[inline]
pub fn raster_of_blk(blk: usize) -> usize {
    (BLK4X4_Y[blk] as usize) * 4 + BLK4X4_X[blk] as usize
}

/// `QPC` from `QPY` and the chroma QP offset (8.5.8 / Table 8-15): `qPI`
/// clips at `-QpBdOffsetC` below (a negative `qPI` maps to itself).
pub(crate) fn chroma_qp(qp: i32, offset: i32, qp_bd_offset: i32) -> i32 {
    let qpi = (qp + offset).clamp(-qp_bd_offset, 51);
    if qpi < 0 {
        qpi
    } else {
        super::tables::CHROMA_QP[qpi as usize] as i32
    }
}

/// `QP_Y` of a macroblock from the previous one's and its `mb_qp_delta`
/// (7.4.5: wraps in `-QpBdOffsetY..=51`).
#[inline]
pub(crate) fn next_qp(prev_qp: i32, qp_delta: i32, bit_depth: u32) -> i32 {
    let bd_off = 6 * (bit_depth as i32 - 8);
    ((prev_qp + qp_delta + 52 + 2 * bd_off) % (52 + bd_off)) - bd_off
}

/// The scaling a coded macroblock's parser applies to each coefficient as
/// it is written (8.5.12.1 / 8.5.13.1 folded into one multiply, add and
/// shift: `(c * LevelScale << shift + 32) >> 6`, exact for every QP): per
/// colour plane the 4x4 and 8x8 tables (raster) and shifts. Plane 0 is
/// luma (or the plane a separate-colour-plane picture is decoding), 1 / 2
/// are Cb / Cr — the 4:4:4 planes coded like luma, or the 4:2:0 / 4:2:2
/// chroma AC blocks. `None` for a lossless (transform bypass) macroblock,
/// whose levels are the residual.
pub struct MbDequant<'a> {
    /// `(table, shift)` per plane for 4x4 blocks.
    pub q4: [(&'a [i32; 16], u32); 3],
    /// The same for 8x8 blocks.
    pub q8: [(&'a [i32; 64], u32); 3],
}

impl<'a> MbDequant<'a> {
    /// The tables for a macroblock of `kind` at `QP_Y = qp` in a slice with
    /// `chroma_offset` (the PPS chroma QP offsets); `None` when it is
    /// lossless.
    pub fn for_mb(dq: &'a super::transform::Dequant, ctx: &SliceCtx, chroma_offset: [i32; 2], kind: MbKind, qp: i32) -> Option<Self> {
        let bd_off = 6 * (ctx.bit_depth as i32 - 8);
        // The primed QPs the scaling uses.
        let qps = [qp + bd_off, chroma_qp(qp, chroma_offset[0], bd_off) + bd_off, chroma_qp(qp, chroma_offset[1], bd_off) + bd_off];
        if ctx.transform_bypass && qps[0] == 0 {
            return None;
        }
        let inter = !kind.is_intra();
        let mut q4 = [(&dq.scale4[0][0], 0u32); 3];
        let mut q8 = [(&dq.scale8[0][0], 0u32); 3];
        // Only the luma-like plane exists for monochrome and for a colour
        // plane coded on its own (whose scaling lists are picked by
        // `scaling_plane`); the chroma entries stay dummies then.
        let planes = if ctx.chroma_format_idc == 0 { 1 } else { 3 };
        for p in 0..planes {
            let q = qps[p];
            let list = p + ctx.scaling_plane;
            q4[p] = (&dq.scale4[list + if inter { 3 } else { 0 }][(q % 6) as usize], (q / 6 + 2) as u32);
            q8[p] = (&dq.scale8[2 * list + inter as usize][(q % 6) as usize], (q / 6) as u32);
        }
        Some(MbDequant { q4, q8 })
    }
}

/// One coefficient scaled: `(level * table << shift + 32) >> 6` (wrapping,
/// so a malformed level cannot panic; a conforming stream stays in range).
#[inline(always)]
pub(crate) fn dequant_level(level: i32, table: i32, shift: u32) -> i32 {
    (level.wrapping_mul(table).wrapping_shl(shift).wrapping_add(32)) >> 6
}

/// `H26X_TRACE_IPM`: trace syntax elements (macroblock starts, intra
/// modes, coded_block_flags, ref_idx, mvd, CAVLC blocks) to stderr, for
/// lining a desync up against a reference decoder's trace.
#[inline]
pub(crate) fn syntax_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("H26X_TRACE_IPM").is_some())
}

/// Slice-level facts the parsers need at every macroblock.
#[derive(Debug, Clone, Copy)]
pub struct SliceCtx {
    /// Slice type.
    pub slice_type: SliceType,
    /// Slice index (for neighbour availability).
    pub slice_num: u16,
    /// Active reference count per list.
    pub num_ref_idx: [u32; 2],
    /// `direct_spatial_mv_pred_flag`.
    pub direct_spatial: bool,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference: bool,
    /// Chroma format idc.
    pub chroma_format_idc: u32,
    /// Whether the slice is CABAC-coded (mvd bookkeeping is only for it).
    pub cabac: bool,
    /// Sample bit depth (luma; chroma is the same in every stream accepted).
    pub bit_depth: u32,
    /// `qpprime_y_zero_transform_bypass_flag`: macroblocks at QP'Y 0 are
    /// lossless (no scaling or transform, 8.5.15's DPCM for H/V intra).
    pub transform_bypass: bool,
    /// `iYCbCr` of the scaling lists (8.5.9): `colour_plane_id` when the
    /// picture is coded as separate colour planes, else 0 (each 4:4:4
    /// plane's index is added on top).
    pub scaling_plane: usize,
    /// Reproduce x264's (builds before 151) 4:4:4 CABAC 8x8
    /// coded_block_flag context bug: a neighbour that is not 8x8-transformed
    /// counts as coded for an intra macroblock and uncoded for an inter one.
    pub x264_old_444: bool,
    /// The slice belongs to a field picture (`field_pic_flag`): field scans
    /// and the field-coded CABAC contexts.
    pub field_pic: bool,
    /// `MbaffFrameFlag`: macroblock pairs, each frame or field.
    pub mbaff: bool,
}
