//! Per-picture decoding state: block-level side data at 4x4 granularity,
//! CTB-level slice/tile membership, the z-scan order tables (6.5.2), SAO
//! parameters and the per-slice loop-filter parameters.

use super::pps::Pps;
use super::sps::Sps;

/// SAO parameters of one CTB for one component.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaoParams {
    /// `SaoTypeIdx`: 0 off, 1 band, 2 edge.
    pub type_idx: u8,
    /// `sao_band_position` (band) or `SaoEoClass` (edge).
    pub band_or_class: u8,
    /// `SaoOffsetVal[1..=4]` (already shifted by `log2OffsetScale`).
    pub offsets: [i16; 4],
}

/// Per-slice loop filter facts, indexed by the slice number stored per CTB.
#[derive(Debug, Clone, Copy, Default)]
pub struct SliceFilterParams {
    /// `slice_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `slice_beta_offset_div2 * 2`.
    pub beta_offset: i32,
    /// `slice_tc_offset_div2 * 2`.
    pub tc_offset: i32,
    /// `slice_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// `pps_cb_qp_offset`, `pps_cr_qp_offset` (chroma deblocking uses the PPS
    /// offsets, not the slice ones).
    pub cb_qp_offset: i32,
    /// See `cb_qp_offset`.
    pub cr_qp_offset: i32,
}

/// The scan-order and tile tables of an SPS/PPS pair (6.5.1, 6.5.2):
/// computed once per parameter-set pair and shared by every picture that
/// uses it (`PicInfo` derefs to it, so `info.min_tb_addr_zs` reads as before).
pub struct Geometry {
    /// Width / height in 4x4 units.
    pub w4: usize,
    /// See `w4`.
    pub h4: usize,
    /// Width / height in CTBs.
    pub wc: usize,
    /// See `wc`.
    pub hc: usize,
    /// `CtbLog2SizeY`.
    pub log2_ctb: u32,
    /// Per CTB: tile id.
    pub ctb_tile: Vec<u16>,
    /// `MinTbAddrZs` per 4x4.
    pub min_tb_addr_zs: Vec<u32>,
    /// `CtbAddrRsToTs`.
    pub ctb_rs_to_ts: Vec<u32>,
    /// `CtbAddrTsToRs`.
    pub ctb_ts_to_rs: Vec<u32>,
    /// `TileId` per CTB in TS order.
    pub tile_id_ts: Vec<u16>,
}

/// The per-picture arrays (plus the shared geometry tables).
pub struct PicInfo {
    /// The scan / tile tables.
    pub geo: std::sync::Arc<Geometry>,
    /// Per 4x4: 1 = intra CU, 0 = inter, 2 = not decoded yet.
    pub pred_mode: Vec<u8>,
    /// Per 4x4: `cu_skip_flag`.
    pub skip: Vec<u8>,
    /// Per 4x4: `CtDepth`.
    pub ct_depth: Vec<u8>,
    /// Per 4x4: luma intra prediction mode (1 = DC where not intra).
    pub intra_mode: Vec<u8>,
    /// Per 4x4: `QpY` of the CU.
    pub qp_y: Vec<i8>,
    /// Per 4x4: bit 0 = samples are exempt from the loop filters (pcm with
    /// `pcm_loop_filter_disabled_flag`, or `cu_transquant_bypass`); bit 1 =
    /// `cu_transquant_bypass_flag` (for the deblocking nDp/nDq rule too).
    pub filter_exempt: Vec<u8>,
    /// Per 4x4: the luma transform block covering it has nonzero coefficients.
    pub cbf_luma: Vec<u8>,
    /// Per 4x4: bit 0 = a transform block edge lies on this block's left side,
    /// bit 1 = a prediction block edge does; bits 2/3 the same for the top.
    pub edges: Vec<u8>,
    /// Per CTB: index into `slices` (u16::MAX = not decoded).
    pub ctb_slice: Vec<u16>,
    /// Per CTB: `SliceAddrRs` (u32::MAX = not decoded).
    pub ctb_slice_addr: Vec<u32>,
    /// SAO parameters per CTB per component.
    pub sao: Vec<[SaoParams; 3]>,
    /// Per slice (by the index stored in `ctb_slice`): its filter
    /// parameters. Fixed size (one per CTB at most), filled as slices arrive.
    pub slices: Vec<SliceFilterParams>,
}

impl std::ops::Deref for PicInfo {
    type Target = Geometry;
    fn deref(&self) -> &Geometry {
        &self.geo
    }
}

impl Geometry {
    /// Build the tables for an SPS/PPS pair.
    pub fn new(sps: &Sps, pps: &Pps) -> Self {
        let w4 = (sps.width as usize).div_ceil(4);
        let h4 = (sps.height as usize).div_ceil(4);
        let wc = sps.pic_width_in_ctbs() as usize;
        let hc = sps.pic_height_in_ctbs() as usize;
        let n4 = w4 * h4;
        let nc = wc * hc;
        let log2_ctb = sps.log2_ctb_size;

        // 6.5.1: CtbAddrRsToTs / TsToRs / TileId.
        let cols = pps.col_bd.len() - 1;
        let rows = pps.row_bd.len() - 1;
        let mut rs_to_ts = vec![0u32; nc];
        let mut ts_to_rs = vec![0u32; nc];
        let mut tile_id_ts = vec![0u16; nc];
        for ctb_rs in 0..nc {
            let tbx = ctb_rs % wc;
            let tby = ctb_rs / wc;
            let mut tile_x = 0;
            for i in 0..cols {
                if tbx >= pps.col_bd[i] as usize {
                    tile_x = i;
                }
            }
            let mut tile_y = 0;
            for j in 0..rows {
                if tby >= pps.row_bd[j] as usize {
                    tile_y = j;
                }
            }
            let mut ts = 0usize;
            for i in 0..tile_x {
                ts += (pps.row_bd[tile_y + 1] - pps.row_bd[tile_y]) as usize * (pps.col_bd[i + 1] - pps.col_bd[i]) as usize;
            }
            for j in 0..tile_y {
                ts += wc * (pps.row_bd[j + 1] - pps.row_bd[j]) as usize;
            }
            ts += (tby - pps.row_bd[tile_y] as usize) * (pps.col_bd[tile_x + 1] - pps.col_bd[tile_x]) as usize
                + tbx
                - pps.col_bd[tile_x] as usize;
            rs_to_ts[ctb_rs] = ts as u32;
            ts_to_rs[ts] = ctb_rs as u32;
            tile_id_ts[ts] = (tile_y * cols + tile_x) as u16;
        }
        // 6.5.2: MinTbAddrZs at 4x4 granularity (log2 min tb = 2 here; the
        // standard's table is at MinTbLog2SizeY, and z-order at a finer
        // granularity is a refinement of the same order).
        let shift = log2_ctb - 2;
        let mut min_tb_addr_zs = vec![0u32; n4];
        for y in 0..h4 {
            for x in 0..w4 {
                let tb_x = (x << 2) >> log2_ctb;
                let tb_y = (y << 2) >> log2_ctb;
                let ctb_rs = tb_y * wc + tb_x;
                let mut v = rs_to_ts[ctb_rs] << (2 * shift);
                for i in 0..shift {
                    let m = 1usize << i;
                    let mut add = 0u32;
                    if m & x != 0 {
                        add += (m * m) as u32;
                    }
                    if m & y != 0 {
                        add += (2 * m * m) as u32;
                    }
                    v += add;
                }
                min_tb_addr_zs[y * w4 + x] = v;
            }
        }
        let mut ctb_tile = vec![0u16; nc];
        for rs in 0..nc {
            ctb_tile[rs] = tile_id_ts[rs_to_ts[rs] as usize];
        }
        Geometry { w4, h4, wc, hc, log2_ctb, ctb_tile, min_tb_addr_zs, ctb_rs_to_ts: rs_to_ts, ctb_ts_to_rs: ts_to_rs, tile_id_ts }
    }
}

/// Recycled per-picture side data: `PicInfo` is close to half a megabyte of
/// arrays for a 720p picture, so allocating and zeroing a fresh one per
/// picture costs page faults and a memset that pictures leaving the DPB can
/// pay for one another instead. Mirrors `FramePool`.
#[derive(Clone, Default)]
pub struct PicInfoPool(std::sync::Arc<std::sync::Mutex<Vec<PicInfo>>>);

impl PicInfoPool {
    /// Side data for `geo`, recycled if the pool holds one built for exactly
    /// that geometry (its arrays are reset to the same values `new` gives).
    pub fn take(&self, geo: std::sync::Arc<Geometry>) -> PicInfo {
        let mut g = self.0.lock().unwrap();
        if let Some(i) = g.iter().position(|p| std::sync::Arc::ptr_eq(&p.geo, &geo)) {
            let mut info = g.swap_remove(i);
            drop(g);
            info.reset();
            return info;
        }
        drop(g);
        PicInfo::new(geo)
    }

    /// Return side data (an emptied shell holds no arrays worth keeping).
    pub fn give(&self, info: PicInfo) {
        if info.pred_mode.is_empty() {
            return;
        }
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(info);
        }
    }
}


/// The current block's side of a z-scan availability test, as
/// `PicInfo::avail_ctx` derives it.
#[derive(Clone, Copy)]
pub struct AvailCtx {
    /// `MinTbAddrZs` of the current block.
    zs: u32,
    /// The current block's CTB, and that CTB's slice address and tile.
    ctb: usize,
    slice_addr: u32,
    tile: u16,
    /// Picture bounds.
    pic_w: i32,
    pic_h: i32,
}

impl PicInfo {
    /// Fresh per-picture arrays over shared geometry.
    pub fn new(geo: std::sync::Arc<Geometry>) -> Self {
        let n4 = geo.w4 * geo.h4;
        let nc = geo.wc * geo.hc;
        PicInfo {
            geo,
            pred_mode: vec![2; n4],
            skip: vec![0; n4],
            ct_depth: vec![0; n4],
            intra_mode: vec![1; n4],
            qp_y: vec![0; n4],
            filter_exempt: vec![0; n4],
            cbf_luma: vec![0; n4],
            edges: vec![0; n4],
            ctb_slice: vec![u16::MAX; nc],
            ctb_slice_addr: vec![u32::MAX; nc],
            sao: vec![[SaoParams::default(); 3]; nc],
            slices: vec![SliceFilterParams::default(); nc],
        }
    }

    /// A shell holding nothing but the geometry, to leave behind when the
    /// arrays are handed to the pool.
    pub fn empty(geo: std::sync::Arc<Geometry>) -> Self {
        PicInfo {
            geo,
            pred_mode: Vec::new(),
            skip: Vec::new(),
            ct_depth: Vec::new(),
            intra_mode: Vec::new(),
            qp_y: Vec::new(),
            filter_exempt: Vec::new(),
            cbf_luma: Vec::new(),
            edges: Vec::new(),
            ctb_slice: Vec::new(),
            ctb_slice_addr: Vec::new(),
            sao: Vec::new(),
            slices: Vec::new(),
        }
    }

    /// Reset the per-picture (not per-sequence) arrays.
    pub fn reset(&mut self) {
        self.pred_mode.fill(2);
        self.skip.fill(0);
        self.ct_depth.fill(0);
        self.intra_mode.fill(1);
        self.qp_y.fill(0);
        self.filter_exempt.fill(0);
        self.cbf_luma.fill(0);
        self.edges.fill(0);
        self.ctb_slice.fill(u16::MAX);
        self.ctb_slice_addr.fill(u32::MAX);
        for s in &mut self.sao {
            *s = [SaoParams::default(); 3];
        }
        self.slices.fill(SliceFilterParams::default());
    }

    /// Index of the 4x4 block containing luma sample (x, y).
    #[inline(always)]
    pub fn idx4(&self, x: usize, y: usize) -> usize {
        (y >> 2) * self.w4 + (x >> 2)
    }

    /// CTB raster address of luma sample (x, y).
    #[inline(always)]
    pub fn ctb_of(&self, x: usize, y: usize) -> usize {
        (y >> self.log2_ctb) * self.wc + (x >> self.log2_ctb)
    }

    /// The current block's half of a z-scan availability test (6.4.1). A
    /// block asks about several neighbours in a row, and all of them compare
    /// against the same z-order address, CTB, slice and tile: derive those
    /// once and hand them to `available_at`.
    ///
    /// Both halves are `inline(always)`: a caller that asks after only four
    /// or five neighbours, as intra prediction does, wants the context in
    /// registers rather than a 32-byte stack temporary, and measures slower
    /// than the unhoisted test without it.
    #[inline(always)]
    pub fn avail_ctx(&self, xc: i32, yc: i32, pic_w: i32, pic_h: i32) -> AvailCtx {
        let (xc, yc) = (xc as usize, yc as usize);
        let cc = self.ctb_of(xc, yc);
        AvailCtx {
            zs: self.min_tb_addr_zs[self.idx4(xc, yc)],
            ctb: cc,
            slice_addr: self.ctb_slice_addr[cc],
            tile: self.ctb_tile[cc],
            pic_w,
            pic_h,
        }
    }

    /// z-scan availability (6.4.1) of the block containing `(xn, yn)` for the
    /// current block `c` describes.
    #[inline(always)]
    pub fn available_at(&self, c: &AvailCtx, xn: i32, yn: i32) -> bool {
        if xn < 0 || yn < 0 || xn >= c.pic_w || yn >= c.pic_h {
            return false;
        }
        let (xn, yn) = (xn as usize, yn as usize);
        let in_ = self.idx4(xn, yn);
        if self.min_tb_addr_zs[in_] > c.zs {
            return false;
        }
        let cn = self.ctb_of(xn, yn);
        // Inside the current CTB the slice and tile are the same by
        // construction (both change only at CTB boundaries).
        if cn != c.ctb {
            if self.ctb_slice_addr[cn] != c.slice_addr || self.ctb_slice_addr[cn] == u32::MAX {
                return false;
            }
            if self.ctb_tile[cn] != c.tile {
                return false;
            }
        }
        // Not yet decoded (a hole from a lost slice, or the block is later in
        // the same CTB): pred_mode 2 means unwritten.
        self.pred_mode[in_] != 2
    }


    /// Fill a rectangle of 4x4 entries in a per-4x4 array. Byte-sized
    /// entries (all the per-4x4 tables) are written as whole words for the
    /// common CU / PU widths — a `memset` call per 2-byte row costs more than
    /// the row, and LLVM turns any small store loop back into one.
    #[inline(always)]
    pub fn fill4<T: Copy>(arr: &mut [T], w4: usize, x: usize, y: usize, w: usize, h: usize, v: T) {
        let (bx0, bx1) = (x >> 2, (x + w) >> 2);
        if bx1 <= bx0 {
            return;
        }
        let n = bx1 - bx0;
        if std::mem::size_of::<T>() == 1 && n.is_power_of_two() && n <= 16 {
            // SAFETY: T is one byte; the transmute copies that byte.
            let b: u8 = unsafe { std::mem::transmute_copy(&v) };
            let word = u64::from_ne_bytes([b; 8]);
            for by in (y >> 2)..((y + h) >> 2) {
                let row = &mut arr[by * w4 + bx0..by * w4 + bx1];
                let p = row.as_mut_ptr() as *mut u8;
                // SAFETY: `row` holds `n` bytes; each store stays inside it.
                unsafe {
                    match n {
                        1 => *p = b,
                        2 => std::ptr::write_unaligned(p as *mut u16, word as u16),
                        4 => std::ptr::write_unaligned(p as *mut u32, word as u32),
                        8 => std::ptr::write_unaligned(p as *mut u64, word),
                        _ => {
                            std::ptr::write_unaligned(p as *mut u64, word);
                            std::ptr::write_unaligned(p.add(8) as *mut u64, word);
                        }
                    }
                }
            }
            return;
        }
        for by in (y >> 2)..((y + h) >> 2) {
            arr[by * w4 + bx0..by * w4 + bx1].fill(v);
        }
    }
}
