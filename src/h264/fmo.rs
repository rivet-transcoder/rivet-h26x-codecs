//! Slice groups — flexible macroblock ordering (H.264 clause 8.2.2): the
//! map from macroblock address to slice group for all seven
//! `slice_group_map_type`s, and from it the next macroblock of a slice
//! (`NextMbAddress`, 8.2.2.8), which is what the slice decoder walks.
//!
//! Without slice groups the next macroblock is `addr + 1` and none of this
//! runs; the slice decoder keeps that as its fast path.

use super::pps::SliceGroups;
use super::sps::Sps;
use crate::{Error, Result};

/// A picture's macroblock-to-slice-group map with its next-address table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceGroupMap {
    /// Per macroblock address in decoding order (raster, or `2 * pair +
    /// bottom` in an MBAFF frame): the next address in the same slice
    /// group, or the macroblock count when the group has no more.
    pub next: Vec<u32>,
    /// Per macroblock address: `mbToSliceGroupMap`.
    pub group: Vec<u8>,
}

/// Bits of `slice_group_change_cycle` in the slice header (7.4.3):
/// `Ceil(Log2(PicSizeInMapUnits ÷ SliceGroupChangeRate + 1))`, the division
/// exact — so the smallest `k` with `2^k * rate >= size + rate`.
pub fn change_cycle_bits(pic_size_in_map_units: u32, change_rate: u32) -> u32 {
    let need = pic_size_in_map_units as u64 + change_rate as u64;
    let mut k = 0;
    while (1u64 << k) * (change_rate as u64) < need {
        k += 1;
    }
    k
}

/// `mapUnitToSliceGroupMap` (8.2.2.1 – 8.2.2.7) for a picture of `w` map
/// units across and `h` down; `change_cycle` is the slice header's
/// `slice_group_change_cycle` (types 3–5 evolve with it).
pub fn map_unit_to_slice_group(
    sg: &SliceGroups,
    w: u32,
    h: u32,
    change_cycle: u32,
) -> Result<Vec<u8>> {
    let size = (w * h) as usize;
    let groups = sg.num_groups;
    let mut map = vec![0u8; size];
    match sg.map_type {
        0 => {
            // Interleaved: runs of each group in turn, round and round.
            let mut i = 0usize;
            while i < size {
                for (g, &run) in sg.run_length.iter().enumerate() {
                    if i >= size {
                        break;
                    }
                    let end = (i + run as usize).min(size);
                    map[i..end].fill(g as u8);
                    i = end;
                }
            }
        }
        1 => {
            // Dispersed.
            for (i, m) in map.iter_mut().enumerate() {
                let (x, y) = (i as u32 % w, i as u32 / w);
                *m = ((x + (y * groups) / 2) % groups) as u8;
            }
        }
        2 => {
            // Foreground boxes over a left-over background: the last group
            // is the background, earlier groups' boxes are laid down last
            // so a lower-numbered group wins where boxes overlap.
            map.fill((groups - 1) as u8);
            for (g, &(tl, br)) in sg.boxes.iter().enumerate().rev() {
                let (x0, y0) = (tl % w, tl / w);
                let (x1, y1) = (br % w, br / w);
                if br >= w * h || x0 > x1 || y0 > y1 {
                    return Err(Error::bitstream("PPS: slice group box out of the picture"));
                }
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        map[(y * w + x) as usize] = g as u8;
                    }
                }
            }
        }
        3 => {
            // Box-out: a spiral from the centre, group 0 growing with the
            // change cycle, the rest group 1.
            let in_group0 = units_in_group0(sg, size, change_cycle);
            map.fill(2);
            let dir = sg.change_direction as i32;
            let (mut x, mut y) = (((w as i32) - dir) / 2, ((h as i32) - dir) / 2);
            let (mut left, mut top, mut right, mut bottom) = (x, y, x, y);
            let (mut xdir, mut ydir) = (dir - 1, dir);
            let mut k = 0usize;
            while k < size {
                let at = (y as u32 * w + x as u32) as usize;
                let vacant = map[at] == 2;
                if vacant {
                    map[at] = (k >= in_group0) as u8;
                }
                if xdir == -1 && x == left {
                    left = (left - 1).max(0);
                    x = left;
                    xdir = 0;
                    ydir = 2 * dir - 1;
                } else if xdir == 1 && x == right {
                    right = (right + 1).min(w as i32 - 1);
                    x = right;
                    xdir = 0;
                    ydir = 1 - 2 * dir;
                } else if ydir == -1 && y == top {
                    top = (top - 1).max(0);
                    y = top;
                    xdir = 1 - 2 * dir;
                    ydir = 0;
                } else if ydir == 1 && y == bottom {
                    bottom = (bottom + 1).min(h as i32 - 1);
                    y = bottom;
                    xdir = 2 * dir - 1;
                    ydir = 0;
                } else {
                    x += xdir;
                    y += ydir;
                }
                k += vacant as usize;
            }
        }
        4 => {
            // Raster scan: the first `sizeOfUpperLeftGroup` units.
            let upper_left = upper_left_size(sg, size, change_cycle);
            for (i, m) in map.iter_mut().enumerate() {
                *m = if i < upper_left {
                    sg.change_direction as u8
                } else {
                    1 - sg.change_direction as u8
                };
            }
        }
        5 => {
            // Wipe: column by column.
            let upper_left = upper_left_size(sg, size, change_cycle);
            let mut k = 0usize;
            for x in 0..w {
                for y in 0..h {
                    map[(y * w + x) as usize] = if k < upper_left {
                        sg.change_direction as u8
                    } else {
                        1 - sg.change_direction as u8
                    };
                    k += 1;
                }
            }
        }
        6 => {
            if sg.slice_group_id.len() != size {
                return Err(Error::bitstream(format!(
                    "PPS: slice_group_id covers {} map units, the picture has {size}",
                    sg.slice_group_id.len()
                )));
            }
            map.copy_from_slice(&sg.slice_group_id);
        }
        t => {
            return Err(Error::bitstream(format!(
                "PPS: slice_group_map_type {t} out of range"
            )));
        }
    }
    if map.iter().any(|&g| g as u32 >= groups) {
        return Err(Error::bitstream(
            "PPS: slice_group_id beyond num_slice_groups",
        ));
    }
    Ok(map)
}

/// `MapUnitsInSliceGroup0 = Min(slice_group_change_cycle * SliceGroupChangeRate, PicSizeInMapUnits)`.
fn units_in_group0(sg: &SliceGroups, size: usize, change_cycle: u32) -> usize {
    ((change_cycle as u64 * sg.change_rate as u64).min(size as u64)) as usize
}

/// `sizeOfUpperLeftGroup` (8.2.2.4 / 8.2.2.5).
fn upper_left_size(sg: &SliceGroups, size: usize, change_cycle: u32) -> usize {
    let g0 = units_in_group0(sg, size, change_cycle);
    if sg.change_direction { size - g0 } else { g0 }
}

/// The map for a picture (8.2.2.8): map units are macroblocks for a frame
/// of frame macroblocks and for a field; macroblock pairs in an MBAFF
/// frame; and for a frame picture of a field-capable sequence decoded
/// without MBAFF, one unit covers the two vertically adjacent macroblocks.
pub fn build(
    sg: &SliceGroups,
    sps: &Sps,
    field_pic: bool,
    mbaff: bool,
    change_cycle: u32,
) -> Result<SliceGroupMap> {
    let w = sps.pic_width_in_mbs;
    let hu = sps.pic_height_in_map_units;
    let units = map_unit_to_slice_group(sg, w, hu, change_cycle)?;
    let pic_h = sps.frame_height_in_mbs() / if field_pic { 2 } else { 1 };
    let total = (w * pic_h) as usize;
    let group: Vec<u8> = (0..total)
        .map(|i| {
            if sps.frame_mbs_only || field_pic {
                units[i]
            } else if mbaff {
                units[i / 2]
            } else {
                units[(i / (2 * w as usize)) * w as usize + i % w as usize]
            }
        })
        .collect();
    let mut next = vec![total as u32; total];
    let mut last = [total as u32; 8];
    for i in (0..total).rev() {
        let g = group[i] as usize;
        next[i] = last[g];
        last[g] = i as u32;
    }
    Ok(SliceGroupMap { next, group })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups(map_type: u32, n: u32) -> SliceGroups {
        SliceGroups {
            num_groups: n,
            map_type,
            run_length: Vec::new(),
            boxes: Vec::new(),
            change_direction: false,
            change_rate: 1,
            slice_group_id: Vec::new(),
        }
    }

    #[test]
    fn change_cycle_bit_count() {
        // 99 units, rate 1: Ceil(Log2(100)) = 7. Rate 99: Ceil(Log2(2)) = 1.
        assert_eq!(change_cycle_bits(99, 1), 7);
        assert_eq!(change_cycle_bits(99, 99), 1);
        // 64 units, rate 64: Log2(2) exactly -> 1 bit; rate 63: Log2(64/63 + 1) -> 2 bits.
        assert_eq!(change_cycle_bits(64, 64), 1);
        assert_eq!(change_cycle_bits(64, 63), 2);
    }

    #[test]
    fn interleaved_runs_wrap_around() {
        let mut sg = groups(0, 2);
        sg.run_length = vec![3, 2];
        let m = map_unit_to_slice_group(&sg, 4, 2, 0).unwrap();
        assert_eq!(m, vec![0, 0, 0, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn dispersed_is_a_checkerboard_for_two_groups() {
        let sg = groups(1, 2);
        let m = map_unit_to_slice_group(&sg, 4, 3, 0).unwrap();
        assert_eq!(m, vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn foreground_boxes_over_background() {
        let mut sg = groups(2, 3);
        // Group 0: units 5..=6 (row 1, cols 1-2); group 1: the whole
        // second column (overlaps group 0 at unit 5, which group 0 wins).
        sg.boxes = vec![(5, 6), (1, 9)];
        let m = map_unit_to_slice_group(&sg, 4, 3, 0).unwrap();
        assert_eq!(m, vec![2, 1, 2, 2, 2, 0, 0, 2, 2, 1, 2, 2]);
    }

    #[test]
    fn box_out_grows_from_the_centre() {
        let mut sg = groups(3, 2);
        sg.change_rate = 1;
        // 4x4 picture, direction 0: the spiral starts at (2, 2) and goes
        // left first; 3 units in group 0.
        let m = map_unit_to_slice_group(&sg, 4, 4, 3).unwrap();
        let mut expect = vec![1u8; 16];
        expect[2 * 4 + 2] = 0;
        expect[2 * 4 + 1] = 0;
        expect[1 * 4 + 1] = 0;
        assert_eq!(m, expect);
        // Everything in group 0 once the cycle covers the picture.
        assert!(
            map_unit_to_slice_group(&sg, 4, 4, 16)
                .unwrap()
                .iter()
                .all(|&g| g == 0)
        );
    }

    #[test]
    fn raster_and_wipe() {
        let mut sg = groups(4, 2);
        sg.change_rate = 2;
        assert_eq!(
            map_unit_to_slice_group(&sg, 3, 2, 2).unwrap(),
            vec![0, 0, 0, 0, 1, 1]
        );
        sg.change_direction = true;
        assert_eq!(
            map_unit_to_slice_group(&sg, 3, 2, 1).unwrap(),
            vec![1, 1, 1, 1, 0, 0]
        );
        let mut sg = groups(5, 2);
        sg.change_rate = 1;
        // Column-wise: 3 units = the first column and the top of the second.
        assert_eq!(
            map_unit_to_slice_group(&sg, 3, 2, 3).unwrap(),
            vec![0, 0, 1, 0, 1, 1]
        );
    }

    #[test]
    fn explicit_map_and_next_address() {
        let mut sg = groups(6, 2);
        sg.slice_group_id = vec![0, 1, 1, 0, 0, 1];
        let mut sps = test_sps(3, 2);
        let m = build(&sg, &sps, false, false, 0).unwrap();
        assert_eq!(m.group, vec![0, 1, 1, 0, 0, 1]);
        assert_eq!(m.next, vec![3, 2, 5, 4, 6, 6]);
        // The same map units over a field-capable frame without MBAFF: each
        // unit covers two vertically adjacent macroblocks (8.2.2.8).
        sps.frame_mbs_only = false;
        let m = build(&sg, &sps, false, false, 0).unwrap();
        assert_eq!(m.group, vec![0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]);
        // An MBAFF frame: unit `i / 2` for decoding-order address `i`.
        sps.mb_adaptive_frame_field = true;
        let m = build(&sg, &sps, false, true, 0).unwrap();
        assert_eq!(m.group, vec![0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
        assert_eq!(m.next[0], 1);
        assert_eq!(m.next[1], 6);
        // A field of it: map units are its macroblocks again.
        let m = build(&sg, &sps, true, false, 0).unwrap();
        assert_eq!(m.group, vec![0, 1, 1, 0, 0, 1]);
    }

    fn test_sps(w: u32, h_units: u32) -> Sps {
        Sps {
            profile_idc: 66,
            constraint_flags: 0,
            level_idc: 10,
            id: 0,
            chroma_format_idc: 1,
            separate_colour_plane: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            transform_bypass: false,
            scaling_lists: None,
            log2_max_frame_num: 4,
            poc_type: 2,
            log2_max_poc_lsb: 4,
            delta_pic_order_always_zero: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_allowed: false,
            pic_width_in_mbs: w,
            pic_height_in_map_units: h_units,
            frame_mbs_only: true,
            mb_adaptive_frame_field: false,
            direct_8x8_inference: true,
            crop: (0, 0, 0, 0),
            vui: None,
        }
    }
}
