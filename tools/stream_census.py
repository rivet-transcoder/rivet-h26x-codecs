#!/usr/bin/env python3
"""stream_census.py <file.264|.265> — one line of `key=value` tokens saying
what an Annex-B stream actually carries, read from its parameter sets, its
NAL headers and the start of each slice header.

This exists so make_fixtures.sh can check a fixture's NAME against its
BYTES. Both ffmpeg encoder wrappers only warn when an option is unknown or
quietly overridden, and the option string the encoder writes into its SEI
says what it was asked for, not what it wrote. A fixture named for a coding
tool it does not contain is a green row that tests nothing; the census is
what catches that.

H.264: profile, chroma format, bit depth, frame_mbs_only / MBAFF, cropping,
CABAC, 8x8 transform, scaling matrices, weighted prediction, constrained
intra, slice groups, data partitioning; pictures, slices per picture, slice
types, IDR count.

HEVC: chroma, depth, CTU and minimum CB size, scaling lists (and whether
explicit), AMP, SAO, PCM, TMVP, strong intra smoothing, WPP, tiles, sign
hiding, cu_qp_delta and its depth, constrained intra, transform skip,
weighted prediction, transquant bypass, deblocking disabled in the PPS;
pictures, slices per picture, slice types, IDR / CRA / RASL / RADL counts,
the highest temporal id.

Only the fields up to the last one wanted are parsed. A stream the parser
cannot follow prints what it got plus `parse_error=<where>` rather than a
traceback, so a check against it fails on a named token, not on Python.
"""
import sys


class Bits:
    """An RBSP bit reader: u(n), ue(v), se(v), more_rbsp_data()."""

    def __init__(self, data):
        self.d = data
        self.pos = 0
        # rbsp_stop_one_bit: the last set bit in the payload.
        self.stop = 0
        for i in range(len(data) - 1, -1, -1):
            if data[i]:
                bit = 0
                while not (data[i] >> bit) & 1:
                    bit += 1
                self.stop = i * 8 + (7 - bit)
                break

    def u(self, n):
        v = 0
        for _ in range(n):
            if self.pos >= len(self.d) * 8:
                raise EOFError("past the end of the RBSP")
            v = (v << 1) | ((self.d[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
            if z > 32:
                raise ValueError("exp-golomb runaway")
        return (1 << z) - 1 + (self.u(z) if z else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)

    def more_rbsp_data(self):
        return self.pos < self.stop


def rbsp(nal):
    """Strip emulation prevention bytes."""
    out = bytearray()
    zeros = 0
    for b in nal:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def nals(data):
    """The NAL units of an Annex-B byte stream, header byte(s) included."""
    starts = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else len(data)
        nal = data[s:e]
        # trailing_zero_8bits, and the leading 00 of a 4-byte start code
        while nal and nal[-1] == 0:
            nal = nal[:-1]
        if nal:
            yield nal


# ---------------------------------------------------------------- H.264

def h264_skip_scaling_list(b, n):
    last = nxt = 8
    for j in range(n):
        if nxt != 0:
            nxt = (last + b.se() + 256) % 256
        last = last if nxt == 0 else nxt


def h264_sps(rb):
    b = Bits(rb)
    profile = b.u(8)
    b.u(8)  # constraint flags + reserved
    b.u(8)  # level_idc
    b.ue()  # seq_parameter_set_id
    chroma, depth, sl = 1, 8, 0
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma = b.ue()
        if chroma == 3:
            b.u(1)  # separate_colour_plane_flag
        depth = b.ue() + 8
        b.ue()  # bit_depth_chroma_minus8
        b.u(1)  # qpprime_y_zero_transform_bypass_flag
        if b.u(1):  # seq_scaling_matrix_present_flag
            sl = 1
            for i in range(8 if chroma != 3 else 12):
                if b.u(1):
                    h264_skip_scaling_list(b, 16 if i < 6 else 64)
    b.ue()  # log2_max_frame_num_minus4
    poc_type = b.ue()
    if poc_type == 0:
        b.ue()
    elif poc_type == 1:
        b.u(1)
        b.se()
        b.se()
        for _ in range(b.ue()):
            b.se()
    b.ue()  # max_num_ref_frames
    b.u(1)  # gaps_in_frame_num_value_allowed_flag
    b.ue()  # pic_width_in_mbs_minus1
    b.ue()  # pic_height_in_map_units_minus1
    frame_mbs_only = b.u(1)
    mbaff = 0 if frame_mbs_only else b.u(1)
    b.u(1)  # direct_8x8_inference_flag
    crop = b.u(1)
    return dict(profile=profile, chroma=chroma, depth=depth, frame_mbs_only=frame_mbs_only,
                mbaff=mbaff, crop=crop, sl_sps=sl)


def h264_pps(rb, chroma):
    b = Bits(rb)
    b.ue()  # pic_parameter_set_id
    b.ue()  # seq_parameter_set_id
    cabac = b.u(1)
    b.u(1)  # bottom_field_pic_order_in_frame_present_flag
    slice_groups = b.ue() + 1
    if slice_groups > 1:
        # The map syntax is long and no fixture encoder writes it; the
        # count is the fact worth having.
        return dict(cabac=cabac, slice_groups=slice_groups)
    b.ue()  # num_ref_idx_l0_default_active_minus1
    b.ue()  # num_ref_idx_l1_default_active_minus1
    wp = b.u(1)
    wbi = b.u(2)
    b.se()  # pic_init_qp_minus26
    b.se()  # pic_init_qs_minus26
    b.se()  # chroma_qp_index_offset
    dfc = b.u(1)  # deblocking_filter_control_present_flag
    cip = b.u(1)
    b.u(1)  # redundant_pic_cnt_present_flag
    t8x8 = sl = 0
    if b.more_rbsp_data():
        t8x8 = b.u(1)
        sl = b.u(1)
        if sl:
            for i in range(6 + (2 if chroma != 3 else 6) * t8x8):
                if b.u(1):
                    h264_skip_scaling_list(b, 16 if i < 6 else 64)
        b.se()  # second_chroma_qp_index_offset
    return dict(cabac=cabac, slice_groups=slice_groups, wp=wp, wbi=wbi, dfc=dfc, cip=cip,
                t8x8=t8x8, sl_pps=sl)


def census_h264(data):
    c = dict(sps=0, pps=0, pics=0, vcl=0, idr=0, islices=0, pslices=0, bslices=0, sp_si=0,
             partitioned=0)
    sps = pps = None
    for nal in nals(data):
        t = nal[0] & 0x1F
        if t == 7:
            c["sps"] += 1
            if sps is None:
                sps = h264_sps(rbsp(nal[1:]))
                c.update(sps)
        elif t == 8:
            c["pps"] += 1
            if pps is None:
                pps = h264_pps(rbsp(nal[1:]), sps["chroma"] if sps else 1)
                c.update(pps)
        elif t in (1, 5):
            c["vcl"] += 1
            if t == 5:
                c["idr"] += 1
            b = Bits(rbsp(nal[1:]))
            if b.ue() == 0:  # first_mb_in_slice
                c["pics"] += 1
            c[("pslices", "bslices", "islices", "sp_si", "sp_si")[b.ue() % 5]] += 1
        elif t in (2, 3, 4):
            c["partitioned"] += 1
    c["sl"] = int(bool(c.get("sl_sps")) or bool(c.get("sl_pps")))
    c["slices_per_pic"] = c["vcl"] // c["pics"] if c["pics"] else 0
    return c


# ----------------------------------------------------------------- HEVC

def hevc_ptl(b, max_sub_layers_minus1):
    b.u(88)  # general profile space .. general reserved bits (2+1+5+32+48)
    b.u(8)  # general_level_idc
    present = [(b.u(1), b.u(1)) for _ in range(max_sub_layers_minus1)]
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            b.u(2)
    for prof, lev in present:
        if prof:
            b.u(88)
        if lev:
            b.u(8)


def hevc_skip_scaling_list_data(b):
    for size_id in range(4):
        for _matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            if not b.u(1):  # scaling_list_pred_mode_flag
                b.ue()  # scaling_list_pred_matrix_id_delta
            else:
                if size_id > 1:
                    b.se()  # scaling_list_dc_coef_minus8
                for _ in range(min(64, 1 << (4 + (size_id << 1)))):
                    b.se()


def hevc_skip_st_ref_pic_set(b, idx, sets):
    """Parse one st_ref_pic_set in an SPS, returning (num_negative,
    num_positive) — the inter-predicted form needs the referenced set's
    counts, so they are carried in `sets`."""
    inter = b.u(1) if idx != 0 else 0
    if inter:
        # In an SPS the referenced set is the previous one.
        ref_neg, ref_pos = sets[idx - 1]
        b.u(1)  # delta_rps_sign
        b.ue()  # abs_delta_rps_minus1
        used, use_delta = [], []
        for _ in range(ref_neg + ref_pos + 1):
            u = b.u(1)
            used.append(u)
            use_delta.append(1 if u else b.u(1))
        # 7-61 / 7-62 only decide which entries survive; without the POC
        # deltas the exact split is unknowable here, but the total is not:
        # every entry with use_delta set, plus the deltaRps entry itself.
        # The split matters to nobody downstream of this census, so the
        # total is stored as negatives.
        kept = sum(1 for u in use_delta if u)
        return (kept, 0)
    neg = b.ue()
    pos = b.ue()
    for _ in range(neg):
        b.ue()
        b.u(1)
    for _ in range(pos):
        b.ue()
        b.u(1)
    return (neg, pos)


def hevc_sps(rb):
    b = Bits(rb)
    b.u(4)  # sps_video_parameter_set_id
    max_sub_layers_minus1 = b.u(3)
    b.u(1)  # sps_temporal_id_nesting_flag
    hevc_ptl(b, max_sub_layers_minus1)
    b.ue()  # sps_seq_parameter_set_id
    chroma = b.ue()
    if chroma == 3:
        b.u(1)  # separate_colour_plane_flag
    w = b.ue()
    h = b.ue()
    crop = b.u(1)
    if crop:
        for _ in range(4):
            b.ue()
    depth = b.ue() + 8
    b.ue()  # bit_depth_chroma_minus8
    log2_max_poc_lsb = b.ue() + 4
    ordering_info = b.u(1)
    for _ in range(0 if ordering_info else max_sub_layers_minus1, max_sub_layers_minus1 + 1):
        b.ue()
        b.ue()
        b.ue()
    log2_min_cb = b.ue() + 3
    log2_ctu = log2_min_cb + b.ue()
    log2_min_tb = b.ue() + 2
    log2_max_tb = log2_min_tb + b.ue()
    b.ue()  # max_transform_hierarchy_depth_inter
    b.ue()  # max_transform_hierarchy_depth_intra
    sl = b.u(1)
    sl_data = 0
    if sl:
        sl_data = b.u(1)
        if sl_data:
            hevc_skip_scaling_list_data(b)
    amp = b.u(1)
    sao = b.u(1)
    pcm = b.u(1)
    if pcm:
        b.u(4)
        b.u(4)
        b.ue()
        b.ue()
        b.u(1)
    sets = []
    for i in range(b.ue()):
        sets.append(hevc_skip_st_ref_pic_set(b, i, sets))
    if b.u(1):  # long_term_ref_pics_present_flag
        for _ in range(b.ue()):
            b.u(log2_max_poc_lsb)
            b.u(1)
    tmvp = b.u(1)
    strong = b.u(1)
    return dict(chroma=chroma, depth=depth, ctu=1 << log2_ctu, min_cb=1 << log2_min_cb,
                max_tb=1 << log2_max_tb, crop=crop, sl=sl, sl_data=sl_data, amp=amp, sao=sao,
                pcm=pcm, tmvp=tmvp, strong=strong, sub_layers=max_sub_layers_minus1 + 1,
                _w=w, _h=h)


def hevc_pps(rb):
    b = Bits(rb)
    b.ue()  # pps_pic_parameter_set_id
    b.ue()  # pps_seq_parameter_set_id
    dep = b.u(1)  # dependent_slice_segments_enabled_flag
    b.u(1)  # output_flag_present_flag
    extra = b.u(3)  # num_extra_slice_header_bits
    sbh = b.u(1)
    b.u(1)  # cabac_init_present_flag
    b.ue()  # num_ref_idx_l0_default_active_minus1
    b.ue()  # num_ref_idx_l1_default_active_minus1
    b.se()  # init_qp_minus26
    cip = b.u(1)
    tskip = b.u(1)
    cuqp = b.u(1)
    qg_depth = b.ue() if cuqp else 0
    b.se()  # pps_cb_qp_offset
    b.se()  # pps_cr_qp_offset
    b.u(1)  # pps_slice_chroma_qp_offsets_present_flag
    wp = b.u(1)
    wbi = b.u(1)
    tqbypass = b.u(1)
    tiles = b.u(1)
    wpp = b.u(1)
    if tiles:
        cols = b.ue() + 1
        rows = b.ue() + 1
        if not b.u(1):  # uniform_spacing_flag
            for _ in range(cols - 1):
                b.ue()
            for _ in range(rows - 1):
                b.ue()
        b.u(1)  # loop_filter_across_tiles_enabled_flag
    b.u(1)  # pps_loop_filter_across_slices_enabled_flag
    deblock_off = 0
    if b.u(1):  # deblocking_filter_control_present_flag
        b.u(1)  # deblocking_filter_override_enabled_flag
        deblock_off = b.u(1)
    return dict(sbh=sbh, cip=cip, tskip=tskip, cuqp=cuqp, qg_depth=qg_depth, wp=wp, wbi=wbi,
                tqbypass=tqbypass, tiles=tiles, wpp=wpp, deblock_off=deblock_off, _dep=dep,
                _extra=extra)


def census_hevc(data):
    c = dict(vps=0, sps=0, pps=0, pics=0, vcl=0, idr=0, cra=0, rasl=0, radl=0, irap=0,
             tid_max=0, islices=0, pslices=0, bslices=0)
    sps = pps = None
    for nal in nals(data):
        t = (nal[0] >> 1) & 0x3F
        tid = (nal[1] & 7) - 1
        if t == 32:
            c["vps"] += 1
        elif t == 33:
            c["sps"] += 1
            if sps is None:
                sps = hevc_sps(rbsp(nal[2:]))
                c.update((k, v) for k, v in sps.items() if not k.startswith("_"))
        elif t == 34:
            c["pps"] += 1
            if pps is None:
                pps = hevc_pps(rbsp(nal[2:]))
                c.update((k, v) for k, v in pps.items() if not k.startswith("_"))
        elif t <= 31:
            c["vcl"] += 1
            c["tid_max"] = max(c["tid_max"], tid)
            if t in (19, 20):
                c["idr"] += 1
            if t == 21:
                c["cra"] += 1
            if t in (8, 9):
                c["rasl"] += 1
            if t in (6, 7):
                c["radl"] += 1
            if 16 <= t <= 23:
                c["irap"] += 1
            b = Bits(rbsp(nal[2:]))
            first = b.u(1)
            if first:
                c["pics"] += 1
            if 16 <= t <= 23:
                b.u(1)  # no_output_of_prior_pics_flag
            b.ue()  # slice_pic_parameter_set_id
            dependent = 0
            if not first:
                if pps and pps["_dep"]:
                    dependent = b.u(1)
                if sps:
                    ctu = sps["ctu"]
                    n = ((sps["_w"] + ctu - 1) // ctu) * ((sps["_h"] + ctu - 1) // ctu)
                    b.u((n - 1).bit_length())  # slice_segment_address
            if not dependent:
                b.u(pps["_extra"] if pps else 0)
                c[("bslices", "pslices", "islices")[b.ue()]] += 1
    c["slices_per_pic"] = c["vcl"] // c["pics"] if c["pics"] else 0
    return c


ORDER_H264 = ("profile chroma depth frame_mbs_only mbaff crop cabac t8x8 sl wp wbi cip dfc "
              "slice_groups partitioned sps pps pics slices_per_pic idr islices pslices "
              "bslices sp_si").split()
ORDER_HEVC = ("chroma depth ctu min_cb max_tb crop sub_layers sl sl_data amp sao pcm tmvp "
              "strong wpp tiles sbh cuqp qg_depth cip tskip wp wbi tqbypass deblock_off vps "
              "sps pps pics slices_per_pic idr cra rasl radl irap tid_max islices pslices "
              "bslices").split()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path, "rb") as f:
        data = f.read()
    hevc = path.lower().endswith((".265", ".hevc", ".h265"))
    c = {}
    err = None
    try:
        c = census_hevc(data) if hevc else census_h264(data)
    except Exception as e:  # noqa: BLE001 — the point is to report, not to die
        err = f"{type(e).__name__}:{str(e).replace(' ', '_')}"
    order = ORDER_HEVC if hevc else ORDER_H264
    toks = [f"{k}={c[k]}" for k in order if k in c]
    toks += [f"{k}={v}" for k, v in c.items() if k not in order and not k.startswith("_")]
    if err:
        toks.append(f"parse_error={err}")
    print(" ".join(toks))


if __name__ == "__main__":
    main()
