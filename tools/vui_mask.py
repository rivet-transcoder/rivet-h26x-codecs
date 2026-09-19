"""Do two Annex-B streams differ only by the VUI frame clock the second adds?

The mask behind identity_encode.sh's MASK=vui. Writing the frame clock on
every stream (timing_info_present_flag, H.264 E.1.1 / H.265 E.2.1) moves
the sequence parameter set of every stream that did not declare a coded
picture buffer, and nothing else. This compares the two streams NAL unit
by NAL unit and accepts exactly that difference:

- every NAL unit but the SPS byte-identical;
- the SPS (unescaped, trailing bits stripped) equal up to one flag that
  goes from 0 to 1, and equal again after a block of bits inserted right
  behind it, where the block is one of:

  - H.264 `vui_parameters_present_flag` 0 -> 1 and a whole VUI holding the
    clock alone (74 bits): four absent groups (aspect ratio, overscan,
    video signal type, chroma siting), timing_info_present_flag 1,
    num_units_in_tick, time_scale, fixed_frame_rate_flag, and four zero
    flags (NAL HRD, VCL HRD, pic_struct, bitstream restriction);
  - H.264 `timing_info_present_flag` 0 -> 1 in a VUI that was there (for
    colour or siting) and the clock after it (65 bits): num_units_in_tick,
    time_scale, fixed_frame_rate_flag;
  - H.265 `vui_parameters_present_flag` 0 -> 1 and a whole VUI holding the
    clock alone (76 bits): eight zero flags (aspect ratio, overscan, video
    signal type, chroma siting, neutral chroma, field_seq, frame_field_info,
    default display window), vui_timing_info_present_flag 1, the two clock
    fields, and three zero flags (poc_proportional_to_timing, HRD,
    bitstream restriction);
  - H.265 `vui_timing_info_present_flag` 0 -> 1 in a VUI that was there and
    the clock after it (66 bits): the two clock fields, poc_proportional 0
    and hrd_parameters_present 0.

    python vui_mask.py h264|h265 A B

Exit 0 when the streams are equal under the mask, printing what was added
(`SPS clock 1/60 fixed`); 1 otherwise, printing the first difference.
"""
import sys


def nal_units(data):
    """The NAL units of an Annex-B byte stream, start codes removed."""
    starts = []
    i, n = 0, len(data)
    while i + 3 <= n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    units = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        # A 4-byte start code's leading zero belongs to the next unit.
        while e > s and data[e - 1] == 0:
            e -= 1
        units.append(data[s:e])
    return units


def unescape(nal):
    """Remove emulation prevention: the 03 of every 00 00 03."""
    out = bytearray()
    zeros = 0
    for b in nal:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return out


def payload_bits(rbsp):
    """The RBSP as a string of bits, rbsp_trailing_bits removed."""
    bits = "".join(f"{b:08b}" for b in rbsp).rstrip("0")
    return bits[:-1] if bits.endswith("1") else bits


# Per codec: (SPS NAL type, header bytes, [(name, inserted length, check)]).
# `check` takes the inserted bits and returns (num_units_in_tick,
# time_scale, fixed flag or None) when the block has the expected shape.
def _h264_vui(x):
    if x[0:4] == "0000" and x[4] == "1" and x[70:74] == "0000":
        return int(x[5:37], 2), int(x[37:69], 2), x[69] == "1"
    return None


def _h264_timing(x):
    return int(x[0:32], 2), int(x[32:64], 2), x[64] == "1"


def _h265_vui(x):
    if x[0:8] == "00000000" and x[8] == "1" and x[73:76] == "000":
        return int(x[9:41], 2), int(x[41:73], 2), None
    return None


def _h265_timing(x):
    if x[64:66] == "00":
        return int(x[0:32], 2), int(x[32:64], 2), None
    return None


CODECS = {
    "h264": (7, 1, [("VUI", 74, _h264_vui), ("clock", 65, _h264_timing)]),
    "h265": (33, 2, [("VUI", 76, _h265_vui), ("clock", 66, _h265_timing)]),
}


def nal_type(codec, nal):
    return nal[0] & 0x1F if codec == "h264" else (nal[0] >> 1) & 0x3F


def added_clock(codec, old_nal, new_nal):
    """What the new SPS adds over the old, or None if it is not the clock."""
    _, header, shapes = CODECS[codec]
    a = payload_bits(unescape(old_nal)[header:])
    b = payload_bits(unescape(new_nal)[header:])
    p = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if p is None or a[p] != "0" or b[p] != "1":
        return None
    for name, k, check in shapes:
        if a[p + 1:] == b[p + 1 + k:] and len(b) == len(a) + k:
            got = check(b[p + 1:p + 1 + k])
            if got is not None:
                units, scale, fixed = got
                flag = "" if fixed is None else (" fixed" if fixed else " not fixed")
                return f"SPS {name} +clock {units}/{scale}{flag}"
    return None


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in CODECS:
        sys.exit("usage: vui_mask.py h264|h265 A B")
    codec = sys.argv[1]
    sps_type = CODECS[codec][0]
    a = nal_units(open(sys.argv[2], "rb").read())
    b = nal_units(open(sys.argv[3], "rb").read())
    if len(a) != len(b):
        print(f"MOVED: {len(a)} NAL units against {len(b)}")
        return 1
    changes = set()
    for k, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        t = nal_type(codec, x)
        if t != sps_type or nal_type(codec, y) != t:
            print(f"MOVED: NAL unit {k} (type {t}) differs and is no sequence parameter set")
            return 1
        added = added_clock(codec, x, y)
        if added is None:
            print(f"MOVED: SPS {k} differs by more than an added clock")
            return 1
        changes.add(added)
    print("; ".join(sorted(changes)) if changes else "identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
