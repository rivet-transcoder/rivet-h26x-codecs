"""Do two Annex-B streams differ only in the profile flags and level they claim?

The mask behind identity_encode.sh's MASK=level. Deriving the level from
the stream (src/encode/level.rs) moved one header field in nearly every
stream the gate encodes, so a plain byte comparison calls every cell moved
and proves nothing about the pictures. This compares the two streams NAL
unit by NAL unit, unescaped, with exactly these fields masked and nothing
else:

- H.264 sequence parameter set: `level_idc`, RBSP byte 3 counting the NAL
  header byte (profile_idc, the constraint flags, then the level; 7.3.2.1.1).
- H.265 video parameter set: `general_tier_flag` (bit 0x20 of RBSP byte 6),
  `general_level_idc` (RBSP byte 17), and the nine format range extensions
  constraint flags `general_max_12bit_constraint_flag` ..
  `general_lower_bit_rate_constraint_flag` (the low four bits of RBSP byte
  11 and the high five of byte 12), counting the two NAL header bytes. The
  profile_tier_level starts at bit 32 of the VPS (7.3.2.1), and the nine
  flags are its bits 44..52 (7.3.3).
- H.265 sequence parameter set: the same fields at RBSP bytes 3 (tier), 14
  (level) and 8 / 9 (the flags); its profile_tier_level starts at bit 8
  (7.3.2.2).

The flags are masked for the change that writes Table A.2's row for each
format range extensions profile where every flag used to be zero; in a
Main or Main 10 stream they are reserved zero bits and cannot differ.

Everything else must be equal: the NAL unit count, every other NAL unit
byte for byte, and every other byte of the parameter sets. Emulation
prevention is removed before comparing, so a level byte that changed an
escape elsewhere would still be judged on the payload. (No level value is
0..3, so none can.)

    python level_mask.py h264|h265 A B

Exit 0 when the streams are equal under the mask, printing each level
change (`SPS level 51->10`); 1 otherwise, printing the first difference.
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


# (name, byte index of the tier flag or None, byte index of the level,
# (byte, bits) of the profile flags) per NAL unit type, in the unescaped
# unit including its header.
FIELDS = {
    "h264": {7: ("SPS", None, 3, ())},
    "h265": {32: ("VPS", 6, 17, ((11, 0x0F), (12, 0xF8))), 33: ("SPS", 3, 14, ((8, 0x0F), (9, 0xF8)))},
}


def profile_flags(u, flags):
    """The nine constraint flags as a string of bits, or '' for none."""
    if not flags:
        return ""
    (b0, _), (b1, _) = flags
    return f"{u[b0] & 0x0F:04b}{u[b1] >> 3:05b}"


def nal_type(codec, nal):
    return nal[0] & 0x1F if codec == "h264" else (nal[0] >> 1) & 0x3F


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in FIELDS:
        sys.exit("usage: level_mask.py h264|h265 A B")
    codec = sys.argv[1]
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
        field = FIELDS[codec].get(t)
        if field is None or nal_type(codec, y) != t:
            print(f"MOVED: NAL unit {k} (type {t}) differs and carries no level")
            return 1
        name, tier, level, flags = field
        ux, uy = unescape(x), unescape(y)
        if len(ux) != len(uy) or len(ux) <= level:
            print(f"MOVED: {name} {k} is {len(ux)} bytes against {len(uy)}")
            return 1
        mx, my = bytearray(ux), bytearray(uy)
        for m in (mx, my):
            m[level] = 0
            if tier is not None:
                m[tier] &= ~0x20 & 0xFF
            for byte, bits in flags:
                m[byte] &= ~bits & 0xFF
        if mx != my:
            first = next(i for i in range(len(mx)) if mx[i] != my[i])
            print(f"MOVED: {name} {k} differs outside the level at RBSP byte {first}")
            return 1
        tiers = "" if tier is None else f" tier {(ux[tier] >> 5) & 1}->{(uy[tier] >> 5) & 1}"
        fx, fy = profile_flags(ux, flags), profile_flags(uy, flags)
        moved_flags = f" flags {fx}->{fy}" if fx != fy else ""
        changes.add(f"{name} level {ux[level]}->{uy[level]}{tiers}{moved_flags}")
    print("; ".join(sorted(changes)) if changes else "identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
