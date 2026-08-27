"""Does this Annex-B stream have exactly one parameter set of each kind?

The BOX property. SELF and CROSS both read Annex-B, where a parameter set
re-sent under the same id simply replaces the previous one, so an encoder
that writes a *different* PPS for its I and P pictures passes both. An MP4
`avc1` / `hvc1` sample entry cannot carry that: the parameter sets are stored
once, out of band, and stripped from the samples, so a decoder reading the
box holds two under one id, keeps whichever it parsed last, and decodes the
pictures written under the other one to garbage from their first macroblock.
libavcodec did exactly that on rivet's first H.264 file (2026-08-27) while the
same bytes as an Annex-B stream decoded clean.

So: one VPS, one SPS, one PPS per stream, byte for byte. Stricter than the
standard requires (distinct ids may coexist) and exactly what these encoders
promise, which is the point — a second one is a change, not a second id.

    python param_sets.py stream.h264|stream.h265

Exit 0 when every kind is unique, 1 otherwise, printing what differed.
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


def kinds(path):
    """{kind: {distinct payloads}} for the parameter sets of `path`."""
    data = open(path, "rb").read()
    hevc = path.lower().endswith(("265", "hevc", "h265"))
    seen = {}
    for u in nal_units(data):
        if not u:
            continue
        if hevc:
            t = (u[0] >> 1) & 0x3F
            kind = {32: "VPS", 33: "SPS", 34: "PPS"}.get(t)
        else:
            t = u[0] & 0x1F
            kind = {7: "SPS", 8: "PPS"}.get(t)
        if kind:
            seen.setdefault(kind, set()).add(bytes(u))
    return seen


def main(path):
    seen = kinds(path)
    bad = {k: v for k, v in seen.items() if len(v) != 1}
    if not seen:
        print(f"{path}: no parameter sets at all")
        return 1
    for k, v in sorted(bad.items()):
        print(f"{path}: {len(v)} distinct {k}: " + ", ".join(sorted(x.hex() for x in v)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
