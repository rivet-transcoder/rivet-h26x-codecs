"""Do two H.264 Annex-B streams differ only by parameter-set repeats the second drops?

The mask behind identity_encode.sh's MASK=psrep. The H.264 encoder used to
open every access unit with its SPS and PPS; it now writes them in IDR
access units and wherever a set is new or changed. A parameter set stays in
force until replaced (7.4.1.2.1), so what went is exactly the repeats: an
SPS or PPS outside an IDR access unit, byte-identical to the one a decoder
already holds under that kind.

Both streams are normalised by removing such repeats — the first stream's
are what the change dropped, the second should have none — and must then
be equal NAL unit for NAL unit. H.265 streams have no repeats to drop and
are compared as they are.

An access unit begins at the first non-VCL NAL unit after a VCL one (H.264
7.4.1.2.3); it is an IDR access unit when one of its slices is
(nal_unit_type 5). A picture that opens with its slice — every non-IDR
picture of the second stream that carries no SEI — falls into the group
before it. That is harmless here: both encoders open every IDR access unit
with its parameter sets, so an IDR slice always lands in its own access
unit's group, and a parameter set, which follows a slice or begins the
stream, always opens the group of the access unit it belongs to.

With --vui (identity_encode.sh's MASK=psrep+vui, for the two changes
landing together) a sequence parameter set left differing after that is
held to tools/vui_mask.py's rule instead of refused: the second adds the
VUI frame clock and nothing else.

    python psrep_mask.py [--vui] h264|h265 A B

Exit 0 when the streams are equal under the mask, printing a label for
what it took — PSREPEAT-ONLY, VUI-ONLY or PSREPEAT+VUI-ONLY — and then
what that was (`12 SPS + 12 PPS repeats, 168 bytes; SPS VUI +clock 1/60
fixed`), or `identical`; 1 otherwise, printing the first difference.
"""
import sys

import vui_mask


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


def access_units(nals):
    """H.264 NAL units grouped into access units."""
    aus, cur, seen_vcl = [], [], False
    for nal in nals:
        vcl = 1 <= (nal[0] & 0x1F) <= 5
        if not vcl and seen_vcl:
            aus.append(cur)
            cur, seen_vcl = [], False
        cur.append(nal)
        seen_vcl |= vcl
    if cur:
        aus.append(cur)
    return aus


def drop_repeats(nals):
    """The stream without its parameter-set repeats, and what was dropped:
    `(nals, sps_count, pps_count, bytes)`. The id is not parsed: this
    encoder writes one SPS and one PPS (id 0), and a set of the same kind
    with other bytes is kept as a change."""
    held = {7: None, 8: None}
    out, counts, dropped = [], {7: 0, 8: 0}, 0
    for au in access_units(nals):
        idr = any(n[0] & 0x1F == 5 for n in au)
        for nal in au:
            t = nal[0] & 0x1F
            if t in held:
                if not idr and held[t] == nal:
                    counts[t] += 1
                    dropped += len(nal) + 4
                    continue
                held[t] = nal
            out.append(nal)
    return out, counts[7], counts[8], dropped


def main():
    args = sys.argv[1:]
    vui = args[:1] == ["--vui"]
    if vui:
        args = args[1:]
    if len(args) != 3 or args[0] not in ("h264", "h265"):
        sys.exit("usage: psrep_mask.py [--vui] h264|h265 A B")
    codec = args[0]
    a = nal_units(open(args[1], "rb").read())
    b = nal_units(open(args[2], "rb").read())
    labels, notes = [], []
    if codec == "h264":
        a, sps, pps, gone = drop_repeats(a)
        b, bsps, bpps, _ = drop_repeats(b)
        if bsps or bpps:
            print(f"MOVED: the second stream repeats parameter sets ({bsps} SPS, {bpps} PPS)")
            return 1
        if sps or pps:
            labels.append("PSREPEAT")
            notes.append(f"{sps} SPS + {pps} PPS repeats, {gone} bytes")
    if len(a) != len(b):
        print(f"MOVED: {len(a)} NAL units against {len(b)} after the repeats")
        return 1
    sps_type = vui_mask.CODECS[codec][0]
    clocks = set()
    for k, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        if vui and vui_mask.nal_type(codec, x) == sps_type == vui_mask.nal_type(codec, y):
            added = vui_mask.added_clock(codec, x, y)
            if added is not None:
                clocks.add(added)
                continue
        print(f"MOVED: NAL unit {k} (first byte {x[0]:02x}) differs")
        return 1
    if clocks:
        labels.append("VUI")
        notes.append("; ".join(sorted(clocks)))
    print(f"{'+'.join(labels)}-ONLY {'; '.join(notes)}" if labels else "identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
