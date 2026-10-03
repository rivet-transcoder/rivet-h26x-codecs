"""Does an independent reader see the colour this stream was told to carry?

The VUI property. A colour description is three H.273 code points and a
range flag in the SPS VUI, and nothing else in the gate can see whether
they are there: SELF and CROSS compare samples, which do not change when
the VUI says BT.2020 instead of nothing, and the crate's own parser is the
inverse of its own writer, so a shared misreading of E.1.1 / E.2.1 would
round-trip cleanly. What settles it is a third reader — MediaInfo
(MediaArea's MediaInfoLib, run as a program), which reports the VUI as
names — agreeing field by field with what the encoder was asked to write.

    python vui_probe.py stream.h264|stream.h265 P:T:M tv|pc
           [--chroma-loc N]
           [--mastering-display G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)]
           [--content-light MAXCLL,MAXFALL]

The chroma siting (`chroma_sample_loc_type`) is checked on every call:
MediaInfo reports a written siting as `Type N` (a written 0 included) and
nothing at all when the VUI carries none, so `--chroma-loc N` must be
reported as `Type N` and, without it, no siting may be reported.

The two HDR arguments extend the same question to the HDR10 static
metadata SEIs (mastering display colour volume, content light level),
compared as the integers the encoder was handed (chromaticities in units of
1/50000, luminances of 1/10000 cd/m2). For HEVC the reader is HM's
TAppDecoder, the reference decoder, which prints every decoded SEI field
(`--OutputDecodedSEIMessagesFilename`). For H.264 no reference program
prints them, so the payloads are read by the small parser in this file —
written from H.264 D.1.29 / D.1.31 (the same syntax as HEVC's D.2.28 /
D.2.35), separately from the crate's writer — and MediaInfo's reading
(the mastering primaries as a named set, the luminances, MaxCLL / MaxFALL)
must agree with it.

Exit 0 when every reader names exactly what was asked for, 1 otherwise,
printing which field disagreed. The names are MediaInfo's for the code
points a caller might ask for; a code outside the table fails by name
rather than passing vacuously.

    MEDIAINFO=path    the mediainfo CLI (default: `mediainfo` on PATH)
    TAPPDECODER=path  HM's TAppDecoder (default: `TAppDecoder` on PATH)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

PRIMARIES = {
    1: "BT.709", 4: "BT.470 System M", 5: "BT.601 PAL", 6: "BT.601 NTSC", 7: "SMPTE 240M",
    8: "Generic film", 9: "BT.2020", 10: "XYZ", 11: "DCI P3", 12: "Display P3",
    22: "EBU Tech 3213",
}
TRANSFER = {
    1: "BT.709", 4: "BT.470 System M", 5: "BT.470 System B/G", 6: "BT.601", 7: "SMPTE 240M",
    8: "Linear", 9: "Logarithmic (100:1)", 10: "Logarithmic (316.22777:1)", 11: "xvYCC",
    12: "BT.1361", 13: "sRGB/sYCC", 14: "BT.2020 (10-bit)", 15: "BT.2020 (12-bit)", 16: "PQ",
    17: "SMPTE 428M", 18: "HLG",
}
MATRIX = {
    0: "Identity", 1: "BT.709", 4: "FCC 73.682", 5: "BT.470 System B/G", 6: "BT.601",
    7: "SMPTE 240M", 8: "YCgCo", 9: "BT.2020 non-constant", 10: "BT.2020 constant",
    11: "Y'D'zD'x", 12: "Chromaticity-derived non-constant", 13: "Chromaticity-derived constant",
    14: "ICtCp",
}
# Mastering display primaries MediaInfo names, in the SEI's units:
# (G, B, R, WP) as (x, y) pairs.
NAMED_PRIMARIES = {
    "Display P3": ((13250, 34500), (7500, 3000), (34000, 16000), (15635, 16450)),
    "BT.2020": ((8500, 39850), (6550, 2300), (35400, 14600), (15635, 16450)),
    "BT.709": ((15000, 30000), (7500, 3000), (32000, 16500), (15635, 16450)),
}
MD_FIELDS = ("green_x", "green_y", "blue_x", "blue_y", "red_x", "red_y",
             "white_point_x", "white_point_y", "max_luminance", "min_luminance")


def parse_mastering(spec):
    """`G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)` -> dict of field names to the
    integers wanted, or None if the text is not that."""
    want = {}
    rest = spec
    for label, fields in (("G", ("green_x", "green_y")), ("B", ("blue_x", "blue_y")),
                          ("R", ("red_x", "red_y")), ("WP", ("white_point_x", "white_point_y")),
                          ("L", ("max_luminance", "min_luminance"))):
        if not rest.startswith(label + "("):
            return None
        end = rest.find(")")
        if end < 0:
            return None
        parts = rest[len(label) + 1:end].split(",")
        if len(parts) != 2:
            return None
        try:
            want[fields[0]], want[fields[1]] = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        rest = rest[end + 1:]
    return want if rest == "" else None


# ---------------------------------------------------------------- SEI reader
def nal_units(data):
    """Annex-B NAL units, emulation prevention removed."""
    i, n = 0, len(data)
    starts = []
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        while e > s and data[e - 1] == 0:
            e -= 1
        raw = data[s:e]
        out = bytearray()
        zeros = 0
        for b in raw:
            if zeros >= 2 and b == 3:
                zeros = 0
                continue
            out.append(b)
            zeros = zeros + 1 if b == 0 else 0
        yield bytes(out)


def sei_payloads(stream, hevc):
    """(payloadType, payload bytes) for every SEI message in the stream."""
    for nal in nal_units(open(stream, "rb").read()):
        if not nal:
            continue
        if hevc:
            if (nal[0] >> 1) & 0x3F not in (39, 40):
                continue
            body = nal[2:]
        else:
            if nal[0] & 0x1F != 6:
                continue
            body = nal[1:]
        i = 0
        while i < len(body) and not (body[i] == 0x80 and i == len(body) - 1):
            ptype = 0
            while body[i] == 0xFF:
                ptype += 255
                i += 1
            ptype += body[i]
            i += 1
            psize = 0
            while body[i] == 0xFF:
                psize += 255
                i += 1
            psize += body[i]
            i += 1
            yield ptype, body[i:i + psize]
            i += psize


def sei_hdr(stream, hevc):
    md = cll = None
    for ptype, p in sei_payloads(stream, hevc):
        if ptype == 137 and len(p) >= 24 and md is None:
            u16 = [int.from_bytes(p[k:k + 2], "big") for k in range(0, 16, 2)]
            md = dict(zip(MD_FIELDS, u16 + [int.from_bytes(p[16:20], "big"), int.from_bytes(p[20:24], "big")]))
        elif ptype == 144 and len(p) >= 4 and cll is None:
            cll = {"max_content": int.from_bytes(p[0:2], "big"), "max_average": int.from_bytes(p[2:4], "big")}
    return md, cll


def hm_hdr(stream):
    """HM's own print of the two SEIs: (mastering dict or None, cll dict or None)."""
    dec = os.environ.get("TAPPDECODER") or shutil.which("TAppDecoder")
    if not dec:
        raise RuntimeError("no TAppDecoder (set TAPPDECODER or put it on PATH)")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sei.txt")
        r = subprocess.run([dec, "-b", os.path.abspath(stream), f"--OutputDecodedSEIMessagesFilename={out}"],
                           cwd=tmp, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"TAppDecoder failed: {(r.stdout + r.stderr).strip()[-200:]}")
        text = open(out).read()
    vals = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.rsplit(":", 1)
            k = k.strip()
            if k not in vals:
                try:
                    vals[k] = int(v.strip())
                except ValueError:
                    pass
    md = None
    if "display_primaries_x[0]" in vals:
        md = {
            "green_x": vals["display_primaries_x[0]"], "green_y": vals["display_primaries_y[0]"],
            "blue_x": vals["display_primaries_x[1]"], "blue_y": vals["display_primaries_y[1]"],
            "red_x": vals["display_primaries_x[2]"], "red_y": vals["display_primaries_y[2]"],
            "white_point_x": vals["white_point_x"], "white_point_y": vals["white_point_y"],
            "max_luminance": vals["max_display_mastering_luminance"],
            "min_luminance": vals["min_display_mastering_luminance"],
        }
    cll = None
    if "max_content_light_level" in vals:
        cll = {"max_content": vals["max_content_light_level"], "max_average": vals["max_pic_average_light_level"]}
    return md, cll


def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print(__doc__)
        return 2
    stream, colour, rng = args[:3]
    mastering = None
    cll = None
    chroma_loc = None
    rest = args[3:]
    while rest:
        if rest[0] == "--chroma-loc" and len(rest) > 1:
            try:
                chroma_loc = int(rest[1])
            except ValueError:
                print(f"vui_probe: --chroma-loc wants a chroma_sample_loc_type, got {rest[1]!r}")
                return 2
            if not 0 <= chroma_loc <= 5:
                print(f"vui_probe: chroma_sample_loc_type {chroma_loc} is not 0..5")
                return 1
            rest = rest[2:]
        elif rest[0] == "--mastering-display" and len(rest) > 1:
            mastering = parse_mastering(rest[1])
            if mastering is None:
                print(f"vui_probe: --mastering-display wants G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min), got {rest[1]!r}")
                return 2
            rest = rest[2:]
        elif rest[0] == "--content-light" and len(rest) > 1:
            try:
                a, b = rest[1].split(",")
                cll = {"max_content": int(a), "max_average": int(b)}
            except ValueError:
                print(f"vui_probe: --content-light wants MAXCLL,MAXFALL, got {rest[1]!r}")
                return 2
            rest = rest[2:]
        else:
            print(__doc__)
            return 2
    try:
        p, t, m = (int(x) for x in colour.split(":"))
    except ValueError:
        print(f"vui_probe: --color wants P:T:M, got {colour!r}")
        return 2
    if rng not in ("tv", "pc"):
        print(f"vui_probe: range must be tv or pc, got {rng!r}")
        return 2
    want = {}
    for field, table, code in (
        ("colour_primaries", PRIMARIES, p),
        ("transfer_characteristics", TRANSFER, t),
        ("matrix_coefficients", MATRIX, m),
    ):
        if code not in table:
            print(f"vui_probe: no MediaInfo name for {field}={code}; extend the table")
            return 1
        want[field] = table[code]
    want["colour_range"] = "Limited" if rng == "tv" else "Full"
    want["ChromaSubsampling_Position"] = f"Type {chroma_loc}" if chroma_loc is not None else "(absent)"

    mediainfo = os.environ.get("MEDIAINFO") or shutil.which("mediainfo") or "mediainfo"
    # MediaInfo does not recognise an elementary stream of a few hundred
    # bytes at all. The stream repeated is still a legal stream (each copy
    # starts with its parameter sets and an IDR), so a short one is handed
    # over as enough copies of itself.
    data = open(stream, "rb").read()
    with tempfile.TemporaryDirectory() as tmp:
        probe = stream
        if len(data) < 8192:
            probe = os.path.join(tmp, os.path.basename(stream))
            open(probe, "wb").write(data * (8192 // max(len(data), 1) + 1))
        out = subprocess.run([mediainfo, "-f", "--Output=JSON", probe], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"vui_probe: mediainfo failed: {out.stderr.strip()[-200:]}")
        return 1
    try:
        tracks = json.loads(out.stdout)["media"]["track"]
    except (ValueError, KeyError, TypeError):
        print("vui_probe: mediainfo printed no tracks")
        return 1
    video = next((tr for tr in tracks if tr.get("@type") == "Video"), {})
    got = {}
    bad = []
    for k, v in want.items():
        g = video.get(k, "(absent)")
        got[k] = g
        if g != v:
            bad.append(f"{k}: wanted {v}, mediainfo says {g}")

    if mastering is not None or cll is not None:
        hevc = stream.lower().endswith((".265", ".h265", ".hevc"))
        try:
            md_got, cll_got = hm_hdr(stream) if hevc else sei_hdr(stream, False)
        except RuntimeError as e:
            print(f"vui_probe: {e}")
            return 1
        reader = "TAppDecoder" if hevc else "the SEI payload"
        if mastering is not None:
            if md_got is None:
                bad.append(f"mastering display: wanted one, {reader} has none")
            else:
                for k, v in mastering.items():
                    if md_got.get(k) != v:
                        bad.append(f"mastering display {k}: wanted {v}, {reader} says {md_got.get(k, '(absent)')}")
                # MediaInfo's reading too: the primaries as a named set when
                # they are one, the luminances in cd/m2.
                name = video.get("MasteringDisplay_ColorPrimaries")
                prim = ((mastering["green_x"], mastering["green_y"]), (mastering["blue_x"], mastering["blue_y"]),
                        (mastering["red_x"], mastering["red_y"]), (mastering["white_point_x"], mastering["white_point_y"]))
                named = next((n for n, v in NAMED_PRIMARIES.items() if v == prim), None)
                if named and name != named:
                    bad.append(f"mastering display primaries: wanted {named}, mediainfo says {name}")
                for k, v in (("MasteringDisplay_Luminance_Max", mastering["max_luminance"]),
                             ("MasteringDisplay_Luminance_Min", mastering["min_luminance"])):
                    try:
                        ok = abs(float(video.get(k, "nan")) - v / 10000) <= 0.00005 + v / 10000 * 1e-6
                    except ValueError:
                        ok = False
                    if not ok:
                        bad.append(f"{k}: wanted {v / 10000:g}, mediainfo says {video.get(k, '(absent)')}")
                got["mastering_display"] = "ok"
        if cll is not None:
            if cll_got is None:
                bad.append(f"content light level: wanted one, {reader} has none")
            else:
                for k, v in cll.items():
                    if cll_got.get(k) != v:
                        bad.append(f"content light level {k}: wanted {v}, {reader} says {cll_got.get(k, '(absent)')}")
                for k, v in (("MaxCLL", cll["max_content"]), ("MaxFALL", cll["max_average"])):
                    if video.get(k) != str(v):
                        bad.append(f"{k}: wanted {v}, mediainfo says {video.get(k, '(absent)')}")
                got["content_light_level"] = "ok"
    if bad:
        print("; ".join(bad))
        return 1
    print(" ".join(f"{k}={v}" for k, v in got.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
