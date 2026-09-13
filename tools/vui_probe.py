"""Does libavcodec read the colour this stream was told to carry?

The VUI property. A colour description is three H.273 code points and a
range flag in the SPS VUI, and nothing else in the gate can see whether
they are there: SELF and CROSS compare samples, which do not change when
the VUI says BT.2020 instead of nothing, and the crate's own parser is the
inverse of its own writer, so a shared misreading of E.1.1 / E.2.1 would
round-trip cleanly. What settles it is a third reader — ffprobe, which
reports the VUI as names — agreeing field by field with what the encoder
was asked to write.

    python vui_probe.py stream.h264|stream.h265 P:T:M tv|pc
           [--chroma-loc N]
           [--mastering-display G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)]
           [--content-light MAXCLL,MAXFALL]

`--chroma-loc` asks the same of the VUI's chroma siting
(`chroma_sample_loc_type`, which ffprobe names as `chroma_location`),
and only then: for a VUI that says nothing about siting libavcodec
reports the type 0 the standard infers ("left") on 4:2:0 and
"unspecified" on 4:2:2 / 4:4:4 — and the latter whatever the VUI says,
since it exports a siting for 4:2:0 alone. So absence is not checkable
here (the crate's parser test holds "unasked, unwritten"), a written 0
is invisible, and a siting is only verifiable on a 4:2:0 stream —
which is why the gate's siting rows name 4:2:0 clips and carry 1 and 2.

The two HDR arguments extend the same question to the HDR10 static
metadata SEIs (mastering display colour volume, content light level),
which ffprobe reports as side data on the first frame, with the values in
the SEI's own units (chromaticities as n/50000, luminances as n/10000) —
compared here as the integers the encoder was handed. The crate has no
reader for these SEIs at all, so this is the only reader.

Exit 0 when ffprobe names exactly what was asked for, 1 otherwise,
printing which field disagreed. The names are libavutil's
(`av_color_*_name`), listed here for the code points a caller might ask
for; a code outside the table fails by name rather than passing vacuously.
"""
import json
import os
import subprocess
import sys

PRIMARIES = {
    1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
    8: "film", 9: "bt2020", 10: "smpte428", 11: "smpte431", 12: "smpte432",
    22: "jedec-p22",
}
TRANSFER = {
    1: "bt709", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 7: "smpte240m",
    8: "linear", 9: "log100", 10: "log316", 11: "iec61966-2-4", 12: "bt1361e",
    13: "iec61966-2-1", 14: "bt2020-10", 15: "bt2020-12", 16: "smpte2084",
    17: "smpte428", 18: "arib-std-b67",
}
MATRIX = {
    0: "gbr", 1: "bt709", 4: "fcc", 5: "bt470bg", 6: "smpte170m",
    7: "smpte240m", 8: "ycgco", 9: "bt2020nc", 10: "bt2020c", 11: "smpte2085",
    12: "chroma-derived-nc", 13: "chroma-derived-c", 14: "ictcp",
}
# H.273 chroma_sample_loc_type -> libavutil's av_chroma_location_name
# (AVCHROMA_LOC_LEFT is 1, so the code is the enum minus one).
CHROMA_LOC = {0: "left", 1: "center", 2: "topleft", 3: "top", 4: "bottomleft", 5: "bottom"}


def parse_mastering(spec):
    """`G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)` -> dict of the ffprobe field
    names to the integers wanted, or None if the text is not that."""
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


def numerator(v):
    """ffprobe prints these as `n/d` strings; the SEI's integer is n."""
    return int(str(v).split("/")[0])


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
            if chroma_loc not in CHROMA_LOC:
                print(f"vui_probe: no libavutil name for chroma_location={chroma_loc}; extend the table")
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
        ("color_primaries", PRIMARIES, p),
        ("color_transfer", TRANSFER, t),
        ("color_space", MATRIX, m),
    ):
        if code not in table:
            print(f"vui_probe: no libavutil name for {field}={code}; extend the table")
            return 1
        want[field] = table[code]
    want["color_range"] = rng
    if chroma_loc is not None:
        want["chroma_location"] = CHROMA_LOC[chroma_loc]

    ffprobe = os.environ.get("FFPROBE", "ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=color_primaries,color_transfer,color_space,color_range,chroma_location",
            "-of", "default=noprint_wrappers=1", stream,
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"vui_probe: ffprobe failed: {out.stderr.strip()[-200:]}")
        return 1
    got = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            got[k.strip()] = v.strip()
    bad = []
    for k, v in want.items():
        if got.get(k) != v:
            bad.append(f"{k}: wanted {v}, ffprobe says {got.get(k, '(absent)')}")

    if mastering is not None or cll is not None:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-read_intervals", "%+#1", "-of", "json", stream,
            ],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"vui_probe: ffprobe -show_frames failed: {out.stderr.strip()[-200:]}")
            return 1
        frames = json.loads(out.stdout).get("frames", [])
        side = {}
        for sd in (frames[0].get("side_data_list", []) if frames else []):
            side[sd.get("side_data_type")] = sd
        if mastering is not None:
            sd = side.get("Mastering display metadata")
            if sd is None:
                bad.append("mastering display: wanted one, ffprobe sees no such side data")
            else:
                for k, v in mastering.items():
                    if k not in sd or numerator(sd[k]) != v:
                        bad.append(f"mastering display {k}: wanted {v}, ffprobe says {sd.get(k, '(absent)')}")
                got["mastering_display"] = "ok"
        if cll is not None:
            sd = side.get("Content light level metadata")
            if sd is None:
                bad.append("content light level: wanted one, ffprobe sees no such side data")
            else:
                for k, v in cll.items():
                    if sd.get(k) != v:
                        bad.append(f"content light level {k}: wanted {v}, ffprobe says {sd.get(k, '(absent)')}")
                got["content_light_level"] = "ok"
    if bad:
        print("; ".join(bad))
        return 1
    print(" ".join(f"{k}={v}" for k, v in got.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
