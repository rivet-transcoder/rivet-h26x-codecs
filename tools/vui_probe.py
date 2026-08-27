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

Exit 0 when ffprobe's four colour fields name exactly the codes given, 1
otherwise, printing which field disagreed. The names are libavutil's
(`av_color_*_name`), listed here for the code points a caller might ask
for; a code outside the table fails by name rather than passing vacuously.
"""
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


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    stream, colour, rng = sys.argv[1:]
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

    ffprobe = os.environ.get("FFPROBE", "ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=color_primaries,color_transfer,color_space,color_range",
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
    if bad:
        print("; ".join(bad))
        return 1
    print(" ".join(f"{k}={v}" for k, v in got.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
