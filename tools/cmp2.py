"""Compare two raw YUV files plane by plane, any chroma format / depth.

usage: cmp2.py ref.yuv mine.yuv W H {420|422|444|400} bytes_per_sample [max_frames]
Prints per frame: OK, or per plane the diff count and the first differing
sample (x, y, ref, mine) plus its 8x8 neighbourhood in both files.
"""
import sys

ref, mine, w, h, fmt, bps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
maxf = int(sys.argv[7]) if len(sys.argv) > 7 else 1 << 30
sub = {"420": (2, 2), "422": (2, 1), "444": (1, 1), "400": None}[fmt]
planes = [(w, h)]
if sub:
    planes += [(w // sub[0], h // sub[1])] * 2
fs = sum(pw * ph for pw, ph in planes) * bps
a = open(ref, "rb").read()
b = open(mine, "rb").read()
n = min(len(a), len(b)) // fs
print("frames", len(a) // fs, len(b) // fs)


def sample(buf, off):
    if bps == 1:
        return buf[off]
    return buf[off] | (buf[off + 1] << 8)


for f in range(min(n, maxf)):
    fa = a[f * fs:(f + 1) * fs]
    fb = b[f * fs:(f + 1) * fs]
    if fa == fb:
        print(f, "OK")
        continue
    off = 0
    for pi, (pw, ph) in enumerate(planes):
        size = pw * ph * bps
        pa = fa[off:off + size]
        pb = fb[off:off + size]
        off += size
        if pa == pb:
            continue
        cnt = 0
        first = None
        for i in range(pw * ph):
            if pa[i * bps:(i + 1) * bps] != pb[i * bps:(i + 1) * bps]:
                cnt += 1
                if first is None:
                    first = i
        x, y = first % pw, first // pw
        print(f"frame {f} plane {pi} ({pw}x{ph}) diffs {cnt} first at x={x} y={y} ref={sample(pa, first * bps)} mine={sample(pb, first * bps)}")
        bx, by = (x // 8) * 8, (y // 8) * 8
        for yy in range(by, min(by + 8, ph)):
            ra = [sample(pa, (yy * pw + xx) * bps) for xx in range(bx, min(bx + 8, pw))]
            rb = [sample(pb, (yy * pw + xx) * bps) for xx in range(bx, min(bx + 8, pw))]
            print("   ref", ra, " mine", rb)
