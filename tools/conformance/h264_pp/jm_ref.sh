#!/bin/bash
# jm_ref.sh <stream-name>... — per-frame MD5s of the JM reference decoder's
# output for professional-profile streams libavcodec cannot decode, written
# to jm/<name>.md5 in the form run_conf.sh reads ("<i> <md5>" per frame).
#
# The MD5 is of the frame as h26xdec packs it: Y, Cb, Cr planes, 16-bit
# little-endian above 8 bits. JM writes 4:4:4 streams whose VUI says
# matrix_coefficients 0 (RGB content — the Sejong FMO streams) as R, G, B,
# i.e. Cr, Y, Cb; that order is undone here, so what is compared is the
# decoded planes, not JM's file layout. ldecod does not report the VUI, so
# the caller says which streams those are: `RGB=1 jm_ref.sh <streams>`.
#
#   LDECOD=path   the ldecod binary (default: the JM 19.1 build under H26X_WORK)
#   H26X_WORK=dir the scratch directory holding JM/ and conf/ (default: ../..,
#                 this script being run from <H26X_WORK>/conf/h264_pp like run_conf.sh)
cd "$(dirname "$0")"
WORK=${H26X_WORK:-$(cd ../.. && pwd)}
LDECOD=${LDECOD:-$WORK/JM/bin/vs18/msvc-19.50/x86_64/release/ldecod.exe}
mkdir -p jm
for name in "$@"; do
  bs=$(find "streams/$name" -maxdepth 1 -type f \( -iname "*.264" -o -iname "*.bits" -o -iname "*.jsv" -o -iname "*.h264" \) | head -1)
  [ -z "$bs" ] && { echo "no stream for $name" >&2; continue; }
  tmp=$(mktemp -d)
  cp "$bs" "$tmp/in.264"
  cp "$WORK/JM/cfg/decoder.cfg" "$tmp/"
  (cd "$tmp" && "$LDECOD" -d decoder.cfg -p InputFile=in.264 -p OutputFile=out.yuv -p Silent=1 -p WriteUV=1 > log.txt 2>&1) \
    || { echo "ldecod failed on $name" >&2; cat "$tmp/log.txt" >&2; rm -rf "$tmp"; continue; }
  # Geometry, format and bit depth from ldecod's own report.
  wh=$(grep -o "Image Format *: *[0-9]*x[0-9]*" "$tmp/log.txt" | head -1 | grep -o "[0-9]*x[0-9]*")
  fmt=$(grep -o "Color Format *: *[0-9:]*" "$tmp/log.txt" | head -1 | grep -o "[0-9]:[0-9]:[0-9]")
  depth=$(grep -o "Color Format *: *[0-9:]* *([0-9]*" "$tmp/log.txt" | head -1 | grep -o "[0-9]*$")
  rgb=${RGB:-0}
  python - "$tmp/out.yuv" "$wh" "$fmt" "${depth:-8}" "$rgb" > "jm/$name.md5" <<'EOF'
import hashlib, sys
path, wh, fmt, depth, rgb = sys.argv[1:6]
w, h = map(int, wh.split('x'))
bps = 2 if int(depth) > 8 else 1
cw, ch = {'4:0:0': (0, 0), '4:2:0': (w // 2, h // 2), '4:2:2': (w // 2, h), '4:4:4': (w, h)}[fmt]
ysz, csz = w * h * bps, cw * ch * bps
if fmt == '4:0:0':
    csz = (w // 2) * (h // 2) * bps  # WriteUV=1: grey 4:2:0 chroma, as h26xdec pads
fs = ysz + 2 * csz
data = open(path, 'rb').read()
for i in range(len(data) // fs):
    f = data[i * fs:(i + 1) * fs]
    if rgb != '0' and fmt == '4:4:4':
        # JM wrote R, G, B = Cr, Y, Cb.
        f = f[ysz:2 * ysz] + f[2 * ysz:3 * ysz] + f[:ysz]
    print(i, hashlib.md5(f).hexdigest())
EOF
  echo "$name: $(wc -l < "jm/$name.md5") frames ($wh $fmt ${depth:-8}-bit$([ "$rgb" != 0 ] && echo ', RGB order undone'))"
  rm -rf "$tmp"
done
