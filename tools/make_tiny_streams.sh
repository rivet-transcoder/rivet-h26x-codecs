#!/bin/bash
# make_tiny_streams.sh [dir] — make the three streams tests/decode.rs
# vendors (tests/data/tiny_cabac.264, tiny_cavlc.264, tiny.265) and print
# the hash each test expects, computed from the REFERENCE decoder's output:
# JM's ldecod for the two H.264 streams, HM's TAppDecoder for HEVC (see
# ref_decode.py). The expectations are anchored to those decoders, not to
# this crate's.
#
# The source is synth_source.py's `detail` recipe, 80x64, twelve frames.
# The encoders are the x264 and x265 command-line programs reading raw YUV:
#   x264 0.165 r3223 0480cb0, built without libavformat / ffms / swscale
#        (`./configure --disable-lavf --disable-ffms --disable-swscale`)
#   x265 4.2, 8-bit
# Both write their build and options into the stream's SEI.
#
#   X264=path X265=path    the encoders (default: on PATH)
#   LDECOD=path TAPPDECODER=path   the reference decoders (ref_decode.py)
set -eu
TOOLS=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-.}
X264=${X264:-x264}
X265=${X265:-x265}
mkdir -p "$OUT" && cd "$OUT"
python "$TOOLS/synth_source.py" detail --size 80x64 --frames 12 --format 420 tiny_src.yuv
raw=(--input-res 80x64 --fps 10 --input-csp i420)
"$X264" --preset veryslow --crf 28 --keyint 6 --bframes 2 --threads 1 "${raw[@]}" -o tiny_cabac.264 tiny_src.yuv
"$X264" --preset veryslow --crf 28 --keyint 6 --bframes 2 --threads 1 --no-cabac "${raw[@]}" -o tiny_cavlc.264 tiny_src.yuv
# --b-adapt 0 keeps the B pictures before the CRA, so the stream has RASL
# pictures; --frame-threads 1 --pools 1 make the output independent of the
# machine's core count.
"$X265" --preset slow --crf 30 --keyint 6 --bframes 2 --b-adapt 0 --repeat-headers \
  --frame-threads 1 --pools 1 --no-wpp "${raw[@]}" -o tiny.265 tiny_src.yuv
for s in tiny_cabac.264 tiny_cavlc.264 tiny.265; do
  python "$TOOLS/ref_decode.py" decode "$s" "$s.ref.yuv"
  python - "$s" "$s.ref.yuv" <<'EOF'
import struct, sys
# The hash tests/decode.rs computes: FNV-1a over each frame's width and
# height (u32 little-endian) and its packed planes, in output order.
name, path = sys.argv[1:3]
d = open(path, 'rb').read()
w, h = 80, 64
fs = w * h * 3 // 2
assert len(d) % fs == 0, 'not whole 80x64 4:2:0 frames'
x = 0xcbf29ce484222325
for i in range(len(d) // fs):
    for b in struct.pack('<II', w, h) + d[i * fs:(i + 1) * fs]:
        x = ((x ^ b) * 0x100000001b3) & 0xffffffffffffffff
print(f'{name}: {len(d) // fs} frames, hash {x:#018x}')
EOF
done
