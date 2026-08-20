#!/bin/bash
# verify_encode.sh — what has to hold before an encoder change lands.
#
# The decoders have conformance suites. The encoders cannot: a standard says
# what a decoder must do with a bitstream and leaves an encoder free to choose
# any legal one, so there is no golden output to compare against. What replaces
# it is three properties, two of them exact.
#
#   1. SELF     the bitstream decodes to what the encoder thought it encoded.
#               The encoder dumps its own reconstruction; our decoder must
#               reproduce it byte for byte. A mismatch is a desync between
#               encoder and decoder state and is always a bug. Needs no
#               reference data, and catches the largest class of faults.
#
#   2. CROSS    libavcodec decoding our output produces the same pictures our
#               decoder does. Property 1 is self-consistent and would pass if
#               both of our sides shared a misreading of the standard; this is
#               what makes the bitstream legal rather than merely
#               self-compatible.
#
#   3. QUALITY  PSNR of the reconstruction against the source. The only one of
#               the three that is a measurement rather than a check — so it is
#               REPORTED, never gated, except in lossless mode where it must be
#               infinite and the check becomes exact like the others.
#
# Usage: verify_encode.sh [encoder] [decoder]
#   H26X_WORK=dir   scratch directory holding the source clips (default: here)
#   JOBS=n          configurations in parallel (default 4)
#
# Safe to run concurrently with itself and with verify.sh: private binary
# copies and a private scratch directory per run, for the reason recorded in
# tools/README.md — a shared copy produces a green run that tested somebody
# else's build.
cd "${H26X_WORK:-$(dirname "$0")}"
ENC=${1:-../release/examples/h26xenc.exe}
DEC=${2:-../release/examples/h26xdec.exe}
[ -f "$ENC" ] || ENC=${ENC%.exe}
[ -f "$DEC" ] || DEC=${DEC%.exe}
FFMPEG=${FFMPEG:-ffmpeg}
TAG=$$
OUT=enc_out_$TAG
JOBS=${JOBS:-4}
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT
fail=0

# Source clips: raw planar YUV, named <name>_<W>x<H>_<fmt>.yuv so the geometry
# travels with the file rather than living in this script.
SOURCES=${SOURCES:-$(ls src_*.yuv 2>/dev/null)}
if [ -z "$SOURCES" ]; then
  echo "no source clips (src_*.yuv); nothing to verify" >&2
  exit 2
fi

# Configurations. Each is a name and the encoder flags for it. The list starts
# at the simplest thing that can be legal and adds one axis at a time, because
# when several are red at once the simplest one names the bug.
#
# Each axis appears in a CAVLC form as well as a CABAC one. That is not
# redundancy: it lets inter prediction be verified without waiting for CABAC
# slice writing and vice versa, so two people can make progress against this
# gate at the same time without one of them being blocked behind the other.
#
# The high-QP rows exist because a fixed quantiser hides a whole class of bug.
# Coding H.264's chroma planes at the luma quantiser — plainly wrong — passed
# every row of this gate, because the chroma QP mapping is the identity up to
# 29 and every row lived below that. At QP 40 the same mutation fails SELF on
# both entropy coders at once. Any table the codec indexes by QP has the same
# shape, so one row per codec sits high enough to leave the identity region.
CONFIGS=${CONFIGS:-"
lossless-intra|--codec h264 --lossless --gop 0
cqp-intra|--codec h264 --qp 26 --gop 0
cqp-ip|--codec h264 --qp 26 --gop 8
cqp-ipb|--codec h264 --qp 26 --gop 8 --bframes 2
cqp40-ip|--codec h264 --qp 40 --gop 8
cavlc-intra|--codec h264 --qp 26 --gop 0 --cavlc
cavlc-ip|--codec h264 --qp 26 --gop 8 --cavlc
cavlc-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc
cavlc40-intra|--codec h264 --qp 40 --gop 0 --cavlc
hevc-lossless-intra|--codec h265 --lossless --gop 0
hevc-cqp-intra|--codec h265 --qp 26 --gop 0
hevc-cqp-ip|--codec h265 --qp 26 --gop 8
hevc-cqp40-intra|--codec h265 --qp 40 --gop 0
"}

one() {
  src=$1; name=$2; flags=$3
  base=$(basename "$src" .yuv)
  geom=$(echo "$base" | sed -n 's/.*_\([0-9]\+x[0-9]\+\)_.*/\1/p')
  fmt=$(echo "$base" | sed -n 's/.*_[0-9]\+x[0-9]\+_\(.*\)/\1/p')
  tag="$base/$name"
  ext=h264; case "$flags" in *"--codec h265"*) ext=h265 ;; esac
  bs="$OUT/$base.$name.$ext"
  rec="$OUT/$base.$name.rec.yuv"

  # The encoder writes the bitstream and, separately, the reconstruction it
  # believes that bitstream carries.
  if ! "$ENC" --input "$src" --size "$geom" --format "$fmt" \
       $flags --output "$bs" --recon "$rec" > "$OUT/$base.$name.enc.log" 2>&1; then
    echo "ENCODE-FAIL $tag: $(tail -1 "$OUT/$base.$name.enc.log" | head -c 100)"
    return 1
  fi

  # 1. SELF.
  ours="$OUT/$base.$name.ours.yuv"
  # H.264 4:0:0: ask the decoder for the samples the codec produced rather
  # than the grey-chroma padding it adds to match libavcodec yuv420p, since
  # the CROSS check below asks ffmpeg for gray.
  if ! H26XDEC_NO_CHROMA_PAD=1 "$DEC" "$bs" "$ours" > /dev/null 2> "$OUT/$base.$name.dec.log"; then
    echo "SELF-FAIL   $tag: our decoder rejected our bitstream: $(tail -1 "$OUT/$base.$name.dec.log" | head -c 80)"
    return 1
  fi
  if ! cmp -s "$rec" "$ours"; then
    echo "SELF-FAIL   $tag: decoded output differs from the encoder's own reconstruction"
    return 1
  fi

  # 2. CROSS.
  theirs="$OUT/$base.$name.ff.yuv"
  # 4:0:0 needs extractplanes, not -pix_fmt gray. libavcodec emits H.264
  # monochrome as yuv420p with grey chroma, so asking swscale for gray makes
  # it convert — and it treats yuv420p as limited range and gray as full, so
  # every luma sample comes out expanded: 68 becomes 61. That is an artefact
  # of the comparison, not a difference in the bitstream, and it cost a false
  # CROSS failure to find.
  ffargs="-pix_fmt $(ffpix "$fmt")"
  case "$fmt" in 400|gray) ffargs="-vf extractplanes=y -pix_fmt gray" ;; esac
  if ! "$FFMPEG" -v error -y -i "$bs" -f rawvideo $ffargs "$theirs" \
       > "$OUT/$base.$name.ff.log" 2>&1; then
    echo "CROSS-FAIL  $tag: libavcodec rejected our bitstream: $(tail -1 "$OUT/$base.$name.ff.log" | head -c 80)"
    return 1
  fi
  if ! cmp -s "$ours" "$theirs"; then
    echo "CROSS-FAIL  $tag: libavcodec decodes our bitstream differently than we do"
    return 1
  fi

  # 3. QUALITY. Gated only when the configuration claims to be lossless.
  psnr=$(psnr_of "$src" "$rec")
  size=$(stat -c %s "$bs")
  case "$flags" in
    *--lossless*)
      if ! cmp -s "$src" "$rec"; then
        echo "LOSSLESS-FAIL $tag: reconstruction differs from the source"
        return 1
      fi
      echo "PASS        $tag (lossless, exact, $size bytes)" ;;
    *)
      echo "PASS        $tag ($size bytes, PSNR $psnr dB)" ;;
  esac
  return 0
}

# Planar chroma format name to the ffmpeg pixel format that matches how this
# decoder packs a picture.
ffpix() {
  case "$1" in
    400|gray) echo gray ;;
    422) echo yuv422p ;;
    444) echo yuv444p ;;
    *) echo yuv420p ;;
  esac
}

psnr_of() {
  python - "$1" "$2" <<'PY'
import sys, math
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
n = min(len(a), len(b))
if n == 0:
    print("n/a"); raise SystemExit
se = 0
for i in range(0, n, max(1, n // 200000) if n > 200000 else 1):
    d = a[i] - b[i]
    se += d * d
cnt = len(range(0, n, max(1, n // 200000) if n > 200000 else 1))
mse = se / cnt
print("inf" if mse == 0 else f"{10 * math.log10(255 * 255 / mse):.2f}")
PY
}
export -f one ffpix psnr_of
export ENC DEC FFMPEG OUT

echo "== encode verification =="
results="$OUT/results.txt"
: > "$results"
for src in $SOURCES; do
  echo "$CONFIGS" | while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    echo "$src|$name|$flags"
  done
done | xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r s n f <<< "{}"; one "$s" "$n" "$f"' \
     | sort | tee "$results"

pass=$(grep -c '^PASS' "$results")
bad=$(grep -cE '^(ENCODE|SELF|CROSS|LOSSLESS)-FAIL' "$results")
echo
echo "encode: $pass passed, $bad failed"
[ "$bad" = 0 ] || fail=1

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
