#!/bin/bash
# ref_roundtrip.sh ENC DEC — a small encode round trip against the ITU-T
# reference decoders, for CI: synthetic clips (synth_source.py) through
# h26xenc in a spread of configurations, then
#   SELF   our decoder reproduces the encoder's reconstruction, and
#   CROSS  JM (H.264) / HM (H.265), via ref_decode.py, decode the stream to
#          exactly the same pictures.
# The full gate is verify_encode.sh; this is the part that fits in a CI job.
# Needs LDECOD and TAPPDECODER (or ldecod / TAppDecoder on the PATH).
ENC=${1:?usage: ref_roundtrip.sh ENC DEC}
DEC=${2:?usage: ref_roundtrip.sh ENC DEC}
TOOLS=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
# clip: recipe WxH frames format
clip() {
  local d=${4#*p}; [ "$d" = "$4" ] && d=8
  python "$TOOLS/synth_source.py" "$1" --size "$2" --frames "$3" --format "$4" --noise $((d - 8)) "$WORK/$1_$2_$4.yuv"
}
clip detail 64x64 8 420
clip motion 64x64 12 420
clip hfade 64x64 12 420
clip detail 50x34 6 420
clip detail 64x64 8 422p10
clip detail 64x64 8 444
clip detail 64x64 8 400
clip detail 64x64 8 420p12
clip interlace 96x96 8 420
# row: clip | flags
while IFS='|' read -r src flags; do
  [ -n "$src" ] || continue
  n=$((n + 1))
  base=${src%.yuv}; geom=$(echo "$base" | cut -d_ -f2); fmt=$(echo "$base" | cut -d_ -f3)
  chroma=${fmt%%p*}; depth=${fmt#*p}; [ "$depth" = "$fmt" ] && depth=8
  ext=h264; case "$flags" in *"--codec h265"*) ext=h265 ;; esac
  bs="$WORK/$n.$ext"; rec="$WORK/$n.rec.yuv"; ours="$WORK/$n.ours.yuv"; ref="$WORK/$n.ref.yuv"
  tag="$src [$flags]"
  if ! "$ENC" --input "$WORK/$src" --size "$geom" --format "$chroma" --depth "$depth" $flags \
       --output "$bs" --recon "$rec" > "$WORK/$n.log" 2>&1; then
    echo "ENCODE-FAIL $tag: $(tail -1 "$WORK/$n.log")"; fail=1; continue
  fi
  H26XDEC_NO_CHROMA_PAD=1 "$DEC" "$bs" "$ours" > /dev/null 2>&1
  if ! cmp -s "$rec" "$ours"; then echo "SELF-FAIL   $tag"; fail=1; continue; fi
  if ! python "$TOOLS/ref_decode.py" decode --codec "$ext" --luma-only "$bs" "$ref" > "$WORK/$n.ref.log" 2>&1; then
    echo "CROSS-FAIL  $tag: the reference decoder rejected it: $(tail -1 "$WORK/$n.ref.log")"; fail=1; continue
  fi
  if ! cmp -s "$ours" "$ref"; then echo "CROSS-FAIL  $tag: the reference decoder decodes it differently"; fail=1; continue; fi
  echo "PASS        $tag"
done <<'EOF'
detail_64x64_420.yuv|--codec h264 --qp 26 --gop 8
detail_64x64_420.yuv|--codec h264 --qp 40 --gop 8 --cavlc
motion_64x64_420.yuv|--codec h264 --qp 26 --gop 8 --bframes 2 --t8x8 --subparts
motion_64x64_420.yuv|--codec h264 --qp 30 --gop 8 --bframes 2 --cavlc --refs 3
hfade_64x64_420.yuv|--codec h264 --qp 26 --gop 12 --bframes 2 --wpred
detail_50x34_420.yuv|--codec h264 --qp 26 --gop 8
detail_64x64_420.yuv|--codec h264 --lossless --gop 8
detail_64x64_422p10.yuv|--codec h264 --qp 30 --gop 8 --bframes 1
detail_64x64_444.yuv|--codec h264 --qp 30 --gop 8 --t8x8
detail_64x64_400.yuv|--codec h264 --qp 30 --gop 8
interlace_96x96_420.yuv|--codec h264 --qp 30 --gop 8 --interlace tff --field-coding paff
interlace_96x96_420.yuv|--codec h264 --qp 30 --gop 8 --bframes 1 --interlace tff --field-coding mbaff
detail_64x64_420.yuv|--codec h265 --qp 30 --gop 8
motion_64x64_420.yuv|--codec h265 --qp 30 --gop 8 --bframes 3 --sao
hfade_64x64_420.yuv|--codec h265 --qp 30 --gop 12 --bframes 2 --wpred
detail_50x34_420.yuv|--codec h265 --qp 26 --gop 8
detail_64x64_420.yuv|--codec h265 --lossless --gop 8
detail_64x64_422p10.yuv|--codec h265 --qp 30 --gop 8 --bframes 1 --sao
detail_64x64_444.yuv|--codec h265 --qp 40 --gop 8
detail_64x64_400.yuv|--codec h265 --qp 30 --gop 8
detail_64x64_420p12.yuv|--codec h265 --qp 30 --gop 8 --bframes 2
EOF
echo "round trips: $n, $([ $fail = 0 ] && echo 'all match the reference decoders' || echo 'SOMETHING FAILED')"
exit $fail
