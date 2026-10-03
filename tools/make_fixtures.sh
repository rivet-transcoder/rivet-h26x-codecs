#!/bin/bash
# make_fixtures.sh <dir> — generate the decoder fixture set that verify.sh
# checks against tools/golden.txt, into DIR (default: $H26X_WORK).
#
# The fixtures are encodes by the x264 and x265 command-line programs of
# deterministic synthetic sources (synth_source.py: `detail`, `zoom`,
# `fade`, the interlaced `tff` / `bff` — nothing that takes a random seed),
# one per coding tool the decoders have to get right: every profile, both
# entropy coders, every chroma format and bit depth, MBAFF in both field
# orders, JVT and custom quantisation matrices, lossless, WPP, SAO, temporal
# layers, open and closed GOPs, transform skip, weighted prediction, and so
# on. They are generated rather than vendored (1.7 MB, and the encoders'
# licences are theirs), and `fixtures.md5` in the output directory records
# the bitstream hashes so a regeneration can be checked against the last one
# before the decoder is asked anything (rerunning the script does that check
# itself; tools/fixtures.md5 is the committed copy).
#
# Reproducibility costs two settings: x264 `--threads 1` and x265
# `--pools 1 --frame-threads 1`. Both encoders change their output with the
# thread count — x265 even with frame-threads=1: four pool sizes gave four
# bitstreams — and a fixture that depends on the core count of the machine
# that made it is not a fixture. WPP stays on at one thread; the syntax
# (entropy_coding_sync, entry points) is what the decoder is tested on, not
# the parallelism. The sources are raw planar YUV at exactly the format and
# depth encoded, so no encoder converts anything, and encodes are 8-12
# frames so the whole set stays small. Deep sources carry noise in their
# low bits (synth_source.py --noise), so those bits are not all zero.
#
# Every encode is checked three ways before it counts, because both
# encoders only WARN about an option they quietly override ("Disabling
# ...", "not supported", ...), and a fixture named for a feature it does
# not contain is a green row that tests nothing:
#   1. the encoder's log must be free of those warnings;
#   2. the encoder's own SEI option string must carry the option tokens the
#      fixture is named for (x264 writes `cabac=0 ... interlaced=tff`, x265
#      `no-wpp temporal-layers=2 ...`);
#   3. stream_census.py — which reads the SPS, PPS and slice headers — must
#      show the profile, chroma format and depth the fixture claims, and the
#      tool in the BYTES: CABAC on, the 8x8 transform on, four slices a
#      picture, two CRAs with leading pictures, a temporal id above zero,
#      transquant bypass enabled, and so on.
# Whether the DECODER agrees with the reference decoders (JM, HM) is
# check.sh's job; record the golden with record_golden.sh once it does.
#
#   X264=path     x264 (default: on PATH). Built with every bit depth and
#                 without libavformat input: `./configure --bit-depth=all
#                 --disable-lavf --disable-ffms --disable-swscale`. The set
#                 was recorded with x264 0.165 r3223 0480cb0.
#   X265=path     x265 (default: on PATH), 8-bit; X265_10 / X265_12 the
#                 10- and 12-bit builds (default: X265, which serves for a
#                 multilib build). Recorded with x265 4.2.
# Both write their build into the SEI of every stream, so a fixture names
# what made it.
OUT=${1:-$H26X_WORK}
[ -n "$OUT" ] || { echo "usage: make_fixtures.sh <dir>   (or set H26X_WORK)" >&2; exit 2; }
TOOLS=$(cd "$(dirname "$0")" && pwd)
CENSUS=$TOOLS/stream_census.py
SYNTH=$TOOLS/synth_source.py
X264=${X264:-x264}
X265=${X265:-x265}
X265_10=${X265_10:-$X265}
X265_12=${X265_12:-$X265}
mkdir -p "$OUT" && cd "$OUT" || exit 2
"$X264" --version | head -1
"$X265" --version 2>&1 | grep -m1 -i "version"
errs=0
made=0
[ -f fixtures.md5 ] && mv fixtures.md5 fixtures.md5.prev
: > fixtures.md5

# ---------------------------------------------------------------- sources
# SOURCE  recipe size rate. All at 25 fps except the 720p pair at 30.
# `detail` has glyph-like detail, moving blocks and colour ramps; `zoom` is
# a Mandelbrot zoom — dense detail with motion everywhere, the clip for
# anything inter. `fade` starts from black so weighted prediction has a fade
# to find. `tff` / `bff` are genuinely interlaced — fields woven from
# consecutive frames of a source at twice the rate, top or bottom first, so
# the encoder's field-order choice is real and both field and frame
# macroblock pairs are worth picking. The odd sizes are not multiples of 16
# (H.264) or 8 (HEVC), so the conformance window / cropping is signalled.
S=160x96
src_detail="detail $S 25"
src_zoom="zoom $S 25"
src_fade="fade $S 25"
src_tff="tff $S 25"
src_bff="bff $S 25"
src_odd264="detail 98x66 25"
src_odd265="detail 194x130 25"
src_720="detail 1280x720 30"
src_720z="zoom 1280x720 30"

# make_source RECIPE SIZE FRAMES FORMAT -> the raw file's name (made once).
make_source() {
  local f="src_${1}_${2}_${3}f_${4}.yuv" depth=${4#*p}
  [ "$depth" = "$4" ] && depth=8
  if [ ! -s "$f" ]; then
    python "$SYNTH" "$1" --size "$2" --frames "$3" --format "$4" --noise $((depth - 8)) "$f" >&2 || return 1
  fi
  echo "$f"
}

# Custom H.264 quantisation matrices in the JM file format x264 reads
# (each list in zigzag order). The values only have to be legal (1..255)
# and unlike the JVT defaults, so the explicit-list syntax is written and
# parsed; these are the JVT defaults with every entry nudged.
cat > x264_custom.cqm <<'EOF'
INTRA4X4_LUMA =
7,14,14,21,21,21,29,29,29,29,33,33,33,38,38,43
INTRA4X4_CHROMAU =
7,14,14,21,21,21,29,29,29,29,33,33,33,38,38,43
INTRA4X4_CHROMAV =
7,14,14,21,21,21,29,29,29,29,33,33,33,38,38,43
INTER4X4_LUMA =
11,15,15,19,19,19,23,23,23,23,25,25,25,28,28,31
INTER4X4_CHROMAU =
11,15,15,19,19,19,23,23,23,23,25,25,25,28,28,31
INTER4X4_CHROMAV =
11,15,15,19,19,19,23,23,23,23,25,25,25,28,28,31
INTRA8X8_LUMA =
7,11,11,14,12,14,17,17,17,17,19,19,19,19,19,22,22,22,22,22,22,25,25,25,25,25,25,25,28,28,28,28,28,28,28,28,30,30,30,30,30,30,30,32,32,32,32,32,32,34,34,34,34,34,36,36,36,36,38,38,38,40,40,42
INTER8X8_LUMA =
10,12,12,14,14,14,16,16,16,16,18,18,18,18,18,20,20,20,20,20,20,22,22,22,22,22,22,22,24,24,24,24,24,24,24,24,26,26,26,26,26,26,26,28,28,28,28,28,28,30,30,30,30,30,31,31,31,31,32,32,32,33,33,34
EOF

# ---------------------------------------------------------------- checks
# What the encoders print when they refused or overrode an option. The one
# harmless x265 line every sub-720p encode prints is dropped first.
bad_log() { grep -v "disabling lookahead-slices" "$1" | grep -iE "error|unknown option|invalid|disabl|not supported|ignor"; }

# The option string the encoder wrote into its user-data SEI.
sei_options() { tr -c '[:print:]' '\n' < "$1" | grep -m1 -o "options:.*"; }

# check_tokens LABEL LINE TOKEN... — each TOKEN against LINE, a
# space-separated list of `key=value` words: `k=v` must appear as a whole
# word, `k>n` needs k present with a value above n.
check_tokens() {
  local label=$1 line=$2 tok key min got; shift 2
  for tok in "$@"; do
    case "$tok" in
      *">"*)
        key=${tok%%>*}; min=${tok#*>}
        got=$(tr ' ' '\n' <<< "$line" | grep -m1 "^$key=" | cut -d= -f2)
        if ! { [ -n "$got" ] && [ "$got" -gt "$min" ]; } 2>/dev/null; then
          echo "  FAIL $label: $key is '${got:-absent}', wanted > $min"; errs=$((errs + 1))
        fi ;;
      *)
        grep -qwF -- "$tok" <<< "$line" || { echo "  FAIL $label: lacks '$tok'"; errs=$((errs + 1)); } ;;
    esac
  done
}

# encode NAME ENC SOURCE FRAMES FORMAT OPT... -- SEI... -- CENSUS...
#   ENC     x264 | x265
#   SOURCE  "recipe size rate"
#   FORMAT  synth_source format: 420, 422p10, 444, 400, 420p12, ...
#   SEI     tokens the encoder's option string must carry
#   CENSUS  tokens stream_census.py must print (pics=FRAMES is implied)
encode() {
  local name=$1 enc=$2 src=$3 n=$4 fmt=$5; shift 5
  local opts=() sei_want=() cen_want=()
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  while [ "$1" != "--" ]; do sei_want+=("$1"); shift; done; shift
  cen_want=("$@")
  printf "%-36s" "$name"
  local recipe size rate file chroma depth csp bin
  read -r recipe size rate <<< "$src"
  file=$(make_source "$recipe" "$size" "$n" "$fmt") || { echo "FAIL (source)"; errs=$((errs + 1)); return; }
  chroma=${fmt%%p*}; depth=${fmt#*p}; [ "$depth" = "$fmt" ] && depth=8
  csp=i$chroma
  case "$enc" in
    x264)
      bin=$X264
      "$bin" --threads 1 --input-res "$size" --fps "$rate" --input-csp "$csp" --output-csp "$csp" \
        --input-depth "$depth" --output-depth "$depth" "${opts[@]}" -o "$name" "$file" > "$name.log" 2>&1 ;;
    x265)
      case "$depth" in 10) bin=$X265_10 ;; 12) bin=$X265_12 ;; *) bin=$X265 ;; esac
      "$bin" --pools 1 --frame-threads 1 --input-res "$size" --fps "$rate" --input-csp "$csp" \
        --input-depth "$depth" --output-depth "$depth" "${opts[@]}" -o "$name" "$file" > "$name.log" 2>&1 ;;
  esac
  if [ $? -ne 0 ] || [ ! -s "$name" ]; then
    echo "FAIL ($enc)"; sed 's/^/    /' "$name.log"; errs=$((errs + 1)); return
  fi
  if bad_log "$name.log" > /dev/null; then
    echo "FAIL (an option was refused or overridden)"; bad_log "$name.log" | sed 's/^/    /'; errs=$((errs + 1)); return
  fi
  local sei cen before=$errs
  sei=$(sei_options "$name")
  cen=$(python "$CENSUS" "$name")
  check_tokens "$name (sei)" "$sei" "${sei_want[@]}"
  check_tokens "$name (census)" "$cen" "pics=$n" "${cen_want[@]}"
  if [ "$errs" = "$before" ]; then
    printf "%7d bytes  %s\n" "$(stat -c %s "$name")" "$cen"
    echo "$(md5sum < "$name" | cut -c1-32) $name" >> fixtures.md5
    made=$((made + 1))
  fi
}

# x264 NAME SOURCE FRAMES FORMAT OPT... -- SEI... -- CENSUS...
x264() { local name=$1; shift; encode "$name.264" x264 "$@"; }
x265() { local name=$1; shift; encode "$name.265" x265 "$@"; }

# ------------------------------------------------------------------ H.264
echo "== H.264 =="
# The census's profile / chroma / depth / frame_mbs_only stand where a
# container-level probe would: what the SPS says, read from the bytes.
P420="chroma=1 depth=8 frame_mbs_only=1"
x264 x264_baseline_cavlc "$src_detail" 10 420 --profile baseline --crf 20 \
  -- cabac=0 bframes=0 8x8dct=0 -- profile=66 $P420 cabac=0 t8x8=0 bslices=0
x264 x264_main_cabac_b "$src_zoom" 12 420 --profile main --crf 20 --bframes 3 --b-pyramid normal --direct spatial --weightb \
  -- cabac=1 bframes=3 direct=1 weightb=1 -- profile=77 $P420 cabac=1 t8x8=0 wbi=2 "bslices>0"
x264 x264_main_cavlc_b_temporal "$src_zoom" 12 420 --profile main --crf 20 --no-cabac --bframes 3 --b-pyramid normal --direct temporal \
  -- cabac=0 bframes=3 direct=2 -- profile=77 $P420 cabac=0 "bslices>0"
x264 x264_high_cabac_8x8_slices "$src_detail" 10 420 --crf 20 --8x8dct --slices 4 --bframes 2 \
  -- cabac=1 8x8dct=1 slices=4 -- profile=100 $P420 cabac=1 t8x8=1 slices_per_pic=4
x264 x264_high_cavlc_8x8 "$src_detail" 10 420 --crf 20 --no-cabac --8x8dct \
  -- cabac=0 8x8dct=1 -- profile=100 $P420 cabac=0 t8x8=1
x264 x264_high_cabac_cip_deblock "$src_detail" 10 420 --crf 20 --constrained-intra --deblock -3:3 --bframes 2 \
  -- constrained_intra=1 deblock=1:-3:3 -- profile=100 $P420 cip=1 dfc=1
x264 x264_high_cqm_jvt "$src_detail" 10 420 --crf 20 --cqm jvt \
  -- cqm=1 8x8dct=1 -- profile=100 $P420 sl=1 t8x8=1
x264 x264_high_cqm_custom "$src_detail" 10 420 --crf 20 --cqmfile x264_custom.cqm \
  -- cqm=2 8x8dct=1 -- profile=100 $P420 sl=1 t8x8=1
x264 x264_high_interlaced_tff "$src_tff" 10 420 --crf 20 --tff --bframes 3 --direct spatial \
  -- interlaced=tff direct=1 -- profile=100 chroma=1 depth=8 frame_mbs_only=0 mbaff=1 "bslices>0"
x264 x264_high_interlaced_bff "$src_bff" 10 420 --crf 20 --bff --bframes 3 --direct temporal \
  -- interlaced=bff direct=2 -- profile=100 chroma=1 depth=8 frame_mbs_only=0 mbaff=1 "bslices>0"
x264 x264_high_weightp_fade "$src_fade" 12 420 --crf 20 --weightp 2 --weightb --bframes 2 \
  -- weightp=2 weightb=1 -- profile=100 $P420 wp=1 wbi=2
x264 x264_high_odd_size "$src_odd264" 10 420 --crf 20 \
  -- -- profile=100 $P420 crop=1
x264 x264_422 "$src_detail" 10 422 --crf 20 \
  -- cabac=1 -- profile=122 chroma=2 depth=8
x264 x264_422_10 "$src_detail" 10 422p10 --crf 20 \
  -- cabac=1 -- profile=122 chroma=2 depth=10
x264 x264_422_10_cavlc "$src_detail" 10 422p10 --crf 20 --no-cabac \
  -- cabac=0 -- profile=122 chroma=2 depth=10 cabac=0
x264 x264_444 "$src_detail" 10 444 --crf 20 \
  -- cabac=1 8x8dct=1 -- profile=244 chroma=3 depth=8
x264 x264_444_cavlc "$src_detail" 10 444 --crf 20 --no-cabac \
  -- cabac=0 -- profile=244 chroma=3 cabac=0
x264 x264_444_10 "$src_detail" 10 444p10 --crf 20 \
  -- cabac=1 -- profile=244 chroma=3 depth=10
x264 x264_444_i16 "$src_detail" 10 444 --crf 20 --partitions none --no-8x8dct \
  -- analyse=0x1:0 8x8dct=0 -- profile=244 chroma=3 t8x8=0
x264 x264_444_no8 "$src_detail" 10 444 --crf 20 --no-8x8dct \
  -- 8x8dct=0 -- profile=244 chroma=3 t8x8=0
x264 x264_444_lossless "$src_detail" 8 444 --qp 0 \
  -- qp=0 -- profile=244 chroma=3
x264 x264_gray "$src_detail" 10 400 --crf 20 \
  -- chroma_me=0 -- profile=100 chroma=0 depth=8
x264 x264_hi10 "$src_detail" 10 420p10 --crf 20 \
  -- cabac=1 8x8dct=1 -- profile=110 chroma=1 depth=10 cabac=1
x264 x264_hi10_cavlc "$src_detail" 10 420p10 --crf 20 --no-cabac \
  -- cabac=0 -- profile=110 chroma=1 depth=10 cabac=0
x264 x264_hi10_cqm "$src_detail" 10 420p10 --crf 20 --cqm jvt \
  -- cqm=1 -- profile=110 depth=10 sl=1
x264 x264_hi10_lossless "$src_detail" 8 420p10 --qp 0 \
  -- qp=0 -- depth=10 profile=244
x264 x264_420_lossless "$src_detail" 8 420 --qp 0 \
  -- qp=0 cabac=1 -- profile=244 chroma=1 cabac=1
x264 x264_420_lossless_cavlc "$src_detail" 8 420 --qp 0 --no-cabac \
  -- qp=0 cabac=0 -- profile=244 chroma=1 cabac=0
x264 x264_720p_cabac "$src_720" 12 420 --crf 20 --bframes 3 --b-pyramid normal \
  -- cabac=1 bframes=3 -- profile=100 $P420 cabac=1 t8x8=1 "bslices>0"
x264 x264_720p_cavlc "$src_720" 12 420 --crf 20 --no-cabac --bframes 3 \
  -- cabac=0 bframes=3 -- profile=100 $P420 cabac=0 "bslices>0"
x264 x264_720p_main "$src_720z" 12 420 --profile main --crf 20 --bframes 3 \
  -- cabac=1 bframes=3 -- profile=77 $P420 t8x8=0 "bslices>0"

# ------------------------------------------------------------------- HEVC
echo "== HEVC =="
# general_profile_idc: 1 Main, 2 Main 10, 4 format range extensions.
x265 x265_main_wpp_sao "$src_detail" 10 420 --crf 22 --wpp --sao \
  -- -- profile=1 chroma=1 depth=8 wpp=1 sao=1 ctu=64
x265 x265_main10_nowpp "$src_detail" 10 420p10 --crf 22 --no-wpp \
  -- no-wpp -- profile=2 depth=10 wpp=0
x265 x265_main12 "$src_detail" 10 420p12 --crf 22 \
  -- -- profile=4 depth=12
x265 x265_closedgop_idr "$src_zoom" 12 420 --crf 22 --keyint 5 --min-keyint 5 --no-open-gop --bframes 2 \
  -- no-open-gop keyint=5 -- profile=1 idr=3 cra=0 "bslices>0"
x265 x265_opengop_cra "$src_zoom" 12 420 --crf 22 --keyint 5 --min-keyint 5 --open-gop --bframes 3 \
  -- open-gop keyint=5 -- profile=1 idr=1 cra=2 "rasl>0"
x265 x265_ctu16_tlayers "$src_zoom" 12 420 --crf 22 --ctu 16 --max-tu-size 16 --qg-size 16 --temporal-layers 2 --bframes 4 --b-pyramid \
  -- ctu=16 temporal-layers=2 -- profile=1 ctu=16 sub_layers=2 "tid_max>0"
x265 x265_cu_lossless "$src_detail" 10 420 --crf 22 --cu-lossless \
  -- cu-lossless -- profile=1 tqbypass=1
x265 x265_lossless "$src_detail" 8 420 --lossless \
  -- lossless -- profile=1 tqbypass=1
x265 x265_intra_only "$src_detail" 8 420 --crf 22 --keyint 1 --min-keyint 1 \
  -- keyint=1 -- profile=4 irap=8 pslices=0 bslices=0
# The default lists only: x265 4.2 refuses every HM-format custom list
# file tried ("can't read matrix"), so explicit scaling_list_data stays
# the conformance suite's (SLIST_A..D).
x265 x265_main_scalinglist "$src_detail" 10 420 --crf 22 --scaling-list default \
  -- -- profile=1 sl=1 sl_data=0
x265 x265_main_tskip_amp "$src_detail" 10 420 --crf 22 --tskip --amp --rect \
  -- tskip amp -- profile=1 tskip=1 amp=1
x265 x265_main_weighted "$src_fade" 12 420 --crf 22 --weightp --weightb --bframes 2 \
  -- weightp weightb -- profile=1 wp=1 wbi=1 "bslices>0"
# no-deblock with WPP on deadlocks x265 (4.2) at every pool size, so this
# one is also the no-WPP 8-bit stream.
x265 x265_notmvp_nodeblock "$src_zoom" 10 420 --crf 22 --no-wpp --no-temporal-mvp --no-deblock \
  -- no-wpp no-temporal-mvp no-deblock -- profile=1 wpp=0 tmvp=0 deblock_off=1
x265 x265_odd_size_ctu32 "$src_odd265" 10 420 --crf 22 --ctu 32 \
  -- ctu=32 -- profile=1 ctu=32 crop=1
x265 x265_slices_wpp_qg16_cip "$src_detail" 10 420 --crf 22 --slices 2 --wpp --qg-size 16 --constrained-intra --aq-mode 2 \
  -- slices=2 qg-size=16 constrained-intra -- profile=1 slices_per_pic=2 wpp=1 cip=1 cuqp=1 qg_depth=2
x265 x265_p_notmvp "$src_zoom" 10 420 --crf 22 --bframes 0 --no-temporal-mvp \
  -- bframes=0 no-temporal-mvp -- profile=1 tmvp=0 bslices=0 "pslices>0"
x265 x265_main_nostrong_nosignhide "$src_detail" 10 420 --crf 22 --no-strong-intra-smoothing --no-signhide \
  -- no-strong-intra-smoothing no-signhide -- profile=1 strong=0 sbh=0
x265 x265_gray "$src_detail" 10 400 --crf 22 \
  -- -- profile=4 chroma=0
x265 x265_main422_10 "$src_detail" 10 422p10 --crf 22 \
  -- -- profile=4 chroma=2 depth=10
x265 x265_main444 "$src_detail" 10 444 --crf 22 \
  -- -- profile=4 chroma=3 depth=8
x265 x265_720p_nowpp "$src_720" 12 420 --crf 22 --no-wpp --bframes 3 \
  -- no-wpp -- profile=1 wpp=0 "bslices>0"
x265 x265_720p_wpp "$src_720z" 12 420 --crf 22 --wpp --sao --bframes 3 \
  -- -- profile=1 wpp=1 sao=1 "bslices>0"

# ------------------------------------------------------------------ summary
echo
if [ -f fixtures.md5.prev ]; then
  same=0; changed=0
  while read -r m f; do
    if grep -q "^$m $f\$" fixtures.md5.prev; then same=$((same + 1))
    elif grep -q " $f\$" fixtures.md5.prev; then echo "REGENERATED DIFFERENTLY: $f"; changed=$((changed + 1)); fi
  done < fixtures.md5
  echo "against the previous run: $same identical, $changed changed"
  errs=$((errs + changed))
  rm -f fixtures.md5.prev
fi
echo "fixtures: $made generated, $(du -ch $(cut -d' ' -f2 fixtures.md5) 2>/dev/null | tail -1 | cut -f1) in $(pwd)"
[ "$errs" = 0 ] && echo "ALL CHECKS PASSED" || echo "SOMETHING FAILED ($errs)"
[ "$errs" = 0 ]
