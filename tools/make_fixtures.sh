#!/bin/bash
# make_fixtures.sh <dir> — generate the decoder fixture set that verify.sh
# checks against tools/golden.txt, into DIR (default: $H26X_WORK).
#
# The fixtures are x264 and x265 encodes of deterministic lavfi sources
# (testsrc2, mandelbrot — nothing that takes a random seed), one per coding
# tool the decoders have to get right: every profile, both entropy coders,
# every chroma format and bit depth, MBAFF in both field orders, JVT and
# custom quantisation matrices, lossless, WPP, SAO, temporal layers, open
# and closed GOPs, transform skip, weighted prediction, and so on — the
# feature list the previous set's names carried, plus a few the decoders
# have since grown into. They are generated rather than vendored because
# the previous set was lost with the directory it lived in (2026-09-09) and
# could not be recreated: its sources and encoder builds were gone. Every
# byte here comes from this script and the ffmpeg build it is run with, and
# `fixtures.md5` in the output directory records the bitstream hashes so a
# regeneration can be checked against the last one before the decoder is
# asked anything (rerunning the script does that check itself).
#
# Reproducibility costs two settings: x264 `threads=1` and x265
# `pools=1:frame-threads=1`. Both encoders change their output with the
# thread count — x265 even with frame-threads=1: four pool sizes gave four
# bitstreams — and a fixture that depends on the core count of the machine
# that made it is not a fixture. WPP stays on at one thread; the syntax
# (entropy_coding_sync, entry points) is what the decoder is tested on, not
# the parallelism. Two more: `-sws_flags +accurate_rnd+bitexact` so the
# pixel-format conversion is the same on every machine, and encodes of
# 8-12 frames so the whole set stays small.
#
# Every encode is checked three ways before it counts, because both ffmpeg
# wrappers only WARN about an option the encoder does not know or quietly
# overrides ("Error parsing option", "Unknown option", the x265
# "Disabling ..." family), and a fixture named for a feature it does not
# contain is a green row that tests nothing:
#   1. the wrapper's log must be free of those warnings;
#   2. the encoder's own SEI option string must carry the option tokens the
#      fixture is named for (x264 writes `cabac=0 ... interlaced=tff`, x265
#      `no-wpp temporal-layers=2 ...`);
#   3. ffprobe must report the profile, pixel format and field order the
#      fixture claims, and stream_census.py — which reads the SPS, PPS and
#      slice headers — must show the tool in the BYTES: CABAC on, the 8x8
#      transform on, four slices a picture, two CRAs with leading pictures,
#      a temporal id above zero, transquant bypass enabled, and so on.
# Whether the DECODER agrees with libavcodec is check.sh's job; record the
# golden with record_golden.sh once it does.
#
#   FFMPEG=path   ffmpeg to use (default: ffmpeg on PATH; the real binary
#                 on the build machine is
#                 C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe,
#                 and ffprobe is taken from beside it). The set was recorded
#                 with ffmpeg 8.1.1 (gyan.dev full build, GCC 15.2.0):
#                 x264 core 165 r3223 0480cb0, x265 4.2+3-3f4120d build 216
#                 (8+10+12-bit) — both write their build into the SEI of
#                 every stream, so a fixture names what made it.
OUT=${1:-$H26X_WORK}
[ -n "$OUT" ] || { echo "usage: make_fixtures.sh <dir>   (or set H26X_WORK)" >&2; exit 2; }
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-${FFMPEG%ffmpeg*}ffprobe${FFMPEG##*ffmpeg}}
CENSUS=$(cd "$(dirname "$0")" && pwd)/stream_census.py
mkdir -p "$OUT" && cd "$OUT" || exit 2
"$FFMPEG" -version | head -1
errs=0
made=0
[ -f fixtures.md5 ] && mv fixtures.md5 fixtures.md5.prev
: > fixtures.md5
# `fixtures.libav_deviates`: "<fixture> <why>" for a fixture where libavcodec
# is the one that is wrong, with the evidence in <why>. check.sh reports
# those as DEVIATES instead of failing; the golden still holds OUR verified
# output. A line here is a claim against the reference and needs proof
# stronger than "we differ" — for a lossless CU, that its samples equal the
# encoder's source frame.
: > fixtures.libav_deviates
libav_deviates() { echo "$1 $2" >> fixtures.libav_deviates; }

# ---------------------------------------------------------------- sources
# All at 25 fps except the 720p pair at 30. `detail` has text, moving
# blocks and colour ramps; `zoom` is a Mandelbrot zoom — dense detail with
# motion everywhere, the clip for anything inter. `fade` starts from black
# so weighted prediction has a fade to find. `tff` / `bff` are genuinely
# interlaced — fields woven from consecutive 50 fps frames, flagged with
# their order, so the encoder's field-order choice is real and both field
# and frame macroblock pairs are worth picking. The odd sizes are not
# multiples of 16 (H.264) or 8 (HEVC), so the conformance window / cropping
# is signalled.
S=160x96
src_detail="testsrc2=size=$S:rate=25"
src_zoom="mandelbrot=size=$S:rate=25"
src_fade="testsrc2=size=$S:rate=25,fade=t=in:s=0:n=10"
src_tff="testsrc2=size=$S:rate=50,tinterlace=mode=interleave_top"
src_bff="testsrc2=size=$S:rate=50,tinterlace=mode=interleave_bottom"
src_odd264="testsrc2=size=98x66:rate=25"
src_odd265="testsrc2=size=194x130:rate=25"
src_720="testsrc2=size=1280x720:rate=30"
src_720z="mandelbrot=size=1280x720:rate=30"

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
# What the wrappers print when the encoder refused or overrode an option
# ("Error parsing option" is libx264's, the rest libx265's). The one
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

# encode NAME MUXER ENCODER SOURCE FRAMES PIX_FMT PROBE OPT... -- SEI... -- CENSUS...
#   PROBE   what `ffprobe -show_entries stream=profile,pix_fmt,field_order`
#           must print
#   SEI     tokens the encoder's option string must carry
#   CENSUS  tokens stream_census.py must print (pics=FRAMES is implied)
encode() {
  local name=$1 mux=$2 enc=$3 src=$4 n=$5 pf=$6 probe=$7; shift 7
  local opts=() sei_want=() cen_want=()
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  while [ "$1" != "--" ]; do sei_want+=("$1"); shift; done; shift
  cen_want=("$@")
  printf "%-36s" "$name"
  if ! "$FFMPEG" -hide_banner -v warning -nostats -nostdin -y -sws_flags +accurate_rnd+bitexact \
       -f lavfi -i "$src" -frames:v "$n" -pix_fmt "$pf" -c:v "$enc" "${opts[@]}" \
       -f "$mux" "$name" > "$name.log" 2>&1; then
    echo "FAIL (ffmpeg)"; sed 's/^/    /' "$name.log"; errs=$((errs + 1)); return
  fi
  if bad_log "$name.log" > /dev/null; then
    echo "FAIL (an option was refused or overridden)"; bad_log "$name.log" | sed 's/^/    /'; errs=$((errs + 1)); return
  fi
  local got sei cen before=$errs
  got=$("$FFPROBE" -v error -show_entries stream=profile,pix_fmt,field_order -of csv=p=0 "$name")
  sei=$(sei_options "$name")
  cen=$(python "$CENSUS" "$name")
  [ "$got" = "$probe" ] || { echo; echo "  FAIL $name: ffprobe says '$got', wanted '$probe'"; errs=$((errs + 1)); }
  check_tokens "$name (sei)" "$sei" "${sei_want[@]}"
  check_tokens "$name (census)" "$cen" "pics=$n" "${cen_want[@]}"
  if [ "$errs" = "$before" ]; then
    printf "%7d bytes  %s\n" "$(stat -c %s "$name")" "$cen"
    echo "$(md5sum < "$name" | cut -c1-32) $name" >> fixtures.md5
    made=$((made + 1))
  fi
}

# x264 NAME SOURCE FRAMES PIX_FMT PROBE PARAMS -- SEI... -- CENSUS...
# PROFILE=baseline|main in the environment restricts the profile (x264 picks
# High and above from the settings otherwise).
x264() {
  local name=$1 src=$2 n=$3 pf=$4 probe=$5 params=$6; shift 6
  encode "$name.264" h264 libx264 "$src" "$n" "$pf" "$probe" \
    ${PROFILE:+-profile:v $PROFILE} -x264-params "threads=1:$params" "$@"
}
x265() {
  local name=$1 src=$2 n=$3 pf=$4 probe=$5 params=$6; shift 6
  encode "$name.265" hevc libx265 "$src" "$n" "$pf" "$probe" \
    -x265-params "pools=1:frame-threads=1:$params" "$@"
}

# ------------------------------------------------------------------ H.264
echo "== H.264 =="
# ffprobe's field order: from the SPS for H.264; "unknown" for every HEVC
# stream here (no VUI field_seq_flag, no pic_struct).
P420="yuv420p,progressive"
H420="yuv420p,unknown"
PROFILE=baseline x264 x264_baseline_cavlc "$src_detail" 10 yuv420p "Constrained Baseline,$P420" \
  "crf=20" -- cabac=0 bframes=0 8x8dct=0 -- profile=66 cabac=0 t8x8=0 bslices=0
PROFILE=main x264 x264_main_cabac_b "$src_zoom" 12 yuv420p "Main,$P420" \
  "crf=20:bframes=3:b-pyramid=normal:direct=spatial:weightb=1" -- cabac=1 bframes=3 direct=1 weightb=1 -- profile=77 cabac=1 t8x8=0 wbi=2 "bslices>0"
PROFILE=main x264 x264_main_cavlc_b_temporal "$src_zoom" 12 yuv420p "Main,$P420" \
  "crf=20:cabac=0:bframes=3:b-pyramid=normal:direct=temporal" -- cabac=0 bframes=3 direct=2 -- profile=77 cabac=0 "bslices>0"
x264 x264_high_cabac_8x8_slices "$src_detail" 10 yuv420p "High,$P420" \
  "crf=20:8x8dct=1:slices=4:bframes=2" -- cabac=1 8x8dct=1 slices=4 -- profile=100 cabac=1 t8x8=1 slices_per_pic=4
x264 x264_high_cavlc_8x8 "$src_detail" 10 yuv420p "High,$P420" \
  "crf=20:cabac=0:8x8dct=1" -- cabac=0 8x8dct=1 -- cabac=0 t8x8=1
x264 x264_high_cabac_cip_deblock "$src_detail" 10 yuv420p "High,$P420" \
  "crf=20:constrained-intra=1:deblock=-3,3:bframes=2" -- constrained_intra=1 deblock=1:-3:3 -- cip=1 dfc=1
x264 x264_high_cqm_jvt "$src_detail" 10 yuv420p "High,$P420" \
  "crf=20:cqm=jvt" -- cqm=1 8x8dct=1 -- sl=1 t8x8=1
x264 x264_high_cqm_custom "$src_detail" 10 yuv420p "High,$P420" \
  "crf=20:cqmfile=x264_custom.cqm" -- cqm=2 8x8dct=1 -- sl=1 t8x8=1
x264 x264_high_interlaced_tff "$src_tff" 10 yuv420p "High,yuv420p,tt" \
  "crf=20:tff=1:bframes=3:direct=spatial" -- interlaced=tff direct=1 -- frame_mbs_only=0 mbaff=1 "bslices>0"
x264 x264_high_interlaced_bff "$src_bff" 10 yuv420p "High,yuv420p,bb" \
  "crf=20:bff=1:bframes=3:direct=temporal" -- interlaced=bff direct=2 -- frame_mbs_only=0 mbaff=1 "bslices>0"
x264 x264_high_weightp_fade "$src_fade" 12 yuv420p "High,$P420" \
  "crf=20:weightp=2:weightb=1:bframes=2" -- weightp=2 weightb=1 -- wp=1 wbi=2
x264 x264_high_odd_size "$src_odd264" 10 yuv420p "High,$P420" \
  "crf=20" -- -- crop=1
x264 x264_422 "$src_detail" 10 yuv422p "High 4:2:2,yuv422p,progressive" \
  "crf=20" -- cabac=1 -- profile=122 chroma=2 depth=8
x264 x264_422_10 "$src_detail" 10 yuv422p10le "High 4:2:2,yuv422p10le,progressive" \
  "crf=20" -- cabac=1 -- chroma=2 depth=10
x264 x264_422_10_cavlc "$src_detail" 10 yuv422p10le "High 4:2:2,yuv422p10le,progressive" \
  "crf=20:cabac=0" -- cabac=0 -- chroma=2 depth=10 cabac=0
x264 x264_444 "$src_detail" 10 yuv444p "High 4:4:4 Predictive,yuv444p,progressive" \
  "crf=20" -- cabac=1 8x8dct=1 -- profile=244 chroma=3 depth=8
x264 x264_444_cavlc "$src_detail" 10 yuv444p "High 4:4:4 Predictive,yuv444p,progressive" \
  "crf=20:cabac=0" -- cabac=0 -- chroma=3 cabac=0
x264 x264_444_10 "$src_detail" 10 yuv444p10le "High 4:4:4 Predictive,yuv444p10le,progressive" \
  "crf=20" -- cabac=1 -- chroma=3 depth=10
x264 x264_444_i16 "$src_detail" 10 yuv444p "High 4:4:4 Predictive,yuv444p,progressive" \
  "crf=20:partitions=none:8x8dct=0" -- analyse=0x1:0 8x8dct=0 -- chroma=3 t8x8=0
x264 x264_444_no8 "$src_detail" 10 yuv444p "High 4:4:4 Predictive,yuv444p,progressive" \
  "crf=20:8x8dct=0" -- 8x8dct=0 -- chroma=3 t8x8=0
x264 x264_444_lossless "$src_detail" 8 yuv444p "High 4:4:4 Predictive,yuv444p,progressive" \
  "qp=0" -- qp=0 -- chroma=3
x264 x264_gray "$src_detail" 10 gray "High,yuvj420p,progressive" \
  "crf=20" -- chroma_me=0 -- chroma=0
x264 x264_hi10 "$src_detail" 10 yuv420p10le "High 10,yuv420p10le,progressive" \
  "crf=20" -- cabac=1 8x8dct=1 -- profile=110 depth=10 cabac=1
x264 x264_hi10_cavlc "$src_detail" 10 yuv420p10le "High 10,yuv420p10le,progressive" \
  "crf=20:cabac=0" -- cabac=0 -- depth=10 cabac=0
x264 x264_hi10_cqm "$src_detail" 10 yuv420p10le "High 10,yuv420p10le,progressive" \
  "crf=20:cqm=jvt" -- cqm=1 -- depth=10 sl=1
x264 x264_hi10_lossless "$src_detail" 8 yuv420p10le "High 4:4:4 Predictive,yuv420p10le,progressive" \
  "qp=0" -- qp=0 -- depth=10 profile=244
x264 x264_420_lossless "$src_detail" 8 yuv420p "High 4:4:4 Predictive,$P420" \
  "qp=0" -- qp=0 cabac=1 -- profile=244 cabac=1
x264 x264_420_lossless_cavlc "$src_detail" 8 yuv420p "High 4:4:4 Predictive,$P420" \
  "qp=0:cabac=0" -- qp=0 cabac=0 -- profile=244 cabac=0
x264 x264_720p_cabac "$src_720" 12 yuv420p "High,$P420" \
  "crf=20:bframes=3:b-pyramid=normal" -- cabac=1 bframes=3 -- cabac=1 t8x8=1 "bslices>0"
x264 x264_720p_cavlc "$src_720" 12 yuv420p "High,$P420" \
  "crf=20:cabac=0:bframes=3" -- cabac=0 bframes=3 -- cabac=0 "bslices>0"
PROFILE=main x264 x264_720p_main "$src_720z" 12 yuv420p "Main,$P420" \
  "crf=20:bframes=3" -- cabac=1 bframes=3 -- profile=77 t8x8=0 "bslices>0"

# ------------------------------------------------------------------- HEVC
echo "== HEVC =="
x265 x265_main_wpp_sao "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:wpp=1:sao=1" -- -- wpp=1 sao=1 ctu=64
x265 x265_main10_nowpp "$src_detail" 10 yuv420p10le "Main 10,yuv420p10le,unknown" \
  "crf=22:wpp=0" -- no-wpp -- wpp=0 depth=10
x265 x265_main12 "$src_detail" 10 yuv420p12le "Rext,yuv420p12le,unknown" \
  "crf=22" -- -- depth=12
x265 x265_closedgop_idr "$src_zoom" 12 yuv420p "Main,$H420" \
  "crf=22:keyint=5:min-keyint=5:open-gop=0:bframes=2" -- no-open-gop keyint=5 -- idr=3 cra=0 "bslices>0"
x265 x265_opengop_cra "$src_zoom" 12 yuv420p "Main,$H420" \
  "crf=22:keyint=5:min-keyint=5:open-gop=1:bframes=3" -- open-gop keyint=5 -- idr=1 cra=2 "rasl>0"
x265 x265_ctu16_tlayers "$src_zoom" 12 yuv420p "Main,$H420" \
  "crf=22:ctu=16:max-tu-size=16:qg-size=16:temporal-layers=2:bframes=4:b-pyramid=1" -- ctu=16 temporal-layers=2 -- ctu=16 sub_layers=2 "tid_max>0"
x265 x265_cu_lossless "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:cu-lossless=1" -- cu-lossless -- tqbypass=1
# libavcodec (8.1.1) applies SAO to the chroma of transquant-bypass CUs in
# inter pictures: in POC 4 (decode order 1) the two intra bypass 8x8 CUs at
# luma (64,40) and (120,72) come out of our decoder equal to the lavfi
# source frame sample for sample — a lossless CU's definition — while
# libavcodec's Cb/Cr there are off by the CTB's band offsets (+2/-2, +3);
# luma of the same CUs agrees. The difference then propagates through
# prediction to every later picture (output frames 1..9 differ).
libav_deviates x265_cu_lossless.265 "libavcodec applies SAO to the chroma of transquant-bypass CUs in inter pictures; ours equals the source there (POC 4, CUs at luma 64,40 and 120,72)"
x265 x265_lossless "$src_detail" 8 yuv420p "Main,$H420" \
  "lossless=1" -- lossless -- tqbypass=1
x265 x265_intra_only "$src_detail" 8 yuv420p "Rext,$H420" \
  "crf=22:keyint=1:min-keyint=1" -- keyint=1 -- irap=8 pslices=0 bslices=0
# The default lists only: x265 4.2 refuses every HM-format custom list
# file tried ("can't read matrix"), so explicit scaling_list_data stays
# the conformance suite's (SLIST_A..D).
x265 x265_main_scalinglist "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:scaling-list=default" -- -- sl=1 sl_data=0
x265 x265_main_tskip_amp "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:tskip=1:amp=1:rect=1" -- tskip amp -- tskip=1 amp=1
x265 x265_main_weighted "$src_fade" 12 yuv420p "Main,$H420" \
  "crf=22:weightp=1:weightb=1:bframes=2" -- weightp weightb -- wp=1 wbi=1 "bslices>0"
# no-deblock with WPP on deadlocks this x265 (4.2) at every pool size, so
# this one is also the no-WPP 8-bit stream.
x265 x265_notmvp_nodeblock "$src_zoom" 10 yuv420p "Main,$H420" \
  "crf=22:wpp=0:temporal-mvp=0:no-deblock=1" -- no-wpp no-temporal-mvp no-deblock -- wpp=0 tmvp=0 deblock_off=1
x265 x265_odd_size_ctu32 "$src_odd265" 10 yuv420p "Main,$H420" \
  "crf=22:ctu=32" -- ctu=32 -- ctu=32 crop=1
x265 x265_slices_wpp_qg16_cip "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:slices=2:wpp=1:qg-size=16:constrained-intra=1:aq-mode=2" -- slices=2 qg-size=16 constrained-intra -- slices_per_pic=2 wpp=1 cip=1 cuqp=1 qg_depth=2
x265 x265_p_notmvp "$src_zoom" 10 yuv420p "Main,$H420" \
  "crf=22:bframes=0:temporal-mvp=0" -- bframes=0 no-temporal-mvp -- tmvp=0 bslices=0 "pslices>0"
x265 x265_main_nostrong_nosignhide "$src_detail" 10 yuv420p "Main,$H420" \
  "crf=22:strong-intra-smoothing=0:signhide=0" -- no-strong-intra-smoothing no-signhide -- strong=0 sbh=0
x265 x265_gray "$src_detail" 10 gray "Rext,gray,unknown" \
  "crf=22" -- -- chroma=0
x265 x265_main422_10 "$src_detail" 10 yuv422p10le "Rext,yuv422p10le,unknown" \
  "crf=22" -- -- chroma=2 depth=10
x265 x265_main444 "$src_detail" 10 yuv444p "Rext,yuv444p,unknown" \
  "crf=22" -- -- chroma=3 depth=8
x265 x265_720p_nowpp "$src_720" 12 yuv420p "Main,$H420" \
  "crf=22:wpp=0:bframes=3" -- no-wpp -- wpp=0 "bslices>0"
x265 x265_720p_wpp "$src_720z" 12 yuv420p "Main,$H420" \
  "crf=22:wpp=1:sao=1:bframes=3" -- -- wpp=1 sao=1 "bslices>0"

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
