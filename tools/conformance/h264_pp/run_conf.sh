#!/bin/bash
# JVT professional-profiles suite (High 10 Intra, High 4:2:2 Intra, High
# 4:4:4 Intra/Predictive, CAVLC 4:4:4 Intra): no reference YUVs ship with it,
# so libavcodec's per-frame MD5s (native pixel format) are the reference —
# and JM ldecod's (jm/*.md5) for the separate-colour-plane streams libavcodec
# refuses. Streams run JOBS at a time (default 8), THREADS per decoder (4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=$(cd ../../../release/examples && pwd)/h26xdec_conf.exe
JOBS=${JOBS:-8}
export DEC H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p out md5
one() {
  d=$1
  name=$(basename "$d")
  bs=$(find "$d" -maxdepth 1 -type f \( -iname "*.264" -o -iname "*.bits" -o -iname "*.jsv" -o -iname "*.h264" \) | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM $name"; return; }
  # Reference: JM ldecod per-frame MD5s (jm/<name>.md5, "i md5" lines) where
  # libavcodec cannot decode the stream (separate colour planes), else
  # libavcodec's framemd5.
  if [ -s "jm/$name.md5" ]; then
    ref="md5/$name.jm.framemd5"
    awk '{print "0, 0, 0, 0, 0, " $2}' "jm/$name.md5" > "$ref"
  else
    ref="md5/$name.framemd5"
    if [ ! -s "$ref" ]; then
      ffmpeg -v error -y -i "$bs" -fps_mode passthrough -f framemd5 "$ref" 2>/dev/null
    fi
  fi
  mine="out/$name.mine"
  "$DEC" "$bs" > "$mine" 2> "out/$name.err"
  st=$?
  if [ $st -ne 0 ]; then
    msg=$(tail -1 "out/$name.err")
    case "$msg" in
      *unsupported*) echo "UNSUPPORTED $name: $msg"; return ;;
      *) echo "FAIL $name: $msg"; return ;;
    esac
  fi
  nref=$(grep -vc '^#' "$ref"); nmine=$(wc -l < "$mine")
  bad=$(paste <(grep -v '^#' "$ref" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "$mine") | awk '$1!=$2' | wc -l)
  if [ "$nref" = "$nmine" ] && [ "$bad" = "0" ]; then
    echo "PASS $name ($nref frames)"
  else
    echo "FAIL $name: frames ref=$nref mine=$nmine mismatching=$bad"
  fi
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > out/results.txt
cat out/results.txt
echo "pass=$(grep -c '^PASS' out/results.txt) fail=$(grep -c '^FAIL' out/results.txt) unsupported=$(grep -c '^UNSUPPORTED' out/results.txt)"
