#!/bin/bash
# JVT professional-profiles suite (High 10 Intra, High 4:2:2 Intra, High
# 4:4:4 Intra/Predictive, CAVLC 4:4:4 Intra): no reference YUVs ship with it,
# so the JM reference decoder's per-frame MD5s are the reference, generated
# on first use through ref_decode.py (jm/<name>.md5 from jm_ref.sh, where
# present, takes precedence).
# Streams run JOBS at a time (default 8), THREADS per decoder (4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
# Two callers running at once must not share a scratch directory or a
# decoder copy: a run that quietly tested somebody else's binary still
# reports green.
OUT=${OUT:-out}
# The reference decoder (JM / HM, via ref_decode.py), for streams whose
# zip carries no reference output. REF_DECODE names the script: run from
# the repository it is ../../ref_decode.py; from a copy under
# $H26X_WORK/conf, point it at the repository's tools/ref_decode.py.
REF_DECODE=${REF_DECODE:-$(cd ../.. && pwd)/ref_decode.py}
export DEC OUT REF_DECODE H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p "$OUT" md5
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs=$(find "$d" -maxdepth 1 -type f \( -iname "*.264" -o -iname "*.bits" -o -iname "*.jsv" -o -iname "*.h264" \) | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM $name"; return; }
  mine="$OUT/$name.mine"
  "$DEC" "$bs" > "$mine" 2> "$OUT/$name.err"
  st=$?
  if [ $st -ne 0 ]; then
    msg=$(tail -1 "$OUT/$name.err")
    case "$msg" in
      *unsupported*) echo "UNSUPPORTED $name: $msg"; return ;;
      *) echo "FAIL $name: $msg"; return ;;
    esac
  fi
  nmine=$(wc -l < "$mine")
  # Reference: jm/<name>.md5 ("i md5" per frame) where jm_ref.sh made one,
  # else the JM decode through ref_decode.py, cached in md5/.
  if [ -s "jm/$name.md5" ]; then
    ref="jm/$name.md5"
  else
    ref="md5/$name.ref.md5"
    # The Sejong 4:4:4 streams' VUI says matrix_coefficients 0 (RGB), and JM
    # writes those planes as R, G, B; --rgb undoes that order.
    rgb=; case "$name" in PPCV444I7_SejongUniv_A|PPH444I7_SejongUniv_A|PPH444P10_SejongUniv_A) rgb=--rgb ;; esac
    [ -s "$ref" ] || python "$REF_DECODE" framemd5 $rgb "$bs" "$nmine" > "$ref" 2> "$OUT/$name.referr" || rm -f "$ref"
  fi
  nref=$(cat "$ref" 2>/dev/null | wc -l)
  bad=$(paste <(awk '{print $2}' "$ref" 2>/dev/null) <(awk -F, '{print $5}' "$mine") | awk '$1!=$2' | wc -l)
  if [ "$nref" = "$nmine" ] && [ "$bad" = "0" ]; then
    echo "PASS $name ($nref frames)"
  else
    echo "FAIL $name: frames ref=$nref mine=$nmine mismatching=$bad"
  fi
  echo "$name $(md5sum < "$mine" | cut -c1-32)" >> "$MD5REC"
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > $OUT/results.txt
cat "$OUT"/md5parts-*.tmp 2>/dev/null | sort > "$OUT/md5s.txt"; rm -f "$OUT"/md5parts-*.tmp
cat $OUT/results.txt
echo "pass=$(grep -c '^PASS' $OUT/results.txt) fail=$(grep -c '^FAIL' $OUT/results.txt) unsupported=$(grep -c '^UNSUPPORTED' $OUT/results.txt)"
