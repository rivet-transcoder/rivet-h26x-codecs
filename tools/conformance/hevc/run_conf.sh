#!/bin/bash
# Run every JCT-VC HEVC_v1 conformance stream through h26xdec and classify it.
# The suite's own whole-YUV MD5 is the reference when the zip carries one;
# otherwise libavcodec's per-frame MD5s (generated on first use) are used.
#   PASS         bit-exact
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
# Two callers running at once must not share a scratch directory or a
# decoder copy: a run that quietly tested somebody else's binary still
# reports green.
OUT=${OUT:-out}
export DEC OUT H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p refs "$OUT"
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs=$(find "$d" -maxdepth 4 -type f \( -name "*.bit" -o -name "*.bin" -o -name "*.265" -o -name "*.hevc" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM    $name"; return; }
  cp "$bs" "$OUT/$name.265" 2>/dev/null
  "$DEC" "$OUT/$name.265" "$OUT/$name.yuv" > "$OUT/$name.mine" 2> "$OUT/$name.err"
  status=$?
  n_mine=$(wc -l < "$OUT/$name.mine")
  # Suite MD5 (of the whole decoded YUV), if any: every 32-hex string in the
  # suite's md5 files (bitstream md5s among them are harmless).
  suite_md5s=$(cat $(find "$d" -maxdepth 4 -type f \( -name "*.md5" -o -name "*md5*.txt" \) 2>/dev/null) 2>/dev/null | grep -io '[0-9a-f]\{32\}' | tr 'A-F' 'a-f' | sort -u)
  mine_md5=$(md5sum "$OUT/$name.yuv" | cut -c1-32)
  if grep -q "unsupported" "$OUT/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 $OUT/$name.err)"
  elif [ $status -eq 0 ] && echo "$suite_md5s" | grep -q "^$mine_md5$"; then
    echo "PASS        $name ($n_mine frames, suite md5)"
  else
    ref="refs/$name.framemd5"
    [ -f "$ref" ] || ffmpeg -v error -y -i "$bs" -fps_mode passthrough -f framemd5 "$ref" 2>/dev/null
    n_ref=$(grep -vc '^#' "$ref")
    match=$(paste <(grep -v '^#' "$ref" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "$OUT/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, ffmpeg framemd5)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match suite=[$(echo $suite_md5s | tr '\n' ' ')] mine_md5=$mine_md5 $(tail -1 $OUT/$name.err | head -c 120)"
    fi
  fi
  [ $status -eq 0 ] && echo "$name $mine_md5" >> "$MD5REC"
  rm -f "$OUT/$name.265" "$OUT/$name.yuv"
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > $OUT/results.txt
cat "$OUT"/md5parts-*.tmp 2>/dev/null | sort > "$OUT/md5s.txt"; rm -f "$OUT"/md5parts-*.tmp
cat $OUT/results.txt
echo "pass=$(grep -c '^PASS' $OUT/results.txt) fail=$(grep -c '^FAIL' $OUT/results.txt) unsupported=$(grep -c '^UNSUPPORTED' $OUT/results.txt)"
