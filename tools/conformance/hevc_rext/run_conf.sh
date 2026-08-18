#!/bin/bash
# Run every JCT-VC RExt conformance stream through h26xdec and classify it.
# The suite's own whole-YUV MD5 is the reference when the zip carries one;
# otherwise libavcodec's per-frame MD5s (generated on first use) are used.
#   PASS         bit-exact
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=$(cd ../../../release/examples && pwd)/h26xdec_conf.exe
JOBS=${JOBS:-8}
export DEC H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p refs out
one() {
  d=$1
  name=$(basename "$d")
  bs=$(find "$d" -maxdepth 4 -type f \( -name "*.bit" -o -name "*.bin" -o -name "*.265" -o -name "*.hevc" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM    $name"; return; }
  cp "$bs" "out/$name.265" 2>/dev/null
  "$DEC" "out/$name.265" "out/$name.yuv" > "out/$name.mine" 2> "out/$name.err"
  status=$?
  n_mine=$(wc -l < "out/$name.mine")
  # Suite MD5 (of the whole decoded YUV), if any: every 32-hex string in the
  # suite's md5 files (bitstream md5s among them are harmless).
  suite_md5s=$(cat $(find "$d" -maxdepth 4 -type f \( -name "*.md5" -o -name "*md5*.txt" \) 2>/dev/null) 2>/dev/null | grep -io '[0-9a-f]\{32\}' | tr 'A-F' 'a-f' | sort -u)
  mine_md5=$(md5sum "out/$name.yuv" | cut -c1-32)
  if grep -q "unsupported" "out/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 out/$name.err)"
  elif [ $status -eq 0 ] && echo "$suite_md5s" | grep -q "^$mine_md5$"; then
    echo "PASS        $name ($n_mine frames, suite md5)"
  else
    ref="refs/$name.framemd5"
    [ -f "$ref" ] || ffmpeg -v error -y -i "$bs" -fps_mode passthrough -f framemd5 "$ref" 2>/dev/null
    n_ref=$(grep -vc '^#' "$ref")
    match=$(paste <(grep -v '^#' "$ref" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "out/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, ffmpeg framemd5)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match suite=[$(echo $suite_md5s | tr '\n' ' ')] mine_md5=$mine_md5 $(tail -1 out/$name.err | head -c 120)"
    fi
  fi
  rm -f "out/$name.265" "out/$name.yuv"
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > out/results.txt
cat out/results.txt
echo "pass=$(grep -c '^PASS' out/results.txt) fail=$(grep -c '^FAIL' out/results.txt) unsupported=$(grep -c '^UNSUPPORTED' out/results.txt)"
