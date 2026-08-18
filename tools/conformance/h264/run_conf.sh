#!/bin/bash
# Run every JVT AVCv1 + FRExt conformance stream through h26xdec and classify it.
# The reference is the suite's decoded YUV in the zip (any *.yuv in the dir; whole-file
# MD5 must match); when the zip carries none, libavcodec's per-frame MD5s are used.
#   PASS         bit-exact
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run in parallel: JOBS at a time (default 8), each decoder on THREADS threads
# (default 4). `run_conf.sh <filter>` runs the streams whose name contains <filter>.
cd "$(dirname "$0")"
DEC=$(cd ../../../release/examples && pwd)/h26xdec_conf.exe
JOBS=${JOBS:-8}
export DEC H26X_THREADS=${THREADS:-4}
mkdir -p refs out
one() {
  d=$1
  name=$(basename "$d")
  bs=$(find "$d" -maxdepth 4 -type f \( -iname "*.264" -o -iname "*.jsv" -o -iname "*.26l" -o -iname "*.h264" -o -iname "*.avc" -o -iname "*.jvt" -o -iname "*.bit" -o -iname "*.dat" -o -iname "*.qcif" \) 2>/dev/null | grep -v "\.yuv$" | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM    $name"; return; }
  cp "$bs" "out/$name.264" 2>/dev/null
  "$DEC" "out/$name.264" "out/$name.yuv" > "out/$name.mine" 2> "out/$name.err"
  status=$?
  n_mine=$(wc -l < "out/$name.mine")
  mine_md5=$(md5sum "out/$name.yuv" | cut -c1-32)
  ref_md5s=$(for y in $(find "$d" -maxdepth 4 -type f -iname "*.yuv" 2>/dev/null); do md5sum "$y" | cut -c1-32; done)
  if grep -q "unsupported" "out/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 out/$name.err | head -c 100)"
  elif [ $status -eq 0 ] && [ -n "$ref_md5s" ] && echo "$ref_md5s" | grep -q "^$mine_md5$"; then
    echo "PASS        $name ($n_mine frames, suite yuv)"
  else
    ref="refs/$name.framemd5"
    [ -f "$ref" ] || ffmpeg -v error -y -i "$bs" -fps_mode passthrough -f framemd5 "$ref" 2>/dev/null
    n_ref=$(grep -vc '^#' "$ref")
    match=$(paste <(grep -v '^#' "$ref" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "out/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, ffmpeg framemd5)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match $(tail -1 out/$name.err | head -c 120)"
    fi
  fi
  rm -f "out/$name.264" "out/$name.yuv"
}
export -f one
ls -d AVCv1/*/ FRExt/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > out/results.txt
cat out/results.txt
echo "pass=$(grep -c '^PASS' out/results.txt) fail=$(grep -c '^FAIL' out/results.txt) unsupported=$(grep -c '^UNSUPPORTED' out/results.txt)"
