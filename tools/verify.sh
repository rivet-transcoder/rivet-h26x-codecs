#!/bin/bash
# verify.sh [decoder.exe] — everything that has to hold before a change lands:
# every workspace fixture against its golden MD5, then all four conformance
# suites (JVT AVCv1+FRExt, JVT professional profiles, JCT-VC HEVC_v1, RExt).
# Defaults to the workspace build; pass a path to check another one.
#   JOBS=n   streams in parallel per suite (default 6)
# Run from the scratch directory holding the fixtures and `conf/` (by
# convention the rivet workspace's target/h26x), or set H26X_WORK to it.
cd "${H26X_WORK:-$(dirname "$0")}"
DEC=${1:-../release/examples/h26xdec.exe}
DEC=$(cd "$(dirname "$DEC")" && pwd)/$(basename "$DEC")
# `golden.txt`: "<fixture> <md5>" per line, the MD5 each fixture must decode
# to. Regenerate deliberately, never to make a red run go green.
GOLD=${GOLD:-golden.txt}
fail=0

echo "== fixtures (golden MD5s) =="
ok=0; bad=0
while read -r f m; do
  [ -f "$f" ] || { echo "MISSING  $f"; bad=$((bad + 1)); continue; }
  g=$(H26X_THREADS=4 "$DEC" "$f" 2>/dev/null | md5sum | cut -c1-32)
  if [ "$g" = "$m" ]; then ok=$((ok + 1)); else echo "MISMATCH $f  $g != $m"; bad=$((bad + 1)); fi
done < "$GOLD"
echo "fixtures: $ok matched, $bad failed"
[ "$bad" = 0 ] || fail=1

echo
echo "== conformance suites =="
# The suites share one frozen copy of the decoder so a rebuild mid-run cannot
# disturb them.
cp "$DEC" ../release/examples/h26xdec_conf.exe || exit 1
J=${JOBS:-6}
for s in h264 h264_pp hevc hevc_rext; do
  JOBS=$J bash conf/$s/run_conf.sh > conf/$s/verify.log 2>&1 &
done
wait
for s in h264 h264_pp hevc hevc_rext; do
  printf "%-10s %s\n" "$s" "$(tail -1 conf/$s/verify.log)"
  grep -E "^FAIL|^NOSTREAM" conf/$s/verify.log
  grep -qE "fail=0" conf/$s/verify.log || fail=1
  grep -qE "^(FAIL|NOSTREAM)" conf/$s/verify.log && fail=1
done

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
