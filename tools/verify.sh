#!/bin/bash
# verify.sh [--baseline FILE] [decoder] — everything that has to hold before a
# change lands: every fixture against its recorded MD5, then all four
# conformance suites (JVT AVCv1+FRExt, JVT professional profiles, JCT-VC
# HEVC_v1, RExt).
#
# --baseline FILE  also record "<suite> <status> <stream> <md5>" per stream to
#                  FILE, and if FILE already exists, diff against it instead.
#                  For a refactor that must not change output, that is a
#                  stricter question than PASS/FAIL — it asks whether every
#                  stream decodes to the same bytes, including the ones no
#                  reference data covers.
#
#   JOBS=n         streams in parallel per suite (default 6)
#   H26X_WORK=dir  the scratch directory holding the fixtures and conf/
#                  (default: this script's directory)
#
# Safe to run while another copy of it is running: the decoder is copied to a
# private name and each suite writes to a private scratch directory. Sharing
# either lets one run report green having tested the other's binary.
cd "${H26X_WORK:-$(dirname "$0")}"
BASE=""
if [ "$1" = "--baseline" ]; then BASE=$2; shift 2; fi
# `golden.txt`: "<fixture> <md5>" per line, the MD5 each fixture must decode
# to. Regenerate deliberately, never to make a red run go green.
GOLD=${GOLD:-golden.txt}
DEC=${1:-../release/examples/h26xdec.exe}
DEC=$(cd "$(dirname "$DEC")" && pwd)/$(basename "$DEC")
# Prove the decoder exists and produces output before anything is compared.
#
# This script cd's to H26X_WORK first, so a relative path given from anywhere
# else resolves to nothing — and a decoder that cannot run produces empty
# output for every fixture and every rung. The fixture check notices, because
# it compares against recorded hashes. The rung sweep does not: it compares
# the rungs against each other, and seven empty outputs agree perfectly. A
# sweep that passes on a decoder that never ran is worse than no sweep, so it
# gets an anchor.
if [ ! -x "$DEC" ] && [ ! -f "$DEC" ]; then
  echo "verify.sh: no decoder at $DEC" >&2
  echo "  (this script cd's to ${H26X_WORK:-its own directory} first, so give an absolute path)" >&2
  exit 2
fi
_probe=$(head -1 "${GOLD:-golden.txt}" | cut -d' ' -f1)
if [ -n "$_probe" ] && [ -f "$_probe" ]; then
  if [ -z "$("$DEC" "$_probe" 2>/dev/null | head -c 1)" ]; then
    echo "verify.sh: $DEC produced no output for $_probe" >&2
    echo "  Refusing to run: every comparison below would be between empty files," >&2
    echo "  and the rung sweep would report all rungs identical because they are." >&2
    exit 2
  fi
fi
TAG=$$
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

# Every rung of the SIMD ladder has to decode to the same bytes as the scalar
# reference. The README says so, so it is gated here rather than checked by
# hand once: a kernel added at one rung and not another is otherwise invisible
# until it reaches a machine that selects that rung, which is never this one.
echo
echo "== SIMD rungs (all fixtures, every rung must agree) =="
case "$(uname -m)" in
  aarch64 | arm64) RUNGS="neon none" ;;
  *) RUNGS="avx512 avx2 avx sse41 ssse3 sse2 none" ;;
esac
ref=""
for r in $RUNGS; do
  all=""
  while read -r f m; do
    [ -f "$f" ] || continue
    all="$all$(H26X_MAX_SIMD=$r H26X_THREADS=4 "$DEC" "$f" 2>/dev/null | md5sum | cut -c1-32)"
  done < "$GOLD"
  h=$(printf %s "$all" | md5sum | cut -c1-32)
  if [ -z "$ref" ]; then ref=$h; printf "  %-6s %s
" "$r" "$h"
  elif [ "$h" = "$ref" ]; then printf "  %-6s %s  same
" "$r" "$h"
  else printf "  %-6s %s  DIFFERS
" "$r" "$h"; fail=1; fi
done

echo
echo "== conformance suites =="
# A private copy, so a rebuild mid-run cannot disturb the suites and a
# concurrent verify cannot swap the binary under them.
PRIV=../release/examples/h26xdec_verify_$TAG.exe
cp "$DEC" "$PRIV" || exit 1
trap 'rm -f "$PRIV"; rm -rf conf/*/out_$TAG' EXIT
J=${JOBS:-6}
for s in h264 h264_pp hevc hevc_rext; do
  DEC=$(cd ../release/examples && pwd)/h26xdec_verify_$TAG.exe OUT=out_$TAG JOBS=$J \
    bash conf/$s/run_conf.sh > conf/$s/verify_$TAG.log 2>&1 &
done
wait
rec=$(mktemp)
for s in h264 h264_pp hevc hevc_rext; do
  printf "%-10s %s\n" "$s" "$(tail -1 conf/$s/verify_$TAG.log)"
  grep -E "^FAIL|^NOSTREAM" conf/$s/verify_$TAG.log
  grep -qE "fail=0" conf/$s/verify_$TAG.log || fail=1
  grep -qE "^(FAIL|NOSTREAM)" conf/$s/verify_$TAG.log && fail=1
  # Per-stream decoded-output MD5s, for --baseline.
  if [ -n "$BASE" ] && [ -f "conf/$s/out_$TAG/md5s.txt" ]; then
    sed "s/^/$s /" "conf/$s/out_$TAG/md5s.txt" >> "$rec"
  fi
  mv "conf/$s/verify_$TAG.log" "conf/$s/verify.log"
done

if [ -n "$BASE" ]; then
  sort -o "$rec" "$rec"
  if [ -f "$BASE" ]; then
    echo
    if diff -u "$BASE" "$rec" > /tmp/baseline.diff; then
      echo "baseline: every stream decodes to the same bytes as $BASE"
    else
      echo "baseline: OUTPUT CHANGED against $BASE"
      head -40 /tmp/baseline.diff
      fail=1
    fi
  else
    cp "$rec" "$BASE"
    echo
    echo "baseline: recorded $(wc -l < "$BASE") streams to $BASE"
  fi
fi
rm -f "$rec"

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
