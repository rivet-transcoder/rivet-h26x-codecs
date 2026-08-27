#!/bin/bash
# verify_enc_ladder.sh — does every rung of the SIMD ladder encode to the
# same bytes?
#
# verify.sh asks this of the decoders: every H26X_MAX_SIMD rung must decode
# the corpus to the same output. This is the encoder's form of the same
# question, and it is a stronger one, because an encoder that takes a
# different decision anywhere — a SATD off by one in a mode search, a
# quantiser that rounds a level differently — writes a different stream,
# whatever its own reconstruction says. For each rung the host can take,
# every 8-bit cell of verify_encode.sh's configuration list is encoded by
# one binary twice, with the scalar reference (H26X_NO_SIMD=1) and with the
# rung, and the two bitstreams and reconstructions are compared byte for
# byte (identity_encode.sh does the cell work). The rung caps *every* table
# the encoder builds — the encode-only ones and the interpolation, inverse
# transform and loop filter it shares with the decoder — so a green ladder
# says the whole encoder is rung-independent, not just its own kernels.
#
# 8-bit cells only: the encode-side SIMD kernels exist for 8-bit samples
# and the 16-bit tables keep the scalar reference, so a 10/12-bit cell
# exercises only the shared decoder kernels, which verify.sh already covers.
#
# Usage: verify_enc_ladder.sh [encoder] [decoder]
#   H26X_WORK=dir   scratch directory holding the source clips
#   JOBS=n          cells in parallel per rung (default 4)
#   RUNGS="..."     override the rung list (default: the host's ladder)
#   The decoder, if given, reports which rung each cap actually selects
#   (`--rung`), so a cap the CPU cannot honour is visible rather than a
#   silent duplicate of the rung below it.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENC=${1:-../release/examples/h26xenc.exe}
DEC=${2:-}
case "$(uname -m)" in
  aarch64 | arm64) DEFAULT_RUNGS="neon" ;;
  *) DEFAULT_RUNGS="sse2 ssse3 sse41 avx avx2 avx512" ;;
esac
RUNGS=${RUNGS:-$DEFAULT_RUNGS}
WORK=${H26X_WORK:-$HERE}
cd "$WORK" || { echo "no such H26X_WORK: $WORK" >&2; exit 2; }
[ -f "$ENC" ] || ENC=${ENC%.exe}
[ -f "$ENC" ] || { echo "no such encoder: $ENC" >&2; exit 2; }

# The 8-bit clips: everything not named with a depth suffix.
SOURCES=$(ls src_*.yuv 2>/dev/null | grep -Ev '_[0-9]+p1[0-9]\.yuv$')
[ -n "$SOURCES" ] || { echo "no 8-bit source clips (src_*.yuv) in $WORK" >&2; exit 2; }
export SOURCES

echo "== encode ladder: scalar reference vs each rung, $(echo $SOURCES | wc -w) clips =="
rungs_ok=0
rungs_bad=0
for r in $RUNGS; do
  if [ -n "$DEC" ] && [ -x "$DEC" ]; then
    got=$(H26X_MAX_SIMD=$r "$DEC" --rung 2>/dev/null)
    echo "-- rung $r (selects: ${got:-?}) --"
  else
    echo "-- rung $r --"
  fi
  out=$(ENV_A="H26X_NO_SIMD=1" ENV_B="H26X_MAX_SIMD=$r" H26X_WORK="$WORK" JOBS="${JOBS:-4}" \
        bash "$HERE/identity_encode.sh" "$ENC" 2>&1)
  rc=$?
  # The per-cell lines are identity_encode.sh's; keep only the moved ones
  # and the tally, since an identical cell says nothing new here.
  echo "$out" | grep -E '^(MOVED|ENCODE-FAIL|identity:|NO CELLS)' | sed 's/^/   /'
  if [ "$rc" = 0 ]; then rungs_ok=$((rungs_ok + 1)); else rungs_bad=$((rungs_bad + 1)); fi
done

echo
echo "ladder: $rungs_ok rungs identical, $rungs_bad failed"
if [ "$rungs_ok" = 0 ]; then echo "NO RUNGS RAN"; exit 2; fi
[ "$rungs_bad" = 0 ] && echo "LADDER IDENTICAL" || echo "LADDER MOVED"
[ "$rungs_bad" = 0 ]
