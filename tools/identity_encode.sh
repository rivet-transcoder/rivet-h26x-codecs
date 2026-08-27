#!/bin/bash
# identity_encode.sh — do the encode-side SIMD kernels change a single bit?
#
# A SIMD kernel that is bit-exact against its scalar reference over random
# inputs is the unit test; this is the integration form of the same claim.
# Every cell of verify_encode.sh's configuration list is encoded twice by one
# binary — once with `H26X_ENC_NO_SIMD=1`, which keeps the encode-only tables
# (distortion, h264_enc, hevc_enc) scalar, once as shipped — and the two
# bitstreams are compared byte for byte. A kernel that changed a decision
# anywhere in the encoder shows up here as a moved cell, whatever it did to
# its own unit test.
#
# The claim this makes is only as strong as the cells that reach a kernel:
# a cell whose configuration never calls the kernel is identical for free.
# The corpus and the configurations are verify_encode.sh's, so every row
# that gate exercises is exercised here.
#
# Usage: identity_encode.sh [encoder]
#   H26X_WORK=dir   scratch directory holding the source clips (default: here)
#   JOBS=n          cells in parallel (default 4)
#   ENV_A / ENV_B   the two environments (default H26X_ENC_NO_SIMD=1 vs none)
#   ENC_B=exe       a second binary for side B, for a change that has no
#                   switch (default: the same binary)
# Resolved before the cd below, which would otherwise turn a relative
# script path into nothing.
VERIFY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verify_encode.sh
cd "${H26X_WORK:-$(dirname "$0")}"
ENC=${1:-../release/examples/h26xenc.exe}
[ -f "$ENC" ] || ENC=${ENC%.exe}
ENC_B=${ENC_B:-$ENC}
TAG=$$
OUT=identity_out_$TAG
JOBS=${JOBS:-4}
# `-`, not `:-`: an explicitly empty ENV_A means "as shipped", not the default.
ENV_A=${ENV_A-H26X_ENC_NO_SIMD=1}
ENV_B=${ENV_B:-}
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT

SOURCES=${SOURCES:-$(ls src_*.yuv 2>/dev/null)}
[ -n "$SOURCES" ] || { echo "no source clips (src_*.yuv)" >&2; exit 2; }
# The configuration list is verify_encode.sh's own, read out of it so the
# two cannot drift: everything between CONFIGS=${CONFIGS:-" and the closing
# quote.
CONFIGS=$(sed -n '/^CONFIGS=\${CONFIGS:-"/,/^"}/p' "$VERIFY" | sed '1d;$d')

one() {
  src=$1; name=$2; flags=$3
  base=$(basename "$src" .yuv)
  geom=$(echo "$base" | sed -n 's/.*_\([0-9]\+x[0-9]\+\)_.*/\1/p')
  fmt=$(echo "$base" | sed -n 's/.*_[0-9]\+x[0-9]\+_\(.*\)/\1/p')
  # `420p10` is chroma 420 at depth 10; a bare `420` is eight bits — the
  # reading verify_encode.sh's chroma_of / depth_of make.
  chroma=${fmt%%p*}; depth=${fmt##*p}; [ "$depth" = "$fmt" ] && depth=8
  tag="$base/$name"
  a="$OUT/$base.$name.a"; b="$OUT/$base.$name.b"
  if ! env $ENV_A "$ENC" --input "$src" --size "$geom" --format "$chroma" --depth "$depth" $flags --output "$a" --recon "$a.yuv" > "$a.log" 2>&1; then
    echo "ENCODE-FAIL $tag (A): $(tail -1 "$a.log" | head -c 100)"; return 1
  fi
  if ! env $ENV_B "$ENC_B" --input "$src" --size "$geom" --format "$chroma" --depth "$depth" $flags --output "$b" --recon "$b.yuv" > "$b.log" 2>&1; then
    echo "ENCODE-FAIL $tag (B): $(tail -1 "$b.log" | head -c 100)"; return 1
  fi
  if ! cmp -s "$a" "$b"; then
    echo "MOVED       $tag: bitstreams differ ($(stat -c %s "$a") vs $(stat -c %s "$b") bytes)"; return 1
  fi
  if ! cmp -s "$a.yuv" "$b.yuv"; then
    echo "MOVED       $tag: reconstructions differ"; return 1
  fi
  echo "SAME        $tag ($(stat -c %s "$a") bytes)"
}
export -f one
export ENC ENC_B OUT ENV_A ENV_B

echo "== encode identity: [$ENV_A] $(basename "$ENC") vs [${ENV_B:-as shipped}] $(basename "$ENC_B") =="
results="$OUT/results.txt"
for src in $SOURCES; do
  echo "$CONFIGS" | while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    # As verify_encode.sh: a clip deeper than 8 bits is visited only by
    # rows that name it with `@p10` / `@p12`; a row without `@` is 8-bit.
    case "$src" in *_[0-9][0-9][0-9]p[0-9]*.yuv) deep=1 ;; *) deep=0 ;; esac
    case "$name" in
      *@*)
        pat=${name##*@}
        case "$src" in *"$pat"*) name=${name%@*} ;; *) continue ;; esac ;;
      *) [ "$deep" = 1 ] && continue ;;
    esac
    echo "$src|$name|$flags"
  done
done | xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r s n f <<< "{}"; one "$s" "$n" "$f"' | sort | tee "$results"

same=$(grep -c '^SAME' "$results")
bad=$(grep -cE '^(MOVED|ENCODE-FAIL)' "$results")
echo
echo "identity: $same identical, $bad moved"
# Zero cells is not a pass: it is the configuration list failing to parse
# or the corpus missing, and a green line over nothing is the vacuity this
# whole gate exists to refuse.
if [ "$same" = 0 ]; then echo "NO CELLS RAN"; exit 2; fi
[ "$bad" = 0 ] && echo "ALL IDENTICAL" || echo "CELLS MOVED"
[ "$bad" = 0 ]
