#!/bin/bash
# wasm.sh — decode inside wasm and check the bytes.
#
# `cargo test` cannot run on wasm32-unknown-unknown: no test harness, no
# filesystem, no clock. So CI can prove the crate *compiles* for wasm and
# nothing more, and a decoder that compiles and aborts on its first picture
# passes every check anyone runs and fails in front of a user. This crate did
# exactly that until the profiling clock reads were guarded — HEVC panicked on
# its first NAL because `Instant::now()` is not implemented there, while H.264
# worked by happening never to take a reading.
#
# Two questions, and this answers both:
#
#   1. Does wasm decode correctly at all? The three vendored streams are
#      decoded inside the module and checked against `tests/decode.rs` — frame
#      counts and output hashes both. Those expectations are read out of that
#      file rather than copied here, because a copy drifts and the stale one is
#      the one somebody trusts; they were anchored frame-by-frame against
#      libavcodec, so passing here means what it means natively.
#
#   2. Does the simd128 tier still decode to the same bytes as the scalar one?
#      With FIXTURES set, every fixture is decoded by both builds and the
#      results compared. That is the wasm half of what `verify.sh` does for the
#      x86 rungs, and it is the check the kernels have to keep passing.
#
#   FIXTURES=dir  also sweep every stream named in dir/golden.txt
#
# Exits nonzero on the first disagreement.
set -u
cd "$(dirname "$0")/.."

command -v node > /dev/null || { echo "wasm.sh: needs node"; exit 1; }
rustup target list --installed 2>/dev/null | grep -qx wasm32-unknown-unknown ||
  { echo "wasm.sh: needs the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown)"; exit 1; }

build() { # build <rustflags> <dest>
  RUSTFLAGS="$1" cargo build --release --target wasm32-unknown-unknown --example wasm_probe > /dev/null 2>&1 ||
    { echo "wasm.sh: build failed ($2)"; exit 1; }
  cp target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm "$2"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "== building =="
build "" "$TMP/scalar.wasm"
echo "  scalar   $(wc -c < "$TMP/scalar.wasm") bytes"
build "-C target-feature=+simd128" "$TMP/simd128.wasm"
echo "  simd128  $(wc -c < "$TMP/simd128.wasm") bytes"

# `<name> <frames> <hash>` per vendored stream, lifted from the test that owns
# the expectations.
EXPECT=$(
  awk '
    /include_bytes!\("data\// { split($0, a, /data\//); split(a[2], b, /"/); name = b[1] }
    /assert_eq!\(frames,/     { split($0, a, /frames, /); split(a[2], b, /,/); frames = b[1] }
    /assert_eq!\(hash,/       { split($0, a, /hash, /);  split(a[2], b, /,/); print name, frames, b[1] }
  ' tests/decode.rs
)
[ -n "$EXPECT" ] || { echo "wasm.sh: could not read the expectations out of tests/decode.rs"; exit 1; }

fail=0
for w in scalar simd128; do
  echo
  echo "== vendored streams, decoded inside wasm ($w) =="
  while read -r name frames hash; do
    got=$(node tools/wasm_run.mjs "$TMP/$w.wasm" "tests/data/$name" 2>&1) ||
      { printf "  %-18s %s\n" "$name" "$got"; fail=1; continue; }
    if [ "$got" = "$frames $hash" ]; then
      printf "  %-18s %s\n" "$name" "$got"
    else
      printf "  %-18s MISMATCH got[%s] want[%s]\n" "$name" "$got" "$frames $hash"
      fail=1
    fi
  done <<< "$EXPECT"
done

if [ -n "${FIXTURES:-}" ]; then
  echo
  echo "== fixtures, simd128 against scalar =="
  ok=0
  while read -r f _; do
    [ -f "$FIXTURES/$f" ] || continue
    a=$(node tools/wasm_run.mjs "$TMP/scalar.wasm" "$FIXTURES/$f" 2>&1)
    b=$(node tools/wasm_run.mjs "$TMP/simd128.wasm" "$FIXTURES/$f" 2>&1)
    if [ "$a" = "$b" ] && [ -n "$a" ]; then
      ok=$((ok + 1))
    else
      printf "  %-32s scalar[%s] simd128[%s]\n" "$f" "$a" "$b"
      fail=1
    fi
  done < "$FIXTURES/golden.txt"
  echo "  $ok fixtures decode identically at both"
fi

echo
if [ "$fail" = 0 ]; then echo "wasm: OK"; else echo "wasm: FAILED"; fi
exit $fail
