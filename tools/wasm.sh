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
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

rustup target list --installed 2>/dev/null | grep -qx wasm32-unknown-unknown ||
  { echo "wasm.sh: needs the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown)"; exit 1; }

build() { # build <rustflags> <dest>
  # An explicit --target-dir, because this crate is a workspace member when it
  # is vendored into rivet and the artifact then lands at the *workspace*
  # root, not under this directory. Asking for a known location beats guessing
  # a relative one that is right in only one of the two checkouts.
  #
  # And the build output is kept: swallowing it turned a compile error into
  # "no such file", which is a worse message about a different problem.
  local log="$TMP/build.log"
  if ! RUSTFLAGS="$1" cargo build --release --target wasm32-unknown-unknown \
       --example wasm_probe --target-dir "$TMP/target" > "$log" 2>&1; then
    echo "wasm.sh: build failed ($2)"
    tail -20 "$log" >&2
    exit 1
  fi
  cp "$TMP/target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm" "$2"
}

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

# A tier that installed nothing would pass every comparison below without
# running one vector instruction, so check which kernels each build actually
# selected before believing that they agree.
echo
echo "== which kernels each build selected =="
for w in scalar:scalar simd128:SIMD128; do
  b=${w%%:*}; want=${w##*:}
  got=$(node tools/wasm_run.mjs "$TMP/$b.wasm" --rung 2>&1)
  printf "  %-8s %s
" "$b" "$got"
  [ "$got" = "$want" ] || { echo "    expected $want"; fail=1; }
done

# The randomised kernel sweep the x86 tiers get from `cargo test`, run inside
# the module instead, because no test harness runs there (see the probe's
# `h26x_selftest`). On the scalar build it compares the reference with itself
# and is vacuous; the rung check above is what makes the simd128 pass mean
# something.
echo
echo "== kernel self-test inside wasm (randomised, against scalar) =="
for w in scalar simd128; do
  got=$(node tools/wasm_run.mjs "$TMP/$w.wasm" --selftest 2>&1)
  printf "  %-8s %s
" "$w" "$got"
  [ "$got" = "OK" ] || fail=1
done

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
