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
#   ENC_CLIPS=dir also encode every src_*_420.yuv there (the 8-bit gate
#                 corpus) in the encode round trip
#
#   3. Do the encode-side kernels (distortion, H.265 forward transforms and
#      quantiser) have the same property, and does an encode inside wasm
#      produce the bytes the scalar build produces? See the encode section.
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

# ----------------------------------------------------------------------
# The encode side
# ----------------------------------------------------------------------
#
# The encode-only kernel tables (distortion; H.265 forward transforms and
# quantiser) have a simd128 tier of their own, and a rung of "SIMD128"
# says nothing about whether *those* tables took it — they were scalar
# for a long time while the rung said that. So: which entries each build
# installed (all six groups on simd128, none on scalar), the randomised
# sweep against the scalar reference inside the module, and then an
# encode round trip on both builds — bitstream, decoded pictures and the
# encoder's own reconstruction hashed inside the module — which must
# agree byte for byte between the builds (the wasm form of
# tools/identity_encode.sh) and, per build, between decoded and
# reconstructed (the SELF property). Timings are best of three, clocked
# from outside because the module has no clock.
echo
echo "== which encode-side kernels each build installed =="
for w in scalar:0 simd128:63; do
  b=${w%%:*}; want=${w##*:}
  got=$(node tools/wasm_enc.mjs "$TMP/$b.wasm" --installed 2>&1)
  printf "  %-8s mask %s
" "$b" "$got"
  [ "$got" = "$want" ] || { echo "    expected $want"; fail=1; }
done

echo
echo "== encode-kernel self-test inside wasm (randomised, against scalar) =="
for w in scalar simd128; do
  got=$(node tools/wasm_enc.mjs "$TMP/$w.wasm" --selftest 2>&1)
  printf "  %-8s %s
" "$w" "$got"
  [ "$got" = "OK" ] || fail=1
done

echo
echo "== encode round trip inside wasm (scalar vs simd128; decoded vs recon) =="
# Cells: codec x (intra, IP, IPB) at QP 26 and one QP 40 row, on the
# synthesised clip; ENC_CLIPS=dir adds every src_*_420.yuv there (the
# 8-bit gate corpus) at 64x64 — the name carries the geometry.
cells="h264:26:0:0 h264:26:8:0 h264:26:8:2 h264:40:8:0 h265:26:0:0 h265:26:8:0 h265:26:8:2 h265:40:8:0"
clips="synth:64x64:"
if [ -n "${ENC_CLIPS:-}" ]; then
  for f in "$ENC_CLIPS"/src_*_420.yuv; do
    [ -f "$f" ] || continue
    geom=$(basename "$f" | sed -n 's/.*_\([0-9]*x[0-9]*\)_420\.yuv$/\1/p')
    [ -n "$geom" ] && clips="$clips $(basename "$f" .yuv):$geom:$f"
  done
fi
cells_ok=0
for clip in $clips; do
  name=${clip%%:*}; r=${clip#*:}; geom=${r%%:*}; file=${r#*:}
  for c in $cells; do
    IFS=: read -r codec qp gop bf <<< "$c"
    a=$(node tools/wasm_enc.mjs "$TMP/scalar.wasm" --encode "$geom" "$codec" "$qp" "$gop" "$bf" $file 2>&1) ||
      { printf "  %-28s %-4s qp%s gop%s b%s  scalar: %s
" "$name" "$codec" "$qp" "$gop" "$bf" "$a"; fail=1; continue; }
    b=$(node tools/wasm_enc.mjs "$TMP/simd128.wasm" --encode "$geom" "$codec" "$qp" "$gop" "$bf" $file 2>&1) ||
      { printf "  %-28s %-4s qp%s gop%s b%s  simd128: %s
" "$name" "$codec" "$qp" "$gop" "$bf" "$b"; fail=1; continue; }
    read -r af al ash adh arh ams <<< "$a"
    read -r bf_ bl bsh bdh brh bms <<< "$b"
    status=SAME
    [ "$af $al $ash $adh" = "$bf_ $bl $bsh $bdh" ] || { status=MOVED; fail=1; }
    [ "$adh" = "$arh" ] && [ "$bdh" = "$brh" ] || { status="$status SELF-FAIL"; fail=1; }
    [ "$status" = SAME ] && cells_ok=$((cells_ok + 1))
    printf "  %-28s %-4s qp%s gop%s b%s  %s frames %s bytes  scalar %sms  simd128 %sms  %s
"       "$name" "$codec" "$qp" "$gop" "$bf" "$af" "$al" "$ams" "$bms" "$status"
  done
done
echo "  $cells_ok cells identical across the builds and self-consistent"
[ "$cells_ok" -gt 0 ] || { echo "  NO CELLS RAN"; fail=1; }

echo
echo "== encode kernels, ns per call group (best of 3, node) =="
printf "  %-22s %10s %10s %8s
" "kernel group" "scalar" "simd128" "ratio"
for spec in "0:0:200000:sad+satd+ssd 4x4" "0:1:100000:sad+satd+ssd 8x8" "0:2:40000:sad+satd+ssd 16x16" "0:3:10000:sad+satd+ssd 32x32" "0:4:3000:sad+satd+ssd 64x64"             "1:0:200000:fdct+quant 4x4" "1:1:100000:fdct+quant 8x8" "1:2:20000:fdct+quant 16x16" "1:3:5000:fdct+quant 32x32"; do
  IFS=: read -r g sh n label <<< "$spec"
  a=$(node tools/wasm_enc.mjs "$TMP/scalar.wasm" --bench "$g" "$sh" "$n" 2>&1)
  b=$(node tools/wasm_enc.mjs "$TMP/simd128.wasm" --bench "$g" "$sh" "$n" 2>&1)
  ratio=$(awk -v a="$a" -v b="$b" 'BEGIN { if (b > 0) printf "%.2fx", a / b; else print "?" }')
  printf "  %-22s %10s %10s %8s
" "$label" "$a" "$b" "$ratio"
done

echo
if [ "$fail" = 0 ]; then echo "wasm: OK"; else echo "wasm: FAILED"; fi
exit $fail
