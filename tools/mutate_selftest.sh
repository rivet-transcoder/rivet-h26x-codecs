#!/bin/bash
# Pins tools/mutate.sh's verdict (mutate_failed) on the lines the tools here
# actually print. Run: bash tools/mutate_selftest.sh — exits non-zero on any
# wrong verdict. The success lines matter most: each contains the word
# "failed", and a case-blind `fail` grep once read every one of them as a
# caught mutation.
. "$(dirname "$0")/mutate.sh"

bad=0
expect() { # expect caught|missed RC OUTPUT
  local want="$1" rc="$2" out="$3" got=missed
  mutate_failed "$rc" "$out" && got=caught
  if [ "$got" = "$want" ]; then echo "ok      $want  rc=$rc  $out"; else echo "WRONG   want $want got $got  rc=$rc  $out"; bad=$((bad + 1)); fi
}

# Success output: must read as MISSED.
expect missed 0 "test result: ok. 330 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.65s"
expect missed 0 "encode: 959 passed, 0 failed"
expect missed 0 "identity: 959 identical, 0 moved"
expect missed 0 "quality: 959 cells against quality_floor.txt (tolerance 0.30 dB): 0 below the floor, 0 above it"
expect missed 0 "ALL GREEN"
expect missed 0 "   clippy error lines: 0 warning lines: 348"
expect missed 0 "tests pass"
expect missed 0 "identity: SAME=959 MOVED=0 ENCODE-FAIL(A)=0 ENCODE-FAIL(B)=0; moved row names outside rule: 0"
expect caught 0 "identity: SAME=957 MOVED=0 ENCODE-FAIL(A)=2 ENCODE-FAIL(B)=0"
# Failure output: must read as CAUGHT.
expect caught 0 "test result: FAILED. 329 passed; 1 failed; 4 ignored"
expect caught 0 "test encode::rc::tests::a_rule ... FAILED"
expect caught 0 "encode: 957 passed, 2 failed"
expect caught 0 "SELF-FAIL    src_cut_64x64_420/hevc-cqp-ip"
expect caught 0 "QUALITY-FAIL src_fade_64x64_420/hevc-abr-la-ipb-64k: Cb -0.40"
expect caught 0 "SOMETHING FAILED"
expect caught 0 "error[E0308]: mismatched types"
expect caught 0 "error: could not compile \`rivet-h26x\`"
expect caught 0 "thread 'main' panicked at src/encode/rc.rs:10:5"
expect caught 0 "tests FAILED"
expect caught 101 "test result: ok. 330 passed; 0 failed"
expect caught 1 ""

echo "=== mutate_selftest: $bad wrong"
[ "$bad" -eq 0 ]
