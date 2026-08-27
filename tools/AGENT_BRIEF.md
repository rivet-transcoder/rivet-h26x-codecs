# Working brief for h26x / rivet agent tracks (lead: 2026-08-27)

Read this whole file before touching code. Every rule here was paid for.

## Repos and layout
- h26x: `C:\Users\elyci\PhpstormProjects\rivet-h26x-codecs` (package `rivet-h26x`, lib `h26x`),
  branch `develop`. Work in YOUR OWN worktree + branch (given in your task); never commit to
  develop; never push. The lead merges and re-verifies.
- rivet: `C:\Users\elyci\PhpstormProjects\rivet` (develop), submodule `crates/h26x` = h26x develop.
  Rivet's software H.264/H.265 encode tier: `crates/codec/src/encode/h26x_sw.rs`.
- Scratch/corpus dir for the gates: `/c/Users/elyci/PhpstormProjects/rivet/target/h26x`
  (`H26X_WORK`). Clips `src_*.yuv` there are NOT versioned; `tools/make_encode_sources.sh` makes them.
- ffmpeg: `C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe` (the REAL binary; the scoop
  shim in `scoop/shims` lies to process accounting). ffprobe beside it.
- Shell is Git Bash on Windows. Python 3 is `python`. No `time`, no `flock`.

## The standard — every encoder change must hold FOUR exact properties + BOX
1. SELF   our decoder reproduces the encoder's own reconstruction byte for byte.
2. CROSS  libavcodec agrees with our decoder (catches what SELF cannot: a shared misreading).
3. QUALITY PSNR, reported never gated (except lossless: exact).
4. RATE/HRD did the encoder hit the objective it was handed (`RATE-FAIL`, `HRD-FAIL`).
6. BOX    exactly one VPS/SPS/PPS per stream (`PS-FAIL`, tools/param_sets.py) — Annex-B cannot
   see this; MP4 avc1/hvc1 breaks without it.
Plus the MODEL CHECK: wherever the encoder chooses freely, assert the decision's predicted
outcome against what actually happened.
Diagnostic: SELF differs + CROSS identical ⟹ bitstream fine, encoder-held recon wrong.

## Gate commands (ABSOLUTE paths always — a relative path silently yields 0 cells)
```
cd <your worktree> && cargo test --quiet && cargo test --release --quiet && cargo doc --no-deps --quiet
rm -f <wt>/target/release/examples/h26xenc.exe   # cargo does not always rebuild; delete first
cargo build --release --examples --quiet
H26X_WORK=/c/Users/elyci/PhpstormProjects/rivet/target/h26x \
FFMPEG="C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe" \
bash tools/verify_encode.sh <wt>/target/release/examples/h26xenc.exe <wt>/target/release/examples/h26xdec.exe
# decoder unchanged? still run the decode baseline if you touched dsp/ or anything under src/h264 src/hevc:
cd /c/Users/elyci/PhpstormProjects/rivet/target/h26x && bash verify.sh --baseline baseline.txt "<wt>/target/release/examples/h26xdec.exe"
```
GATE LOCK: at most TWO verify sweeps may run on the machine at once (spurious failures
otherwise). Before a sweep: `until mkdir /c/Users/elyci/PhpstormProjects/rivet/target/h26x/.gate_lock_$$ 2>/dev/null && [ $(ls -d /c/Users/elyci/PhpstormProjects/rivet/target/h26x/.gate_lock_* | wc -l) -le 2 ]; do rmdir /c/Users/elyci/PhpstormProjects/rivet/target/h26x/.gate_lock_$$ 2>/dev/null; sleep 20; done`
and after: `rmdir /c/Users/elyci/PhpstormProjects/rivet/target/h26x/.gate_lock_$$`. Always release it (trap EXIT).

## House rules
- Mirror the reader — better, CALL it. Where the decoder's fn is pub, call it. Writers beside
  their readers, exact inverses, round-tripped through the production parser.
- Refuse by name: `Error::unsupported("... (encoder in progress)")`. Never narrow silently.
- A gate row is meaningful iff it FAILS when the feature is stubbed out. RUN the mutation.
  Every failure prefix `one` prints must be in verify_encode.sh's tally grep.
- Mutate one side at a time — a symmetric mutation of an inverse pair is invisible.
- Commit before mutating, and before going idle. Small commits, one concern each, message says
  what was RUN (quote output). Never record reasoned-about as executed.
- Measure before building when payoff is uncertain. For "does less work": COUNT, don't time.
  For timing: one binary, two paths behind an env var, same-binary control first, median of
  paired ratios (`tools/ab.py`). A control you print but do not apply is worse than none.
- Zero warnings (crate denies missing_docs); MSRV 1.89; `rustfmt --edition 2024 <file>` per
  file only (never `cargo fmt`, the crate is not rustfmt-clean).
- Scratch files: `/c/Users/elyci/AppData/Local/Temp/claude/.../scratchpad` and `/tmp` are SHARED
  between agents — prefix every filename with your track name.
- Never `;`-chain a conflict-resolve step with `git add`; use `&&`.
- Use the Write tool for any file containing backslashes/quotes (heredocs mangle them).
- Scan your tree for `DB_CYCLES|_rdtsc|_meas|STAT_|black_box|pub static mut|eprintln!` before
  declaring done — instrumentation gets committed invisibly.
- Byte-identity claims: an identity result is evidence only if a row exercises the path.
- The vacuity family (7 ways a green run lies): a test that cannot fail; a combination no test
  reaches; a reporter that cannot report; phantom cells; a counter reading zero; a combination
  that did not exist to be tested (fix = CONTENT, add a clip); two wrongs cancelling (a green
  gate after a fix is weaker evidence than before one).

## Report format (your final message = the record; the lead reads nothing else)
1. Branch + commit hashes. 2. What was RUN, with quoted result lines (test counts, gate
"encode: N passed, M failed", baseline line). 3. Numbers: size/PSNR/speed deltas with the
control. 4. What you did NOT do and why. 5. Traps found. Keep it under ~80 lines.
