# Verification and measurement tools

How this decoder is held to "bit-exact": every change is checked against the
ITU-T / JVT and JCT-VC conformance suites and a set of workspace fixtures
before it lands. These are the scripts that do it. They are written for a
scratch directory holding the fixtures and the downloaded suites — by
convention `target/h26x/` in the rivet workspace — and take the decoder to
test as an argument, so a build under test is never confused with the
reference one.

Build the decoder first:

```sh
cargo build --release --examples          # target/release/examples/h26xdec
```

`h26xdec <input.264|.265> [out.yuv]` decodes a raw Annex-B file and prints one
CSV line per output frame — index, POC, decode index, dimensions, and the MD5
of the packed planar picture, the same layout `ffmpeg -f framemd5` hashes — so
a decode can be compared frame by frame against libavcodec or a reference
decoder without writing the YUV out.

## Correctness

| | |
|---|---|
| `verify.sh [--baseline FILE] [decoder]` | Everything that has to hold before a change lands: every fixture against its recorded MD5, then all four conformance suites. Prints one tally per suite and `ALL GREEN` or not. Safe to run concurrently with another copy of itself. |
| `verify.sh --baseline FILE` | Also records `<suite> <stream> <md5>` for every stream that decodes, and diffs against `FILE` when it already exists. For a refactor that must not change output this is the stricter question — not "does it still pass" but "does every one of 412 streams decode to the same bytes", including the ones no reference data covers. |
| `check.sh <file...>` | One or more fixtures against libavcodec's per-frame MD5s (`<name>.framemd5`, generated on first use), reporting the first mismatching frame. |
| `cmp2.py ref mine W H fmt bps [frames]` | Where two YUVs differ, per plane and per frame — for when `check.sh` says a frame is wrong and you need to know which samples. |
| `conformance/run_all.sh` | All four suites at once, each running its streams in parallel (`JOBS`, default 6). About ninety seconds on a 16-core machine. |
| `conformance/<suite>/run_conf.sh [filter]` | One suite; the filter selects streams by name substring. Classifies each stream `PASS` / `FAIL` / `UNSUPPORTED` — a decoder that *refuses* a stream up front is not a failure, it is a stream the caller should hand to another decoder. |
| `conformance/<suite>/fetch.sh` | Downloads that suite from the ITU-T / JCT-VC servers. The suites are not redistributable, so they are fetched, not vendored. |
| `conformance/<suite>/mt_stress.sh` | Decodes every stream on 1 and on 12 threads and compares — threading must not change a single sample. |

The suites, and what "accepted" means for each:

- `h264` — JVT AVCv1 + FRExt. Reference is the suite's own decoded YUV where the
  zip ships one, else libavcodec's per-frame MD5s.
- `h264_pp` — JVT professional profiles (High 10 / 4:2:2 / 4:4:4 Intra, CAVLC
  4:4:4 Intra, High 4:4:4 Predictive). No reference YUVs ship with it;
  libavcodec is the reference, except for the separate-colour-plane streams it
  refuses, which are checked against the JM reference decoder (`jm/*.md5`).
- `hevc` — JCT-VC HEVC_v1. The suite's whole-YUV MD5, else libavcodec.
- `hevc_rext` — JCT-VC range extensions, same rule.

## Measurement

| | |
|---|---|
| `ab.py <exeA> <exeB> <stream> [runs]` | Interleaved A/B: alternates the two builds, reports min and median **CPU** seconds and the ratio. Pins to one core and forces a single decoding thread, so the number is about the code and not the scheduler. Interleaving is what makes it survive a machine that is doing something else. |
| `bench.py <stream[,stream]> [runs]` | This decoder against ffmpeg, single-threaded and multi-threaded, wall and CPU time. `MODE=st` for single-thread only. |
| `benchmark.py [--runs N] [--streams a,b]` | The published comparison: Markdown tables, one per stream, a row per instruction-set rung and one for libavcodec, naming the processor. Cost in CPU seconds, throughput in frames per wall second. This is what the README's benchmark section is generated from — regenerate it rather than editing numbers by hand. |

A caution learned the hard way: a 1–2% difference measured on a loaded machine
is noise, and it is worse than that — measurements taken on a *contended core*
were not merely noisy but systematically misleading, giving opposite signs on
two streams for the same change. Two habits that fix it:

**The cheap rdtsc instrument is sound only when the variants differ in how they
compute, not in how they store.** Wrapping a function in `_rdtsc()` and
accumulating into an `AtomicU64` with `fetch_add` is the best tool here for a
change worth a few per cent of one function — but `fetch_add` compiles to
`lock xadd`, which is a full barrier, so it drains the store buffer at the
closing timestamp. Two variants that issue different amounts of store traffic
are charged differently for it, and one measurement of a 512-bit kernel against
its 128-bit equivalent came out **5x** wrong that way. Two variants that write
identical bytes in identical order drain the same thing and it cancels in the
ratio: re-measuring a CABAC change under `lock xadd`, a relaxed load/add/store,
and `lfence`-serialised rdtsc gave 0.9001, 0.9076 and 0.9093 — the artefact
moved it by under a point. So: if your change alters what gets written or in
what order, serialise the instrument or use a relaxed accumulator, and say
which one you used.

**The ladder is its own control.** `benchmark.py` checks that no rung comes
out ahead of the rung above it, and says so above any table where one does
rather than printing it as a result. Best-of-N defends against a brief
interruption; it cannot defend against something running for the whole
benchmark, which shifts every row together and leaves a table that looks
entirely plausible. That happened here — a run put SSE4.1 20% ahead of AVX2 on
a CAVLC clip — and the check is what caught it. It also prints how busy the
machine was before the run started, because that is the fact most likely to
explain a number nobody can reproduce, and it is the all-threads column it
distorts: a single-threaded run on a 32-thread box can find an idle core, a
run that wants every thread is competing for them.

**Measure the process tree, not the process.** `benchmark.py` puts each run in
a Windows job object and reads the job's accounting, because asking a process
how much CPU it used misses anything it spawned. `ffmpeg` on this machine is a
136 KB scoop shim that runs the real binary as a child: timed as a process it
reported **0.03 CPU seconds for 1.29 seconds of wall-clock work**, a 46x
understatement, and since it was the thing being compared against, the error
pointed the only direction that flatters us. It also refuses to report a
single-threaded run whose CPU time is under half its wall time, which is what
that failure looks like from the inside — a measurement that cannot be right
should stop the tool, not quietly become a row in a table.

**Know your floor before you trust a number.** Run `ab.py` with the *same
binary as both arguments*. Whatever spread that reports is the smallest
difference the machine can currently resolve; a change smaller than it has not
been measured, it has been guessed at. `ab.py` leads with the **median of the
paired ratios** — A and B run back to back within a round, so pairing them
cancels drift that neither a ratio of medians nor a ratio of minimums does, and
the median discards the round where something else woke up. It is deliberately
not the minimum: with interleaving on a live machine one lucky run poisons it,
and a same-binary control has read 1.120 and 0.862 on consecutive rounds while
its medians read 1.018 and 1.000. It prints the span of those ratios and tells
you when the span is wider than any change worth believing.

`ab.py` picks the core itself, sampling per-processor load at start-up and
taking the quieter half of the quietest SMT *pair* — a sibling shares the
execution units, so pinning to an idle logical processor whose sibling is busy
measures the sibling. It prints what it chose and warns when nothing is idle.
It does this because the previous fixed default became self-defeating: the
moment a good core is written down, everyone pins to it, and a same-binary
control on the documented "good" core read a **15%** spread — worse than the
contention the default was introduced to avoid. `AFFINITY` still overrides.

**Give the clock something to measure.** Process CPU time comes in ~15.6 ms
steps on Windows, so a clip that decodes in a fifth of a second is quantised to
within a few per cent of itself and every comparison reads exactly 1.000 —
which is indistinguishable from "no change" and is how a real 2% improvement
went unnoticed here. Concatenate a short clip until the run takes a second or
two (`cat x.265 x.265 ... > long.265` works: each copy carries its own
parameter sets and IDR).

**Measure the function, not the program,** when the change is a few per cent of
one hot function. Wrapping that function in `rdtsc` and reporting cycles per
macroblock row separated variants that whole-program wall clock could not see
at all, and gave a 2–3% noise floor while the end-to-end number was useless.
That is how the deblocking rewrite was measured at 0.66x when end to end it
only showed as 3-4%.

## Traps

`cargo fmt -- src/some/file.rs` does **not** scope to that path: it reformats
the whole crate. The crate is not rustfmt-clean as a whole — parts of it are
hand-laid-out — so that is a several-thousand-line diff sitting on top of
whatever you were doing. To format one file, run `rustfmt --edition 2024
<file>`, and check `git diff --stat` before committing either way.

The conformance runners work from a frozen copy of the decoder, so rebuilding
mid-run cannot disturb a suite — but it also means **a suite run tests the
copy, not your latest build**. `verify.sh` makes its own copy under a private
name and gives each suite a private scratch directory, so two people can run it
at once; invoke a runner directly and you get the shared defaults, where a
concurrent run can swap the binary underneath you. The failure mode is the
nasty one: a **green** run that tested somebody else's build.

## Profiling

`prof.sh <name> <cmd...>` records with [samply](https://github.com/mstange/samply),
symbolicates, and aggregates in one step; `symprof.py` does the aggregation
(self time, inclusive time, and with `LINES=<substring>` a line-level breakdown
that follows inlining); `symaddr.py` gives the instruction-address histogram
inside a function, for when the line attribution is not enough and you want to
disassemble.

These need a build with debug info, which the release profile strips:

```sh
CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=none \
  cargo build --release --examples
```

## Debug switches

`H26X_NO_SIMD=1` forces the scalar kernels — the executable specification the
SIMD paths are tested against, and the first thing to try when a SIMD kernel is
suspected. `H26X_MAX_SIMD=avx|sse41|neon|none` caps the ladder one rung at a
time instead, which is how one machine checks that every rung decodes to the
same bytes. `H26X_THREADS=1` decodes on the calling thread. The rest
(`H26X_NO_DEBLOCK`, `H26X_TRACE=<mbaddr>|all`, `H26X_TRACE_IPM`,
`H26X_TRACE_DPB`, `H26X_TRACE_PS`, `H26X_TRACE_CU`, `H26X_TRACE_PU`,
`H26X_TRACE_TB`, `H26X_VERIFY_HASH`) are listed in the crate README.
