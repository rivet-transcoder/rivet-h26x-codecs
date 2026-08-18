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
| `verify.sh [decoder]` | Everything that has to hold before a change lands: every fixture against its recorded MD5, then all four conformance suites. Prints one tally per suite and `ALL GREEN` or not. |
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

A caution learned the hard way: a 1–2% difference measured on a loaded machine
is noise. If something else is building, either stop it or do not believe the
number.

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
suspected. `H26X_THREADS=1` decodes on the calling thread. The rest
(`H26X_NO_DEBLOCK`, `H26X_TRACE=<mbaddr>|all`, `H26X_TRACE_IPM`,
`H26X_TRACE_DPB`, `H26X_TRACE_PS`, `H26X_TRACE_CU`, `H26X_TRACE_PU`,
`H26X_TRACE_TB`, `H26X_VERIFY_HASH`) are listed in the crate README.
