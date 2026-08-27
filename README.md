# rivet-h26x

[![crates.io](https://img.shields.io/crates/v/rivet-h26x.svg)](https://crates.io/crates/rivet-h26x)
[![docs.rs](https://docs.rs/rivet-h26x/badge.svg)](https://docs.rs/rivet-h26x)
[![CI](https://github.com/rivet-transcoder/rivet-h26x-codecs/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-h26x-codecs/actions/workflows/ci.yml)

Native **H.264/AVC** and **H.265/HEVC** decoders in Rust: no C, no system
libraries, no build script, nothing to install on a build host. Bit-exact
against the JVT and JCT-VC conformance suites, frame- and wavefront-threaded,
with a run-time ladder of SSE2-through-AVX2 and NEON kernels, and a
supplementary AVX-512 tier over AVX2 for the shapes where 512-bit lanes pay.

The SIMD is written in Rust intrinsics, not assembly, with one exception:
AArch64's `sdot` has no intrinsic on stable Rust yet
([rust-lang/rust#117224](https://github.com/rust-lang/rust/issues/117224)), so
that one instruction goes through a stable `asm!` wrapper carrying the
signature the intrinsic will have. Swapping it back is a one-line change.

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, where they are the software decode tier for the two codecs every
camera, phone and broadcast chain emits — under the GPU decoders (NVDEC / AMF /
QSV) so a machine without a usable GPU still decodes — and usable on their own
by anything that has Annex-B NAL units and wants planar pictures back.

Published as `rivet-h26x`; **imported as `h26x`** (`use h26x::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
h26x = { package = "rivet-h26x", version = "0.2" }
```

## What it decodes

| | supported | refused with `Error::Unsupported` |
|---|---|---|
| **H.264** | Baseline / Main / High / High 10 / High 4:2:2 / High 4:4:4 Predictive / CAVLC 4:4:4 Intra (and the Intra profiles): frames, field pictures (PAFF — field / frame reference lists and marking, field POCs, colocated field / frame mapping) and MBAFF (macroblock-adaptive frame / field: pair-wise neighbour derivation, field-scan and field-context entropy coding, mixed frame / field prediction and direct-mode mapping, mixed-edge deblocking), 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8–14-bit, separate colour planes, lossless (transform bypass), CAVLC + CABAC, I/P/B, spatial + temporal direct, explicit + implicit weighting, 8x8 transform, scaling matrices, PCM, MMCO, multi-slice, frame-num gaps, all three POC types, deblocking, VUI reorder hints, the old-x264 4:4:4 CABAC quirk | unequal luma / chroma bit depths, FMO / ASO, data partitioning, SP / SI |
| **H.265** | Main / Main 10 / Main 12 and the format range extensions — Main 4:2:2 / 4:4:4 (10, 12, 16 Intra), the High Throughput 4:4:4 (16 Intra) and Monochrome profiles (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–16-bit, unequal luma / chroma depths): CTB 16–64, AMP, transform skip (any size, rotation, single-context), scaling lists, sign hiding, PCM, `cu_transquant_bypass` (lossless), cu_qp_delta, chroma QP offset lists, cross-component prediction, implicit / explicit RDPCM, persistent Rice adaptation, high-precision weighted-prediction offsets, intra smoothing disabling, extended precision processing, CABAC bypass alignment, tiles, WPP (and both together), dependent slice segments, merge / AMVP / TMVP, explicit weighting, deblocking, SAO, long-term references, CRA / BLA / RASL handling, `pic_output_flag`, `no_output_of_prior_pics`, decoded-picture-hash SEI verification (`H26X_VERIFY_HASH=1`) | separate colour planes, SCC, multi-layer |

Both decoders are **bit-exact**. H.264 passes **199 of the 199** JVT
conformance bitstreams (AVCv1 + FRExt — every profile set, every field-picture
and MBAFF stream) it does not refuse, against the suite's reconstructed YUV
(libavcodec's per-frame MD5s where the zip ships none) — the other 5 are
refused up front (FMO 3, SP/SI 2) — and **35 of the 35** JVT
professional-profile bitstreams it accepts (High 10 / 4:2:2 / 4:4:4 Intra,
CAVLC 4:4:4 Intra, High 4:4:4 Predictive at up to 14-bit, ten of them coded as
separate colour planes, which libavcodec refuses; those ten are checked
against the JM reference decoder, the rest against libavcodec — the other 3
are FMO). Decoding is deterministic across thread counts (every suite stream
decoded on 1 and 12 threads gives the same bytes). It matches libavcodec on
the workspace fixtures too (CAVLC/CABAC, B-pyramids, weighting, 8x8, slices,
CQM, 10-bit, 4:2:2, 4:0:0, 4:4:4, lossless, x264 interlaced). H.265 passes **147 of the 147** JCT-VC HEVC_v1 conformance bitstreams
and **all 49** RExt bitstreams against the suite's own MD5s — among them the
17 libavcodec declines (16-bit, extended precision, CABAC bypass alignment,
unequal luma / chroma bit depths), which the suite's MD5 and the stream's own
decoded-picture-hash SEI are the only witnesses for — plus fifteen x265
feature fixtures; every HM stream's decoded-picture-hash SEI is checked as
well. Bit depths above 12, and extended precision at any depth, decode on a
scalar `i32` pipeline beside the `i16` SIMD one the 8–12-bit streams keep.

`Unsupported` is a *classification*, not a failure: rivet's decode tier list
falls through to the next backend (libavcodec when built with the `ffmpeg`
feature, openh264 for H.264).

## What it encodes (in progress)

Encoders for both codecs are being built to the same standard as the
decoders, and the part that is settled is how they are verified, because an
encoder cannot have a conformance suite: a standard constrains what a
*decoder* must do with a bitstream and leaves an encoder free to choose any
legal one, so there is no golden output to compare against. Three properties
replace it, two of them exact:

1. **SELF** — the encoder records the reconstruction it believes its
   bitstream carries, and this crate's own decoder must reproduce it byte for
   byte. A mismatch is encoder/decoder state desync: always a bug, never a
   quality question, and it needs no reference data.
2. **CROSS** — libavcodec must decode the bitstream to the same pictures our
   decoder does. SELF alone would pass if both of our sides shared a
   misreading; CROSS is what makes the output *legal* rather than merely
   self-compatible.
3. **QUALITY** — PSNR against the source, reported rather than gated, except
   in lossless mode where the reconstruction must equal the source exactly
   and the measurement becomes a check like the other two.

`tools/verify_encode.sh` gates 1 and 2 and reports 3, over seven generated
clips (4:0:0 through 4:4:4, plus a 50x34 one because cropping is a common
place to be wrong) and a configuration list that grows one axis at a time.

Current state: H.264 produces legal streams — verified against libavcodec —
for all-intra content through both entropy coders (I_PCM, exactly lossless)
and CAVLC P/B envelopes (all-skip); the intra transform path, both residual
writers, forward transforms and quantisation for both codecs, and the
distortion metrics all exist and are converging on real compression. H.265
has its parameter sets (accepted by this crate's own conformance-proven
parsers) and refuses at the coding tree, which H.265 requires for even a PCM
picture since every slice payload is CABAC. Anything not yet built refuses
with `Error::Unsupported` naming the missing piece — the gate distinguishes
"not built" from "wrong".

## Threading and SIMD

Both decoders are threaded two ways, on one FIFO worker pool sized to the
machine (`H26X_THREADS`, default every core, up to 32):

- **Frame threading.** Pictures decode concurrently; a picture that references
  another waits only for the rows it needs (motion compensation waits on the
  reference's *filtered* row progress, temporal motion prediction on its
  *decoded* row progress), and the deblocking / SAO of a picture trail its
  decode row by row so the next picture can start before this one ends. Up to
  `H26X_INFLIGHT` pictures are in flight (default `threads.clamp(2, 16)`).
- **Intra-picture parallelism (H.265).** Wavefront rows (WPP), tiles and slice
  segments each run as their own task; a row waits on the CTB above-right, and
  the next row is spawned as soon as this one is two CTBs in. Dependent slice
  segments resume their predecessor's CABAC state.

The pixel kernels — interpolation, weighting, inverse transforms, residual add,
SAO for H.265; the sixteen quarter-sample positions, chroma bilinear, averaging
and weighting for H.264 — sit behind a function table filled at run time from
what the CPU has. On x86-64 that is a ladder — **SSE2**, **SSSE3**,
**SSE4.1**, **AVX**, **AVX2** — climbed one rung at a time, each rung
replacing only the kernels it improves on the one below, so a CPU ends up with
the best available version of every kernel. SSE2 is baseline on x86-64, so the
bottom rung always applies: no x86-64 machine runs the scalar kernels, which
are the executable specification the others are tested against and not a
fallback. AArch64 has **NEON**, likewise baseline, with the ARMv8.2-A **dot
product** extension above it where present. **AVX-512** (F + BW + VL) sits
above AVX2 as a supplement rather than a rung: it replaces the handful of
H.265 kernels whose block shape genuinely fits 512-bit lanes and leaves the
rest of the AVX2 table alone.

Measured on one core against the scalar reference, **SSE2 alone is worth
2.5–3.2x and is nearly all of the total.** What the rungs above it add depends
on the codec, and the split is large: everything from SSSE3 to AVX-512
together is worth about **5% on H.264** but **13–17% on H.265**. Block width
is the reason. H.264's blocks are at most sixteen samples wide, so a 256-bit
vector spans a row at best and pays cross-lane permutes to undo per-lane
packing that 128-bit code never does; H.265 has 32- and 64-wide blocks, and
its x86 kernels already put two rows in a vector. So the individual rungs —
SSSE3 (`pmaddubsw` for the six-tap and bilinear filters), SSE4.1
(`pblendvb`, `pminsd` / `pmaxsd`), AVX (VEX's three-operand encoding; AVX adds
no 256-bit *integer* operation, that is AVX2) — are worth low single digits
each on a modern out-of-order core, and should be worth more on the processors
that actually need them, which have far less slack to hide the extra
instructions the rung below spends. On AArch64, `sdot` is worth having for the
one kernel whose shape it fits, HEVC's eight-tap horizontal luma filter, and
measurably not worth it for H.264's six-tap, whose taps LLVM already folds
into `umlal` pairs.

`H26X_NO_SIMD=1` forces the scalar kernels and
`H26X_MAX_SIMD=avx512 | avx2 | avx | sse41 | ssse3 | sse2 | neon | none` caps
the ladder one rung at a time, which is how one machine checks that every rung
decodes to the same bytes — `tools/verify.sh` does exactly that on every run
and fails if any two rungs disagree. Every SIMD kernel is also checked
bit-exact against the scalar reference by the crate's tests, on both
architectures in CI. `h26xdec --rung` (or [`dsp::Cpu::rung`]) prints which rung
was selected, which is worth knowing before quoting a number from this
machine or any other. `H26X_PROF=1` prints where the time went.

## Performance

Against libavcodec, on the same clips, with both decoders materialising every
frame — ffmpeg writing rawvideo to the null device, this decoder packing each
picture and dropping it. Neither writes to disk. **Cost** is CPU seconds
(user+kernel) for the whole process tree, best of five; **throughput** is
frames per wall second, which is the question a multi-threaded run is actually
asking. Regenerate with `tools/benchmark.py`.

Rows are the instruction-set rungs, because "how fast is it" has no answer
without saying which instructions it was allowed to use — and the rung is
chosen at run time, so the table doubles as a map of what different hardware
gets. `H26X_MAX_SIMD` caps the ladder, which is how one machine produces every
row; the rung it selects on its own is in bold, and is what a user on that
hardware actually runs.

One line under each table is generated and says "Widest rung against
libavcodec". For the two H.264 tables that widest rung is byte-identical to
the row beneath it, since no H.264 kernel is AVX-512 — so it names a
configuration that is not distinct for this codec, and those two rows should
be read as a control pair rather than as two rungs.

The single-threaded rows are pinned to one quiet core, both decoders alike, so
that column survives a machine that is doing other things. The all-threads
rows cannot be — pinning them would measure something other than what they
claim — so they are the rows to distrust on a busy box, and the header says
how busy it was.

**AMD Ryzen 9 9950X 16-Core Processor**, 32 hardware threads, selecting **AVX-512**. Machine 17% busy before the run. Single-threaded rows are pinned to core 29 (both decoders). Best of 5. Cost is CPU seconds, throughput is frames per wall second.

#### `cabac3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 2.266 | 1.49x | 319 | 3.766 | 2169 |
| AVX2 | 2.219 | 1.46x | 326 | 3.766 | 2221 |
| AVX (VEX-128) | 2.266 | 1.49x | 319 | 3.781 | 2206 |
| SSE4.1 | 2.266 | 1.49x | 319 | 3.875 | 2123 |
| SSSE3 | 2.281 | 1.51x | 317 | 3.703 | 2153 |
| SSE2 | 2.297 | 1.52x | 315 | 3.969 | 2163 |
| scalar | 5.469 | 3.61x | 132 | 7.734 | 1329 |
| libavcodec | 1.516 | 1.00x | 477 | 2.625 | 2119 |

Widest rung against libavcodec: **1.49x** its single-threaded CPU time, **1.43x** with every thread; SIMD is worth **2.4x** over the scalar reference.

#### `cavlc3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 1.672 | 1.55x | 432 | 3.031 | 3116 |
| AVX2 | 1.656 | 1.54x | 437 | 3.078 | 2997 |
| AVX (VEX-128) | 1.734 | 1.61x | 417 | 3.344 | 2890 |
| SSE4.1 | 1.734 | 1.61x | 417 | 3.500 | 2856 |
| SSSE3 | 1.766 | 1.64x | 409 | 3.594 | 2831 |
| SSE2 | 1.703 | 1.58x | 425 | 3.062 | 2985 |
| scalar | 4.906 | 4.55x | 147 | 7.406 | 1550 |
| libavcodec | 1.078 | 1.00x | 671 | 2.406 | 3063 |

Widest rung against libavcodec: **1.55x** its single-threaded CPU time, **1.26x** with every thread; SIMD is worth **2.9x** over the scalar reference.

#### `hevc6.265` — H.265, 1280x720, 1446 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 3.328 | 1.12x | 434 | 4.219 | 1288 |
| AVX2 | 3.406 | 1.15x | 425 | 4.391 | 1277 |
| AVX (VEX-128) | 3.656 | 1.23x | 395 | 4.266 | 1241 |
| SSE4.1 | 3.688 | 1.24x | 392 | 4.562 | 1255 |
| SSSE3 | 3.734 | 1.26x | 387 | 4.578 | 1228 |
| SSE2 | 3.844 | 1.29x | 376 | 4.500 | 1220 |
| scalar | 8.578 | 2.89x | 169 | 9.375 | 693 |
| libavcodec | 2.969 | 1.00x | 487 | 4.141 | 1387 |

Widest rung against libavcodec: **1.12x** its single-threaded CPU time, **1.02x** with every thread; SIMD is worth **2.6x** over the scalar reference.

#### `wpp10.265` — H.265, 1280x720, 1200 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 1.938 | 1.02x | 619 | 4.656 | 1963 |
| AVX2 | 1.984 | 1.04x | 605 | 4.062 | 1946 |
| AVX (VEX-128) | 2.172 | 1.14x | 553 | 4.609 | 1799 |
| SSE4.1 | 2.203 | 1.16x | 545 | 4.359 | 1913 |
| SSSE3 | 2.188 | 1.15x | 549 | 4.859 | 1749 |
| SSE2 | 2.281 | 1.20x | 526 | 4.297 | 1792 |
| scalar | 5.125 | 2.69x | 234 | 7.984 | 1067 |
| libavcodec | 1.906 | 1.00x | 630 | 2.656 | 1774 |

Widest rung against libavcodec: **1.02x** its single-threaded CPU time, **1.75x** with every thread; SIMD is worth **2.6x** over the scalar reference.

### Reading it

**Two things in the table say how much precision it can carry, and both are
worth reading before any of the numbers.**

The first is inside each H.264 table. No H.264 kernel is AVX-512 — the tier
carries H.265 kernels only — so the top two rows of those two tables run
*byte-identical code*, and every difference between them is measurement noise.
There is no single number for it, because it depends on the column: 2.1% and
1.0% in single-threaded CPU seconds, 0.0% and 1.6% in all-thread CPU seconds,
and **2.4% and 4.0% in all-thread frames per second**. The threaded columns
are the noisiest, and they are the ones the multi-threaded claims below are
made in, so read those against 4% and not against 2%.

The second is between runs. This table replaced one taken earlier on the same
machine, and no H.265 code changed in between — yet `wpp10.265` moved from
0.96x to 1.02x against libavcodec and `hevc6.265` from 1.11x to 1.12x. The
absolute costs of *both* decoders drift with whatever else the machine has
been doing, and the ratio inherits it. So **treat the comparison column as
good to about ±5% between sessions**, and do not read a change of a few per
cent in these figures as a change in the code. A few per cent in the code is
measurable, but only by holding everything else fixed — one binary, two code
paths, switched by an environment variable — which is what
`tools/verify.sh --baseline` and the per-change measurements in the commit log
do, and what a table like this cannot.

**With that said: single-threaded, H.265 is at parity with libavcodec** (1.02x
and 1.12x) and H.264 costs about half as much again — 1.46-1.49x on a CABAC
stream, 1.54-1.55x on a CAVLC one. Those are ranges rather than figures
because the two H.264 rows they come from are the same code: the bolded row is
whichever rung this machine selects, not whichever sample came out faster, and
on both clips it happens to be the slower one. Quoting it alone would inflate
the gap by the width of the control. The gap itself is not in the pixel
kernels; it is entropy decoding and per-macroblock bookkeeping, which is where
the remaining work is.

**With every thread, two of the four comparisons say anything.** Ahead on WPP
(1963 against 1774 fps, +10.7%) and behind on the H.265 clip without
wavefronts (1288 against 1387, -7.1%). The other two do not clear the floor:
CAVLC (3116 against 3063) is +1.7% and CABAC (2169 against 2119) is +2.4%,
against a byte-identical control spread of 4.0% and 2.4% in that same column.
Frame threading is the whole of it for H.264; H.265 adds wavefront rows, tiles
and slice segments inside a picture.

**Nearly all of the SIMD win is on the bottom rung, and on H.264 there is now
almost nothing above it.** Scalar to SSE2 is worth 2.4–2.9x. Everything above
SSE2 put together adds **1–2% on H.264** — at or under the floor the table
itself reports — against **15–18% on H.265**. Block width is why: H.264's
blocks are at most sixteen samples wide, so a 256-bit vector spans a row at
best and pays cross-lane permutes to undo per-lane packing that 128-bit code
never does, while H.265 has 32- and 64-wide blocks and its x86 kernels already
put two rows in a vector. The H.264 figure has *fallen* as this decoder got
faster: removing redundant work from the interpolation filters took out the
part that the wider rungs were helping with, leaving a profile even more
dominated by entropy decoding than before. On x86-64 nothing ever falls back
to scalar, since SSE2 is baseline, so the practical reading is that an old
x86-64 machine gives up almost nothing on H.264 and about a sixth on H.265.

**AVX-512 is a supplement, not a rung.** It installs over a finished AVX2
table and replaces only the H.265 kernels whose shape suits 512-bit lanes: the
vertical 14-bit stage of a diagonal interpolation, the wide luma filters at 32
samples and up, and the 32-point inverse transform, whose row *is* a 512-bit
vector. Everything it does not carry keeps the AVX2 version — H.265 chroma
interpolation was written for it, measured worse, and was left behind, as were
four H.264 kernels.

Its end-to-end contribution is the one figure in this section with no control
behind it. The byte-identical pair exists only in the H.264 tables, because
AVX-512 genuinely does carry H.265 kernels — so the H.265 tables have no two
rows running the same code, and the H.264 floor cannot be borrowed across to
them when that floor is itself anything from 0.0% to 4.0% depending on the
column. Per-kernel it is 1.6-2.3x on the shapes it carries; what that is worth
inside a decode was measured separately, against its own control, and is
recorded in the commit that added the tier rather than inferred from here.

Numbers move with the clip. These are 720p Big Buck Bunny encodes — H.264 High
profile from x264 at CRF 20 with B-pyramid and the 8x8 transform, H.265 Main
from x265 with and without WPP — repeated to a length the clock can measure
(process CPU time comes in ~15.6 ms steps, so a fifth-of-a-second clip cannot
be compared with itself, let alone with something else).

## Provenance and licensing

This is not a translation of libavcodec's C, which is LGPL and could not be
carried under this crate's license. The decoders were written from the ITU-T
Recommendations (H.264 08/2024, H.265 V11 01/2026) — the decoding process is
fully specified there — using libavcodec's *architecture* as the model:
parameter-set tables, a slice-driven decode loop over a decoded-picture buffer,
entropy decoding into per-block syntax, then prediction, inverse transform,
reconstruction and in-loop filters, with the pixel kernels behind a
runtime-dispatched DSP layer. Numeric tables (VLC codes, CABAC initialisation
values, scan orders, filter taps) are the standard's tables.

**Patents.** The AVC and HEVC patent pools (Via LA, Access Advance, Velos)
license *products and services* — encoders, decoders, and content shipped to
end users — not the act of writing or publishing an implementation; several
pools additionally waive royalties for application software distributed at no
charge. Nothing here is a licence to any patent, and anyone shipping a product
built on this crate is responsible for their own patent position, exactly as
with x264, x265, OpenH264 or FFmpeg. Source availability was never the
licensed act; distribution of a decoding product is.

## Using it

Feed whole NAL units (Annex-B framed, parameter sets included) in decode order;
pictures come back in output order as [`Picture`](src/picture.rs) — cropped
planar samples (8-bit as bytes, 9–16-bit as little-endian 16-bit words; with
unequal luma / chroma depths every plane takes the wider word, as HM writes
them) with the chroma format, both bit depths and picture order count of the
stream:

```rust
let mut dec = h26x::hevc::HevcDecoder::new();   // or h26x::h264::H264Decoder
dec.push_annexb(&bytes)?;                        // whole NAL units, Annex-B framed
while let Some(pic) = dec.next_picture() { /* h26x::Picture, output order */ }
dec.flush()?;                                    // drain the reorder buffer
```

`examples/h26xdec.rs` decodes a raw `.264` / `.265` file and prints one line per
output frame with the MD5 of the packed planar picture — the same layout
`ffmpeg -f framemd5` hashes — or writes raw YUV. Debug environment variables:
`H26X_NO_DEBLOCK`, `H26X_NO_SAO`, `H26X_TRACE=<mbaddr>|all` (a macroblock's
parsed layer and motion), `H26X_TRACE_IPM=1` (every syntax element with its
bit / CABAC position — lines up with the JM reference decoder's trace) and
`H26X_TRACE_DPB=1` (picture starts, marking, lists) (H.264), `H26X_TRACE_PS`
(parsed SPS/PPS), `H26X_TRACE_CU`, `H26X_TRACE_PU=x,y`,
`H26X_TRACE_TB=c,x,y`, `H26X_VERIFY_HASH=1` (check every decoded-picture-hash
SEI, count a warning per mismatch) (H.265).

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
