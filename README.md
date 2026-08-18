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
| **H.265** | Main / Main 10 / Main 12 and the format range extensions (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–12-bit): CTB 16–64, AMP, transform skip (any size, rotation, single-context), scaling lists, sign hiding, PCM, `cu_transquant_bypass` (lossless), cu_qp_delta, chroma QP offset lists, cross-component prediction, implicit / explicit RDPCM, persistent Rice adaptation, high-precision weighted-prediction offsets, intra smoothing disabling, tiles, WPP (and both together), dependent slice segments, merge / AMVP / TMVP, explicit weighting, deblocking, SAO, long-term references, CRA / BLA / RASL handling, `pic_output_flag`, `no_output_of_prior_pics`, decoded-picture-hash SEI verification (`H26X_VERIFY_HASH=1`) | unequal luma/chroma bit depth, > 12-bit, extended precision processing, CABAC bypass alignment, separate colour planes, SCC, multi-layer |

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
CQM, 10-bit, 4:2:2, 4:0:0, 4:4:4, lossless, x264 interlaced). H.265 passes **146 of the 147** JCT-VC HEVC_v1 conformance bitstreams
against the suite's own MD5s (the one exception is the unequal-bit-depth
stream, which is refused) and **32 of the 32** RExt bitstreams it accepts (the
other 17 are refused up front: 16-bit, extended precision, CABAC bypass
alignment, unequal bit depths — the same set libavcodec declines), plus
fifteen x265 feature fixtures; every HM stream's decoded-picture-hash SEI is
checked as well.

`Unsupported` is a *classification*, not a failure: rivet's decode tier list
falls through to the next backend (libavcodec when built with the `ffmpeg`
feature, openh264 for H.264).

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

The single-threaded rows are pinned to one quiet core, both decoders alike, so
that column survives a machine that is doing other things. The all-threads
rows cannot be — pinning them would measure something other than what they
claim — so they are the rows to distrust on a busy box, and the header says
how busy it was.

**AMD Ryzen 9 9950X 16-Core Processor**, 32 hardware threads, selecting **AVX-512**. Machine 19% busy before the run. Single-threaded rows are pinned to core 28 (both decoders). Best of 5. Cost is CPU seconds, throughput is frames per wall second.

#### `cabac3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 2.250 | 1.52x | 321 | 4.141 | 1799 |
| AVX2 | 2.359 | 1.59x | 306 | 4.344 | 1797 |
| AVX (VEX-128) | 2.391 | 1.61x | 302 | 4.359 | 1779 |
| SSE4.1 | 2.391 | 1.61x | 302 | 4.281 | 1782 |
| SSSE3 | 2.422 | 1.63x | 299 | 4.453 | 1796 |
| SSE2 | 2.453 | 1.65x | 295 | 4.156 | 1771 |
| scalar | 5.703 | 3.84x | 127 | 9.312 | 1114 |
| libavcodec | 1.484 | 1.00x | 487 | 2.859 | 1844 |

Widest rung against libavcodec: **1.52x** its single-threaded CPU time, **1.45x** with every thread; SIMD is worth **2.5x** over the scalar reference.

#### `cavlc3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 1.766 | 1.61x | 409 | 3.797 | 2398 |
| AVX2 | 1.781 | 1.63x | 406 | 4.000 | 2364 |
| AVX (VEX-128) | 1.812 | 1.66x | 399 | 3.719 | 2356 |
| SSE4.1 | 1.812 | 1.66x | 399 | 3.797 | 2305 |
| SSSE3 | 1.859 | 1.70x | 389 | 3.922 | 2316 |
| SSE2 | 1.891 | 1.73x | 382 | 3.859 | 2300 |
| scalar | 5.203 | 4.76x | 139 | 8.562 | 1289 |
| libavcodec | 1.094 | 1.00x | 661 | 2.938 | 2190 |

Widest rung against libavcodec: **1.61x** its single-threaded CPU time, **1.29x** with every thread; SIMD is worth **2.9x** over the scalar reference.

#### `hevc6.265` — H.265, 1280x720, 1446 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 3.547 | 1.11x | 408 | 4.828 | 1097 |
| AVX2 | 3.594 | 1.12x | 402 | 5.141 | 1086 |
| AVX (VEX-128) | 3.797 | 1.19x | 381 | 5.422 | 1027 |
| SSE4.1 | 3.906 | 1.22x | 370 | 5.656 | 1042 |
| SSSE3 | 3.906 | 1.22x | 370 | 5.797 | 940 |
| SSE2 | 4.266 | 1.33x | 339 | 5.719 | 960 |
| scalar | 8.953 | 2.80x | 162 | 10.516 | 576 |
| libavcodec | 3.203 | 1.00x | 451 | 4.719 | 1212 |

Widest rung against libavcodec: **1.11x** its single-threaded CPU time, **1.02x** with every thread; SIMD is worth **2.5x** over the scalar reference.

#### `wpp10.265` — H.265, 1280x720, 1200 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 2.109 | 0.96x | 569 | 5.547 | 1402 |
| AVX2 | 2.250 | 1.03x | 533 | 5.797 | 1371 |
| AVX (VEX-128) | 2.766 | 1.26x | 434 | 6.375 | 914 |
| SSE4.1 | 3.062 | 1.40x | 392 | 7.547 | 855 |
| SSSE3 | 3.078 | 1.41x | 390 | 7.188 | 824 |
| SSE2 | 3.375 | 1.54x | 356 | 7.281 | 892 |
| scalar | 7.172 | 3.28x | 167 | 11.969 | 608 |
| libavcodec | 2.188 | 1.00x | 549 | 3.703 | 1375 |

Widest rung against libavcodec: **0.96x** its single-threaded CPU time, **1.50x** with every thread; SIMD is worth **3.4x** over the scalar reference.

### Reading it

**How this machine's own noise floor is visible in the table.** No H.264 kernel
is AVX-512 — the tier carries H.265 kernels only — so on the two H.264 clips
the top two rows are running *byte-identical code*. They differ by 4.6% and
0.8%. That is the resolution limit of this measurement, sitting in the table
where it can be checked rather than asserted, and it is the number to hold any
other difference here up against.

**Single-threaded, against libavcodec, it depends on how much of the work is
entropy decoding.** H.265 is at parity — 1.11x on the plain clip, and **0.96x
on the WPP one, which is to say faster**. H.264 is 1.52x on a CABAC stream and
1.61x on a CAVLC one, and that is the honest number for it. The gap is not in
the pixel kernels; it is entropy decoding and per-macroblock bookkeeping, which
is where the remaining work is.

**With every thread it trades wins with libavcodec**: ahead on CAVLC (2398 vs
2190 fps) and on WPP (1402 vs 1375), level on CABAC, behind on the H.265 clip
without wavefronts (1097 vs 1212). Frame threading is the whole of it for
H.264; H.265 adds wavefront rows, tiles and slice segments inside a picture,
which is why the WPP clip is where this decoder does best.

**Nearly all of the SIMD win is on the bottom rung.** Scalar to SSE2 is worth
2.5–3.4x. Everything above SSE2 put together adds **7–9% on H.264** — barely
above the noise floor the table itself reports — but **20% on H.265, and 60%
on the WPP clip**. Block width is the reason: H.264's blocks are at most
sixteen samples wide, so a 256-bit vector spans a row at best and pays
cross-lane permutes to undo per-lane packing that 128-bit code never does,
while H.265 has 32- and 64-wide blocks and its x86 kernels already put two
rows in a vector. This is why the low rungs are worth having — a CPU with only
SSE2 gives up single digits on H.264, not the 2.5x it would give up falling
back to scalar, and on x86-64 nothing ever falls back to scalar, because SSE2
is baseline.

**AVX-512 is a supplement, not a rung.** It installs over a finished AVX2 table
and replaces only the H.265 kernels whose shape suits 512-bit lanes: the
vertical 14-bit stage of a diagonal interpolation, the wide luma filters at 32
samples and up, and the 32-point inverse transform, whose row *is* a 512-bit
vector. Everything it does not carry keeps the AVX2 version — H.265 chroma
interpolation was written for it, measured worse, and was left behind. End to
end it is worth 1.3% on the plain H.265 clip and 6.3% on the WPP one; the
first of those is under this machine's noise floor and the second is only just
over it, so treat the tier as worth having and not as a step change.

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
planar samples (8-bit as bytes, 9–14-bit as little-endian 16-bit words) with
the chroma format, bit depth and picture order count of the stream:

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
