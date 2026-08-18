# rivet-h26x

[![crates.io](https://img.shields.io/crates/v/rivet-h26x.svg)](https://crates.io/crates/rivet-h26x)
[![docs.rs](https://docs.rs/rivet-h26x/badge.svg)](https://docs.rs/rivet-h26x)
[![CI](https://github.com/rivet-transcoder/rivet-h26x-codecs/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-h26x-codecs/actions/workflows/ci.yml)

Native **H.264/AVC** and **H.265/HEVC** decoders in Rust: no C, no system
libraries, no build script, nothing to install on a build host. Bit-exact
against the JVT and JCT-VC conformance suites, frame- and wavefront-threaded,
with AVX2, AVX / SSE4.1 and NEON kernels chosen at run time.

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
what the CPU has: **AVX2** on x86-64, **AVX** or **SSE4.1** on the x86-64 CPUs
without it, **NEON** on AArch64 and the ARMv8.2-A **dot product** extension on
top of it where present, scalar otherwise. The rungs are cumulative — each
replaces only the kernels its instructions actually improve, so a CPU ends up
with the best available version of every one. The AVX / SSE4.1 tier is the same
kernels at 128 bits: AVX adds no 256-bit *integer* operation — that is AVX2 —
so what it buys over SSE4.1 is VEX's three-operand encoding, worth a couple of
per cent, while the step from scalar to either is worth 2.3–2.7x. `sdot` is
worth having for the one kernel whose shape it fits, HEVC's eight-tap
horizontal luma filter, and measurably not worth it for H.264's six-tap, whose
taps LLVM already folds into `umlal` pairs. `H26X_NO_SIMD=1` forces scalar and
`H26X_MAX_SIMD=avx | sse41 | neon | none` caps the ladder, so one machine can
exercise every rung. Every SIMD kernel is checked
bit-exact against the scalar reference by the crate's tests, on both
architectures in CI. `H26X_PROF=1` prints where the time went.

## Performance

Against libavcodec, on the same clips, with both decoders materialising every
frame — ffmpeg writing rawvideo to the null device, this decoder packing each
picture and dropping it. Neither writes to disk. **Cost** is CPU seconds
(user+kernel), best of five; **throughput** is frames per wall second, which is
the question a multi-threaded run is actually asking. Regenerate with
`tools/benchmark.py`, on a machine doing nothing else — these are wall-clock
and CPU-time measurements and a busy box does not merely add noise to them, it
biases them.

Rows are the instruction-set rungs, because "how fast is it" has no answer
without saying which instructions it was allowed to use — and the rung is
chosen at run time, so the table doubles as a map of what different hardware
gets. `H26X_MAX_SIMD` caps the ladder, which is how one machine produces every
row.

**AMD Ryzen 9 9950X 16-Core Processor**, 32 hardware threads. Best of 5. Cost is CPU seconds, throughput is frames per wall second.

#### `cabac3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|
| AVX2 | 2.422 | 299 | 3.938 | 2016 |
| AVX (VEX-128) | 2.281 | 317 | 3.969 | 2028 |
| SSE4.1 | 2.422 | 299 | 3.859 | 2002 |
| scalar | 5.562 | 130 | 8.141 | 1249 |
| libavcodec | 1.375 | 526 | 2.719 | 1804 |

#### `cavlc3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|
| AVX2 | 1.828 | 395 | 4.031 | 2595 |
| AVX (VEX-128) | 1.922 | 376 | 3.578 | 2731 |
| SSE4.1 | 1.875 | 386 | 3.594 | 2778 |
| scalar | 5.062 | 143 | 8.703 | 1320 |
| libavcodec | 1.219 | 593 | 2.188 | 2854 |

#### `hevc6.265` — H.265, 1280x720, 1446 frames

| instructions | 1 thread: CPU s | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|
| AVX2 | 3.469 | 417 | 4.797 | 1189 |
| AVX (VEX-128) | 3.812 | 379 | 4.922 | 1171 |
| SSE4.1 | 3.844 | 376 | 5.453 | 1085 |
| scalar | 8.703 | 166 | 9.016 | 673 |
| libavcodec | 3.109 | 465 | 4.188 | 1355 |

#### `wpp10.265` — H.265, 1280x720, 1200 frames

| instructions | 1 thread: CPU s | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|
| AVX2 | 2.141 | 561 | 4.438 | 1780 |
| AVX (VEX-128) | 2.328 | 515 | 4.688 | 1739 |
| SSE4.1 | 2.422 | 495 | 4.781 | 1605 |
| scalar | 5.438 | 221 | 7.938 | 951 |
| libavcodec | 2.156 | 557 | 2.984 | 1722 |

### Reading it

**Single-threaded, this decoder costs more CPU than libavcodec** — about 1.8x
on H.264 CABAC, 1.5x on CAVLC, 1.1x on HEVC, and parity on HEVC with wavefront
rows. That gap is the honest one to quote, and it is where the remaining work
is: entropy decoding and per-block bookkeeping, not the pixel kernels.

**With every thread it is at or ahead of libavcodec on throughput** for the
clips whose structure gives it parallelism, and behind where the clip does not.
Frame threading is the whole of it for H.264; H.265 adds wavefront rows, tiles
and slice segments inside a picture, which is why the WPP clip scales further
than the one without it.

**The step from scalar to SIMD is worth 2.3–2.8x**, and the step from 128-bit
to 256-bit is worth almost nothing — AVX2 is within a couple of per cent of the
VEX-128 tier, and loses to it outright on one clip. H.264's blocks are at most
sixteen samples wide, so a 256-bit vector spans a row at best and pays
cross-lane permutes to undo per-lane packing that 128-bit code never does. This
is why the SSE4.1 rung matters: a CPU without AVX2 gives up a couple of per
cent, not the 2.5x it would give up falling back to scalar.

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
