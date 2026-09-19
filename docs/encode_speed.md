# Encoder speed: where the time goes, and which kernels have SIMD

Started 2026-08-27 on the `agent/enc-speed` track. The decoders have a
runtime kernel ladder (SSE2 → SSSE3 → SSE4.1 → AVX → AVX2, AVX-512 over the
top; NEON; wasm simd128) checked bit-exact against scalar references. This
note records the same question for the encoders: what each encode-only
kernel table carries at each tier, and what a profile of a real encode says
is worth writing.

## Inventory of the encode-side kernel tables (at 5cb468c)

The three tables an encoder builds with `*::new(cpu)`. A tier cell says
which rung *replaces* the scalar entry; `—` means the scalar reference runs
on that tier.

| table | kernel | SSE2 | SSSE3 | SSE4.1 | AVX | AVX2 | AVX-512 | NEON | simd128 |
|---|---|---|---|---|---|---|---|---|---|
| `distortion` | `sad` | — | — | — | — | — | — | — | — |
| `distortion` | `satd` | — | — | — | — | — | — | — | — |
| `distortion` | `ssd` | — | — | — | — | — | — | — | — |
| `h264_enc` | `fdct4` | — | — | — | — | — | — | — | — |
| `h264_enc` | `fdct8` | — | — | — | — | — | — | — | — |
| `h264_enc` | `hadamard4` | — | — | — | — | — | — | — | — |
| `h264_enc` | `hadamard2x2` | — | — | — | — | — | — | — | — |
| `h264_enc` | `hadamard2x4` | — | — | — | — | — | — | — | — |
| `h264_enc` | `quant4` | — | — | — | — | — | — | — | — |
| `h264_enc` | `quant8` | — | — | — | — | — | — | — | — |
| `hevc_enc` | `fdct[4]` (4/8/16/32) | — | — | — | — | — | — | — | — |
| `hevc_enc` | `fdst4` | — | — | — | — | — | — | — | — |
| `hevc_enc` | `quant` | — | — | — | — | — | — | — | — |
| `hevc_enc` | `fskip` | — | — | — | — | — | — | — | — |

Every cell is `—`: at 5cb468c **no encode-only kernel has a SIMD tier on
any architecture**. `H264EncDsp::new`, `HevcEncDsp::new` and
`DistortionDsp::new` all return the scalar table with the `cpu` field set.
(Checked by grepping every tier file for `fdct|hadamard|quant|satd|sad|ssd|
fskip|fdst|EncDsp|Distortion`: the only hits are comments about
*de*quantised coefficients in the inverse-transform kernels.)

What the encoders *do* get from the decoder's ladder: intra prediction,
sub-pel interpolation (`qpel_impl`, `fir_h`, `uni_impl` in the profiles
below), the inverse transforms used for reconstruction, deblocking and SAO
filtering. Those are shared with the decoder and already SIMD.

## Method

Profiled with `tools/prof.sh` (samply, 8 kHz, single process) on a build
with `CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=none`, on a
320x240 4:2:0 clip of 60 frames of ffmpeg `testsrc2` (generated beside the
gate corpus, not part of it). AMD Ryzen 9 9950X, AVX-512 rung selected for
the decoder-shared kernels. "self %" is the outermost non-inlined function.

Three names recur in every table and are not pixel kernels:

- `log2` is `f64::log2` called from `CabacEncoder::encode_decision` for
  every context-coded bin, in the *emitting* encoder as well as the
  counting one — the bit accounting `fractional_bits` reports. 8–24% of
  self time in every CABAC configuration.
- `hevc::intra::predict` / `h264::intra::predict_4x4` are the decoder's
  intra predictors, run once per candidate mode.
- `write_residual` / `write_residual_block_cabac` are the entropy writers,
  run for real and again for every RDOQ candidate.

## Profiles at 5cb468c

#### H.265 all-intra, CABAC, QP 26 (10355 samples at 8 kHz)

| self % | function |
|---:|---|
| 23.50% | `log2` |
| 16.58% | `hevc::residual::write_residual` |
| 11.92% | `dsp::distortion::satd_scalar` |
| 11.77% | `hevc::intra::predict` |
| 3.82% | `encode::h265_intra::rdoq_trim::closure$0` |
| 3.81% | `cabac_enc::CabacEncoder::encode_decision` |
| 3.54% | `dsp::hevc_enc::fdct_scalar<16>` |
| 2.80% | `cabac_enc::CabacEncoder::put_bit` |
| 2.70% | `encode::h265_intra::code_residual` |
| 1.88% | `dsp::hevc_enc::quant_scalar` |

#### H.265 IP (GOP 8), QP 26 (3675 samples at 8 kHz)

| self % | function |
|---:|---|
| 23.32% | `dsp::distortion::satd_scalar` |
| 9.06% | `log2` |
| 8.98% | `hevc::intra::predict` |
| 6.88% | `dsp::hevc_enc::fdct_scalar<16>` |
| 5.14% | `hevc::residual::write_residual` |
| 4.98% | `dsp::distortion::sad_scalar` |
| 4.44% | `dsp::hevc_enc::quant_scalar` |
| 2.07% | `cabac_enc::CabacEncoder::encode_decision` |
| 1.99% | `dsp::hevc_avx2_u8::uni_impl` |
| 1.85% | `dsp::hevc_enc::fdct_scalar<8>` |

#### H.265 IPB (GOP 8, 2 B), QP 26 (4328 samples at 8 kHz)

| self % | function |
|---:|---|
| 27.22% | `dsp::distortion::satd_scalar` |
| 9.27% | `log2` |
| 7.60% | `hevc::intra::predict` |
| 6.45% | `dsp::distortion::sad_scalar` |
| 6.17% | `dsp::hevc_enc::fdct_scalar<16>` |
| 4.34% | `hevc::residual::write_residual` |
| 3.05% | `dsp::hevc_enc::quant_scalar` |
| 2.43% | `dsp::hevc_avx2_u8::uni_impl` |
| 2.29% | `cabac_enc::CabacEncoder::encode_decision` |
| 1.52% | `dsp::hevc_avx2_u8::fir_h<8,0>` |

#### H.264 all-intra, CABAC, QP 26 (2156 samples at 8 kHz)

| self % | function |
|---:|---|
| 21.24% | `dsp::distortion::satd_scalar` |
| 17.72% | `encode::h264_intra::code_i4x4` |
| 14.66% | `log2` |
| 8.81% | `h264::intra::predict_4x4` |
| 3.85% | `h264::cabac_mb::write_residual_block_cabac` |
| 3.66% | `cabac_enc::CabacEncoder::encode_decision` |
| 3.43% | `dsp::h264_enc::quant4_scalar` |
| 2.41% | `encode::h264_intra::code_block_4x4` |
| 1.95% | `dsp::h264_avx2::residual4_impl` |
| 1.90% | `encode::h264_intra::code_macroblock` |

#### H.264 IP (GOP 8), CABAC, QP 26 (1654 samples at 8 kHz)

| self % | function |
|---:|---|
| 35.67% | `dsp::distortion::satd_scalar` |
| 8.52% | `log2` |
| 6.71% | `dsp::distortion::sad_scalar` |
| 4.29% | `dsp::h264_enc::quant4_scalar` |
| 4.17% | `encode::h264_intra::code_i4x4` |
| 3.02% | `cabac_enc::CabacEncoder::encode_decision` |
| 2.72% | `dsp::h264_avx2::residual4_impl` |
| 1.75% | `h264::intra::predict_4x4` |
| 1.75% | `h264::cabac_mb::write_residual_block_cabac` |
| 1.63% | `encode::h264_me::code_inter_4x4` |

#### H.264 IPB (GOP 8, 2 B), CABAC, QP 26 (2104 samples at 8 kHz)

| self % | function |
|---:|---|
| 44.06% | `dsp::distortion::satd_scalar` |
| 8.08% | `dsp::distortion::sad_scalar` |
| 7.56% | `log2` |
| 3.56% | `dsp::h264_enc::quant4_scalar` |
| 2.66% | `encode::h264_intra::code_i4x4` |
| 1.85% | `h264::intra::predict_4x4` |
| 1.71% | `cabac_enc::CabacEncoder::encode_decision` |
| 1.57% | `h264::cabac_mb::write_residual_block_cabac` |
| 1.52% | `dsp::h264_avx2::residual4_impl` |
| 1.47% | `dsp::h264_avx2::qpel_impl<2,2>` |

#### H.264 all-intra, CAVLC, QP 26 (1674 samples at 8 kHz)

| self % | function |
|---:|---|
| 23.78% | `dsp::distortion::satd_scalar` |
| 22.88% | `encode::h264_intra::code_i4x4` |
| 9.68% | `h264::intra::predict_4x4` |
| 4.42% | `h264::cavlc::write_residual_block_cavlc` |
| 4.24% | `dsp::h264_enc::quant4_scalar` |
| 3.46% | `encode::h264_intra::code_block_4x4` |
| 2.03% | `h264::intra::predict_planar_block` |
| 1.97% | `encode::h264_intra::code_macroblock` |
| 1.91% | `dsp::h264_avx2::residual4_impl` |
| 1.91% | `dsp::h264_enc::fdct4_scalar` |

#### H.264 IP (GOP 8), CAVLC, QP 26 (1321 samples at 8 kHz)

| self % | function |
|---:|---|
| 43.00% | `dsp::distortion::satd_scalar` |
| 6.66% | `dsp::distortion::sad_scalar` |
| 4.92% | `dsp::h264_enc::quant4_scalar` |
| 4.84% | `encode::h264_intra::code_i4x4` |
| 3.10% | `dsp::h264_avx2::residual4_impl` |
| 2.80% | `encode::h264_me::code_inter_4x4` |
| 2.57% | `h264::cavlc::write_residual_block_cavlc` |
| 2.20% | `fun_6bb340` |
| 2.20% | `dsp::h264_avx2::qpel_impl<2,2>` |
| 1.82% | `h264::intra::predict_4x4` |

#### H.264 IP (GOP 8), CABAC, --t8x8 --subparts, QP 26 (7877 samples at 8 kHz)

| self % | function |
|---:|---|
| 49.19% | `dsp::distortion::satd_scalar` |
| 10.88% | `dsp::distortion::sad_scalar` |
| 5.89% | `dsp::h264_avx2::qpel_impl<2,2>` |
| 3.78% | `encode::h264_me::search_rect` |
| 2.63% | `encode::h264_me::luma_pred_into` |
| 2.25% | `encode::h264_me::code_macroblock_p` |
| 1.88% | `log2` |
| 1.31% | `dsp::h264_avx2::qpel_impl<2,0>` |
| 1.19% | `dsp::h264_avx2::qpel_impl<0,2>` |
| 1.09% | `dsp::h264_avx2::qpel_impl<3,3>` |

## What was built, and what it measured (2026-08-27)

Every number below is from one binary under two environments (or, where
a change has no switch, two builds of the same tree one commit apart),
interleaved, pinned to the quietest physical core, median of nine paired
CPU-second ratios (`tools/ab_enc.py`). Every group ran a same-binary
control; the control's spread is the smallest difference the machine
could resolve at the time, and it is quoted beside the result it bounds.
Clip: 640x360 4:2:0 `testsrc2`, 30 frames (90 for the whole-track row),
QP 26. The machine was shared with two other gate sweeps throughout.

### Inventory after this work

| table | kernel | SSE2 | SSSE3 | AVX | AVX2 | NEON | simd128 |
|---|---|---|---|---|---|---|---|
| `distortion` | `sad` / `satd` / `ssd` (u8) | yes | satd (`pabsw`) | yes (VEX) | yes, widths ≥16 | written, compile-checked only | — |
| `hevc_enc` | `fdct` 4/8/16/32, `fdst4` | yes | — | yes (VEX) | 16 and 32 | — | — |
| `hevc_enc` | `quant` | yes | `pabsw` | yes (VEX) | yes | — | — |
| `hevc_enc` | `fskip` | — | — | — | — | — | — |
| `h264_enc` | all seven | — | — | — | — | — | — |

SSE4.1 is not a rung for any of these: nothing in them has a better
SSE4.1 instruction. `H264EncDsp` stays scalar on purpose — see below.
*(It has SIMD since the simdfull round at the end of this note.)*
`DistortionDsp<u16>` keeps the scalar reference.

### Per-kernel (microbench in each module's `kernel_bench`, ns per call group, release)

Scalar is listed twice: the pair is the control.

| distortion (sad+satd+ssd) | scalar | scalar again | SSE2 | AVX | AVX2 |
|---|---:|---:|---:|---:|---:|
| 4x4 | 60.9 | 59.5 | 21.7 | 17.3 | 17.2 |
| 8x8 | 174 | 167 | 35.8 | 29.1 | 30.5 |
| 16x16 | 644 | 633 | 90.0 | 78.6 | 36.6 |
| 32x32 | 1947 | 1867 | 230 | 249 | 130 |
| 64x64 | 8161 | 8484 | 1109 | 892 | 472 |

| hevc_enc (fdct+quant) | scalar | scalar again | SSE2 | AVX | AVX2 |
|---|---:|---:|---:|---:|---:|
| 4x4 | 58.5 | 60.2 | 24.0 | 22.3 | 24.9 |
| 8x8 | 238 | 247 | 58.2 | 65.1 | 52.9 |
| 16x16 | 2173 | 2149 | 317 | 251 | 177 |
| 32x32 | 17087 | 16602 | 1905 | 1678 | 1173 |

H.264 `fdct4` and `quant4` were written the same way (i32-lane 4x4
transform, `pmuludq` quantiser reproducing the i64 product and the two
truncating casts), bit-exact, and measured: `fdct4` scalar 8.4 ns against
8.8–9.5 ns SIMD, `quant4` 10.1 against 9.0–11.6. The compiler already
vectorises those fixed-size loops and the 64-bit product costs the SIMD
form what it gains, so the module was not kept. Recorded so nobody writes
it a third time; the remaining scalar hot spot in H.264 (`quant4` at 3–5%)
wants a different idea — a 32-bit product under a proven bound — not
wider lanes. *(They were written a third time, measured faster, and kept:
see the simdfull round at the end of this note.)*

### End-to-end, one step at a time

`B/A` is the encode's CPU time after over before; the range is the nine
paired ratios.

| step | switch | H.265 intra | H.265 IP | H.265 IPB | H.264 intra | H.264 IP | H.264 IPB | H.264 t8x8+subparts |
|---|---|---|---|---|---|---|---|---|
| distortion SIMD | `H26X_ENC_NO_SIMD=distortion` | 0.823 (0.78–0.98) | 0.705 (0.67–1.05) | — | 0.727 (0.71–0.76) | — | 0.458 (0.42–0.48) | 0.456 (0.44–0.47) |
| HEVC forward SIMD | `H26X_ENC_NO_SIMD=hevc_enc` | 0.878 (0.87–0.89) | 0.815 (0.78–0.89) | 0.857 (0.79–0.89) | — | — | — | — |
| CABAC cost table | two builds | 0.897 (0.89–0.93) | 0.955 (0.91–1.00) | — | 0.938 (0.88–1.07) | 1.000 (0.90–1.00) | 0.909 (0.91–1.00) | — |
| RDOQ early-out | two builds | 0.967 (0.95–1.00) at QP 26; 0.941 (0.91–1.29) at QP 22; 1.000 (0.85–1.02) at QP 34 | n/a (intra only) | | | | | |
| controls (same binary, same env) | | 1.026 (0.93–1.03), 1.000 (0.96–1.04), 1.000 (0.92–1.05), 1.000 (0.95–1.05) | | | 1.000 (0.91–1.10), 1.000 (0.90–1.10), 1.000 (1.00–1.11) | | | |

The H.264 rows for the CABAC table are at the resolution of the CPU-time
tick (0.15 s runs, 15.6 ms ticks — the controls span ±10%); the
whole-track row below, on three times the frames, is the one to read.

### The whole track, 5cb468c against 1905e57 (640x360, 90 frames)

| | H.264 intra | H.264 IP | H.264 IPB | H.264 t8x8+subparts | H.264 CAVLC IP | H.265 intra | H.265 IP | H.265 IPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after / before | **0.689** | **0.490** | **0.420** | **0.446** | **0.477** | **0.653** | **0.527** | **0.485** |
| paired range | 0.667–0.694 | 0.469–0.521 | 0.406–0.441 | 0.445–0.452 | 0.455–0.488 | 0.646–0.660 | 0.513–0.549 | 0.481–0.492 |

Controls in the same session: 1.000 (0.931–1.037) and 1.000 (0.991–1.009).

### Identity

`tools/identity_encode.sh` — all 280 cells of the eight-bit corpus
(9 clips x the configuration list, `@src_cut` row on one), bitstream and
reconstruction compared byte for byte:

- encode-side SIMD off vs on, binary at 7695ece: **280 identical, 0 moved**
- 7695ece vs 7b547c0 (CABAC table): 280 identical, 0 moved
- 7b547c0 vs 9644eac (RDOQ early-out): 280 identical, 0 moved
- `tools/bd_rate.py`, RDOQ early-out, all-intra H.265 at QP 22/27/32/37:
  +0.000% on encspeed_320x240, src_detail, src_cut (byte-identical streams;
  the only divergence the count found is two blocks in 544 at QP 40).

### Threading

`--threads` is parsed into `Config::threads` and read by nothing: neither
encoder has a thread. `--threads 1` vs `0` on 640x360: H.264 IPB 255 vs
246 ms, H.265 IP 398 vs 452 ms — noise. Making an encoder scale is a
design, not a serialisation fix, and was not attempted.

### Not done

- **NEON** kernels are compile-checked against `aarch64-unknown-linux-gnu`
  (`--tests` too) and carry the x86 module's bit-exactness test, which no
  machine here can run. They need the CI runners (a PR against develop);
  nothing was pushed. *(Taken up by the portability round below.)*
- **wasm simd128** tier: not written. *(Written in the portability round
  below.)*
- **`DistortionDsp<u16>`**: scalar. The u8 kernels' shapes carry over
  (`psadbw` does not; a u16 SAD is `pabsw` of a difference and `pmaddwd`
  against ones), an afternoon's work once the u16 encoder path has a
  profile to point at.
- **H.264 `quant4`** stays scalar for the reason above. *(SIMD since the
  simdfull round below.)*
- **`fskip`, `hadamard*`, `fdct8`**: not in any profile's top ten.
- **A README benchmark table per rung** for the encoders was not
  generated; the method (`tools/ab_enc.py`, `H26X_MAX_SIMD`) and this
  note's tables are what exist.

## Portability round: wasm simd128 and NEON for the encode kernels (`agent/wasm-enc`, 2026-08-27)

The decode kernels run on x86 (SSE2 to AVX-512), NEON and wasm `simd128`;
after `agent/enc-speed` the encode-only kernels ran only on x86, with a
NEON distortion kernel written blind. This round gives every encode kernel
that has an x86 tier the other two, and adds the ladder gate the encoders
were missing.

### Inventory, before and after

A cell says which rung replaces the scalar entry on that tier; `—` means
the scalar reference runs there. H.264's seven encode kernels have no x86
tier (the note above says why) and therefore none here either.

Before (develop 64166be):

| table | kernel | x86-128 (SSE2/SSSE3/AVX) | AVX2 | NEON | wasm simd128 |
|---|---|---|---|---|---|
| `distortion` | `sad` / `satd` / `ssd` (u8) | yes | widths ≥16 | written, never executed | — |
| `hevc_enc` | `fdct` 4/8/16/32, `fdst4` | yes | 16 and 32 | — | — |
| `hevc_enc` | `quant` | yes | yes | — | — |
| `hevc_enc` | `fskip` | — | — | — | — |
| `h264_enc` | all seven | — | — | — | — |

After (this branch):

| table | kernel | x86-128 (SSE2/SSSE3/AVX) | AVX2 | NEON | wasm simd128 |
|---|---|---|---|---|---|
| `distortion` | `sad` / `satd` / `ssd` (u8) | yes | widths ≥16 | yes (`distortion_neon.rs`, CI-executed) | yes (`distortion_wasm128.rs`) |
| `hevc_enc` | `fdct` 4/8/16/32, `fdst4` | yes | 16 and 32 | yes (`hevc_enc_neon.rs`, CI-executed) | yes (`hevc_enc_wasm128.rs`) |
| `hevc_enc` | `quant` | yes | yes | yes | yes |
| `hevc_enc` | `fskip` | — | — | — | — |
| `h264_enc` | all seven | — | — | — | — |

The H.265 matrix now lives in three compile-time layouts in
`hevc_enc::layouts` — the `pmaddwd` pair table (x86 and wasm), the
transposed matrix and the matrix itself (NEON's `smlal`-by-lane wants the
weights of eight consecutive outputs contiguous in stage one and the
weights of one output row contiguous in stage two) — every one built from
the decoder's `TRANSFORM32` and read back against it by a test, so the
three tiers cannot disagree with the reference about a coefficient.

### wasm, verified inside the module (`tools/wasm.sh`, node 22)

`cargo test` does not run on wasm32, so the x86 modules' randomised sweeps
were exported from `examples/wasm_probe` (`h26x_enc_dsp_check`) and run
inside both builds, and an encode → decode round trip (`h26x_encode`)
hashes bitstream, decoded pictures and the encoder's reconstruction
inside the module. What a run prints:

- which encode-side entries each build's tables took: scalar mask 0,
  simd128 mask 63 (all six groups) — the rung name alone cannot say this;
- the sweep: `scalar OK`, `simd128 OK` (24 rounds over sixteen distortion
  shapes with random strides and offsets and pinned extreme rows; 40
  rounds per transform size at bit depths 8/10/12; the quantiser at every
  third QP, intra and inter, with i16-extreme rows);
- the round trip, h264 and h265 × intra / IP / IPB at QP 26 and IP at QP
  40, on synthesised frames and the seven 8-bit gate clips (`ENC_CLIPS`):
  **56 cells identical across the builds and self-consistent** (decoded ==
  reconstruction, the SELF property, on both builds);
- kernel timings from outside the module, ns per call group, best of three:

| group | scalar | simd128 | ratio |
|---|---:|---:|---:|
| sad+satd+ssd 4x4 | 40.8 | 17.3 | 2.36x |
| sad+satd+ssd 8x8 | 127.8 | 24.3 | 5.26x |
| sad+satd+ssd 16x16 | 478.5 | 68.9 | 6.94x |
| sad+satd+ssd 32x32 | 1812.8 | 219.7 | 8.25x |
| sad+satd+ssd 64x64 | 7103.6 | 805.5 | 8.82x |
| fdct+quant 4x4 | 51.2 | 16.5 | 3.10x |
| fdct+quant 8x8 | 235.7 | 42.7 | 5.52x |
| fdct+quant 16x16 | 2150.1 | 204.2 | 10.53x |
| fdct+quant 32x32 | 15809.1 | 1490.6 | 10.61x |

Whole encodes inside wasm, scalar build → simd128 build, best of three
(these include the decode-shared kernels' tier, which wasm cannot switch
off separately — there is no environment there): `src_cut` 96 frames
64x64, H.264 IP 64.6 → 23.9 ms, IPB 82.5 → 28.7 ms, H.265 IP 108.6 → 39.3
ms, IPB 141.5 → 42.3 ms; H.264 intra 62.5 → 50.2 ms and H.265 intra 402.7
→ 262.9 ms, the intra paths being the ones that spend least in these
kernels.

Re-run on the rebase onto develop 4128e49 (08657cc, 2026-09-13): masks 0
and 63, both sweeps `OK`, the same 56 cells identical across the builds
and self-consistent. The timings were not re-recorded — the box was
carrying two other tracks' gate sweeps at the time, and the table above
is the quiet-box record at d0dde0a (the kernels have not changed since).

Two things learned porting: wasm has no `psadbw` and no `pmulhuw`. SAD is
`u8x16_sub_sat` both ways or'd together and `u16x8_extadd_pairwise_u8x16`
(NEON's shape, not x86's); the quantiser's exact 32-bit product is a
widening and an `i32x4_mul`, whose low half is the whole product because
it is under 2^31. Everything else is the x86 128-bit kernel instruction
for instruction.

### NEON, verified on the CI runners

No ARM here; the only execution route is a PR against develop, whose
`ubuntu-24.04-arm` and `macos-latest` jobs run `cargo test --release`
with `H26X_REQUIRE_DOTPROD=1`. **PR #5** (`agent/wasm-enc`, do not
merge), run 33116493939 at d0dde0a: `tests (linux arm64 (NEON))` pass
in 57 s, `tests (macos arm64 (NEON))` pass in 1 m 16 s, and both job logs
carry the lines that matter —
`dsp::distortion_neon::tests::{neon_matches_scalar,
full_deflection_is_exact}` and
`dsp::hevc_enc_neon::tests::{forward_transforms_match_scalar,
quantiser_matches_scalar, new_installs_neon}` all `ok`, 239 passed on
Linux, alongside the `hevc_neon_u8` dot-product tests that
`H26X_REQUIRE_DOTPROD=1` forbids from skipping. The x86 and MSRV jobs
pass too. This is the first execution of `distortion_neon.rs`, which had
been on develop installed-but-unexecuted since `agent/enc-speed`.
Re-run on the rebase onto develop 4128e49 (08657cc, run 34785319047,
2026-09-13): the same five tests `ok` on both arm64 runners, 239 passed
on each, all six jobs green.

The NEON transform is not the pair-table shape: NEON has no `pmaddwd`,
and its natural idiom is `smlal`/`smlal2` by lane, so stage one takes
each input sample as a lane against eight consecutive outputs' weights
(the transposed matrix) and stage two takes one output row's weights as
the lanes against the intermediate's rows (the matrix itself) — no
transpose in either stage. The rounding term is the accumulator's initial
value, the shift is `sshl` by a negative count, `sqxtn` is the clamp. The
quantiser is `umull` of the u16 magnitude by the scale, `ushl` by
`-qbits`, and the zero count comes from subtracting `cmeq`-with-zero
masks into a lane counter. No timing: nothing here runs it.

### The encoder ladder

`tools/verify_enc_ladder.sh`: every `H26X_MAX_SIMD` rung the host can
take, against the scalar reference (`H26X_NO_SIMD=1`), over every 8-bit
cell of `verify_encode.sh`'s list, bitstream and reconstruction compared
byte for byte through `identity_encode.sh`. The cap applies to every
table the encoder builds, the shared decoder kernels included, so it is
the whole encoder's rung-independence that is asserted.

Run on the rebase onto develop 4128e49 (2026-09-13, a Zen 5 box that
takes every rung to AVX-512, `JOBS=6`): 34 8-bit rows over 9 clips is 309
cells a rung, each encoded twice —

```
-- rung sse2 (selects: SSE2) --           identity: 309 identical, 0 moved
-- rung ssse3 (selects: SSSE3) --         identity: 309 identical, 0 moved
-- rung sse41 (selects: SSE4.1) --        identity: 309 identical, 0 moved
-- rung avx (selects: AVX (VEX-128)) --   identity: 309 identical, 0 moved
-- rung avx2 (selects: AVX2) --           identity: 309 identical, 0 moved
-- rung avx512 (selects: AVX-512) --      identity: 309 identical, 0 moved
ladder: 6 rungs identical, 0 failed
LADDER IDENTICAL
```

The gate bites. A binary built aside with the AVX2 SAD dropping the last
row of a 16-wide block, run over two clips on rungs `avx` and `avx2`:
`avx` stays 68 identical, `avx2` reports 34 of 68 cells MOVED, bitstreams
differing by 1 to 57 bytes — all of them H.264 inter rows. What did not
move is instructive rather than vacuous: the H.264 intra-only rows, two IP
rows on the detail clip whose search happened not to flip, and all twelve
H.265 rows on both clips — the H.265 encoder of that measurement searched
at the CTB size (`log2_cu = log2_ctb_size`; the writer took 32 for a
64x64 clip and 16 only where that padded less, as for `odd`), so on these
clips its SAD calls never took the 16-wide path; that combination did not
exist in that encoder. (Since the coding quadtree it searches every unit
size down to 8x8, and the CTB is 32x32, partial along the right and bottom
edges, at every picture size 64 or more in either direction; whole 16x16
CTBs remain, where they pad less, at `max_cu_depth` 0 and below 64 both
ways, as for `odd`.) The H.265 rows are reached through the quantiser: the same two
clips with the AVX2 quantiser's rounding offset doubled give `avx` 68
identical again and `avx2` 18 of 68 MOVED — every H.265 row that
quantises, on both clips, while the six lossless cells and all 44 H.264
cells (which never call `hevc_enc.quant`) stay identical. A first
attempt at that mutation, `offset + 1`, moved nothing on either rung:
it changes a level only when `|c| * scale + offset` sits exactly one
below a multiple of `2^qbits`, and `qbits` is 20 to 23 at 8 bits — a
mutation the gate cannot be blamed for missing.

## The simdfull round: the H.264 encode kernels on every tier (`agent/simdfull`, 2026-09-18)

A 1080p profile (one thread, 10 frames IP at QP 27, `--t8x8 --subparts`,
the AVX-512 rung for everything shared with the decoder) put the scalar
`quant4`, `quant8`, `fdct4` and `fdct8` at 6.9% of an 8-bit encode's self
time and 5.9% of a 10-bit one's. At that size, with the 8x8 transform
on, `fdct8` and `quant8` are in the profile; on the 640x360 clip above
they were not.

### Inventory, before and after

| table | kernel | x86-128 (SSE2 to AVX) | AVX2 | AVX-512 | NEON | simd128 |
|---|---|---|---|---|---|---|
| `h264_enc` | `fdct4`, `hadamard4` | — → yes | — → the AVX kernel | — → the AVX kernel | — → yes | — → yes |
| `h264_enc` | `fdct8`, `quant4`, `quant8` | — → yes | — → yes (256-bit) | — → the AVX2 kernel | — → yes | — → yes |
| `h264_enc` | `hadamard2x2`, `hadamard2x4` | — | — | — | — | — |
| `distortion` | `satd` on 8-wide blocks (u8, u16) | yes | 128-bit → 256-bit | as AVX2 | yes | yes |

The two chroma DC Hadamards stay scalar: four or eight values a
macroblock are not a vector's worth of work, and neither is in a
profile. The AVX2 SATD handles an 8x8 as one pass, rows 0 to 3 in the low
128-bit lane and 4 to 7 in the high one; a 4-wide block stays on the
128-bit kernel, which it half fills already.

The kernels are the design the 2026-08-27 note describes: the reference's
integer butterflies on i32 lanes with transposes around the first pass,
and the quantiser's `(|c| * mf + offset) >> qbits` in 64 bits through
`pmuludq` (NEON `umull` / `umull2`, simd128 `u64x2_extmul`), the level
truncated to i16 as the reference's casts truncate. Each module's tests
compare them with the scalar reference at bit depths 8 to 14.

### What they measured

Per kernel: `h264_enc_x86::tests::kernel_bench` (ignored; run it with
`--ignored --nocapture`). Both sides are called through the table, as the
encoder calls them. The figures are ns per call, the median of seven
paired rounds, at 44–59% machine load, and the scalar row twice is the
control.

| kernel | scalar | scalar again | SSE2 | SSE4.1 | AVX | AVX2 |
|---|---:|---:|---:|---:|---:|---:|
| `fdct4` | 7.7 | 7.7 | 5.7 | 3.6 | 2.9 | 2.9 |
| `fdct8` | 31.2 | 32.2 | 17.2 | 16.7 | 15.8 | 10.6 |
| `hadamard4` | 9.1 | 9.1 | 4.0 | 3.6 | 2.9 | 2.9 |
| `quant4` | 10.2 | 10.2 | 6.5 | 7.0 | 6.9 | 5.9 |
| `quant8` | 39.3 | 39.2 | 21.9 | 17.8 | 17.4 | 12.3 |

A run at 97–100% load gave the same ordering at lower ratios: `fdct4`
1.6x, `fdct8` 3.3x, `hadamard4` 1.9x, `quant4` 1.2x, `quant8` 2.1x at
AVX2. That is not the parity the 2026-08-27 note recorded for the same
design. Its code is not in the tree, so the two cannot be compared; this
bench is, and can be re-run.

End to end: 1080p, 10 frames IP, QP 27, `--t8x8 --subparts --threads 1`,
`tools/ab_enc.py`, seven paired rounds at 96–97% load.

| | 8-bit | 10-bit |
|---|---:|---:|
| one binary, `H26X_ENC_NO_SIMD=h264_enc` against the table as shipped | 0.960 (0.855–1.080) | 0.935 (0.882–0.985) |
| develop c5be83f against the batch (with the 4x4 inverse transforms and the AVX2 8x8 SATD) | 0.945 (0.895–0.985) | 0.933 (0.825–1.054) |
| same-environment control | 1.000 (0.934–1.036) | 1.041 (0.909–1.097) |

### Identity and the gates

Every cell of `verify_encode.sh`'s 966 is byte for byte the same:
- with the encode-side tables on and off (`identity_encode.sh`,
  `H26X_ENC_NO_SIMD`);
- against develop c5be83f's encoder;
- on every rung of `verify_enc_ladder.sh`, SSE2 to AVX-512 (`LADDER
  IDENTICAL`).

The gate bites: doubling the AVX2 quantiser's rounding offset fails
`quantisers_match_scalar`. The NEON tests pass under Docker Desktop's
arm64 emulation (`rust:1-slim-bookworm`, `linux/arm64`). There is no
arm64 hardware behind that result. On simd128, `tools/wasm.sh` now expects
installed-mask 2047: bits 512 and 1024 are the H.264 transforms and
quantisers.

## The simdfull round, continued: intra prediction and the SAO statistics (2026-09-18)

The same 1080p profile, H.265 with `--sao`, put two scalar passes at the top:
- the intra predictor, called once for each candidate mode: 22.3% of an 8-bit encode's self time and 20.9% of a 10-bit one's;
- the SAO decision's edge classification and per-category error sums: 11.2% and 9.0%.

### What changed

- **Intra prediction.** `hevc::intra` gathers and substitutes the references once, in `prepare` (8.4.4.2.2). `predict_prepared` then smooths on first need, keeping the result for the block's later modes (8.4.4.2.3), and predicts.
  - The three predictors are `HevcDsp` entries: `intra_planar`, `intra_dc` and `intra_angular`. The horizontal angular modes predict the transposed block and transpose it on the way out.
  - SIMD on every x86 rung (128-bit; AVX2 and AVX-512 keep those), NEON and simd128.
  - The encoder's luma search prepares a block once for its 35 trials. The chroma search does the same per plane when the chroma block is one transform block.
  - A test keeps the old predictor verbatim and checks the new one against it on every table the host builds: every mode, size and availability pattern, and every filter configuration.
- **SAO statistics.** A new `DistortionDsp` entry, `sao_edge_stats`: per edge category, the count and the error sum over a region whose neighbours are all inside the picture.
  - The picture-edge rows and columns keep the scalar loop.
  - Samples within reach of a rail come back one by one, as the reference reports them.
  - x86 SSE2 / SSE4.1 / AVX. Every sum is an integer, so lane order cannot change a result. NEON and simd128 run the scalar reference.
  - The decision's "leave the CTB alone" distortion is now the table's SSD.

### What they measured

Per kernel: ns per call, through the table, median of seven paired rounds, 44–59% machine load. The bench crate that produced these is outside the tree; each call includes a small buffer allocation. Speedups are over scalar, at AVX2.

| kernel | 4x4 | 8x8 | 16x16 | 32x32 |
|---|---:|---:|---:|---:|
| `intra_angular`, four modes, 8-bit | 1.6x | 3.0x | 4.0x | 4.2x |
| `intra_angular`, four modes, 10-bit | 1.5x | 2.7x | 4.2x | 4.3x |
| `intra_planar` + `intra_dc`, 8-bit | 1.1x | 1.4x | 1.8x | 1.8x |

`sao_edge_stats`:
- 8-bit: 3.1x on a 64x64 region, 2.5x on 32x32;
- 10-bit: 4.2x on 64x64, 3.5x on 32x32.

End to end: 1080p, 10 frames IP, QP 27, `--sao --threads 1`. `tools/ab_enc.py`, seven paired rounds, at 14–17% load.

| | 8-bit | 10-bit |
|---|---:|---:|
| encode, before these two against after | 0.777 (0.766–0.783) | 0.829 (0.816–0.836) |
| same-binary control | 0.995 (0.913–1.073) | 0.989 (0.979–1.005) |

The decoder runs the same predictors, but intra prediction is a few percent of a decode. An H.265 decode read 0.974 (0.925–1.054) at 8 bits and 1.000 (0.959–1.061) at 10, which is no measurable change.

The whole simdfull round was also measured in the same session: develop c5be83f against all three batches, 1080p, one thread.

| | H.264 8-bit | H.264 10-bit | H.265 8-bit | H.265 10-bit |
|---|---:|---:|---:|---:|
| encode | 0.924 (0.913–0.924) | 0.940 (0.934–0.958) | 0.722 (0.685–0.755) | 0.770 (0.737–0.779) |
| decode | 1.000 (0.966–1.051) | 0.985 (0.951–1.031) | 0.975 (0.952–1.027) | 0.881 (0.800–1.185) |

The 10-bit H.265 decode row is the fused 16-bit interpolation. Against its own base that change read 0.824 (0.640–0.978) over 25 rounds.

Every stream is byte for byte what it was:
- all 966 cells of `verify_encode.sh`, with the encode-side tables on and off;
- the same 966 cells against the previous encoder;
- every rung of `verify_enc_ladder.sh`;
- every rung of the decoder fixtures against `baseline.txt`.
