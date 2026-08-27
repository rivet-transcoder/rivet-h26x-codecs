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
wider lanes.

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
  nothing was pushed.
- **wasm simd128** tier: not written.
- **`DistortionDsp<u16>`**: scalar. The u8 kernels' shapes carry over
  (`psadbw` does not; a u16 SAD is `pabsw` of a difference and `pmaddwd`
  against ones), an afternoon's work once the u16 encoder path has a
  profile to point at.
- **H.264 `quant4`** stays scalar for the reason above.
- **`fskip`, `hadamard*`, `fdct8`**: not in any profile's top ten.
- **A README benchmark table per rung** for the encoders was not
  generated; the method (`tools/ab_enc.py`, `H26X_MAX_SIMD`) and this
  note's tables are what exist.
