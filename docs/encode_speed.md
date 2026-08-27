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
