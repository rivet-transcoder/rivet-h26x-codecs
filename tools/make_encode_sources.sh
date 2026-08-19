#!/bin/bash
# Generate the raw source clips verify_encode.sh encodes.
#
# Generated rather than vendored: they are a quarter of a megabyte, they are
# reproducible from this script in a second, and unlike the decode fixtures
# nothing about them needs to be pinned — an encoder gate compares its output
# against its own input, so the input only has to be varied, not identical to
# anyone else's.
#
# The naming carries the geometry (src_<name>_<W>x<H>_<fmt>.yuv) so
# verify_encode.sh does not need a table mapping clips to dimensions, which is
# the sort of table that goes stale silently.
#
# Usage: make_encode_sources.sh [dir]
set -e
cd "${1:-$(dirname "$0")}"
FFMPEG=${FFMPEG:-ffmpeg}

# Content chosen so that a broken encoder shows up rather than averaging out:
#   grad    smooth gradients and slow pans — intra prediction and sub-pel
#           motion have somewhere to be wrong
#   detail  high-frequency detail — the transform and quantiser carry it, and
#           residual coding has real coefficients to write
#   motion  fast motion — motion search, B pictures and reference handling
gen() { # name filter frames fmt pix
  out="src_$1_${4}.yuv"
  [ -f "$out" ] && { echo "have $out"; return; }
  "$FFMPEG" -v error -y -f lavfi -i "$2" -frames:v "$3" \
            -f rawvideo -pix_fmt "$5" "$out"
  echo "made $out ($(stat -c %s "$out") bytes)"
}

gen grad   "gradients=size=64x64:rate=25"                     8 64x64_420 yuv420p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_420 yuv420p
gen motion "testsrc=size=64x64:rate=25"                      12 64x64_420 yuv420p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_422 yuv422p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_444 yuv444p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_400 gray
# One clip whose dimensions are not a multiple of the coding block size, since
# cropping is signalled in the SPS and is a common place to be wrong.
gen odd    "testsrc2=size=50x34:rate=25"                      6 50x34_420  yuv420p
