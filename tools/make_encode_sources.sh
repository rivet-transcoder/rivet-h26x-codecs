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
#   static  the SAME picture held, frame after frame — detailed, but with
#           nothing moving. Every other clip here is moving detail, and that
#           uniformity hid a real bug: with no quantiser to round residual
#           away, a lossless CU on moving content ALWAYS carries some, so no
#           lossless CU is ever a skip, so the rule that
#           cu_transquant_bypass_flag precedes cu_skip_flag is never
#           exercised. Omitting the flag on a skip left the whole
#           hevc-lossless-ip row green. A held frame predicts exactly, the
#           residual really is zero, skips appear, and the same mutation
#           fails SELF at once.
#
#           It is not exotic content. A title card, a slate, a letterbox, a
#           paused shot — "nothing changed" is most of the frames in a great
#           deal of real video, and the skip, merge-everything and
#           residual-quantised-to-nothing families are all thin without it.
#
#   cut     the only clip here that is SECONDS rather than frames: 96 of
#           them, with a hard scene cut partway. Everything above is six to
#           twelve frames, and that one property of the corpus has now
#           blocked three separate features from being tested at all:
#
#             - rate control's achievable range came out 4.5x across the
#               corpus, because one keyframe dominates a clip that short and
#               caps how far a target can be moved;
#             - convergence cannot be measured over eight pictures, so the
#               rate band had to be [0.5x, 2.0x] and the residual error is
#               all opening transient;
#             - a conforming coded picture buffer must be at least as large
#               as the biggest access unit, and on these clips that is 33%
#               to 62% of the ENTIRE clip — so any buffer a stream could
#               conform to is bigger than the stream, and the buffer never
#               completes one fill-and-drain cycle. Both branches of a VBV
#               gate row are vacuous: a buffer at or above the largest
#               access unit passes for every encoder including one that
#               ignores it, and below it fails for every encoder.
#
#           Ninety-six frames at thirty a second is 3.2 seconds, so a
#           one-second buffer cycles about three times.
#
#           The cut is HARD, not a dissolve, and lands at frame 51 — which
#           is deliberately not a multiple of the gate's GOP of 8. A cut on
#           a keyframe boundary is the easy case: the encoder was going to
#           code an intra picture there anyway. A cut mid-GOP leaves an
#           inter picture with nothing to predict from, which is what
#           actually stresses a buffer and a reference chain, and it is what
#           real content does at every shot boundary. Both halves are
#           expensive and structurally unrelated — a cut into cheap content
#           is easy, because the encoder simply spends less.
gen() { # name filter frames fmt pix
  out="src_$1_${4}.yuv"
  [ -f "$out" ] && { echo "have $out"; return; }
  "$FFMPEG" -v error -y -f lavfi -i "$2" -frames:v "$3" \
            -f rawvideo -pix_fmt "$5" "$out"
  echo "made $out ($(stat -c %s "$out") bytes)"
}

gen grad   "gradients=size=64x64:rate=25:c0=0x2050a0:c1=0xe0b040:x0=0:y0=0:x1=63:y1=63:nb_colors=2:seed=1:speed=0.01:type=linear"                     8 64x64_420 yuv420p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_420 yuv420p
gen motion "testsrc=size=64x64:rate=25"                      12 64x64_420 yuv420p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_422 yuv422p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_444 yuv444p
gen detail "testsrc2=size=64x64:rate=25"                      8 64x64_400 gray
# One clip whose dimensions are not a multiple of the coding block size, since
# cropping is signalled in the SPS and is a common place to be wrong.
gen odd    "testsrc2=size=50x34:rate=25"                      6 50x34_420  yuv420p
# The held frame: `loop` repeats source frame 0 for the whole clip, so every
# picture is byte-identical to the first while still carrying real detail.
gen static "testsrc2=size=64x64:rate=25,loop=loop=-1:size=1:start=0" 8 64x64_420 yuv420p
# The scene cut: 51 frames of one source, then 45 of a structurally
# unrelated one, spliced with no transition. `trim` takes the head of each
# and `setpts` restarts the timestamps so `concat` joins them cleanly.
gen cut    "testsrc2=size=64x64:rate=25,trim=end_frame=51,setpts=PTS-STARTPTS[a];mandelbrot=size=64x64:rate=25,trim=end_frame=45,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0" 96 64x64_420 yuv420p
# The fade: every picture is the one before it at a lower luma gain,
# `Y * (1 - N/16)` over twelve frames, chroma untouched. Nothing above
# changes brightness between pictures, so weighted prediction — a gain and
# an offset applied to the reference before it predicts — had no clip on
# which it could win, and a gate row for it would have proved the syntax
# and nothing else. This is the clip where a picture predicted from an
# unweighted reference always carries residual and one predicted from a
# scaled reference need not.
#
# Its left half is testsrc2 and its right half a flat grey, deliberately:
# the four 32x32 coding tree blocks of the 64x64 clips above all have about
# the same luma variance, so a zero-mean per-block quantiser offset rounds
# to zero on every one of them and adaptive quantisation moved nothing on
# this corpus except the odd-sized clip. Two busy blocks beside two flat
# ones is the smallest picture on which it has something to move.
gen fade   "testsrc2=size=32x64:rate=25,format=yuv420p[a];color=c=0x808080:size=32x64:rate=25,format=yuv420p[b];[a][b]hstack=inputs=2,geq=lum='p(X,Y)*(1-N/16)':cb='p(X,Y)':cr='p(X,Y)'" 12 64x64_420 yuv420p

# The gain-and-offset fade: the fade above with an offset as well as a gain,
# luma of picture N `p * (1 - N/16) - 3N` (clipped at 0), chroma untouched.
# The fade above is a pure gain, so the weighting fitted to it carries offsets
# that round to zero, and a writer spelling every weighting offset with its
# sign flipped left 14 of the 25 weighted-prediction cells of the gate green
# (2026-09-14). Here the fit has an offset to carry. Its format token carries
# its depth, `420p8`, so the untagged rows skip it the way they skip the deep
# clips and only rows tagged `@wpoff` visit it.
gen wpoff  "testsrc2=size=32x64:rate=25,format=yuv420p[a];color=c=0x808080:size=32x64:rate=25,format=yuv420p[b];[a][b]hstack=inputs=2,geq=lum='max(0,p(X,Y)*(1-N/16)-3*N)':cb='p(X,Y)':cr='p(X,Y)'" 12 64x64_420p8 yuv420p

# Deep samples. The format token grows a depth suffix — `420p10` — which
# verify_encode.sh splits into `--format 420 --depth 10` and maps to
# ffmpeg's `yuv420p10le` for the CROSS decode; a token without a suffix is
# 8-bit, as every clip above is. Little-endian 16-bit planar throughout,
# the layout the decoders emit and the encoders take.
#
# The content is NOT an 8-bit picture shifted up. `testsrc2` is drawn at
# 8 bits and `-pix_fmt yuv420p10le` alone would scale it, leaving the low
# two bits of every sample zero — and a depth bug that only touched those
# bits (a quantiser shift short by two, a clip at 255 << 2) would then be
# invisible to the whole gate. So a `geq` stage adds two bits of noise per
# sample AFTER the conversion, at the deep format, and a probe of the
# result shows every low-bit pattern present. Four bits at 12.
deep() { # name source frames geom fmt depth
  noise=$(( 1 << ($6 - 8) ))
  gen "$1" "$2,format=yuv${5}p${6}le,geq=lum='p(X,Y)+floor(random(0)*$noise)':cb='p(X,Y)+floor(random(1)*$noise)':cr='p(X,Y)+floor(random(2)*$noise)'" "$3" "${4}_${5}p${6}" "yuv${5}p${6}le"
}
deep detail10 "testsrc2=size=64x64:rate=25"  8 64x64 420 10
deep motion10 "testsrc=size=64x64:rate=25"  12 64x64 420 10
deep detail10 "testsrc2=size=64x64:rate=25"  8 64x64 422 10
deep detail10 "testsrc2=size=64x64:rate=25"  8 64x64 444 10
deep detail12 "testsrc2=size=64x64:rate=25"  8 64x64 420 12

# The one clip larger than 64x64, for the H.265 coding quadtree. Every
# clip above is four 32x32 coding tree blocks (twelve 16x16 ones on the
# odd clip), so a split decision there sees at most four CTBs of one
# content each, and a probe of what splitting buys cannot tell a win from
# the clip's one texture. This is forty CTBs of four unrelated contents in
# quarters — moving detail (testsrc2), a zooming fractal (mandelbrot),
# smooth drifting gradients, and static bars with hard vertical edges —
# so one picture holds regions where a whole 32x32 unit is right beside
# regions where only 8x8 units are.
#
# 256x160 rather than, say, 256x144: the encoder picks the CTB size that
# pads least, larger on a tie, and 144 is a whole number of 16s but not
# of 32s — at 256x144 it would choose 16x16 CTBs, where the quadtree can
# split only once.
#
# VISITED ONLY BY ROWS THAT NAME IT (`@big`). Its format token spells the
# depth, `420p8`, though it is the 8-bit 4:2:0 every token without a
# suffix means: a token with a depth suffix is what verify_encode.sh and
# identity_encode.sh skip for every row without an `@` (the deep clips'
# rule), so this clip's arrival changed no existing row's cost — in every
# checkout of those scripts, including ones older than the clip.
gen big    "testsrc2=size=128x80:rate=25,format=yuv420p[a];mandelbrot=size=128x80:rate=25,format=yuv420p[b];gradients=size=128x80:rate=25:c0=0x2050a0:c1=0xe0b040:x0=0:y0=0:x1=127:y1=79:nb_colors=2:seed=1:speed=0.01:type=linear,format=yuv420p[c];smptehdbars=size=128x80:rate=25,format=yuv420p[d];[a][b]hstack=inputs=2[top];[c][d]hstack=inputs=2[bot];[top][bot]vstack=inputs=2" 16 256x160_420p8 yuv420p

# The partial-CTB clip. Under the coding quadtree the CTB is 32x32 and the
# coded picture the smallest legal size, so a picture that is not whole
# CTBs ends in partial ones whose splits the reader infers. 88x44 codes as
# 88x48: a right column of CTBs 24 wide (a 32 node crossing the edge, its
# right 16 children crossing again) and a bottom row 16 high (a crossing 32
# whose lower children lie outside) behind a conformance window of 4 rows —
# the remainder 1280x720 and 3840x2160 leave at the bottom. It is the one
# partial-CTB clip: the odd clip (50x34) is below 64 both ways, where the
# encoder keeps whole CTBs (`Geometry::new`). 88x44 is 64 or more one way,
# which is enough. testsrc2 moves, so inter pictures split at the edges too.
#
# VISITED ONLY BY ROWS THAT NAME IT (`@edge`): the `420p8` depth token keeps
# every untagged row off it, as on the big clip.
gen edge   "testsrc2=size=88x44:rate=25" 12 88x44_420p8 yuv420p

# The interlaced clip. Every clip above is progressive — each frame one
# instant — so an interlaced encode of them has fields that agree and a
# frame/field decision with nothing to decide. This one is 16 progressive
# frames at 50 per second woven into 8 interlaced ones (`tinterlace`
# interleave: the top field from one instant, the bottom from the next), so
# its fields really are 20 ms apart. Its left half moves and its right half
# is one picture held (`loop`), so the same frame holds a region where the
# two fields disagree (field coding pays) beside one where they are the
# same picture (frame coding pays) — which is what a per-picture and a
# per-macroblock-pair decision need to have something to choose between.
#
# The left half scrolls (`scroll`, 6% of its width per source frame, about
# three samples between a frame's two fields) because testsrc2 alone moves
# too little to comb: its neighbouring rows still differ less than rows of
# one field, and the encoder's PAFF screen offers field pictures only to a
# combed frame — so on that source the PAFF rows coded frame pictures and
# nothing else, and a broken field decision would have passed them. This
# one measures 1.5-1.8 (frame / field vertical SAD, every frame), and its
# PAFF rows code field pictures as well as frame pictures.
#
# 96x96 rather than 64x64 so an MBAFF frame has eighteen macroblock pairs,
# enough for pairs of both kinds to sit beside each other. The `p8` depth
# suffix keeps every untagged row off it: only `@interlace` rows visit it.
gen interlace "testsrc2=size=48x96:rate=50,scroll=horizontal=0.06[a];testsrc2=size=48x96:rate=50,loop=loop=-1:size=1:start=0[b];[a][b]hstack=inputs=2,tinterlace=mode=interleave_top" 8 96x96_420p8 yuv420p

# The same interlaced clip at 10 bits, through `deep` (two bits of noise
# below the up-shift, like every deep clip). A PAFF row on the progressive
# @p10 clips codes frame pictures only; this is where 10-bit PAFF has field
# pictures to choose. Its name carries `ilace`, one of verify_encode.sh's
# EXCLUSIVE_TOKENS, so only `@ilace10` rows visit it: its `420p10` token
# alone would have put it under every `@p10` row.
deep ilace10 "testsrc2=size=48x96:rate=50,scroll=horizontal=0.06[a];testsrc2=size=48x96:rate=50,loop=loop=-1:size=1:start=0[b];[a][b]hstack=inputs=2,tinterlace=mode=interleave_top" 8 96x96 420 10
