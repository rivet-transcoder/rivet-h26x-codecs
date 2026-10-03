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
# Every clip comes from synth_source.py (deterministic synthetic recipes,
# described there); nothing here needs any other program.
#
# Usage: make_encode_sources.sh [dir]
set -e
SYNTH=$(cd "$(dirname "$0")" && pwd)/synth_source.py
cd "${1:-$(dirname "$0")}"

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
gen() { # name recipe frames <W>x<H>_<fmt> [synth_source options...]
  local name=$1 recipe=$2 frames=$3 tok=$4; shift 4
  out="src_${name}_${tok}.yuv"
  [ -f "$out" ] && { echo "have $out"; return; }
  python "$SYNTH" "$recipe" --size "${tok%%_*}" --frames "$frames" --format "${tok#*_}" "$@" "$out"
  echo "made $out ($(stat -c %s "$out") bytes)"
}

gen grad   grad    8 64x64_420
gen detail detail  8 64x64_420
gen motion motion 12 64x64_420
gen detail detail  8 64x64_422
gen detail detail  8 64x64_444
gen detail detail  8 64x64_400
# One clip whose dimensions are not a multiple of the coding block size, since
# cropping is signalled in the SPS and is a common place to be wrong.
gen odd    detail  6 50x34_420
# The held frame: `static` repeats frame 0 of `detail` for the whole clip, so every
# picture is byte-identical to the first while still carrying real detail.
gen static static  8 64x64_420
# The scene cut: 51 frames of one source, then 45 of a structurally
# unrelated one (`detail`, then `zoom` from its first frame), spliced with
# no transition.
# With this corpus (2026-10) it found a fault the previous (lavfi) one
# hid: the H.265 lookahead priced the pictures past the cut at the old
# scene's bits per cost, saw them as nearly free, and gave the pictures
# before the cut up to four times their share — hevc-abr-la-64k / -96k
# landed the 3-GOP window around it at 1.20x / 1.24x, outside 4b's band.
# The window now stops at a cut (encode::h265 SCENE_CUT_RATIO): 1.12x /
# 1.16x.
gen cut    cut    96 64x64_420
# The fade: every picture is the one before it at a lower luma gain,
# `Y * (1 - N/16)` over twelve frames, chroma untouched. Nothing above
# changes brightness between pictures, so weighted prediction — a gain and
# an offset applied to the reference before it predicts — had no clip on
# which it could win, and a gate row for it would have proved the syntax
# and nothing else. This is the clip where a picture predicted from an
# unweighted reference always carries residual and one predicted from a
# scaled reference need not.
#
# Its left half is `detail` and its right half a flat grey, deliberately:
# the four 32x32 coding tree blocks of the 64x64 clips above all have about
# the same luma variance, so a zero-mean per-block quantiser offset rounds
# to zero on every one of them and adaptive quantisation moved nothing on
# this corpus except the odd-sized clip. Two busy blocks beside two flat
# ones is the smallest picture on which it has something to move.
gen fade   hfade  12 64x64_420

# The gain-and-offset fade: the fade above with an offset as well as a gain,
# luma of picture N `p * (1 - N/16) - 3N` (clipped at 0), chroma untouched.
# The fade above is a pure gain, so the weighting fitted to it carries offsets
# that round to zero, and a writer spelling every weighting offset with its
# sign flipped left 14 of the 25 weighted-prediction cells of the gate green
# (2026-09-14). Here the fit has an offset to carry. Its format token carries
# its depth, `420p8`, so the untagged rows skip it the way they skip the deep
# clips and only rows tagged `@wpoff` visit it.
gen wpoff  wpoff  12 64x64_420p8

# The settling shot: 24 frames of `detail` scrolling sideways, then its frame 24 held
# for the remaining 72 — motion that stops, a pause, a slate after a pan.
# Ninety-six frames so the rate gate's sustained-spend property (4b) sees
# twelve GOPs at the gate's usual length and forty-eight at two.
#
# It exists for rate control's insensitivity verdict (encode::rc), which no
# other clip reaches: once the picture holds, every P picture is a skip
# that costs the same bits at any quantiser above the keyframe's, so a
# controller under its target lowers the P quantiser into bits that do not
# answer — the silent walk, the raised picture that stays silent, the
# verdict and its probes. The moving head matters: it lets the P quantiser
# converge high before the hold, so the walk down is a real one rather than
# a first picture already at the floor, and short GOPs keep the keyframes
# able to spend what the held P pictures cannot. `p8` keeps untagged rows
# off it; only `@settle` rows visit it.
# With this corpus (2026-10) the h264-verdict-g2-256k row releases its
# verdict late in the hold and repays the budget the hold saved: 1.27x over
# three of its two-picture GOPs, 1.12x over the 24 pictures 4b's thresholds
# were measured on, which is the window 4b now takes (verify_encode.sh 4b).
gen settle settle 96 64x64_420p8

# Deep samples. The format token grows a depth suffix — `420p10` — which
# verify_encode.sh splits into `--format 420 --depth 10`; a token without a
# suffix is 8-bit, as every clip above is. Little-endian 16-bit planar throughout,
# the layout the decoders emit and the encoders take.
#
# The content is NOT an 8-bit picture shifted up. The recipes are drawn in
# 8-bit units and scaling alone would leave the low two bits of every
# sample zero — and a depth bug that only touched those
# bits (a quantiser shift short by two, a clip at 255 << 2) would then be
# invisible to the whole gate. So `--noise` adds two bits of noise per
# sample AFTER the conversion, at the deep format, and a probe of the
# result shows every low-bit pattern present. Four bits at 12.
deep() { # name recipe frames geom fmt depth
  gen "$1" "$2" "$3" "${4}_${5}p${6}" --noise $(( $6 - 8 ))
}
deep detail10 detail  8 64x64 420 10
deep motion10 motion 12 64x64 420 10
deep detail10 detail  8 64x64 422 10
deep detail10 detail  8 64x64 444 10
deep detail12 detail  8 64x64 420 12

# The one clip larger than 64x64, for the H.265 coding quadtree. Every
# clip above is four 32x32 coding tree blocks (twelve 16x16 ones on the
# odd clip), so a split decision there sees at most four CTBs of one
# content each, and a probe of what splitting buys cannot tell a win from
# the clip's one texture. This is forty CTBs of four unrelated contents in
# quarters — moving detail (`detail`), a zooming fractal (`zoom`),
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
gen big    big    16 256x160_420p8

# The partial-CTB clip. Under the coding quadtree the CTB is 32x32 and the
# coded picture the smallest legal size, so a picture that is not whole
# CTBs ends in partial ones whose splits the reader infers. 88x44 codes as
# 88x48: a right column of CTBs 24 wide (a 32 node crossing the edge, its
# right 16 children crossing again) and a bottom row 16 high (a crossing 32
# whose lower children lie outside) behind a conformance window of 4 rows —
# the remainder 1280x720 and 3840x2160 leave at the bottom. It is the one
# partial-CTB clip: the odd clip (50x34) is below 64 both ways, where the
# encoder keeps whole CTBs (`Geometry::new`). 88x44 is 64 or more one way,
# which is enough. `detail` moves, so inter pictures split at the edges too.
#
# VISITED ONLY BY ROWS THAT NAME IT (`@edge`): the `420p8` depth token keeps
# every untagged row off it, as on the big clip.
gen edge   detail 12 88x44_420p8

# The interlaced clip. Every clip above is progressive — each frame one
# instant — so an interlaced encode of them has fields that agree and a
# frame/field decision with nothing to decide. This one is 16 progressive
# frames at 50 per second woven into 8 interlaced ones (the top field from
# one instant, the bottom from the next), so its fields really are 20 ms
# apart. Its left half moves and its right half is one picture held, so the
# same frame holds a region where the
# two fields disagree (field coding pays) beside one where they are the
# same picture (frame coding pays) — which is what a per-picture and a
# per-macroblock-pair decision need to have something to choose between.
#
# The left half is colour bars scrolling sideways (7% of its width per
# source frame, about three samples between a frame's two fields): hard
# vertical edges moving across the field interval are what combs. Content
# that moves too little does not — its neighbouring rows still differ less
# than rows of one field, and the encoder's PAFF screen offers field
# pictures only to a combed frame, so on such a source the PAFF rows code
# frame pictures and nothing else, and a broken field decision would pass
# them. This one measures 1.6-1.9 (frame / field vertical SAD, every
# frame).
#
# 96x96 rather than 64x64 so an MBAFF frame has eighteen macroblock pairs,
# enough for pairs of both kinds to sit beside each other. The `p8` depth
# suffix keeps every untagged row off it: only `@interlace` rows visit it.
gen interlace interlace 8 96x96_420p8

# The same interlaced clip at 10 bits, through `deep` (two bits of noise
# below the up-shift, like every deep clip). A PAFF row on the progressive
# @p10 clips codes frame pictures only; this is where 10-bit PAFF has field
# pictures to choose. Its name carries `ilace`, one of verify_encode.sh's
# EXCLUSIVE_TOKENS, so only `@ilace10` rows visit it: its `420p10` token
# alone would have put it under every `@p10` row.
deep ilace10 interlace 8 96x96 420 10

# The gain-and-offset fade at 10 bits: the wpoff fade above with its offset
# at the depth, luma of picture N `p * (1 - N/16) - 12N` (clipped at 0),
# chroma untouched, and two bits of noise on every plane after the
# conversion, as `deep` adds them — written out rather than through `deep`
# so the fade and the noise are one expression. The corpus had no deep fade,
# so H.265's weighted rows at 10 bits proved the syntax and little else.
# Its name carries `fdeep`, one of the EXCLUSIVE_TOKENS, so only `@fdeep10`
# rows visit it: its `420p10` token alone would have put it under every
# `@p10` row. md5 ea58501aa75f495c99f7c956bdffd261.
gen fdeep10 half 12 64x64_420p10 --fade16 --luma-offset-per-frame 12 --noise 2

# A native 10-bit weighted fade. fdeep10 above is 8-bit content scaled up,
# and from QP 26 on it codes like its own 8-bit twin, so it checks the
# 10-bit path's consistency and nothing about 10-bit rate-distortion. This
# one is computed at 10 bits: luma a sinusoidal texture drifting by a
# quarter radian per picture, `512 + 300 sin(X/5 + N/4) cos(Y/7)`, under
# the fdeep10 gain and offset (`* (1 - N/16) - 12N`), chroma two drifting
# waves about 512, and two bits of noise on every plane. Its low bits carry
# the texture's gradients, so truncating it to 8 bits costs luma PSNR at
# QP 26. Its name carries `wsine`, one of the EXCLUSIVE_TOKENS, so only
# `@wsine10` rows visit it. md5 094b6a0f44798be875894ed1b6093503 (sin / cos
# come from the platform's libm, so another machine may differ in a
# sample).
gen wsine10 wsine 12 64x64_420p10 --noise 2
