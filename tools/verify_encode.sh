#!/bin/bash
# verify_encode.sh — what has to hold before an encoder change lands.
#
# The decoders have conformance suites. The encoders cannot: a standard says
# what a decoder must do with a bitstream and leaves an encoder free to choose
# any legal one, so there is no golden output to compare against. What replaces
# it is three properties, two of them exact.
#
#   1. SELF     the bitstream decodes to what the encoder thought it encoded.
#               The encoder dumps its own reconstruction; our decoder must
#               reproduce it byte for byte. A mismatch is a desync between
#               encoder and decoder state and is always a bug. Needs no
#               reference data, and catches the largest class of faults.
#
#   2. CROSS    libavcodec decoding our output produces the same pictures our
#               decoder does. Property 1 is self-consistent and would pass if
#               both of our sides shared a misreading of the standard; this is
#               what makes the bitstream legal rather than merely
#               self-compatible.
#
#   3. QUALITY  PSNR of the reconstruction against the source. The only one of
#               the three that is a measurement rather than a check — so it is
#               REPORTED, never gated, except in lossless mode where it must be
#               infinite and the check becomes exact like the others.
#
#   4. RATE     Only for --bitrate rows, and a different kind of property from
#               the three above: did the encoder achieve the objective it was
#               HANDED, rather than describe correctly what it did? A rate
#               controller that ignores its target produces a perfectly legal
#               stream that passes SELF, passes CROSS and reports a fine PSNR.
#               Nothing above can see it.
#
#               So these rows assert a band: achieved within [0.5x, 2.0x] of
#               target. That is wide, and wide on purpose - the clips here are
#               six to twelve frames, which gives a controller almost no time
#               to converge and lets the opening keyframe dominate. A tighter
#               tolerance would be flaky rather than rigorous, and a flaky row
#               teaches people to re-run it. The tightness is bought back by
#               the targets instead: they sit inside every clip's achievable
#               range but on opposite sides of different clips' natural rates,
#               so some clips must compress harder and others must spend more,
#               and a controller stuck at one quantiser fails on both counts.
#
#               The test that a rate row is worth having: replace the
#               controller with a constant quantiser and it must go red.
#
#   5. BUFFER   Only for --cpb-ms rows. A stream that underflows the coded
#               picture buffer it declares is non-conforming - determined
#               integer arithmetic, not a judgement - so unlike RATE this
#               has a right answer. But neither of our conformance
#               instruments can see it: a decoder is NOT required to check
#               the hypothetical reference decoder and ours does not, and
#               libavcodec decodes a violating stream as happily as any
#               other. So h26xhrd checks it, reading the declaration out of
#               the stream itself rather than being told.
#
#               These rows need a clip with seconds in it. On a six-to-
#               twelve-frame clip the minimum conforming buffer is 33% to
#               62% of the whole stream, so the buffer never completes a
#               fill-and-drain cycle and both branches are vacuous - which
#               is why the row is restricted to src_cut below.
#
#               Its mutation: make the controller ignore the buffer it was
#               given, and the row must go red. So does forbidding the
#               encoder to code a picture twice - at this buffer size the
#               cap alone lands 632 bits short, so the row is carried by
#               the re-code and not merely by the aim.
#
#   6. BOX      Every stream: exactly one VPS / SPS / PPS, byte for byte.
#               SELF and CROSS both read Annex-B, where a parameter set
#               re-sent under the same id replaces the previous one, so an
#               encoder writing a DIFFERENT PPS for its I and P pictures
#               passes both. An MP4 avc1/hvc1 sample entry cannot carry that:
#               the sets are stored once, out of band, and stripped from the
#               samples, so a decoder reading the box holds two under one id
#               and decodes the pictures written under the other one to
#               garbage from their first macroblock. That is how rivet's first
#               H.264 file failed (2026-08-27) with this whole gate green.
#
#               Checked by tools/param_sets.py. Its mutation: put the picture
#               quantiser back into pic_init_qp, and every H.264 row with an
#               I and a P picture must go red.
#
# Usage: verify_encode.sh [encoder] [decoder]
#   H26X_WORK=dir   scratch directory holding the source clips (default: here)
#   JOBS=n          configurations in parallel (default 4)
#
# Safe to run concurrently with itself and with verify.sh: private binary
# copies and a private scratch directory per run, for the reason recorded in
# tools/README.md — a shared copy produces a green run that tested somebody
# else's build.
cd "${H26X_WORK:-$(dirname "$0")}"
ENC=${1:-../release/examples/h26xenc.exe}
DEC=${2:-../release/examples/h26xdec.exe}
[ -f "$ENC" ] || ENC=${ENC%.exe}
[ -f "$DEC" ] || DEC=${DEC%.exe}
# The buffer checker lives beside the encoder it was built with.
HRD=${HRD:-$(dirname "$ENC")/h26xhrd.exe}
[ -f "$HRD" ] || HRD=${HRD%.exe}
FFMPEG=${FFMPEG:-ffmpeg}
# The BOX checker (property 6) lives in the repo, beside this script.
PARAM_SETS=${PARAM_SETS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/param_sets.py}
TAG=$$
OUT=enc_out_$TAG
JOBS=${JOBS:-4}
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT
fail=0

# Source clips: raw planar YUV, named <name>_<W>x<H>_<fmt>.yuv so the geometry
# travels with the file rather than living in this script.
#
# THE CORPUS IS NOT VERSIONED BY GIT, and that has bitten once. This reads
# whatever clips are on disk in the work directory, not whatever the checked
# out commit's make_encode_sources.sh would generate. The generator is
# versioned; its output is not; the two can disagree silently.
#
# So a clean checkout of an older commit can fail on a clip that commit never
# knew about — which is exactly what happened when a clip with a scene cut
# was generated into a shared work directory before the fix for the bug it
# found had been pushed. The commit was not broken; the environment around it
# had moved. If a result surprises you, check `ls src_*.yuv` against the
# generator in your checkout before believing the commit is at fault.
#
# A SECOND WAY TO VERIFY THE WRONG THING: cargo does not always rebuild
# after an edit here. A source change followed by `cargo build` has been
# observed finishing in hundredths of a second with no `Compiling` line,
# leaving the previous binary in place — so the gate then runs against
# code you did not write. Touching the file is not always enough;
# bumping its mtime into the future forces it, and deleting the target
# binary always does. The failure mode is the one this whole file exists
# to hunt: no error, a confident wrong answer, and a run that quietly
# tested something other than the change. If a result is surprising in
# either direction, confirm the binary is newer than the source before
# believing it.
SOURCES=${SOURCES:-$(ls src_*.yuv 2>/dev/null)}
if [ -z "$SOURCES" ]; then
  echo "no source clips (src_*.yuv); nothing to verify" >&2
  exit 2
fi

# A configuration's name may carry an `@substring` suffix, which restricts
# it to sources whose filename contains that substring. Rows are not all
# meaningful on all clips and pretending otherwise costs either coverage or
# a red cell: the buffer rows below need a clip with seconds in it, the SAO
# rows are carried by two clips and merely pass on the rest, and the rate
# rows had to be calibrated per chroma format. Saying so in the row is
# cheaper than a table somewhere else that goes stale.
#
# Configurations. Each is a name and the encoder flags for it. The list starts
# at the simplest thing that can be legal and adds one axis at a time, because
# when several are red at once the simplest one names the bug.
#
# Each axis appears in a CAVLC form as well as a CABAC one. That is not
# redundancy: it lets inter prediction be verified without waiting for CABAC
# slice writing and vice versa, so two people can make progress against this
# gate at the same time without one of them being blocked behind the other.
#
# The high-QP rows exist because a fixed quantiser hides a whole class of bug.
# Coding H.264's chroma planes at the luma quantiser — plainly wrong — passed
# every row of this gate, because the chroma QP mapping is the identity up to
# 29 and every row lived below that. At QP 40 the same mutation fails SELF on
# both entropy coders at once. Any table the codec indexes by QP has the same
# shape, so one row per codec sits high enough to leave the identity region.
#
# It has now happened twice, which is what makes it a rule rather than an
# anecdote. When H.265's inter path gained its first format-dependent chroma
# QP derivation, telling that mapping "4:2:0" whatever the real format is was
# invisible at QP 26 and failed 4:2:2 and 4:4:4 at once at QP 40. Intra and
# inter reach the table through different code, so a high-QP row for one buys
# nothing for the other: every combination of codec, entropy coder and
# prediction mode that indexes a QP table needs its own row above 29.
#
# The third instance sharpened it again, and this time about the clips
# rather than the configurations. Coding 4:4:4's luma-style chroma planes
# at the luma quantiser under the 8x8 transform is a *literal no-op* at QP
# 26 — the two expressions are the same integer, the bitstream is byte for
# byte what the unmutated encoder writes — and fatal at QP 40. But it lives
# inside a 4:4:4 branch, so a high-QP row catches it only because a 4:4:4
# clip is in the source list. The quantiser axis lives here; the chroma
# format axis lives in the sources; a QP table reached under one format
# alone needs both, and a row above 29 is necessary rather than
# sufficient.
# Nothing below this line may be a comment. CONFIGS is a quoted string, so
# a leading # is data: the reader takes the whole line as a configuration
# name with no flags and runs the encoder's defaults under it, which
# passes, tests nothing, and inflates the count. Seven such lines once
# added forty-nine cells that all quietly re-ran the same default
# configuration. Notes go above.
#
# The SAO rows are carried by two of the seven clips, and it is worth
# knowing which. On the gradient and odd clips the decision selects "off"
# for every component of every coding tree block — correctly, there is
# nothing there for SAO to shape — so those cells prove the syntax and
# prove nothing about the filter. detail and motion are where the filter
# actually runs. A row is only as strong as the clips that make it do
# something.
#
# The third instance of that, and the sharpest, is about a shape rather
# than a filter. H.264's sub-16x16 partitions were built against a corpus
# that could barely exercise them: across every clip here before the one
# with a hard cut, the whole gate produced FOUR Inter8x8 macroblocks. Add
# a clip whose halves genuinely move differently and the same encoder at
# the same quantiser produces 204 Inter16x8, 74 Inter8x16 and 164
# Inter8x8. Nothing about the configuration list changed; a cut is simply
# a lot of macroblocks that one vector cannot describe.
#
# So this is now three for three - SAO, lossless, and the partition
# shapes - and the general form is worth stating: a configuration row
# turns a code path ON, and only the source decides whether anything
# TAKES it. Where a feature is chosen per block rather than set per
# stream, a row proves the syntax and a clip proves the feature. Reach
# for the corpus before reaching for the configuration list.
CONFIGS=${CONFIGS:-"
lossless-intra|--codec h264 --lossless --gop 0
cqp-intra|--codec h264 --qp 26 --gop 0
cqp-ip|--codec h264 --qp 26 --gop 8
cqp-ipb|--codec h264 --qp 26 --gop 8 --bframes 2
cqp40-ip|--codec h264 --qp 40 --gop 8
cavlc-intra|--codec h264 --qp 26 --gop 0 --cavlc
cavlc-ip|--codec h264 --qp 26 --gop 8 --cavlc
cavlc-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc
cavlc40-intra|--codec h264 --qp 40 --gop 0 --cavlc
cqp-t8x8|--codec h264 --qp 26 --gop 8 --t8x8
cqp40-t8x8|--codec h264 --qp 40 --gop 8 --t8x8
cavlc-t8x8|--codec h264 --qp 26 --gop 8 --cavlc --t8x8
cavlc40-t8x8|--codec h264 --qp 40 --gop 8 --cavlc --t8x8
cqp-subparts|--codec h264 --qp 26 --gop 8 --subparts
cqp40-subparts|--codec h264 --qp 40 --gop 8 --subparts
cavlc-subparts|--codec h264 --qp 26 --gop 8 --cavlc --subparts
cavlc40-subparts|--codec h264 --qp 40 --gop 8 --cavlc --subparts
hevc-lossless-intra|--codec h265 --lossless --gop 0
hevc-lossless-ip|--codec h265 --lossless --gop 8
hevc-lossless-ipb|--codec h265 --lossless --gop 8 --bframes 2
hevc-cqp-intra|--codec h265 --qp 26 --gop 0
hevc-cqp-ip|--codec h265 --qp 26 --gop 8
hevc-cqp40-intra|--codec h265 --qp 40 --gop 0
hevc-cqp40-ip|--codec h265 --qp 40 --gop 8
hevc-cqp-ipb|--codec h265 --qp 26 --gop 8 --bframes 2
hevc-cqp40-sao-intra|--codec h265 --qp 40 --gop 0 --sao
hevc-cqp40-sao-ip|--codec h265 --qp 40 --gop 8 --sao
hevc-abr-64k|--codec h265 --bitrate 64000 --gop 8
hevc-abr-96k|--codec h265 --bitrate 96000 --gop 8
abr-64k|--codec h264 --bitrate 64000 --gop 8
abr-128k|--codec h264 --bitrate 128000 --gop 8
hevc-vbv-125@src_cut|--codec h265 --bitrate 64000 --cpb-ms 125 --gop 8
"}

one() {
  src=$1; name=$2; flags=$3
  base=$(basename "$src" .yuv)
  geom=$(echo "$base" | sed -n 's/.*_\([0-9]\+x[0-9]\+\)_.*/\1/p')
  fmt=$(echo "$base" | sed -n 's/.*_[0-9]\+x[0-9]\+_\(.*\)/\1/p')
  tag="$base/$name"
  ext=h264; case "$flags" in *"--codec h265"*) ext=h265 ;; esac
  bs="$OUT/$base.$name.$ext"
  rec="$OUT/$base.$name.rec.yuv"

  # The encoder writes the bitstream and, separately, the reconstruction it
  # believes that bitstream carries.
  if ! "$ENC" --input "$src" --size "$geom" --format "$fmt" \
       $flags --output "$bs" --recon "$rec" > "$OUT/$base.$name.enc.log" 2>&1; then
    echo "ENCODE-FAIL $tag: $(tail -1 "$OUT/$base.$name.enc.log" | head -c 100)"
    return 1
  fi

  # 1. SELF.
  ours="$OUT/$base.$name.ours.yuv"
  # H.264 4:0:0: ask the decoder for the samples the codec produced rather
  # than the grey-chroma padding it adds to match libavcodec yuv420p, since
  # the CROSS check below asks ffmpeg for gray.
  if ! H26XDEC_NO_CHROMA_PAD=1 "$DEC" "$bs" "$ours" > /dev/null 2> "$OUT/$base.$name.dec.log"; then
    echo "SELF-FAIL   $tag: our decoder rejected our bitstream: $(tail -1 "$OUT/$base.$name.dec.log" | head -c 80)"
    return 1
  fi
  if ! cmp -s "$rec" "$ours"; then
    echo "SELF-FAIL   $tag: decoded output differs from the encoder's own reconstruction"
    return 1
  fi

  # 2. CROSS.
  theirs="$OUT/$base.$name.ff.yuv"
  # 4:0:0 needs extractplanes, not -pix_fmt gray. libavcodec emits H.264
  # monochrome as yuv420p with grey chroma, so asking swscale for gray makes
  # it convert — and it treats yuv420p as limited range and gray as full, so
  # every luma sample comes out expanded: 68 becomes 61. That is an artefact
  # of the comparison, not a difference in the bitstream, and it cost a false
  # CROSS failure to find.
  ffargs="-pix_fmt $(ffpix "$fmt")"
  case "$fmt" in 400|gray) ffargs="-vf extractplanes=y -pix_fmt gray" ;; esac
  if ! "$FFMPEG" -v error -y -i "$bs" -f rawvideo $ffargs "$theirs" \
       > "$OUT/$base.$name.ff.log" 2>&1; then
    echo "CROSS-FAIL  $tag: libavcodec rejected our bitstream: $(tail -1 "$OUT/$base.$name.ff.log" | head -c 80)"
    return 1
  fi
  if ! cmp -s "$ours" "$theirs"; then
    echo "CROSS-FAIL  $tag: libavcodec decodes our bitstream differently than we do"
    return 1
  fi

  # 6. BOX. One parameter set of each kind for the whole stream, so the
  # stream can be put in an MP4 with the sets out of band. Neither decoder
  # above can see this: both read Annex-B, where a re-sent set replaces
  # the old one.
  if ! python "$PARAM_SETS" "$bs" > "$OUT/$base.$name.ps.log" 2>&1; then
    echo "PS-FAIL     $tag: $(tail -1 "$OUT/$base.$name.ps.log" | head -c 100)"
    return 1
  fi

  # 4. RATE. Only where a target was given. The encoder reports what it
  # achieved rather than this script recomputing it: it knows the frame
  # count, the frame rate and the exact bytes emitted, and a second
  # implementation of that division here is a second thing that can be
  # wrong.
  case "$flags" in
    *--bitrate*)
      ratio=$(sed -n 's/.*ratio \([0-9.]*\).*/\1/p' "$OUT/$base.$name.enc.log" | tail -1)
      if [ -z "$ratio" ]; then
        echo "RATE-FAIL   $tag: the encoder reported no achieved rate"
        return 1
      fi
      if ! awk -v r="$ratio" 'BEGIN { exit !(r >= 0.5 && r <= 2.0) }'; then
        echo "RATE-FAIL   $tag: achieved $(printf '%.2f' "$ratio")x of target, outside [0.50, 2.00]"
        return 1
      fi
      ;;
  esac

  # 5. BUFFER. Only where a buffer was declared. h26xhrd reads the
  # declaration out of the stream - rate and size from the sequence
  # parameter set's VUI, the removal interval from the frame rate beside
  # it, the initial delay from the buffering period SEI - so nothing here
  # tells it what to expect.
  case "$flags" in
    *--cpb-ms*)
      if ! out=$("$HRD" "$bs" 2>&1); then
        echo "HRD-FAIL    $tag: $(echo "$out" | tail -1 | head -c 100)"
        return 1
      fi
      ;;
  esac

  # 3. QUALITY. Gated only when the configuration claims to be lossless.
  psnr=$(psnr_of "$src" "$rec")
  size=$(stat -c %s "$bs")
  case "$flags" in
    *--lossless*)
      if ! cmp -s "$src" "$rec"; then
        echo "LOSSLESS-FAIL $tag: reconstruction differs from the source"
        return 1
      fi
      echo "PASS        $tag (lossless, exact, $size bytes)" ;;
    *--bitrate*)
      echo "PASS        $tag ($size bytes, PSNR $psnr dB, rate $(printf '%.2f' "$ratio")x)" ;;
    *)
      echo "PASS        $tag ($size bytes, PSNR $psnr dB)" ;;
  esac
  return 0
}

# Planar chroma format name to the ffmpeg pixel format that matches how this
# decoder packs a picture.
ffpix() {
  case "$1" in
    400|gray) echo gray ;;
    422) echo yuv422p ;;
    444) echo yuv444p ;;
    *) echo yuv420p ;;
  esac
}

psnr_of() {
  python - "$1" "$2" <<'PY'
import sys, math
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
n = min(len(a), len(b))
if n == 0:
    print("n/a"); raise SystemExit
se = 0
for i in range(0, n, max(1, n // 200000) if n > 200000 else 1):
    d = a[i] - b[i]
    se += d * d
cnt = len(range(0, n, max(1, n // 200000) if n > 200000 else 1))
mse = se / cnt
print("inf" if mse == 0 else f"{10 * math.log10(255 * 255 / mse):.2f}")
PY
}
export -f one ffpix psnr_of
export ENC DEC HRD FFMPEG OUT PARAM_SETS

echo "== encode verification =="
results="$OUT/results.txt"
: > "$results"
for src in $SOURCES; do
  echo "$CONFIGS" | while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    # A configuration with no flags is always a mistake — most often a
    # comment line, which is data inside this quoted string rather than a
    # comment. Refuse it instead of running the defaults under its name.
    case "$name" in
      *' '*|'#'*)
        echo "verify_encode.sh: not a configuration: $name" >&2
        exit 2 ;;
    esac
    # An `@substring` suffix restricts the row to matching sources.
    case "$name" in
      *@*)
        pat=${name##*@}
        case "$src" in
          *"$pat"*) name=${name%@*} ;;
          *) continue ;;
        esac
        ;;
    esac
    echo "$src|$name|$flags"
  done
done | xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r s n f <<< "{}"; one "$s" "$n" "$f"' \
     | sort | tee "$results"

pass=$(grep -c '^PASS' "$results")
# Every failure prefix `one` can print must appear here. A row whose
# failure prefix is missing from this pattern reports its failure and is
# then counted as green - which is how the RATE rows first shipped, caught
# only by running the mutation they exist to catch.
bad=$(grep -cE '^(ENCODE|SELF|CROSS|LOSSLESS|RATE|HRD|PS)-FAIL' "$results")
echo
echo "encode: $pass passed, $bad failed"
[ "$bad" = 0 ] || fail=1

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
