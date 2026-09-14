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
#   7. SPEED    Reported, never gated, like QUALITY: every PASS line carries
#               the encoder's wall time for the cell and its frames per
#               second, and the run ends with the total over all cells. A
#               kernel that made a decision path ten times slower passes
#               every property above; this is where it becomes visible.
#               Wall time under JOBS parallel cells on a shared machine is
#               a coarse instrument — it catches an order of magnitude, not
#               ten percent; for that use tools/ab_enc.py. H26X_SPEED_TABLE
#               names a file to append one tab-separated row per cell to
#               (clip, configuration, bytes, PSNR, seconds, frames/s), so
#               two runs can be set side by side.
#
#   8. VUI      Only for --color rows. The colour description is three
#               H.273 code points and a range flag in the SPS VUI, and
#               nothing above can see whether they are there: SELF and
#               CROSS compare samples, which do not change when the VUI
#               says BT.2020 PQ instead of nothing, and the crate's own
#               parser is the inverse of its own writer, so a shared
#               misreading of E.1.1 / E.2.1 round-trips cleanly. What
#               settles it is a third reader: tools/vui_probe.py asks
#               ffprobe, which reports the VUI as names, and the row is
#               green only when all four fields name exactly the codes
#               the encoder was handed. A player that shows BT.2020 PQ as
#               BT.709 is what this row exists to prevent.
#
#               Its mutation: write the transfer code into the primaries
#               field, and every --color row must go red naming the field.
#               Which is why the H.264 rows carry three DIFFERENT codes
#               (sRGB-on-709 full range, 1:13:6; P3 / ST 428 / 601,
#               12:17:6) rather than the BT.709 triple 1:1:1 they first
#               had: under that mutation 1:1:1 writes 1 where 1 belonged
#               and stayed green in ten cells at once. The HDR rows have
#               primaries equal to matrix (9:16:9, 9:18:9) because that
#               is what HDR10 and HLG are; the H.264 rows are where a
#               primaries/matrix swap shows.
#
#               The same property covers the chroma siting
#               (`chroma_sample_loc_type`, ffprobe's chroma_location) on
#               the rows that write one — and only those: for an absent
#               field libavcodec reports the type 0 the standard infers
#               ("left"), never "unspecified", so a written 0 is invisible
#               to it and absence is not checkable here (the parser test
#               holds "unasked, unwritten"). And it reports a siting for
#               4:2:0 only — 4:2:2 / 4:4:4 read "unspecified" whatever the
#               VUI says (E.2.1 wants the flag 0 there, and the encoder
#               refuses a siting off 4:2:0 by name) — so the siting rows
#               name 4:2:0 clips and carry non-zero codes: the H.264
#               `@src_cut` row 1 ("center", the 2x2 box siting), the H.265
#               `@420p10` row 2 ("topleft", BT.2100's 4:2:0 siting). Its
#               mutation: stub the writer to the zero flag, and both rows
#               must go red naming chroma_location while every other
#               --color row stays green.
#
#               The same property covers the HDR10 static-metadata SEIs
#               (mastering display colour volume, content light level)
#               on the rows that write them: the probe is handed the
#               values and asks ffprobe's frame side data. Its mutation:
#               swap the red and green primaries in the writer, and the
#               row must go red naming red_x. The H.264 SEI row is a
#               buffer row so the two SEIs travel beside a buffering
#               period and a pic timing, where an ordering or framing
#               slip would show — with a 250 ms buffer, not the 125 ms
#               of the other buffer rows: at 125 ms picture 0 was already
#               carried by the re-code (632 bits short on the cap alone),
#               and the 45 bytes of SEI NAL put it past what the re-code
#               can recover, so the encoder refuses by name ("picture 0
#               needs 3432 bits and the declared buffer affords 2893").
#
# Usage: verify_encode.sh [encoder] [decoder]
#   H26X_WORK=dir   scratch directory holding the source clips (default: here)
#   JOBS=n          configurations in parallel (default 4)
#   H26X_SPEED_TABLE=file  append a per-cell speed row here (see 7)
#
# Safe to run concurrently with itself and with verify.sh: private binary
# copies and a private scratch directory per run, for the reason recorded in
# tools/README.md — a shared copy produces a green run that tested somebody
# else's build.
# Where this script lives, resolved BEFORE the cd below: a relative
# invocation (`bash tools/verify_encode.sh`) resolves to nothing afterwards,
# and MSYS turned that nothing into `C:\Program Files\Git\param_sets.py` —
# every cell red on the BOX check with a message that named the wrong bug.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${H26X_WORK:-$SCRIPT_DIR}"
ENC=${1:-../release/examples/h26xenc.exe}
DEC=${2:-../release/examples/h26xdec.exe}
[ -f "$ENC" ] || ENC=${ENC%.exe}
[ -f "$DEC" ] || DEC=${DEC%.exe}
# The buffer checker lives beside the encoder it was built with.
HRD=${HRD:-$(dirname "$ENC")/h26xhrd.exe}
[ -f "$HRD" ] || HRD=${HRD%.exe}
FFMPEG=${FFMPEG:-ffmpeg}
# The BOX checker (property 6) lives in the repo, beside this script.
PARAM_SETS=${PARAM_SETS:-$SCRIPT_DIR/param_sets.py}
# The VUI probe (property 8) too; it asks ffprobe, which lives beside the
# ffmpeg named above (the same suffix, `.exe` or none), or is on the PATH
# when ffmpeg is.
VUI_PROBE=${VUI_PROBE:-$SCRIPT_DIR/vui_probe.py}
case "$FFMPEG" in
  */*) FFPROBE=${FFPROBE:-$(dirname "$FFMPEG")/ffprobe${FFMPEG##*/ffmpeg}} ;;
  *) FFPROBE=${FFPROBE:-ffprobe} ;;
esac
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

# EXCLUSIVE_TOKENS: a source whose name contains one of these tokens is
# visited ONLY by rows whose `@` tag contains that same token. A row's tag is
# otherwise a plain substring of the clip name, so a new clip is visited by
# every row whose tag happens to occur in its name — a 10-bit clip is spelled
# `..._420p10` and would join every `@p10` row — and its arrival would change
# the cost of rows that never asked for it. `ilace`: the 10-bit interlaced
# clip, src_ilace10_96x96_420p10, visited only by `@ilace10` rows.
# Defined identically in identity_encode.sh, whose cells must be these.
EXCLUSIVE_TOKENS="ilace"

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
# The H.265 encoder codes its coding quadtree by default (depth 2), so every
# untagged hevc row exercises it; the `hevc*-cu0-*` rows (`--cu-depth 0`)
# keep the whole-CTB path — the geometry of every stream before the
# quadtree — under the same properties, and `hevc-cu1-ipb` the one-split
# depth. Which clip carries what is worth knowing. Depth 2 (8x8 units,
# PART_NxN) is reachable only on 32x32 CTBs: the odd clip's 16x16 CTBs stop at depth
# 1 whatever the row asks, and grad's smooth gradients split almost nowhere
# (its cells prove the syntax). detail, motion, cut and fade split at every
# depth in every picture kind; NxN is taken on every clip in I pictures and
# on seven of them inside P/B. The @big clip, src_big_256x160_420p8 — the
# one clip larger than 64x64, forty CTBs of four unrelated contents — is
# spelled with a depth token so every row without an `@` skips it (the deep
# clips' rule); its `hevc-cu0-*@big` rows are the depth-0 twins of the
# `hevc-cu2-*@big` ones.
#
# The quadtree's mutations, each run once against these rows: the split
# decision ignored by the writer, the split_cu_flag neighbour context
# reported at depth 0, and the quantiser prediction read at the unit
# instead of its quantisation group all fail SELF (the last is invisible
# on the odd clip, whose 8x8 groups are the minimum unit — the AQ rows on
# detail, cut and @big carry it); 4:4:4 PART_NxN's four chroma modes
# written in reverse fails SELF on the 4:4:4 clips.
#
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
#
# The h264-aq rows are H.264's adaptive quantisation: the H.265 model
# (encode::aq) over each 16x16 macroblock, carried as mb_qp_delta. A 64x64
# clip is sixteen macroblocks rather than four coding tree blocks, so the
# uniform clips move quantisers here where the H.265 rows move none;
# h26xenc's `aq` census line says which cells did. One row per entropy
# coder and prediction mode, one of each above QP 29 (the chroma QP table
# is indexed per macroblock now), and deep rows at 10 and 12 bits.
#
# The h264-wp rows are H.264's explicit weighted prediction for P slices.
# Only the fade clip changes brightness, so it carries most of them. Two rows
# run over every 8-bit clip, for opposite reasons. At QP 26 the fit has to
# decline on content that does not fade (h26xenc's `wp` line counts the P
# pictures that took a weighting: none on detail, motion or static). At QP 40
# it does not decline: a reconstruction that coarse has drifted in level from
# its source, the fit takes a weighting to correct it (21 of 84 P pictures on
# the cut clip at QP 38), and that is the weighted path on content the fade
# rows never show it — a row at QP 26 alone proved it only on the fade.
#
# The fade is a pure gain, so the offsets its weighting carries round to zero,
# and a writer that flipped the sign of every weighting offset failed only 11
# of the 25 weighted cells. The `@wpoff` rows visit the gain-and-offset fade
# (src_wpoff_64x64_420p8, see make_encode_sources.sh), where every weighted P
# picture carries an offset, so that regression cannot pass the gate. The corpus has no deep or non-4:2:0
# fade, so the @p10 rows prove the syntax at depth; a deep, 4:2:2, 4:4:4 and
# monochrome fade run in the unit test.
#
# The hevc-wp*-ipb rows are H.265's weighted bi-prediction. With --wpred and
# B pictures the PPS sets weighted_bipred_flag and every B slice carries a
# table with an entry for each list's anchor (encode::h265_wp's fit), which
# the B walk's one-list and bi predictions apply. So every H.265 --wpred row
# with --bframes moved when that landed — `hevc-wp-ipb@fade` and
# `hevc-cu0-wp-ipb@fade` included — and no IP row did (no B picture, no
# flag). Against the encoder before it (P weighted, B default) on the fade at
# --bframes 2: -6.8% bytes at QP 26, +0.7% at QP 40 with PSNR up, BD-rate
# -4.6% over QP 22..40, and only the B slices and the PPS differ. The
# untagged rows show the skip: at QP 26 no fit is used on a clip that does
# not fade, the reconstruction is the default-weighted one to the byte, and
# the stream is a table of defaults (about ten bits a B slice) larger. At
# QP 40 a few B pictures of those clips take a weighting, mostly chroma, as
# the P rows do. The corpus has no deep fade, so the @p10 row is mostly the
# syntax at depth: no fit is used on the three detail10 clips (the
# reconstruction is the default-weighted one), motion10 takes a weighting in
# a B picture, and a 10-bit fade with B pictures runs in the unit test; the
# @wpoff rows put an offset in both lists; refs2 mixes a two-entry P table
# with one-entry B lists; the ABR row runs the fit under a lookahead.
# The h264-paff / h264-mbaff rows are H.264 interlaced coding. They visit the
# two interlaced clips: src_interlace_96x96_420p8 (`@interlace`: fields 20 ms
# apart, a scrolling half beside a held one, combed so that PAFF has field
# pictures to choose), whose `p8` suffix keeps every untagged row off it, and
# its 10-bit twin src_ilace10_96x96_420p10 (`@ilace10`), whose name carries an
# EXCLUSIVE_TOKENS token so that no `@p10` row visits it. Two deep rows also
# code the progressive @p10 / @420p10 clips as interlaced. `field` codes every
# frame as two field pictures, `paff` and `mbaff` decide per picture and per
# macroblock pair, and h26xenc's `interlace` census line counts what each cell
# actually coded. A wrong bottom_field_flag is red under SELF, not CROSS:
# libavcodec follows the flag exactly as our decoder does, so the two agree on
# the misread stream while both differ from the encoder's reconstruction.
# CROSS checks the field reference lists and field-geometry filtering with a
# decoder that shares none of our code.
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
cqp-ipb-subparts|--codec h264 --qp 26 --gop 8 --bframes 2 --subparts
cavlc-ipb-subparts|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --subparts
cqp-ipb-t8x8-subparts|--codec h264 --qp 26 --gop 8 --bframes 2 --t8x8 --subparts
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
hevc10-cqp-intra@p10|--codec h265 --qp 26 --gop 0
hevc10-cqp-ip@p10|--codec h265 --qp 26 --gop 8
hevc10-cqp-ipb@p10|--codec h265 --qp 26 --gop 8 --bframes 2
hevc10-cqp40-ip@p10|--codec h265 --qp 40 --gop 8
hevc10-lossless-ip@p10|--codec h265 --lossless --gop 8
hevc10-lossless-ipb@p10|--codec h265 --lossless --gop 8 --bframes 2
hevc10-cqp40-sao-ip@p10|--codec h265 --qp 40 --gop 8 --sao
hevc10-abr-96k@p10|--codec h265 --bitrate 96000 --gop 8
hevc12-cqp-ip@p12|--codec h265 --qp 26 --gop 8
hevc12-cqp40-sao-ip@p12|--codec h265 --qp 40 --gop 8 --sao
hevc12-lossless-ip@p12|--codec h265 --lossless --gop 8
hevc-aq-intra|--codec h265 --qp 26 --gop 0 --aq 1.0
hevc-aq-ip|--codec h265 --qp 26 --gop 8 --aq 1.0
hevc-aq-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --aq 1.0
hevc-aq40-ip|--codec h265 --qp 40 --gop 8 --aq 1.0
hevc-aq40-sao-ip|--codec h265 --qp 40 --gop 8 --sao --aq 1.0
hevc-abr-aq-64k|--codec h265 --bitrate 64000 --gop 8 --aq 1.0
hevc10-aq-ip@p10|--codec h265 --qp 26 --gop 8 --aq 1.0
hevc12-aq40-ip@p12|--codec h265 --qp 40 --gop 8 --aq 1.0
hevc-abr-la-64k|--codec h265 --bitrate 64000 --gop 8 --lookahead 8
hevc-abr-la-96k|--codec h265 --bitrate 96000 --gop 8 --lookahead 8
hevc-abr-la-ipb-64k|--codec h265 --bitrate 64000 --gop 8 --bframes 2 --lookahead 4
hevc-vbv-la-125@src_cut|--codec h265 --bitrate 64000 --cpb-ms 125 --gop 8 --lookahead 8
hevc10-abr-la-96k@p10|--codec h265 --bitrate 96000 --gop 8 --lookahead 8
hevc-wp-ip|--codec h265 --qp 26 --gop 8 --wpred
hevc-wp-ipb@fade|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred
hevc-wp40-ip@fade|--codec h265 --qp 40 --gop 8 --wpred
hevc-wp-sao-ip@fade|--codec h265 --qp 26 --gop 8 --sao --wpred
hevc-wp-abr-64k@fade|--codec h265 --bitrate 64000 --gop 8 --wpred
hevc10-wp-ip@p10|--codec h265 --qp 26 --gop 8 --wpred
hevc-refs2-ip|--codec h265 --qp 26 --gop 8 --refs 2
hevc-refs2-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --refs 2
hevc-refs2-40-ip|--codec h265 --qp 40 --gop 8 --refs 2
hevc-refs2-wp-ip@fade|--codec h265 --qp 26 --gop 8 --refs 2 --wpred
hevc10-refs2-ip@p10|--codec h265 --qp 26 --gop 8 --refs 2
abr-64k-cpb@src_cut|--codec h264 --bitrate 64000 --cpb-ms 125 --gop 8
abr-64k-cavlc-cpb@src_cut|--codec h264 --bitrate 64000 --cpb-ms 125 --gop 8 --cavlc
h264-10-lossless-intra@p10|--codec h264 --lossless --gop 0
h264-10-lossless-cavlc-intra@p10|--codec h264 --lossless --gop 0 --cavlc
h264-10-cqp-intra@p10|--codec h264 --qp 26 --gop 0
h264-10-cavlc-intra@p10|--codec h264 --qp 26 --gop 0 --cavlc
h264-10-cqp-ip@p10|--codec h264 --qp 26 --gop 8
h264-10-cavlc-ip@p10|--codec h264 --qp 26 --gop 8 --cavlc
h264-10-cqp-ipb@p10|--codec h264 --qp 26 --gop 8 --bframes 2
h264-10-cavlc-ipb@p10|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc
h264-10-cqp40-t8x8@p10|--codec h264 --qp 40 --gop 8 --t8x8
h264-10-cavlc-t8x8@p10|--codec h264 --qp 26 --gop 8 --cavlc --t8x8
h264-10-cqp-subparts@p10|--codec h264 --qp 26 --gop 8 --subparts
h264-10-cavlc40-subparts@p10|--codec h264 --qp 40 --gop 8 --cavlc --subparts
h264-10-cqp-ipb-t8x8-subparts@p10|--codec h264 --qp 26 --gop 8 --bframes 2 --t8x8 --subparts
h264-10-abr-128k@p10|--codec h264 --bitrate 128000 --gop 8
h264-12-cqp-ip@p12|--codec h264 --qp 26 --gop 8
h264-12-cavlc40-ipb-t8x8-subparts@p12|--codec h264 --qp 40 --gop 8 --bframes 2 --cavlc --t8x8 --subparts
h264-12-lossless-intra@p12|--codec h264 --lossless --gop 0
h264-aq-intra|--codec h264 --qp 26 --gop 0 --aq 1.0
h264-aq-ip|--codec h264 --qp 26 --gop 8 --aq 1.0
h264-aq-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --aq 1.0
h264-aq40-ip|--codec h264 --qp 40 --gop 8 --aq 1.0
h264-aq-cavlc-intra|--codec h264 --qp 26 --gop 0 --cavlc --aq 1.0
h264-aq-cavlc-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --aq 1.0
h264-aq40-cavlc-ip|--codec h264 --qp 40 --gop 8 --cavlc --aq 1.0
h264-aq-t8x8-subparts-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --t8x8 --subparts --aq 1.0
h264-abr-aq-64k|--codec h264 --bitrate 64000 --gop 8 --aq 1.0
h264-10-aq-ip@p10|--codec h264 --qp 26 --gop 8 --aq 1.0
h264-10-aq40-cavlc-ipb@p10|--codec h264 --qp 40 --gop 8 --bframes 2 --cavlc --aq 1.0
h264-12-aq40-ip@p12|--codec h264 --qp 40 --gop 8 --aq 1.0
h264-wp-ip|--codec h264 --qp 26 --gop 8 --wpred
h264-wp-cavlc-ip@fade|--codec h264 --qp 26 --gop 8 --cavlc --wpred
h264-wp-ipb@fade|--codec h264 --qp 26 --gop 8 --bframes 2 --wpred
h264-wp40-ip|--codec h264 --qp 40 --gop 8 --wpred
h264-wp40-cavlc-ip@fade|--codec h264 --qp 40 --gop 8 --cavlc --wpred
h264-wp-t8x8-subparts-ip@fade|--codec h264 --qp 26 --gop 8 --t8x8 --subparts --wpred
h264-wp-aq-ip@fade|--codec h264 --qp 26 --gop 8 --aq 1.0 --wpred
h264-wp-abr-64k@fade|--codec h264 --bitrate 64000 --gop 8 --wpred
h264-10-wp-ip@p10|--codec h264 --qp 26 --gop 8 --wpred
h264-10-wp-cavlc-ipb@p10|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --wpred
h264-wpoff-ip@wpoff|--codec h264 --qp 26 --gop 8 --wpred
h264-wpoff-cavlc-ipb@wpoff|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --wpred
h264-wpoff40-ip@wpoff|--codec h264 --qp 40 --gop 8 --wpred
hevc-wpoff-ip@wpoff|--codec h265 --qp 26 --gop 8 --wpred
cqp-ip-srgb-pc|--codec h264 --qp 26 --gop 8 --color 1:13:6 --full-range
abr-64k-cpb-p3@src_cut|--codec h264 --bitrate 64000 --cpb-ms 125 --gop 8 --color 12:17:6 --chroma-loc 1
hevc-vbv-125-hdr10@src_cut|--codec h265 --bitrate 64000 --cpb-ms 125 --gop 8 --color 9:16:9
hevc10-hdr10-ip@p10|--codec h265 --qp 26 --gop 8 --color 9:16:9
hevc10-hlg-ipb@p10|--codec h265 --qp 26 --gop 8 --bframes 2 --color 9:18:9
hevc10-hlg-topleft-ip@420p10|--codec h265 --qp 26 --gop 8 --color 9:18:9 --chroma-loc 2
abr-64k-cpb250-hdr10-sei@src_cut|--codec h264 --bitrate 64000 --cpb-ms 250 --gop 8 --color 9:16:9 --mastering-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1) --content-light 1000,400
hevc10-hdr10-sei-ip@p10|--codec h265 --qp 26 --gop 8 --color 9:16:9 --mastering-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1) --content-light 1000,400
hevc-cu0-intra|--codec h265 --qp 26 --gop 0 --cu-depth 0
hevc-cu0-ip|--codec h265 --qp 26 --gop 8 --cu-depth 0
hevc-cu0-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 0
hevc-cu1-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 1
hevc-cu0-40-intra|--codec h265 --qp 40 --gop 0 --cu-depth 0
hevc-cu0-40-ip|--codec h265 --qp 40 --gop 8 --cu-depth 0
hevc-cu0-lossless-ipb|--codec h265 --lossless --gop 8 --bframes 2 --cu-depth 0
hevc-cu0-40-sao-ip|--codec h265 --qp 40 --gop 8 --sao --cu-depth 0
hevc-cu0-aq-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --aq 1.0 --cu-depth 0
hevc-cu0-aq40-ip|--codec h265 --qp 40 --gop 8 --aq 1.0 --cu-depth 0
hevc-cu0-abr-64k|--codec h265 --bitrate 64000 --gop 8 --cu-depth 0
hevc-cu0-refs2-ip|--codec h265 --qp 26 --gop 8 --refs 2 --cu-depth 0
hevc-cu0-vbv-125@src_cut|--codec h265 --bitrate 64000 --cpb-ms 125 --gop 8 --cu-depth 0
hevc-cu0-wp-ipb@fade|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred --cu-depth 0
hevc10-cu0-ipb@p10|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 0
hevc10-cu0-aq-ip@p10|--codec h265 --qp 26 --gop 8 --aq 1.0 --cu-depth 0
hevc10-cu0-lossless-ip@p10|--codec h265 --lossless --gop 8 --cu-depth 0
hevc12-cu0-40-sao-ip@p12|--codec h265 --qp 40 --gop 8 --sao --cu-depth 0
hevc-cu0-intra@big|--codec h265 --qp 26 --gop 0 --cu-depth 0
hevc-cu0-ipb@big|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 0
hevc-cu2-intra@big|--codec h265 --qp 26 --gop 0 --cu-depth 2
hevc-cu2-ipb@big|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 2
hevc-cu2-aq40-ipb@big|--codec h265 --qp 40 --gop 8 --bframes 2 --aq 1.0 --cu-depth 2
hevc-cu2-40-sao-ip@big|--codec h265 --qp 40 --gop 8 --sao --cu-depth 2
hevc-wp40-ipb|--codec h265 --qp 40 --gop 8 --bframes 2 --wpred
hevc-wp-sao-ipb|--codec h265 --qp 26 --gop 8 --bframes 2 --sao --wpred
hevc10-wp-ipb@p10|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred
hevc-wpoff-ipb@wpoff|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred
hevc-wpoff40-ipb@wpoff|--codec h265 --qp 40 --gop 8 --bframes 2 --wpred
hevc-wp-refs2-ipb@fade|--codec h265 --qp 26 --gop 8 --bframes 2 --refs 2 --wpred
hevc-wp-abr-la-ipb-64k@fade|--codec h265 --bitrate 64000 --gop 8 --bframes 2 --lookahead 4 --wpred
h264-paff-field-ip@interlace|--codec h264 --qp 26 --gop 8 --interlace tff --field-coding field
h264-paff-field-cavlc-ip@interlace|--codec h264 --qp 26 --gop 8 --cavlc --interlace bff --field-coding field
h264-paff-field-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace tff --field-coding field
h264-paff-field-cavlc-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --interlace bff --field-coding field
h264-paff-field40-t8x8-subparts-ipb@interlace|--codec h264 --qp 40 --gop 8 --bframes 2 --t8x8 --subparts --interlace tff --field-coding field
h264-paff-field-cavlc40-t8x8-subparts-ip@interlace|--codec h264 --qp 40 --gop 8 --cavlc --t8x8 --subparts --interlace bff --field-coding field
h264-10-paff-field-ipb@p10|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace tff --field-coding field
h264-paff-ip@interlace|--codec h264 --qp 26 --gop 8 --interlace tff --field-coding paff
h264-paff-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace bff --field-coding paff
h264-paff-cavlc-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --interlace tff --field-coding paff
h264-paff40-cavlc-t8x8-subparts-ip@interlace|--codec h264 --qp 40 --gop 8 --cavlc --t8x8 --subparts --interlace bff --field-coding paff
h264-10-paff-ipb@ilace10|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace bff --field-coding paff
h264-mbaff-intra@interlace|--codec h264 --qp 26 --gop 0 --interlace tff --field-coding mbaff
h264-mbaff-ip@interlace|--codec h264 --qp 26 --gop 8 --interlace tff --field-coding mbaff
h264-mbaff-cavlc-ip@interlace|--codec h264 --qp 26 --gop 8 --cavlc --interlace bff --field-coding mbaff
h264-mbaff-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace bff --field-coding mbaff
h264-mbaff-cavlc-ipb@interlace|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --interlace tff --field-coding mbaff
h264-mbaff40-t8x8-subparts-ipb@interlace|--codec h264 --qp 40 --gop 8 --bframes 2 --t8x8 --subparts --interlace tff --field-coding mbaff
h264-mbaff-cavlc40-t8x8-subparts-ip@interlace|--codec h264 --qp 40 --gop 8 --cavlc --t8x8 --subparts --interlace bff --field-coding mbaff
h264-10-mbaff-ipb@420p10|--codec h264 --qp 26 --gop 8 --bframes 2 --interlace tff --field-coding mbaff
h264-10-paff-cavlc-ip@ilace10|--codec h264 --qp 26 --gop 8 --cavlc --interlace tff --field-coding paff
h264-10-mbaff-ip@ilace10|--codec h264 --qp 26 --gop 8 --interlace tff --field-coding mbaff
h264-10-mbaff-cavlc-ipb@ilace10|--codec h264 --qp 26 --gop 8 --bframes 2 --cavlc --interlace bff --field-coding mbaff
"}

# Split a clip's format token into its chroma format and sample depth:
# `420` is 4:2:0 at 8 bits, `420p10` the same at 10. The depth travels in
# the filename with the geometry, for the same reason the geometry does.
chroma_of() { echo "${1%%p*}"; }
depth_of() { local d=${1##*p}; [ "$d" = "$1" ] && d=8; echo "$d"; }

one() {
  src=$1; name=$2; flags=$3
  base=$(basename "$src" .yuv)
  geom=$(echo "$base" | sed -n 's/.*_\([0-9]\+x[0-9]\+\)_.*/\1/p')
  fmt=$(echo "$base" | sed -n 's/.*_[0-9]\+x[0-9]\+_\(.*\)/\1/p')
  chroma=$(chroma_of "$fmt")
  depth=$(depth_of "$fmt")
  tag="$base/$name"
  ext=h264; case "$flags" in *"--codec h265"*) ext=h265 ;; esac
  bs="$OUT/$base.$name.$ext"
  rec="$OUT/$base.$name.rec.yuv"

  # The encoder writes the bitstream and, separately, the reconstruction it
  # believes that bitstream carries. The depth is the clip's, never the
  # row's: a row that asked for a depth the clip does not have would code
  # the wrong number of bytes per sample and fail on the first picture.
  # Timed (7): wall nanoseconds around the one process, nothing else
  # in the cell.
  t0=$(date +%s%N)
  if ! "$ENC" --input "$src" --size "$geom" --format "$chroma" --depth "$depth" \
       $flags --output "$bs" --recon "$rec" > "$OUT/$base.$name.enc.log" 2>&1; then
    echo "ENCODE-FAIL $tag: $(tail -1 "$OUT/$base.$name.enc.log" | head -c 100)"
    return 1
  fi
  t1=$(date +%s%N)
  frames=$(( $(stat -c %s "$src") / $(frame_bytes "$geom" "$chroma" "$depth") ))
  speed=$(awk -v ns="$((t1 - t0))" -v f="$frames" 'BEGIN { s = ns / 1e9; printf "%.3f s, %.0f f/s", s, (s > 0 ? f / s : 0) }')

  # Every source picture has to come out. SELF and CROSS compare the stream
  # with itself, so an encoder that drops pictures — a flush that forgets the
  # ones a lookahead is holding — writes a shorter stream whose every picture
  # still decodes to its reconstruction on both decoders, at a PSNR over the
  # pictures that exist and a rate divided by them. Counting the
  # reconstruction is the check nothing else here makes.
  fb=$(frame_bytes "$geom" "$chroma" "$depth")
  if [ "$(stat -c %s "$rec")" != "$((frames * fb))" ]; then
    echo "ENCODE-FAIL $tag: the reconstruction holds $(( $(stat -c %s "$rec") / fb )) pictures, the source $frames"
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
  # Above 8 bits libavcodec emits little-endian 16-bit planes natively
  # (`yuv420p10le` and friends) — the layout our decoder packs, so the
  # comparison stays a plain `cmp` at every depth.
  ffargs="-pix_fmt $(ffpix "$fmt")"
  case "$chroma" in 400|gray) ffargs="-vf extractplanes=y -pix_fmt $(ffpix "$fmt")" ;; esac
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

  # 8. VUI. Only where a colour was given. The probe is told the codes the
  # encoder was handed and the range flag beside them, and asks ffprobe
  # whether the stream says so — the one reader here that is neither the
  # writer nor its own inverse.
  case "$flags" in
    *--color*)
      colour=$(echo "$flags" | sed -n 's/.*--color \([0-9:]*\).*/\1/p')
      range=tv; case "$flags" in *--full-range*) range=pc ;; esac
      # The HDR10 static-metadata SEIs, when the row wrote them: the probe
      # is told the same values and asks ffprobe's frame side data.
      hdr=""
      md=$(echo "$flags" | sed -n 's/.*--mastering-display \([^ ]*\).*/\1/p')
      [ -n "$md" ] && hdr="$hdr --mastering-display $md"
      cl=$(echo "$flags" | sed -n 's/.*--content-light \([^ ]*\).*/\1/p')
      [ -n "$cl" ] && hdr="$hdr --content-light $cl"
      # The chroma siting, when the row wrote one; without it the probe
      # insists the stream says nothing about siting.
      loc=$(echo "$flags" | sed -n 's/.*--chroma-loc \([0-9]*\).*/\1/p')
      [ -n "$loc" ] && hdr="$hdr --chroma-loc $loc"
      if ! out=$(FFPROBE="$FFPROBE" python "$VUI_PROBE" "$bs" "$colour" "$range" $hdr 2>&1); then
        echo "VUI-FAIL    $tag: $(echo "$out" | tail -1 | head -c 120)"
        return 1
      fi
      ;;
  esac

  # 3. QUALITY. Gated only when the configuration claims to be lossless.
  psnr=$(psnr_of "$src" "$rec" "$depth")
  size=$(stat -c %s "$bs")
  # 7. SPEED. One row per cell if a table was asked for; the PASS line
  # carries it regardless.
  if [ -n "$H26X_SPEED_TABLE" ]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$base" "$name" "$size" "$psnr" "$(echo "$speed" | sed 's/ s, /\t/; s/ f\/s//')" >> "$H26X_SPEED_TABLE"
  fi
  case "$flags" in
    *--lossless*)
      if ! cmp -s "$src" "$rec"; then
        echo "LOSSLESS-FAIL $tag: reconstruction differs from the source"
        return 1
      fi
      echo "PASS        $tag (lossless, exact, $size bytes, $speed)" ;;
    *--bitrate*)
      echo "PASS        $tag ($size bytes, PSNR $psnr dB, rate $(printf '%.2f' "$ratio")x, $speed)" ;;
    *)
      echo "PASS        $tag ($size bytes, PSNR $psnr dB, $speed)" ;;
  esac
  return 0
}

# Bytes in one raw picture of the clip, from its geometry and format —
# what turns a clip's size into a frame count for the speed line.
frame_bytes() {
  w=${1%x*}; h=${1#*x}
  bps=1; [ "${3:-8}" -gt 8 ] && bps=2
  case "$2" in
    400|gray) echo $((w * h * bps)) ;;
    422) echo $((w * h * 2 * bps)) ;;
    444) echo $((w * h * 3 * bps)) ;;
    *) echo $((w * h * 3 / 2 * bps)) ;;
  esac
}

# Planar chroma format token to the ffmpeg pixel format that matches how
# this decoder packs a picture: bytes at 8 bits, little-endian 16-bit
# planes above (`420p10` -> `yuv420p10le`, `400p10` -> `gray10le`).
ffpix() {
  local d
  d=$(depth_of "$1")
  [ "$d" = 8 ] && d="" || d="${d}le"
  case "$(chroma_of "$1")" in
    400|gray) echo "gray$d" ;;
    422) echo "yuv422p$d" ;;
    444) echo "yuv444p$d" ;;
    *) echo "yuv420p$d" ;;
  esac
}

# PSNR of two raw files at a sample depth: bytes at 8, little-endian
# 16-bit samples above, against the peak that depth has. Reading a 10-bit
# file as bytes would compare the low and high halves of each sample as
# if they were two samples and report nonsense with a straight face.
psnr_of() {
  python - "$1" "$2" "$3" <<'PY'
import sys, math, array
depth = int(sys.argv[3])
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
if depth > 8:
    sa = array.array('H'); sa.frombytes(a[:len(a) & ~1])
    sb = array.array('H'); sb.frombytes(b[:len(b) & ~1])
    if sys.byteorder != 'little':
        sa.byteswap(); sb.byteswap()
    a, b = sa, sb
n = min(len(a), len(b))
if n == 0:
    print("n/a"); raise SystemExit
step = max(1, n // 200000) if n > 200000 else 1
se = 0
for i in range(0, n, step):
    d = a[i] - b[i]
    se += d * d
cnt = len(range(0, n, step))
mse = se / cnt
peak = (1 << depth) - 1
print("inf" if mse == 0 else f"{10 * math.log10(peak * peak / mse):.2f}")
PY
}
export -f one ffpix psnr_of chroma_of depth_of frame_bytes
export ENC DEC HRD FFMPEG FFPROBE OUT PARAM_SETS VUI_PROBE H26X_SPEED_TABLE JOBS

echo "== encode verification =="
results="$OUT/results.txt"
: > "$results"
for src in $SOURCES; do
  # A clip deeper than 8 bits (format token `420p10`, `420p12`, ...) is
  # visited only by rows that name it with `@p10` / `@p12`. A row without
  # a suffix is an 8-bit row: most of them are H.264, which codes 8 bits
  # only, and the H.265 rows have their deep twins listed explicitly so
  # the tally says how many deep cells ran rather than folding them in.
  case "$src" in *_[0-9][0-9][0-9]p[0-9]*.yuv) deep=1 ;; *) deep=0 ;; esac
  # The exclusive tokens (EXCLUSIVE_TOKENS, above) this source carries.
  excl=
  for tok in $EXCLUSIVE_TOKENS; do case "$src" in *"$tok"*) excl="$excl $tok" ;; esac; done
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
        for tok in $excl; do case "$pat" in *"$tok"*) ;; *) continue 2 ;; esac; done
        case "$src" in
          *"$pat"*) name=${name%@*} ;;
          *) continue ;;
        esac
        ;;
      *)
        [ "$deep" = 1 ] && continue
        [ -n "$excl" ] && continue
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
bad=$(grep -cE '^(ENCODE|SELF|CROSS|LOSSLESS|RATE|HRD|PS|VUI)-FAIL' "$results")
echo
echo "encode: $pass passed, $bad failed"
[ "$bad" = 0 ] || fail=1
# 7. The speed total: encoder wall seconds summed over every PASS cell.
# Not a gate. The number to compare is this one against the last run's on
# the same machine; a per-cell table (H26X_SPEED_TABLE) says where it went.
sed -n 's/^PASS .*, \([0-9.]*\) s, \([0-9]*\) f\/s)$/\1 \2/p' "$results" \
  | awk -v cells="$pass" -v jobs="$JOBS" '{ s += $1 } END { printf "encode speed: %d cells, %.2f s of encoder wall time (summed over cells, %d in parallel)\n", cells, s, jobs }'

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
