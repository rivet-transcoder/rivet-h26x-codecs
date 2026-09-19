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
#               infinite and the check becomes exact like the others — and
#               against a floor recorded from an earlier run (9).
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
#   4b. SUSTAINED  The band is blind to a controller that converges and
#               then stops steering. That happened: rc.rs concluded from one
#               quantiser move that the content ignored the quantiser, held
#               it for the rest of the clip, and spent a third to a half of
#               the plan on every picture after - H.265 cut at 96 kbps ended
#               at 0.83x, H.264 at 128 kbps at 0.89x, ten of the eighteen
#               cut-clip rate cells between 0.81x and 0.95x, all green.
#               So on a clip long enough to converge (at least RATE_WINDOW
#               GOPs after the first; of this corpus src_cut, 12 GOPs, and
#               src_settle, 48 at the two-picture GOP of its rows)
#               every RATE_WINDOW consecutive GOPs after the first must
#               spend within [RATE_WINDOW_LO, RATE_WINDOW_HI] of target,
#               measured from the stream's own access units (gop_spend_of).
#               The first GOP is exempt for the reason the band is wide.
#
#               Thresholds from the measured distribution (2026-09-14, the
#               18 cut-clip rate cells): after the fix every 3-GOP window
#               lies in [0.93, 1.07]; single GOPs dip to 0.86 after the cut,
#               which is why the window is three. On the pre-fix encoder
#               (h26x d88e24a) the nine cells the rule froze bottom out at
#               0.56 to 0.79. 0.85 sits between the two, and the ceiling is
#               its reciprocal, 1.18. Its mutation: the pre-fix encoder must
#               fail those nine cells and nothing else.
#
#   4c. VERDICT  Only for --bitrate rows whose name carries `-verdict`. The
#               insensitivity rule has three paths - a verdict, the probe
#               below its floor, the release - and when it was rewritten no
#               cell of this corpus reached the first, so every property
#               above was silent about it. The encoder reports what the rule
#               did ("rate: insensitivity verdicts V, probes P, releases R")
#               and a `-verdict` row must show V >= 1 and P >= 1.
#
#               The rows are H.264 at a two-picture GOP on src_settle (24
#               frames of motion, then one frame held for 72): once the
#               picture holds, every P picture is a skip whose bits do not
#               answer the quantiser, and the P quantiser, converged high on
#               the motion, walks down into them - silent walk, silent
#               raised picture, verdict, then a silent probe every ninth P.
#               The GOP is two so the keyframes can spend what the held P
#               pictures cannot, which keeps these rows inside 4b too (every
#               3-GOP window 0.90 to 1.07, 2026-09-14). A release is not
#               asserted: every configuration measured whose verdict was
#               released - motion resuming, or a probe that answered - then
#               spent the budget the hold had left, correctly, above 4b's
#               ceiling. The release path is held by encode::rc's tests.
#               H.265 is absent for a measured reason: its P picture after
#               each keyframe codes that keyframe's noise, the bits alternate
#               between GOPs and the walk restarts; no H.265 configuration
#               measured reached a verdict inside 4b.
#
#               Its mutation: a rule that never turns a confirmed walk into
#               a verdict must fail both rows, and so must one that never
#               probes.
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
#   9. QUALITY FLOOR  Only with --quality-baseline FILE, and modelled on
#               verify.sh --baseline: when FILE does not exist the run
#               records every cell's luma, Cb and Cr PSNR to it (a red run
#               records nothing — a floor taken from a red run holds the
#               red); when it does, a cell fails as QUALITY-FAIL if any plane
#               lands more than H26X_QUALITY_TOL below the PSNR recorded for
#               it. Cells more than the tolerance above their floor pass and
#               are listed, and so are cells FILE does not know and cells it
#               knows that this run did not measure — listed, not failed,
#               because a new row or a new clip is not a regression.
#
#               Why it exists: SELF and CROSS both compare a bitstream with
#               itself, and 3 only reports, so an encoder whose pictures got
#               worse passed every property above. That happened. The H.264
#               encoder quantised the I_16x16 luma DC one shift short from
#               its first commit, every I_16x16 DC came back at twice its
#               coded mean, and at QP 38 — where every intra macroblock is
#               I_16x16 — luma read 14.6 dB against H.265's 27.3 on the same
#               clip, with this whole gate green.
#
#               PSNR here is per plane and exact (no sampling, which could
#               step over the region that regressed), over every picture.
#
#               The tolerance, 0.30 dB by default, is not for noise: the
#               encoder is deterministic, and a same-binary rerun holds every
#               cell exactly. It is what an intended decision change may cost
#               a plane before its floor has to be recorded again, and it was
#               chosen on the I_16x16 fix itself, fixed floor against pre-fix
#               (337 of 754 cells moved, -3% to -24% BD-rate per clip). That
#               fix lowered 37 planes by more than 0.10 dB, 13 by more than
#               0.30 and 3 by more than 0.50. The 13 are Cr at -0.35 to -0.46
#               on ten QP 40 detail cells whose luma rose 11.9 dB (cqp40-ip on
#               39% fewer bytes), and one rate row, cut abr-128k, that spent
#               0.88x of its target where the pre-fix run spent 0.98x. At 0.30
#               the pre-fix encoder fails 248 of the 337 against the fixed
#               floor: every cell where the bug cost some plane more than
#               0.30 dB. The 89 it passes lost less than that.
#
#               Its mutation: disable the below-the-floor comparison, and
#               the pre-fix encoder must pass against a floor recorded from
#               the fixed one.
#
# Usage: verify_encode.sh [--quality-baseline FILE] [encoder] [decoder]
#   H26X_WORK=dir   scratch directory holding the source clips (default: here)
#   H26X_QUALITY_TOL=dB  how far below its floor a plane may land (see 9)
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
# --quality-baseline FILE (9), made absolute before the cd below for the
# same reason.
QBASE=""
while :; do
  case "$1" in
    --quality-baseline)
      [ -n "$2" ] || { echo "verify_encode.sh: --quality-baseline needs a file" >&2; exit 2; }
      QBASE=$2; shift 2 ;;
    *) break ;;
  esac
done
if [ -n "$QBASE" ]; then
  qdir=$(cd "$(dirname "$QBASE")" 2>/dev/null && pwd) \
    || { echo "verify_encode.sh: --quality-baseline: no directory $(dirname "$QBASE")" >&2; exit 2; }
  QBASE=$qdir/$(basename "$QBASE")
fi
QTOL=${H26X_QUALITY_TOL:-0.30}
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
# clip, src_ilace10_96x96_420p10, visited only by `@ilace10` rows. `fdeep`:
# the 10-bit gain-and-offset fade, src_fdeep10_64x64_420p10, visited only by
# `@fdeep10` rows. `wsine`: the native 10-bit weighted fade,
# src_wsine10_64x64_420p10, visited only by `@wsine10` rows.
# Defined identically in identity_encode.sh, whose cells must be these.
EXCLUSIVE_TOKENS="ilace fdeep wsine"

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
# depth. Which clip carries what is worth knowing. Under the quadtree the CTB
# is 32x32 and the coded picture the smallest legal size, partial CTBs
# along the right and bottom edges — except below 64 both ways, where every
# row keeps whole CTBs, 16 or 32 by least padding (`Geometry::new` has the
# measurement that exception is fitted to). So the odd clip (50x34) codes
# whole 16x16 CTBs (64x48) in every row, where the quadtree stops at depth
# 1 whatever the row asks. grad's smooth gradients split almost nowhere
# (its cells prove the syntax). detail, motion, cut and fade split at
# every depth in every picture kind; NxN is taken on every clip in I pictures
# and on seven of them inside P/B. The @big clip, src_big_256x160_420p8 — the
# one clip larger than 64x64, forty CTBs of four unrelated contents — is
# spelled with a depth token so every row without an `@` skips it (the deep
# clips' rule); its `hevc-cu0-*@big` rows are the depth-0 twins of the
# `hevc-cu2-*@big` ones. The @edge clip, src_edge_88x44_420p8 (coded 88x48),
# is the one partial-CTB clip: a bottom row 16 high, the remainder 1280x720
# and 3840x2160 leave, beside a right column 24 wide. Its `hevc-cu1-edge-ipb`
# row splits past its depth where the edge forces it, and
# `hevc-cu0-edge-ipb` is the whole-CTB geometry (16x16 CTBs, 96x48).
#
# The quadtree's mutations, each run once against these rows: the split
# decision ignored by the writer, the split_cu_flag neighbour context
# reported at depth 0, and the quantiser prediction read at the unit
# instead of its quantisation group all fail SELF (the last on 5 of 7 AQ
# cells, on cut, detail and @big; it is invisible on the odd clip, whose
# 8x8 groups on 16x16 CTBs are the minimum unit); 4:4:4 PART_NxN's four
# chroma modes written in reverse fails SELF on the 4:4:4 clips.
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
# its source, and the fit takes a weighting to correct it — a weak one, which
# the picture-level check prices against a table of defaults (23 of the cut
# clip's 84 P pictures at QP 40, 21 of them going back to the defaults, the
# `priced` and `kept them` counts of the `wp` line). That is the weighted
# path, and the check, on content the fade rows never show them — a row at
# QP 26 alone proved them only on the fade, whose fits are strong and are
# never priced.
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
# The `@fdeep10` rows are that fade at 10 bits: src_fdeep10_64x64_420p10, the
# gain-and-offset fade with its offset at the depth (luma p * (1 - N/16) -
# 12N, two bits of noise; see make_encode_sources.sh), where the weighted
# rows at 10 bits get a table that weights something in both lists rather
# than the syntax alone. Its name carries `fdeep`, one of EXCLUSIVE_TOKENS,
# so no `@p10` or `@420p10` row visits it and its arrival changed no other
# cell. h26xenc's `shapes B` census line says whether each cell's B pictures
# took a table (wp_on), and how many kept the defaults (wp_rd_default).
# fdeep10 is 8-bit content at four times the scale: its two noise bits
# quantise away from QP 26 on, and it codes to within a few bytes and a
# tenth of a dB of its own 8-bit twin, so those rows check the 10-bit path's
# consistency rather than its rate-distortion. The `@wsine10` rows are the
# 10-bit case proper: src_wsine10_64x64_420p10 is computed at 10 bits (a
# drifting sinusoidal texture under the same gain and offset), and at QP 26
# its reconstruction is 0.84 dB (luma) to 1.6 dB (chroma) better than the
# same encoder's on its 8-bit twin, on fewer bytes; its B pictures take both
# outcomes of the table-against-defaults check. Its name carries `wsine`,
# one of EXCLUSIVE_TOKENS.
# The H.264 --wpred rows with --bframes are H.264's explicit weighted
# bi-prediction. The PPS sets weighted_bipred_idc 1 and every B slice carries
# a table with an entry for each list's anchor; a B picture whose table
# weights something is priced against a table of defaults, and a component
# class left at the defaults writes denominator 0. A B pair whose weights
# would sum past 8.4.2.3's bound at sixty-fourths takes a coarser
# denominator (on the fade the first B picture of each GOP codes its luma in
# thirty-seconds). So every H.264 --wpred row with --bframes moved when that
# landed — `h264-wp-ipb@fade`, `h264-10-wp-cavlc-ipb@p10` and
# `h264-wpoff-cavlc-ipb@wpoff` included — and with them, since P pictures'
# weak fits are priced the same way and their default tables got shorter,
# every H.264 --wpred row. Against the encoder before it (P weighted, B
# default) at --bframes 2 over QP 22..40: BD-rate -13.5% on the fade, -17.3%
# on the gain-and-offset fade, -12.4% on fdeep10, -22.9% on wsine10; on the
# untagged clips the reconstruction is the default-weighted one to the byte
# and each B slice a table of defaults (about a byte) larger. The rows below
# `hevc10-wp40-ipb@wsine10` add the B-slice weighting's own cells: QP 40 over
# every 8-bit clip, where B pictures take and decline tables; the
# gain-and-offset fade under CABAC (its CAVLC twin was already there); the
# fade with the 8x8 transform and sub-partitions, and under adaptive
# quantisation; and both 10-bit fades. h26xenc's `wp B` line counts the B
# pictures that took a table, were priced, and kept the defaults.
# The h264-*imp* rows are H.264's implicit B weighting (`--bweight implicit`,
# `weighted_bipred_idc` 2): no table, every bi-predicted block weighted by
# the picture's distances to its anchors through the decoder's own
# `implicit_pair`, so SELF holds the encoder to that derivation and CROSS
# holds both to libavcodec's. It is opt-in (see `Config::b_weighting` for
# the measurement: fades and motion gain, detail loses), so no other row
# moved. One row over every 8-bit clip at two B pictures, one at three with
# the 8x8 transform and sub-partitions under CAVLC at QP 40, the deep clips,
# beside explicitly weighted P pictures on the fade, and on the native
# 10-bit fade, where it gains most.
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
# The hevc-refs3-ipb / hevc-refs4-b3 rows are H.265 with more references than
# a B picture predicts from, over one long GOP (--gop 250). A B picture uses
# one past anchor and the future one, and its reference picture set must
# still keep the older anchors a later P picture uses, flagged unused. When
# it listed only the two it uses, the older anchors were marked unused, and
# libavcodec refused the next P picture that named one ("Could not find ref
# with POC 0"). Our decoder generates a stand-in for a missing reference and
# counts a warning (h26xdec prints the count); SELF passed and only CROSS was
# red, which is why SELF now also fails on any warning our decoder counts. They visit motion (and, through the tag, its 10-bit twin
# motion10), fade and cut; --gop 250 because at --gop 8 the GOP ends before
# --bframes 3 leaves a B picture below an anchor it does not use.
# The *-gNbM rows are short GOPs with B pictures, several GOPs per clip. An
# H.264 IDR empties the decoder's reference lists and restarts POC, but the
# encoder kept the previous GOP's references: where one of its anchors
# survived the IDR at the POC of the new GOP's, the first B picture after
# the IDR predicted from it and the decoder, which no longer had it, did
# not — every B picture from there differed from the encoder's
# reconstruction. That happened in a GOP of one mini-GOP (`--gop` =
# `--bframes` + 2) at any reference count, and with three references at
# most GOP lengths, while every row here ran --gop 8 or longer with at most
# two. Under that fault SELF was red on 29 of the 34 H.264 cells here (from
# display index 4 at --gop 3 --bframes 1), all but the static clip, whose
# anchors do not differ, and the six-picture odd clip where the second GOP
# holds no B picture; CROSS stayed green, as libavcodec decodes the stream
# exactly as we do. The H.265 rows are the same GOPs, which H.265 always
# coded correctly (it drops its references at an IDR), held there.
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
hevc-refs3-ipb@motion|--codec h265 --qp 26 --gop 250 --bframes 2 --refs 3
hevc-refs3-ipb@fade|--codec h265 --qp 26 --gop 250 --bframes 2 --refs 3
hevc-refs3-ipb@cut|--codec h265 --qp 26 --gop 250 --bframes 2 --refs 3
hevc-refs4-b3@motion|--codec h265 --qp 26 --gop 250 --bframes 3 --refs 4
hevc-refs4-b3@fade|--codec h265 --qp 26 --gop 250 --bframes 3 --refs 4
hevc-refs4-b3@cut|--codec h265 --qp 26 --gop 250 --bframes 3 --refs 4
h264-ipb-g3b1|--codec h264 --qp 26 --gop 3 --bframes 1
h264-ipb-g4b2|--codec h264 --qp 26 --gop 4 --bframes 2
h264-refs3-ipb-g5b1|--codec h264 --qp 26 --gop 5 --bframes 1 --refs 3
h264-10-ipb-g3b1@p10|--codec h264 --qp 26 --gop 3 --bframes 1
hevc-ipb-g3b1|--codec h265 --qp 26 --gop 3 --bframes 1
hevc-ipb-g4b2|--codec h265 --qp 26 --gop 4 --bframes 2
hevc-refs3-ipb-g5b1|--codec h265 --qp 26 --gop 5 --bframes 1 --refs 3
hevc10-ipb-g3b1@p10|--codec h265 --qp 26 --gop 3 --bframes 1
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
hevc-edge-intra@edge|--codec h265 --qp 26 --gop 0
hevc-edge-ipb@edge|--codec h265 --qp 26 --gop 8 --bframes 2
hevc-edge-40-sao-ip@edge|--codec h265 --qp 40 --gop 8 --sao
hevc-edge-aq40-ipb@edge|--codec h265 --qp 40 --gop 8 --bframes 2 --aq 1.0
hevc-edge-lossless-ip@edge|--codec h265 --lossless --gop 8
hevc-cu1-edge-ipb@edge|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 1
hevc-cu0-edge-ipb@edge|--codec h265 --qp 26 --gop 8 --bframes 2 --cu-depth 0
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
hevc10-wp-ipb@fdeep10|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred
hevc10-wp40-ipb@fdeep10|--codec h265 --qp 40 --gop 8 --bframes 2 --wpred
hevc10-wp-sao-ipb@fdeep10|--codec h265 --qp 26 --gop 8 --bframes 2 --sao --wpred
hevc10-wp-ip@fdeep10|--codec h265 --qp 26 --gop 8 --wpred
h264-verdict-g2-192k@settle|--codec h264 --bitrate 192000 --gop 2
h264-verdict-g2-256k@settle|--codec h264 --bitrate 256000 --gop 2
hevc10-wp-ipb@wsine10|--codec h265 --qp 26 --gop 8 --bframes 2 --wpred
hevc10-wp-ip@wsine10|--codec h265 --qp 26 --gop 8 --wpred
hevc10-wp40-ipb@wsine10|--codec h265 --qp 40 --gop 8 --bframes 2 --wpred
h264-wp40-ipb|--codec h264 --qp 40 --gop 8 --bframes 2 --wpred
h264-wp-ipb@wpoff|--codec h264 --qp 26 --gop 8 --bframes 2 --wpred
h264-wp-t8x8-subparts-ipb@fade|--codec h264 --qp 26 --gop 8 --bframes 2 --t8x8 --subparts --wpred
h264-wp-aq-ipb@fade|--codec h264 --qp 26 --gop 8 --bframes 2 --aq 1.0 --wpred
h264-10-wp-ipb@wsine10|--codec h264 --qp 26 --gop 8 --bframes 2 --wpred
h264-10-wp40-ipb@fdeep10|--codec h264 --qp 40 --gop 8 --bframes 2 --wpred
h264-imp-ipb|--codec h264 --qp 26 --gop 8 --bframes 2 --bweight implicit
h264-imp40-cavlc-t8x8-subparts-b3|--codec h264 --qp 40 --gop 8 --bframes 3 --cavlc --t8x8 --subparts --bweight implicit
h264-10-imp-ipb@p10|--codec h264 --qp 26 --gop 8 --bframes 2 --bweight implicit
h264-imp-wp-ipb@fade|--codec h264 --qp 26 --gop 8 --bframes 2 --wpred --bweight implicit
h264-10-imp-ipb@wsine10|--codec h264 --qp 26 --gop 8 --bframes 2 --bweight implicit
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
  # Our decoder conceals what a stream gets wrong — a missing reference gets a
  # generated stand-in, an out-of-range syntax element a clamp — and counts
  # each time it did ("N frames, W warnings" on stderr). A stream from our own
  # encoder has nothing to conceal, so any warning is a defect even when the
  # pictures still match the reconstruction: the RPS that dropped a later P
  # picture's anchor passed the two checks above and was caught only by CROSS.
  selfwarn=$(sed -nE 's/^[0-9]+ frames, ([0-9]+) warnings$/\1/p' "$OUT/$base.$name.dec.log" | tail -n 1)
  if [ -z "$selfwarn" ]; then
    echo "SELF-FAIL   $tag: our decoder printed no frame/warning count"
    return 1
  elif [ "$selfwarn" -ne 0 ]; then
    detail=$(grep -vE '^[0-9]+ frames, ' "$OUT/$base.$name.dec.log" | head -1 | head -c 80)
    echo "SELF-FAIL   $tag: our decoder concealed $selfwarn problem(s) in our bitstream${detail:+: $detail}"
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
      # 4b. SUSTAINED. On a clip long enough to converge, the spend over
      # every RATE_WINDOW consecutive GOPs after the first must sit inside
      # [RATE_WINDOW_LO, RATE_WINDOW_HI] of target (see 4 above).
      pics=$(sed -n 's/^\([0-9]*\) pictures, [0-9]* bytes$/\1/p' "$OUT/$base.$name.enc.log" | tail -1)
      if ! out=$(gop_spend_of "$bs" "$ratio" "$pics"); then
        echo "RATE-FAIL   $tag: $out"
        return 1
      fi
      verdict=$(echo "$out" | awk -v w="$RATE_WINDOW" -v lo="$RATE_WINDOW_LO" -v hi="$RATE_WINDOW_HI" '{
        if (NF < w + 1) { print "short"; exit }
        for (i = 2; i + w - 1 <= NF; i++) {
          s = 0; for (j = i; j < i + w; j++) s += $j
          if (s / w < lo || s / w > hi) { printf "GOPs %d-%d of %d spent %.2fx of target, outside [%.2f, %.2f] (per GOP: %s)\n", i, i + w - 1, NF, s / w, lo, hi, $0; exit }
        }
        print "held"
      }')
      case "$verdict" in
        short|held) ;;
        *) echo "RATE-FAIL   $tag: $verdict"; return 1 ;;
      esac
      # 4c. VERDICT. A `-verdict` row was built to reach the insensitivity
      # verdict: it must report one, and a probe (see 4c above).
      case "$name" in
        *-verdict*)
          ins=$(sed -n 's/^rate: insensitivity verdicts \([0-9]*\), probes \([0-9]*\), releases \([0-9]*\)$/\1 \2 \3/p' "$OUT/$base.$name.enc.log" | tail -1)
          read -r ins_v ins_p ins_r <<< "$ins"
          if [ -z "$ins" ] || [ "$ins_v" -lt 1 ] || [ "$ins_p" -lt 1 ]; then
            echo "RATE-FAIL   $tag: insensitivity verdicts ${ins_v:-unreported}, probes ${ins_p:-unreported} (releases ${ins_r:-unreported}); a -verdict row must reach at least one verdict and one probe"
            return 1
          fi
          ;;
      esac
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
  # 9. QUALITY FLOOR. Only with --quality-baseline. Lossless rows are
  # recorded but held by their exact check below rather than a tolerance.
  if [ -n "$QBASE" ]; then
    planes=$(planes_psnr_of "$src" "$rec" "$depth" "$geom" "$chroma")
    echo "$tag $planes" > "$OUT/$base.$name.quality"
    floor=""
    [ -f "$QBASE" ] && floor=$(awk -v t="$tag" '$1 == t { print $2, $3, $4; exit }' "$QBASE")
    case "$flags" in *--lossless*) floor="" ;; esac
    if [ -n "$floor" ]; then
      verdict=$(quality_verdict "$planes" "$floor" "$QTOL")
      case "$verdict" in
        below*) echo "QUALITY-FAIL $tag: ${verdict#below }"; return 1 ;;
        above*) echo "$tag: ${verdict#above }" > "$OUT/$base.$name.improved" ;;
      esac
    fi
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
# Property 9's measurement: luma, Cb and Cr PSNR over every picture,
# exact, "-" for a plane the format does not have and "inf" for one
# reproduced exactly. Geometry and chroma format give the plane sizes.
planes_psnr_of() {
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import sys, math, array
depth = int(sys.argv[3])
w, h = (int(v) for v in sys.argv[4].split("x"))
cw, ch = {"400": (0, 0), "gray": (0, 0), "420": ((w + 1) // 2, (h + 1) // 2),
          "422": ((w + 1) // 2, h), "444": (w, h)}[sys.argv[5]]
def samples(path):
    data = open(path, "rb").read()
    if depth <= 8:
        return data
    s = array.array("H"); s.frombytes(data[:len(data) & ~1])
    if sys.byteorder != "little":
        s.byteswap()
    return s
a, b = samples(sys.argv[1]), samples(sys.argv[2])
fs = w * h + 2 * cw * ch
n = min(len(a), len(b)) // fs
peak = (1 << depth) - 1
out = []
for off, size in ((0, w * h), (w * h, cw * ch), (w * h + cw * ch, cw * ch)):
    if size == 0 or n == 0:
        out.append("-")
        continue
    se = 0
    for f in range(n):
        o = f * fs + off
        se += sum((x - y) * (x - y) for x, y in zip(a[o:o + size], b[o:o + size]))
    mse = se / (n * size)
    out.append("inf" if mse == 0 else f"{10 * math.log10(peak * peak / mse):.2f}")
print(" ".join(out))
PY
}

# Property 9's comparison of a cell's planes against its floor: "below"
# when any plane fell more than the tolerance under its recorded PSNR,
# else "above" when any rose more than it, else "held". An exact plane
# ("inf") ranks above every number.
quality_verdict() {
  awk -v got="$1" -v floor="$2" -v tol="$3" 'BEGIN {
    split(got, g, " "); split(floor, f, " "); split("luma Cb Cr", plane, " ")
    below = ""; above = ""
    for (i = 1; i <= 3; i++) {
      if (g[i] == "" || f[i] == "" || g[i] == "-" || f[i] == "-") continue
      gv = (g[i] == "inf") ? 1e9 : g[i] + 0
      fv = (f[i] == "inf") ? 1e9 : f[i] + 0
      d = gv - fv
      if (gv == 1e9 || fv == 1e9) note = sprintf("%s %s dB, recorded %s", plane[i], g[i], f[i])
      else note = sprintf("%s %s dB, recorded %s (%+.2f)", plane[i], g[i], f[i], d)
      if (d < -tol) below = below (below == "" ? "" : "; ") note
      else if (d > tol) above = above (above == "" ? "" : "; ") note
    }
    if (below != "") print "below " below "; tolerance " tol " dB"
    else if (above != "") print "above " above
    else print "held"
  }'
}
# Property 4b's measurement: each GOP's spend against the target, from the
# bytes of the stream alone. Splits the Annex-B stream into access units
# (H.264 or H.265, told apart by the first NAL: an SPS or a VPS), cuts it
# into GOPs at keyframes in coding order, and prints one number per GOP:
# its bits per picture over the clip's bits per picture, times the ratio
# the encoder reported for the whole clip. So the numbers are relative to
# the target without this script knowing the frame rate, and they average
# to the encoder's own ratio. The split is checked against the file size
# and against the encoder's picture count, and a mismatch is a failure —
# a splitter that miscounted would otherwise measure the wrong pictures
# and pass.
gop_spend_of() {
  python - "$1" "$2" "$3" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
ratio, pictures = float(sys.argv[2]), int(sys.argv[3] or 0)
starts = []
i = data.find(b"\x00\x00\x01")
while i >= 0:
    starts.append((i - 1 if i > 0 and data[i - 1] == 0 else i, i + 3))
    i = data.find(b"\x00\x00\x01", i + 3)
h264 = bool(starts) and data[starts[0][1]] & 0x1F == 7
units, begin, vcl_seen, key = [], 0, False, False
for k, (s, h) in enumerate(starts):
    end = starts[k + 1][0] if k + 1 < len(starts) else len(data)
    if h264:
        t = data[h] & 0x1F
        vcl = 1 <= t <= 5
        first = vcl and h + 1 < end and data[h + 1] & 0x80
        opens = t in (6, 7, 8, 9, 14, 15, 16, 17, 18)
        is_key = t == 5
    else:
        t = (data[h] >> 1) & 0x3F
        vcl = t <= 31
        first = vcl and h + 2 < end and data[h + 2] & 0x80
        opens = 32 <= t <= 39 or 41 <= t <= 44 or 48 <= t <= 55
        is_key = 16 <= t <= 21
    if vcl_seen and (first or (not vcl and opens)):
        units.append((s - begin, key))
        begin, vcl_seen, key = s, False, False
    vcl_seen |= bool(vcl)
    key |= is_key
units.append((len(data) - begin, key))
if len(units) != pictures or sum(u for u, _ in units) != len(data):
    print(f"the stream splits into {len(units)} access units of {sum(u for u, _ in units)} bytes; the encoder reported {pictures} pictures in {len(data)}")
    sys.exit(1)
gops = []
for size, is_key in units:
    if is_key or not gops:
        gops.append([])
    gops[-1].append(size)
per_picture = len(data) / len(units)
print(" ".join(f"{ratio * sum(g) / len(g) / per_picture:.3f}" for g in gops))
PY
}
# Property 4b's thresholds (see 4).
RATE_WINDOW=${RATE_WINDOW:-3}
RATE_WINDOW_LO=${RATE_WINDOW_LO:-0.85}
RATE_WINDOW_HI=${RATE_WINDOW_HI:-1.18}
export -f one ffpix psnr_of planes_psnr_of quality_verdict chroma_of depth_of frame_bytes gop_spend_of
export ENC DEC HRD FFMPEG FFPROBE OUT PARAM_SETS VUI_PROBE H26X_SPEED_TABLE JOBS QBASE QTOL RATE_WINDOW RATE_WINDOW_LO RATE_WINDOW_HI

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
bad=$(grep -cE '^(ENCODE|SELF|CROSS|LOSSLESS|RATE|HRD|PS|VUI|QUALITY)-FAIL' "$results")
echo
echo "encode: $pass passed, $bad failed"
[ "$bad" = 0 ] || fail=1
# 7. The speed total: encoder wall seconds summed over every PASS cell.
# Not a gate. The number to compare is this one against the last run's on
# the same machine; a per-cell table (H26X_SPEED_TABLE) says where it went.
sed -n 's/^PASS .*, \([0-9.]*\) s, \([0-9]*\) f\/s)$/\1 \2/p' "$results" \
  | awk -v cells="$pass" -v jobs="$JOBS" '{ s += $1 } END { printf "encode speed: %d cells, %.2f s of encoder wall time (summed over cells, %d in parallel)\n", cells, s, jobs }'

# 9. The quality floor's ledger: record it, or say what moved against it.
if [ -n "$QBASE" ]; then
  qrec="$OUT/quality.txt"
  cat "$OUT"/*.quality 2>/dev/null | sort > "$qrec"
  cells=$(wc -l < "$qrec")
  echo
  if [ "$cells" = 0 ]; then
    # Zero measured cells is the vacuity the tally above refuses too.
    echo "quality: no cell was measured"
    fail=1
  elif [ -f "$QBASE" ]; then
    cat "$OUT"/*.improved 2>/dev/null | sort > "$OUT/above.txt"
    awk 'NR == FNR { seen[$1] = 1; next } !($1 in seen) { print $1 }' "$QBASE" "$qrec" > "$OUT/unrecorded.txt"
    awk 'NR == FNR { seen[$1] = 1; next } !($1 in seen) { print $1 }' "$qrec" "$QBASE" > "$OUT/unmeasured.txt"
    echo "quality: $cells cells against $QBASE (tolerance $QTOL dB): $(grep -c '^QUALITY-FAIL' "$results") below the floor, $(wc -l < "$OUT/above.txt") above it, $(wc -l < "$OUT/unrecorded.txt") not recorded, $(wc -l < "$OUT/unmeasured.txt") recorded but not measured"
    sed 's/^/  above         /' "$OUT/above.txt"
    sed 's/^/  not recorded  /' "$OUT/unrecorded.txt"
    sed 's/^/  not measured  /' "$OUT/unmeasured.txt"
  elif [ "$fail" = 0 ]; then
    cp "$qrec" "$QBASE"
    echo "quality: recorded $cells cells to $QBASE"
  else
    echo "quality: NOT recorded to $QBASE: a floor recorded from a red run holds the red"
  fi
fi

echo
[ "$fail" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit $fail
