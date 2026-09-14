#!/bin/bash
# check.sh <fixture...> — each fixture against libavcodec's per-frame MD5s
# (`<name>.framemd5`, made with ffmpeg on first use), reporting the frame
# count on each side and the first mismatching frame. Exits 1 if any
# fixture differs in any frame or frame count, so it can gate: run it on
# a new fixture set before record_golden.sh, because a golden records
# what the decoder did, and this is what says whether that was right.
#
# A fixture named in `fixtures.libav_deviates` beside it (written by
# make_fixtures.sh, with the evidence) is one where libavcodec is known to
# be wrong: its mismatch is reported as DEVIATES and does not fail the run.
#
#   DEC=path     the decoder (default ../release/examples/h26xdec.exe,
#                i.e. run from target/h26x)
#   FFMPEG=path  ffmpeg for the reference (default: ffmpeg on PATH)
DEC=${DEC:-../release/examples/h26xdec.exe}
FFMPEG=${FFMPEG:-ffmpeg}
fail=0
for f in "$@"; do
  base="${f%.*}"
  if why=$(grep -m1 "^$(basename "$f") " "$(dirname "$f")/fixtures.libav_deviates" 2>/dev/null); then
    "$DEC" "$f" > "$base.mine" 2> "$base.err" || { echo "$f: the decoder failed: $(tail -1 "$base.err")"; fail=1; continue; }
    printf "%-40s DEVIATES (libavcodec): %s\n" "$f" "${why#* }"
    continue
  fi
  if [ ! -s "$base.framemd5" ]; then
    "$FFMPEG" -v error -nostdin -y -i "$f" -f framemd5 "$base.framemd5" 2>/dev/null \
      || { echo "$f: ffmpeg could not decode it"; rm -f "$base.framemd5"; fail=1; continue; }
  fi
  "$DEC" "$f" > "$base.mine" 2> "$base.err"
  status=$?
  n_ref=$(grep -vc '^#' "$base.framemd5")
  n_mine=$(wc -l < "$base.mine")
  # compare md5 columns in order
  paste <(grep -v '^#' "$base.framemd5" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "$base.mine") \
    | awk -v f="$f" -v nr="$n_ref" -v nm="$n_mine" '
    BEGIN{ok=0; bad=0; first=-1}
    { if ($1==$2) ok++; else { bad++; if (first<0) first=NR-1 } }
    END{ printf "%-40s frames ref=%s mine=%s  match=%d mismatch=%d first_bad=%d\n", f, nr, nm, ok, bad, first
         exit (bad > 0 || nr != nm || nr == 0) ? 1 : 0 }' || fail=1
  if [ $status -ne 0 ]; then echo "   -> $(tail -1 "$base.err")"; fail=1; fi
done
exit $fail
