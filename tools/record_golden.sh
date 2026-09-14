#!/bin/bash
# record_golden.sh <decoder> [dir] — print "<fixture> <md5>" for every
# fixture named in DIR/fixtures.md5 (DIR defaults to $H26X_WORK), in the
# form verify.sh compares: the MD5 of the decoder's frame listing — index,
# POC, decode index, size and per-frame MD5 — not of the YUV, so a picture
# dropped or reordered fails as surely as a wrong sample.
#
# Redirect to tools/golden.txt. Do that only after check.sh has shown every
# fixture identical to libavcodec: a golden records what the decoder DID,
# and recording one to make a red run green records the bug.
DEC=$1
DIR=${2:-$H26X_WORK}
[ -n "$DEC" ] && [ -f "$DEC" ] && [ -n "$DIR" ] \
  || { echo "usage: record_golden.sh <decoder> [dir]   (dir defaults to H26X_WORK)" >&2; exit 2; }
DEC=$(cd "$(dirname "$DEC")" && pwd)/$(basename "$DEC")
cd "$DIR" || exit 2
[ -s fixtures.md5 ] || { echo "record_golden.sh: no fixtures.md5 in $DIR — run make_fixtures.sh first" >&2; exit 2; }
while read -r _ f; do
  m=$(H26X_THREADS=4 "$DEC" "$f" 2>/dev/null | md5sum | cut -c1-32)
  # the MD5 of nothing: the decoder refused the stream or is not the decoder
  [ "$m" = d41d8cd98f00b204e9800998ecf8427e ] && { echo "record_golden.sh: $DEC produced no output for $f" >&2; exit 1; }
  echo "$f $m"
done < fixtures.md5
