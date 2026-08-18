#!/bin/bash
# For each fixture: decode with h26xdec, compare per-frame MD5s with libavcodec's framemd5.
DEC=../release/examples/h26xdec.exe
for f in "$@"; do
  base="${f%.*}"
  $DEC "$f" > "$base.mine" 2> "$base.err"
  status=$?
  n_ref=$(grep -vc '^#' "$base.framemd5")
  n_mine=$(wc -l < "$base.mine")
  # compare md5 columns in order
  paste <(grep -v '^#' "$base.framemd5" | awk -F', *' '{print $6}') <(awk -F, '{print $5}' "$base.mine") | awk -v f="$f" -v nr="$n_ref" -v nm="$n_mine" '
    BEGIN{ok=0; bad=0; first=-1}
    { if ($1==$2) ok++; else { bad++; if (first<0) first=NR-1 } }
    END{ printf "%-40s frames ref=%s mine=%s  match=%d mismatch=%d first_bad=%d\n", f, nr, nm, ok, bad, first }'
  if [ $status -ne 0 ]; then echo "   -> $(tail -1 $base.err)"; fi
done
