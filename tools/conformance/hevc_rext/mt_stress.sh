#!/bin/bash
# Decode each stream N times at 12 threads and once at 1; report any md5 disagreement.
DEC=../../../release/examples/h26xdec_conf.exe
N=${N:-3}
bad=0
for d in streams/*/; do
  name=$(basename "$d")
  case "$1" in "") ;; *) echo "$name" | grep -qi -- "$1" || continue ;; esac
  bs=$(find "$d" -maxdepth 2 -type f \( -name "*.bit" -o -name "*.bin" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && continue
  cp "$bs" "out/$name.265"
  st=$(H26X_THREADS=1 $DEC "out/$name.265" 2>/dev/null | md5sum | cut -c1-8)
  res=""
  for i in $(seq 1 $N); do res="$res $(H26X_THREADS=12 $DEC "out/$name.265" 2>/dev/null | md5sum | cut -c1-8)"; done
  if [ "$(echo $res | tr ' ' '\n' | sort -u | wc -l)" != "1" ] || [ "$(echo $res | awk '{print $1}')" != "$st" ]; then
    echo "NONDET $name st=$st mt=$res"; bad=$((bad+1))
  fi
  rm -f "out/$name.265"
done
echo "nondeterministic=$bad"
