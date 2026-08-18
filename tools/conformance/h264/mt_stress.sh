#!/bin/bash
DEC=../../../release/examples/h26xdec_conf.exe
N=${N:-2}
bad=0
for d in AVCv1/*/ FRExt/*/; do
  name=$(basename "$d")
  bs=$(find "$d" -maxdepth 2 -type f \( -iname "*.264" -o -iname "*.jsv" -o -iname "*.26l" -o -iname "*.h264" -o -iname "*.avc" -o -iname "*.jvt" -o -iname "*.bit" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && continue
  cp "$bs" "out/$name.264"
  st=$(H26X_THREADS=1 $DEC "out/$name.264" 2>/dev/null | md5sum | cut -c1-8)
  res=""
  for i in $(seq 1 $N); do res="$res $(H26X_THREADS=12 $DEC "out/$name.264" 2>/dev/null | md5sum | cut -c1-8)"; done
  if [ "$(echo $res | tr ' ' '\n' | sort -u | wc -l)" != "1" ] || [ "$(echo $res | awk '{print $1}')" != "$st" ]; then
    echo "NONDET $name st=$st mt=$res"; bad=$((bad+1))
  fi
  rm -f "out/$name.264"
done
echo "nondeterministic=$bad"
