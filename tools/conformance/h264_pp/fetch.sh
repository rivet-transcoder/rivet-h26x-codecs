#!/bin/bash
# Fetch the JVT professional-profiles conformance suite (4:4:4, 4:2:2 and 10-bit intra profiles).
set -u
base=https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance/professional_profiles
mkdir -p zips streams
while read -r z; do
  [ -f "zips/$z" ] || curl -s --max-time 900 -o "zips/$z" "$base/$z"
done < list.txt
echo fetched
for z in zips/*; do
  n=$(basename "$z"); d="streams/${n%.*}"
  if [ -d "$d" ]; then continue; fi
  mkdir -p "$d"
  case "$n" in
    *.zip) (cd "$d" && unzip -qq -o "../../$z" >/dev/null 2>&1) ;;
    *) cp "$z" "$d/" ;;
  esac
done
echo unzipped
