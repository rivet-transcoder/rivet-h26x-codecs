#!/bin/bash
# Fetch the JVT professional-profiles conformance suite (4:4:4, 4:2:2 and 10-bit intra profiles).
set -u
base=https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance/professional_profiles
# list.txt names the archives to fetch; when it is absent (the work directory
# was deleted once and took the list with it) it is rebuilt from the site's own
# directory listing, which is what it was made from in the first place.
if [ ! -s list.txt ]; then
  curl -s -A 'Mozilla/5.0' --max-time 120 "$base/" | grep -io 'HREF="[^"]*"' | sed 's/HREF="//;s/"//' | sed 's#.*/##' \
    | grep -iE '\.(zip|bit|bin|264|jsv|jvt|avc|26l)$' | sort -u > list.txt
fi
mkdir -p zips streams
while read -r z; do
  [ -f "zips/$z" ] || curl -s --max-time 900 -o "zips/$z" "$base/$z"
done < list.txt
echo fetched
for z in zips/*; do
  [ -f "$z" ] || continue  # an empty zips/ leaves the glob unexpanded: no "streams/*" directory
  n=$(basename "$z"); d="streams/${n%.*}"
  if [ -d "$d" ]; then continue; fi
  mkdir -p "$d"
  case "$n" in
    *.zip) (cd "$d" && unzip -qq -o "../../$z" >/dev/null 2>&1) ;;
    *) cp "$z" "$d/" ;;
  esac
done
echo unzipped
