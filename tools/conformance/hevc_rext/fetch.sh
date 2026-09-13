#!/bin/bash
# Fetch the JCT-VC RExt conformance suite and unzip each stream into its own directory.
set -u
base=https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/RExt
# list.txt names the archives to fetch; when it is absent (the work directory
# was deleted once and took the list with it) it is rebuilt from the site's own
# directory listing, which is what it was made from in the first place.
if [ ! -s list.txt ]; then
  curl -s -A 'Mozilla/5.0' --max-time 120 "$base/" | grep -io 'HREF="[^"]*"' | sed 's/HREF="//;s/"//' | sed 's#.*/##' \
    | grep -iE '\.(zip|bit|bin|264|jsv|jvt|avc|26l)$' | sort -u > list.txt
fi
mkdir -p zips
while read -r z; do
  [ -f "zips/$z" ] || curl -s --max-time 900 -o "zips/$z" "$base/$z"
done < list.txt
echo fetched
mkdir -p streams
for z in zips/*.zip; do
  [ -f "$z" ] || continue  # an empty zips/ leaves the glob unexpanded: no "streams/*" directory
  d="streams/$(basename "${z%.zip}")"
  [ -d "$d" ] || (mkdir -p "$d" && cd "$d" && unzip -qq -o "../../$z" >/dev/null 2>&1)
done
echo unzipped
