#!/bin/bash
# Fetch the JCT-VC RExt conformance suite and unzip each stream into its own directory.
set -u
base=https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/RExt
mkdir -p zips
while read -r z; do
  [ -f "zips/$z" ] || curl -s --max-time 900 -o "zips/$z" "$base/$z"
done < list.txt
echo fetched
mkdir -p streams
for z in zips/*.zip; do
  d="streams/$(basename "${z%.zip}")"
  [ -d "$d" ] || (mkdir -p "$d" && cd "$d" && unzip -qq -o "../../$z" >/dev/null 2>&1)
done
echo unzipped
