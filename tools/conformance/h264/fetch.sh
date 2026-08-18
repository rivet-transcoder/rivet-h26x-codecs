#!/bin/bash
# Fetch the JVT AVCv1 + FRExt conformance suites (progressive-relevant streams included; the
# interlaced ones are downloaded too so the runner can prove they are refused cleanly).
set -u
base=https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance
for dir in AVCv1 FRExt; do
  mkdir -p $dir
  for z in $(curl -s --max-time 120 $base/$dir/ | grep -io 'HREF="[^"]*\.zip"' | sed 's/HREF="//;s/"//' | sed 's#.*/##'); do
    [ -f $dir/$z ] || curl -s --max-time 600 -o $dir/$z "$base/$dir/$z"
  done
done
echo fetched
for dir in AVCv1 FRExt; do
  cd $dir
  for z in *.zip; do d="${z%.zip}"; [ -d "$d" ] || (mkdir -p "$d" && cd "$d" && unzip -qq -o "../$z" >/dev/null 2>&1); done
  cd ..
done
echo unzipped
