#!/bin/bash
# Every conformance suite (JVT AVCv1+FRExt, JVT professional profiles, JCT-VC
# HEVC_v1, JCT-VC RExt) at once, from a frozen copy of the current h26xdec
# build (so a rebuild does not disturb the run). Prints each suite's tally.
cd "$(dirname "$0")"
cp ../../release/examples/h26xdec.exe ../../release/examples/h26xdec_conf.exe || exit 1
J=${JOBS:-6}
JOBS=$J bash h264/run_conf.sh > h264/run_all.log 2>&1 &
JOBS=$J bash h264_pp/run_conf.sh > h264_pp/run_all.log 2>&1 &
JOBS=$J bash hevc/run_conf.sh > hevc/run_all.log 2>&1 &
JOBS=$J bash hevc_rext/run_conf.sh > hevc_rext/run_all.log 2>&1 &
wait
for s in h264 h264_pp hevc hevc_rext; do
  printf "%-10s %s\n" "$s" "$(tail -1 $s/run_all.log)"
  grep -E "^FAIL|^NOSTREAM" $s/run_all.log
done
