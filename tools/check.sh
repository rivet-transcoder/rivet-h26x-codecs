#!/bin/bash
# check.sh <fixture...> — each fixture against the ITU-T reference decoder
# (JM's ldecod for .264, HM's TAppDecoder for .265, run by ref_decode.py),
# frame by frame: the frame count on each side, how many frames match, and
# the first mismatching frame. Exits 1 if any fixture differs in any frame
# or frame count, so it can gate: run it on a new fixture set before
# record_golden.sh, because a golden records what the decoder did, and this
# is what says whether that was right.
#
#   DEC=path          the decoder (default ../release/examples/h26xdec.exe,
#                     i.e. run from target/h26x)
#   LDECOD=path       JM's ldecod     } see ref_decode.py; default: on PATH
#   TAPPDECODER=path  HM's TAppDecoder}
DEC=${DEC:-../release/examples/h26xdec.exe}
[ -f "$DEC" ] || DEC=${DEC%.exe}
exec python "$(cd "$(dirname "$0")" && pwd)/ref_decode.py" check --dec "$DEC" "$@"
