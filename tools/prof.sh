#!/bin/bash
# prof.sh <out-name> <cmd...>: samply-record at 8 kHz (single thread env from caller), then symbolicate + aggregate.
# Needs the profiling build: CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=none cargo build --release -p rivet-h26x --examples --target-dir target/prof
name=$1; shift
samply record -r 8000 --save-only -o "$name.json.gz" "$@" > /dev/null 2> "$name.samply.log"
port=$((3000 + RANDOM % 2000))
samply load --no-open -P $port "$name.json.gz" > "$name.load.log" 2>&1 &
pid=$!
for i in $(seq 1 50); do grep -q "Local server" "$name.load.log" 2>/dev/null && break; sleep 0.2; done
token=$(grep -o "127.0.0.1%3A$port%2F[a-z0-9]*" "$name.load.log" | head -1 | sed 's/.*%2F//')
python "$(dirname "$0")/symprof.py" "$name.json.gz" "http://127.0.0.1:$port/$token" 0 ${TOP:-40} > "$name.txt" 2>&1
kill $pid 2>/dev/null
cat "$name.txt"
