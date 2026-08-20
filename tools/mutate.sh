#!/bin/bash
# mutate.sh — seed one fault at a time and require the checks to notice.
#
# A test suite is the last thing anyone checks, and an untested test is worth
# less than none: it reports green whether or not the code works, and it does
# so forever. Mutation testing is the only cheap way to find that out — break
# the code deliberately, and if everything still passes, the thing that did
# not fail is the thing that is not testing anything.
#
# It has already paid for itself twice in this repo. A B-picture replay test
# passed FIVE of six seeded faults, because its fixture reached only one
# signalling shape and its vacuity guards counted decisions that had gone
# intra. And omitting `cu_transquant_bypass_flag` on a skipped CU left the
# entire lossless encode row green, because every clip in the corpus was
# moving detail and a lossless CU on moving content is never a skip.
#
# ---------------------------------------------------------------------------
# RESTORE BY SNAPSHOT, NEVER BY `git checkout --`
#
# This harness copies the file before it edits it and moves the copy back
# afterwards. That is not a style preference. `git checkout -- <file>` reverts
# to HEAD, so it discards any UNCOMMITTED work in that file — and the code
# under test is very often uncommitted, because mutating it is how you decide
# whether it is finished. An earlier version of this harness used the git form
# and destroyed the code under test twice, once mid-way through writing it;
# the tell both times was later mutations reporting "anchor not found",
# because by then the function they were editing no longer existed.
#
# A harness must not be able to reach past its own edit. This one cannot.
# ---------------------------------------------------------------------------
#
# Usage: source this, define `mutate_check`, then call `mutate_try` per fault.
#
#   . tools/mutate.sh
#   mutate_check() { cargo test --quiet 2>&1 | grep -E 'test result'; }
#   mutate_try "cbf inverted" src/encode/h265.rs 'cbf as u32' '!cbf as u32'
#   mutate_report
#
# `mutate_try` takes NAME FILE FROM TO. FROM must appear EXACTLY ONCE in FILE
# — an anchor matching twice usually means it also matches a sibling writer,
# and applying it to both tests something other than what the name says. A
# non-unique or absent anchor is reported and skipped, never guessed at.
#
# Read the result as: a mutation your checks CATCH is evidence they work; a
# mutation they MISS is a finding. Sometimes the finding is that the mutation
# cannot fail — a value that is inert in this configuration, or a decision
# that only costs quality and no correctness check can see. Say so by name
# rather than counting it as a pass; a mutation that cannot fail is not a
# passing test, it is an absent one.

_MUTATE_CAUGHT=0
_MUTATE_MISSED=0
_MUTATE_SKIPPED=0
_MUTATE_MISSED_NAMES=()

# mutate_try NAME FILE FROM TO
mutate_try() {
  local name="$1" file="$2" from="$3" to="$4"
  echo
  echo "==== MUTATION: $name"
  if [ ! -f "$file" ]; then
    echo "     SKIPPED: no such file: $file"
    _MUTATE_SKIPPED=$((_MUTATE_SKIPPED + 1))
    return
  fi
  # Snapshot first, so every exit path below can restore.
  cp -f "$file" "$file.mutbak" || return
  if ! python - "$file" "$from" "$to" <<'PY'
import io, sys
path, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(path, encoding='utf-8').read()
n = s.count(a)
if n != 1:
    sys.stderr.write("anchor appears %d times, expected exactly 1\n" % n)
    sys.exit(1)
io.open(path, 'w', encoding='utf-8', newline='').write(s.replace(a, b, 1))
PY
  then
    echo "     SKIPPED: anchor not applied"
    mv -f "$file.mutbak" "$file"
    _MUTATE_SKIPPED=$((_MUTATE_SKIPPED + 1))
    return
  fi

  local out
  out="$(mutate_check 2>&1)"
  local rc=$?
  echo "$out" | sed 's/^/     /'
  # Caught when the checks failed: a non-zero status, or any line the caller's
  # own output marks as a failure.
  if [ $rc -ne 0 ] || echo "$out" | grep -qiE 'fail|error'; then
    echo "     -> CAUGHT"
    _MUTATE_CAUGHT=$((_MUTATE_CAUGHT + 1))
  else
    echo "     -> MISSED  <<< the checks did not notice this"
    _MUTATE_MISSED=$((_MUTATE_MISSED + 1))
    _MUTATE_MISSED_NAMES+=("$name")
  fi
  mv -f "$file.mutbak" "$file"
}

mutate_report() {
  echo
  echo "=== mutations: $_MUTATE_CAUGHT caught, $_MUTATE_MISSED missed, $_MUTATE_SKIPPED skipped"
  local n
  for n in "${_MUTATE_MISSED_NAMES[@]}"; do
    echo "    MISSED: $n"
  done
  # Any leftover snapshot means a run was interrupted mid-mutation; the file
  # on disk is then the MUTATED one, which must not be mistaken for source.
  local stray
  stray="$(find . -name '*.mutbak' -not -path './target/*' 2>/dev/null)"
  if [ -n "$stray" ]; then
    echo
    echo "    WARNING: leftover snapshots — the files beside them are MUTATED:"
    echo "$stray" | sed 's/^/      /'
    echo "    restore each with: mv -f F.mutbak F"
  fi
}
