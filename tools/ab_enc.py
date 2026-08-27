# ab_enc.py "ENV_A" "ENV_B" [runs] -- <encoder command...>: interleaved A/B
# of ONE encoder binary under two environments, min and median CPU seconds,
# and the median of the paired ratios.
#
# The encoder counterpart of ab.py, and the same method: one binary, two
# paths behind an environment variable, alternated so that whatever the
# machine is doing drifts across both, pinned to the quietest physical core.
# What differs is only what is being switched — the encoder has no stream to
# take as its one argument, so the whole command follows `--`, and the two
# sides are environment assignments rather than two executables:
#
#   python tools/ab_enc.py "H26X_ENC_NO_SIMD=1" "" 9 -- \
#       target/release/examples/h26xenc.exe --input clip.yuv --size 320x240 \
#       --format 420 --codec h265 --qp 26 --gop 8 --output /tmp/x.265
#
# `H26X_ENC_NO_SIMD=1` keeps the encode-only kernel tables scalar while the
# decoder-shared kernels keep their rungs, which is what isolates the
# encode-side SIMD; `H26X_NO_SIMD=1` would measure both at once. An empty
# side is the binary as it runs in the field. Several assignments may be
# space-separated in one side. `EXE=path` in a side replaces the binary for
# that side instead of setting a variable — for a change that has no
# switch, where the honest comparison is two builds.
#
# Before believing any number: run this with the same environment on both
# sides. That spread is the smallest difference this machine can resolve.
import ctypes
import os
import statistics
import subprocess
import sys
import time
from ctypes import wintypes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

kernel32 = ctypes.windll.kernel32
HIGH_PRIORITY_CLASS = 0x00000080


def cpu_seconds(handle):
    c, e, k, u = (wintypes.FILETIME() for _ in range(4))
    kernel32.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(c), ctypes.byref(e),
                             ctypes.byref(k), ctypes.byref(u))
    ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return ft(k) + ft(u)


def quietest_core():
    """The least busy physical core as (mask, index, load%) — see ab.py."""
    n = os.cpu_count() or 1
    if os.environ.get("AFFINITY"):
        m = int(os.environ["AFFINITY"], 0)
        return m, (m.bit_length() - 1), None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(*)\\% Processor Time' -SampleInterval 1"
             " -MaxSamples 1).CounterSamples | "
             "% { \"$($_.InstanceName) $($_.CookedValue)\" }"],
            capture_output=True, text=True, timeout=60).stdout
        load = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                load[int(parts[0])] = float(parts[1])
        if load:
            pairs = [(load.get(i, 0.0) + load.get(i + 1, 0.0), i) for i in range(0, n, 2)]
            _, i = min(pairs)
            j = i if load.get(i, 0.0) <= load.get(i + 1, 0.0) else i + 1
            return 1 << j, j, load.get(i, 0.0) + load.get(i + 1, 0.0)
    except Exception:
        pass
    return 1 << (n - 1), n - 1, None


def parse_env(spec):
    e = {}
    for item in spec.split():
        k, _, v = item.partition("=")
        e[k] = v
    return e


args = sys.argv[1:]
if "--" not in args or len(args[:args.index("--")]) < 2:
    sys.exit(__doc__ or "usage: ab_enc.py ENV_A ENV_B [runs] -- <command...>")
head = args[:args.index("--")]
cmd = args[args.index("--") + 1:]
env_a, env_b = parse_env(head[0]), parse_env(head[1])
n = int(head[2]) if len(head) > 2 else 7

MASK, CORE, LOAD = quietest_core()
where = f"core {CORE}" + (f" (its SMT pair {LOAD:.0f}% busy)" if LOAD is not None else " (AFFINITY)")
if LOAD is not None and LOAD > 50 and os.environ.get("AB_FORCE") != "1":
    sys.exit(f"refusing to measure: the quietest SMT pair on this machine is {LOAD:.0f}% busy. "
             f"Wait for it to drop, or set AB_FORCE=1 if you have a reason to want the numbers anyway.")
if LOAD is not None and LOAD > 15:
    where += ("\n  WARNING: the quietest pair on this machine is already busy."
              "\n  Run the same environment on both sides first, and do not believe"
              "\n  a difference smaller than the spread that reports.")
print(f"pinned to {where}")


def run(extra):
    e = dict(os.environ)
    e.update({k: v for k, v in extra.items() if k != "EXE"})
    argv = [extra.get("EXE", cmd[0])] + cmd[1:]
    t = time.perf_counter()
    p = subprocess.Popen(argv, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=HIGH_PRIORITY_CLASS)
    kernel32.SetProcessAffinityMask(wintypes.HANDLE(p._handle), ctypes.c_size_t(MASK))
    p.wait()
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}) under {extra or '{}'}")
    return time.perf_counter() - t, cpu_seconds(p._handle)


ra, rb = [], []
for i in range(n):
    ra.append(run(env_a))
    rb.append(run(env_b))
ca = [c for _, c in ra]
cb = [c for _, c in rb]
wa = [w for w, _ in ra]
wb = [w for w, _ in rb]
paired = sorted(y / x for x, y in zip(ca, cb))
ratio = statistics.median(paired)
print(f"A [{head[0] or 'as shipped'}]: cpu med={statistics.median(ca):.3f} min={min(ca):.3f}  wall min={min(wa):.3f}")
print(f"B [{head[1] or 'as shipped'}]: cpu med={statistics.median(cb):.3f} min={min(cb):.3f}  wall min={min(wb):.3f}")
print(f"B/A = {ratio:.3f}  (paired ratios {paired[0]:.3f}..{paired[-1]:.3f}, {n} rounds)")
spread = paired[-1] - paired[0]
if spread > 0.02:
    print(f"  the paired ratios span {spread * 100:.0f}% — wider than a change worth "
          f"believing, so treat {ratio:.3f} as 'no measurable difference' unless a\n"
          f"  same-environment control on this machine spans less than that")
