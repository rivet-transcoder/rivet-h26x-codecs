# ab.py <exeA> <exeB> <stream> [runs]: interleaved A/B, min and median CPU seconds.
#
# Alternates the two builds so that whatever the machine is doing drifts across
# both. Pins to one core, chosen at start-up as the least busy one rather than a
# fixed number: a fixed default becomes the contended core the moment it is
# written down and everyone pins to it — that happened, and a same-binary control
# on the "good" core read a 15% spread, worse than the problem the default was
# introduced to fix. `AFFINITY=0x100000` overrides; `MT=1` measures every thread
# and does not pin.
#
# Before believing any number: run this with the same binary as both arguments.
# That spread is the smallest difference this machine can currently resolve.
import ctypes
import os
import statistics
import subprocess
import sys
import time
from ctypes import wintypes

kernel32 = ctypes.windll.kernel32
HIGH_PRIORITY_CLASS = 0x00000080


def cpu_seconds(handle):
    c, e, k, u = (wintypes.FILETIME() for _ in range(4))
    kernel32.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(c), ctypes.byref(e),
                             ctypes.byref(k), ctypes.byref(u))
    ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return ft(k) + ft(u)


def quietest_core():
    """The least busy *physical* core, as a (mask, index, load%) triple.

    Sampled once at start-up. Picking the quietest logical processor is not
    enough on an SMT machine: its sibling shares the execution units, so a busy
    sibling costs you as much as a busy core. Logical processors are numbered
    in sibling pairs, so this scores pairs and takes the quieter half of the
    quietest pair. Falls back to the highest-numbered core if the counters
    cannot be read — still better than a low one, where the rest of the system
    lands.
    """
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


MT = os.environ.get("MT") == "1"
MASK, CORE, LOAD = (0, 0, None) if MT else quietest_core()


def run(exe, f):
    e = dict(os.environ)
    e.update({"H26XDEC_NOMD5": "1"})
    if not MT:
        e["H26X_THREADS"] = "1"
    t = time.perf_counter()
    p = subprocess.Popen([exe, f], env=e, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=HIGH_PRIORITY_CLASS)
    if not MT:
        kernel32.SetProcessAffinityMask(wintypes.HANDLE(p._handle), ctypes.c_size_t(MASK))
    p.wait()
    return time.perf_counter() - t, cpu_seconds(p._handle)


a, b, f = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 7
if not MT:
    where = f"core {CORE}" + (f" (its SMT pair {LOAD:.0f}% busy)" if LOAD is not None else " (AFFINITY)")
    if LOAD is not None and LOAD > 15:
        where += ("\n  WARNING: the quietest pair on this machine is already busy."
                  "\n  Run the same binary against itself first, and do not believe"
                  "\n  a difference smaller than the spread that reports.")
    print(f"pinned to {where}")
ra, rb = [], []
for i in range(n):
    ra.append(run(a, f))
    rb.append(run(b, f))
ca = [c for _, c in ra]
cb = [c for _, c in rb]
wa = [w for w, _ in ra]
wb = [w for w, _ in rb]
# The headline is the median of the *paired* ratios, one per round. A and B run
# back to back within a round, so pairing them cancels drift that neither the
# ratio-of-medians nor the ratio-of-minimums does; and the median discards the
# round where something else woke up. Minimum is not the statistic to lead
# with here — with interleaving on a live machine one lucky run poisons it, and
# a same-binary control has been seen to read 1.120 and 0.862 on consecutive
# rounds while its medians read 1.018 and 1.000.
paired = sorted(y / x for x, y in zip(ca, cb))
ratio = statistics.median(paired)
print(f"A {os.path.basename(a)}: cpu med={statistics.median(ca):.3f} min={min(ca):.3f}  wall min={min(wa):.3f}")
print(f"B {os.path.basename(b)}: cpu med={statistics.median(cb):.3f} min={min(cb):.3f}  wall min={min(wb):.3f}")
print(f"B/A = {ratio:.3f}  (paired ratios {paired[0]:.3f}..{paired[-1]:.3f}, {n} rounds)")
spread = paired[-1] - paired[0]
if spread > 0.02:
    print(f"  the paired ratios span {spread * 100:.0f}% — wider than a change worth "
          f"believing, so treat {ratio:.3f} as 'no measurable difference' unless a\n"
          f"  same-binary control on this machine spans less than that")
