#!/usr/bin/env python3
"""benchmark.py — this decoder against libavcodec, per instruction-set tier.

Emits Markdown tables: one per stream, rows for each SIMD tier this CPU can
run and for ffmpeg, columns for single-threaded and all-threads. The point of
splitting by tier is that "how fast is it" has no answer without saying which
instructions it was allowed to use — and the tier a machine actually takes is
chosen at run time, so the table doubles as a map of what different hardware
gets.

Both sides decode and materialise every frame: ffmpeg writes rawvideo to the
null device, this decoder packs each picture and drops it. Neither writes to
disk. Cost is CPU seconds (user+kernel) — wall time on a machine doing
anything else measures the scheduler — and throughput is frames per wall
second, which is what a multi-threaded run is actually for.

Best of N, not the median: these are absolute costs, where the fastest run is
the one least disturbed by everything else on the machine. That is the
opposite of the choice `ab.py` makes, and for the opposite reason — comparing
two builds divides one noisy measurement by another, and there the minimum is
the statistic a single lucky run can poison.

  python benchmark.py [--runs N] [--dec PATH] [--ffmpeg PATH] [--streams a,b]
"""
import argparse
import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from ctypes import wintypes

WINDOWS = os.name == "nt"
# The tables are Markdown destined for a UTF-8 README, and Windows
# consoles still default to a code page that cannot hold an em dash;
# redirected to a file that silently becomes a replacement character.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HIGH_PRIORITY = 0x00000080 if WINDOWS else 0


def cpu_name():
    if WINDOWS:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            if out:
                return " ".join(out.split())
        except Exception:
            pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cpu_threads():
    return os.cpu_count() or 1


class JobAccounting(ctypes.Structure):
    """`JOBOBJECT_BASIC_ACCOUNTING_INFORMATION`."""
    _fields_ = [("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD)]


def job_cpu_seconds(job):
    """CPU time of every process that ran in `job`, finished or not.

    A process is the wrong unit to measure. `ffmpeg` on this machine is a
    136 KB scoop shim that spawns the real binary as a child, so asking the
    process we launched how much CPU it used answered 0.03 seconds for work
    that took 1.44 — a 46x understatement that reads as a spectacular win for
    whatever it is being compared against. A job object catches the children.
    """
    info = JobAccounting()
    ok = ctypes.windll.kernel32.QueryInformationJobObject(
        job, 1, ctypes.byref(info), ctypes.sizeof(info), None)
    if not ok:
        raise OSError("QueryInformationJobObject failed")
    return (info.TotalUserTime + info.TotalKernelTime) / 1e7


PIN_MASK = None


def run_once(cmd, env, pin=False):
    """(CPU seconds of the whole process tree, wall seconds)."""
    if not WINDOWS:
        t = time.perf_counter()
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        _, _, r = os.wait4(p.pid, 0)
        wall = time.perf_counter() - t
        # ru_children is not available per-call here; the direct child's own
        # usage is what os.wait4 reports, which is right when nothing wraps it.
        return r.ru_utime + r.ru_stime, wall
    k32 = ctypes.windll.kernel32
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")
    try:
        t = time.perf_counter()
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             creationflags=HIGH_PRIORITY)
        k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(p._handle)))
        if pin and PIN_MASK:
            k32.SetProcessAffinityMask(wintypes.HANDLE(int(p._handle)),
                                       ctypes.c_size_t(PIN_MASK))
        p.wait()
        wall = time.perf_counter() - t
        return job_cpu_seconds(job), wall
    finally:
        k32.CloseHandle(job)


def run_best(cmd, env_extra, runs):
    """(best CPU seconds, best wall seconds) over `runs` runs.

    Refuses a result whose CPU time is far below its wall time on a run that
    should be CPU-bound: that means work escaped the measurement rather than
    that the program was fast, and a table built from it is worse than no
    table. Single-threaded runs are the check, because a threaded one can
    legitimately spend its wall time waiting.
    """
    env = dict(os.environ)
    env.update(env_extra)
    single = env.get("H26X_THREADS") == "1" or "-threads" in cmd
    best_cpu, best_wall = float("inf"), float("inf")
    for _ in range(runs):
        cpu, wall = run_once(cmd, env, pin=single)
        if single and wall > 0.3 and cpu < 0.5 * wall:
            raise SystemExit(
                f"refusing to report {cpu:.3f} CPU s against {wall:.3f} wall s"
                f" for a single-threaded run of: {' '.join(map(str, cmd))}."
                " Work escaped the measurement — the usual cause is a launcher"
                " or shim that runs the real program as a child. Point --ffmpeg"
                " or --dec at the real executable.")
        best_cpu, best_wall = min(best_cpu, cpu), min(best_wall, wall)
    return best_cpu, best_wall


def stream_info(dec, path):
    """(frames, width, height) from the decoder's own frame list."""
    out = subprocess.run([dec, path], capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return 0, 0, 0
    m = re.search(r"(\d+)x(\d+)", lines[-1])
    return len(lines), (int(m.group(1)) if m else 0), (int(m.group(2)) if m else 0)


def available_tiers():
    """The rungs of this architecture's ladder, widest first.

    `H26X_MAX_SIMD` caps the ladder rather than selecting a rung, so on a CPU
    that lacks a rung the cap lands on the next one down and its row simply
    repeats the row below — which is itself informative, and never wrong.
    """
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return [("NEON + DotProd", {}), ("NEON", {"H26X_MAX_SIMD": "neon"}),
                ("scalar", {"H26X_NO_SIMD": "1"})]
    return [("AVX-512", {}), ("AVX2", {"H26X_MAX_SIMD": "avx2"}),
            ("AVX (VEX-128)", {"H26X_MAX_SIMD": "avx"}),
            ("SSE4.1", {"H26X_MAX_SIMD": "sse41"}),
            ("SSSE3", {"H26X_MAX_SIMD": "ssse3"}),
            ("SSE2", {"H26X_MAX_SIMD": "sse2"}),
            ("scalar", {"H26X_NO_SIMD": "1"})]


def control_line(name, selected_row, control_row, f1, fm, frames):
    """Per-column spread between two runs of the *same* configuration.

    Every figure in a table is worth exactly as much as the difference
    between two runs of identical code, and that difference is not one
    number — it depends on the column. On this machine the same binary has
    disagreed with itself by 0.0% in all-thread CPU seconds and 4.0% in
    all-thread frames per second, in the same table. Quoting a 2% win from
    the second of those is quoting noise, and it happened here: two
    published multi-threaded comparisons turned out to be inside their own
    control, which nobody noticed because the control was printed once and
    applied never.

    So it is printed per column, under every table, including the ones where
    no two rungs happen to run identical code.
    """
    a, b = selected_row, control_row
    cols = [
        ("1-thread CPU", a[1], b[1]),
        ("vs libav", a[1] / f1 if f1 else 0, b[1] / f1 if f1 else 0),
        ("1-thread fps", a[2], b[2]),
        ("all-thread CPU", a[3], b[3]),
        ("all-thread fps", a[4], b[4]),
    ]
    parts = []
    worst = 0.0
    for label, x, y in cols:
        lo, hi = min(x, y), max(x, y)
        d = (hi / lo - 1) * 100 if lo else 0.0
        worst = max(worst, d)
        parts.append(f"{label} {d:.1f}%")
    print(f"\nControl for `{name}`, the selected rung run twice: "
          + ", ".join(parts)
          + ". Read each figure against its OWN column here, not against the"
            " smallest of them: a difference under that column number is noise,"
            " whatever the other columns say.")
    return worst


def check_ladder(name, rows, tol=0.05):
    """Rungs that beat the rung above them, which cannot happen honestly.

    Best-of-N defends against a brief interruption; it cannot defend against
    something that runs for the whole benchmark, which shifts every row and
    leaves the table looking perfectly plausible. The ladder itself is the
    control: it is ordered by construction, so SSE4.1 coming out 20% ahead of
    AVX2 is not a measurement, it is a machine that was busy. Reported loudly,
    because a wrong table is worse than no table.
    """
    bad = []
    for i in range(len(rows) - 1):
        higher, lower = rows[i], rows[i + 1]
        if higher[1] > lower[1] * (1 + tol):
            bad.append("{} ({:.3f} s) beats {} ({:.3f} s) by {:.0f}%".format(
                lower[0], lower[1], higher[0], higher[1],
                (higher[1] / lower[1] - 1) * 100))
    if bad:
        print()
        print("> **These `" + name + "` numbers are not trustworthy.** A lower "
              "rung of the ladder came out ahead of a higher one, which does "
              "not happen on a quiet machine:")
        for b in bad:
            print("> - " + b)
        print("> Re-run on an idle machine before believing this table.")
    return not bad


def quietest_core():
    """A (mask, index) for the quieter half of the least busy SMT pair.

    Single-threaded rows are pinned to it, both this decoder's and ffmpeg's,
    so that the column the ladder comparison rests on survives a machine that
    is doing other things. It is the same reasoning as tools/ab.py: picking
    the quietest logical processor is not enough, because a busy SMT sibling
    shares the execution units and costs as much as a busy core.

    The all-threads rows are deliberately NOT pinned — pinning them would
    measure something other than what they claim to. That column is the one
    an ambient load distorts, which is why the load is printed in the header.
    """
    n = os.cpu_count() or 1
    if os.environ.get("AFFINITY"):
        m = int(os.environ["AFFINITY"], 0)
        return m, m.bit_length() - 1
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(*)\\% Processor Time' -SampleInterval 1"
             " -MaxSamples 1).CounterSamples | "
             "% { \"$($_.InstanceName) $($_.CookedValue)\" }"],
            capture_output=True, text=True, timeout=90).stdout
        load = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                load[int(parts[0])] = float(parts[1])
        if load:
            pairs = [(load.get(i, 0.0) + load.get(i + 1, 0.0), i)
                     for i in range(0, n, 2)]
            _, i = min(pairs)
            j = i if load.get(i, 0.0) <= load.get(i + 1, 0.0) else i + 1
            return 1 << j, j
    except Exception:
        pass
    return 1 << (n - 1), n - 1


def ambient_load():
    """Percentage of the machine already busy, sampled before the run.

    Printed with the table because it is the single fact most likely to
    explain a number somebody cannot reproduce. The all-threads column is the
    one it distorts: a single-threaded run on a 32-thread box can find an idle
    core, a run that wants every thread is competing for them.
    """
    if not WINDOWS:
        try:
            return os.getloadavg()[0] / (os.cpu_count() or 1) * 100
        except OSError:
            return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\Processor(_Total)\% Processor Time' "
             "-SampleInterval 2 -MaxSamples 1).CounterSamples.CookedValue"],
            capture_output=True, text=True, timeout=90).stdout.strip()
        return float(out.replace(",", "."))
    except Exception:
        return None


def selected_rung(dec):
    """The rung the decoder picks when nothing caps it — what a user gets."""
    try:
        return subprocess.run([dec, "--rung"], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dec", default="../release/examples/h26xdec.exe" if WINDOWS
                    else "../release/examples/h26xdec")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    ap.add_argument("--streams", default="")
    args = ap.parse_args()

    null = "NUL" if WINDOWS else "/dev/null"
    # Long enough that the clock is not the instrument: process CPU time comes
    # in ~15.6 ms steps on Windows, so a clip that decodes in a fifth of a
    # second is quantised to within a few per cent of itself and every
    # comparison reads 1.000. These are the short clips repeated.
    streams = [s for s in args.streams.split(",") if s] or [
        f for f in ("cabac3.264", "cavlc3.264", "hevc6.265", "wpp10.265",
                    "bbb_720p_cabac.264", "bbb_720p_cavlc.264",
                    "bbb_720p_hevc.265", "bbb_720p_wpp.265") if os.path.exists(f)][:4]
    threads = cpu_threads()
    tiers = available_tiers()
    picked = selected_rung(args.dec)
    selected = next((i for i, (n, _) in enumerate(tiers) if n == picked), -1)

    global PIN_MASK
    PIN_MASK, pin_core = quietest_core()
    load = ambient_load()
    busy = f" Machine {load:.0f}% busy before the run." if load is not None else ""
    print(f"**{cpu_name()}**, {threads} hardware threads, "
          f"selecting **{picked or 'unknown'}**.{busy} Single-threaded rows "
          f"are pinned to core {pin_core} (both decoders). Best of {args.runs}. "
          f"Cost is CPU seconds, throughput is frames per wall second.\n")

    clean = True
    for path in streams:
        frames, w, h = stream_info(args.dec, path)
        codec = "H.265" if path.endswith((".265", ".hevc")) else "H.264"
        print(f"### `{path}` — {codec}, {w}x{h}, {frames} frames\n")
        print("| instructions | 1 thread: CPU s | vs libav | fps | "
              "all threads: CPU s | fps |")
        print("|---|---:|---:|---:|---:|---:|")
        f1, _ = run_best([args.ffmpeg, "-threads", "1", "-i", path, "-f",
                          "rawvideo", "-y", null], {}, args.runs)
        fm, fmw = run_best([args.ffmpeg, "-i", path, "-f", "rawvideo", "-y",
                            null], {}, args.runs)
        rows = []
        for name, extra in tiers:
            e1 = dict(extra, H26X_THREADS="1", H26XDEC_NOMD5="1")
            em = dict(extra, H26XDEC_NOMD5="1")
            c1, _ = run_best([args.dec, path], e1, args.runs)
            cm, wm = run_best([args.dec, path], em, args.runs)
            rows.append((name, c1, frames / c1 if c1 else 0, cm,
                         frames / wm if wm else 0))
        for i, (name, c1, fps1, cm, fpsm) in enumerate(rows):
            # The rung this CPU selects on its own is the one a user gets.
            tag = f"**{name}**" if i == selected else name
            print(f"| {tag} | {c1:.3f} | {c1 / f1:.2f}x | {fps1:.0f} | "
                  f"{cm:.3f} | {fpsm:.0f} |")
        print(f"| libavcodec | {f1:.3f} | 1.00x | {frames / f1 if f1 else 0:.0f} | "
              f"{fm:.3f} | {frames / fmw if fmw else 0:.0f} |")
        # The control: the selected rung again, nothing changed.
        sel = tiers[selected] if 0 <= selected < len(tiers) else tiers[0]
        ce1 = dict(sel[1], H26X_THREADS="1", H26XDEC_NOMD5="1")
        cem = dict(sel[1], H26XDEC_NOMD5="1")
        cc1, _ = run_best([args.dec, path], ce1, args.runs)
        ccm, ccw = run_best([args.dec, path], cem, args.runs)
        ctl = ("control", cc1, frames / cc1 if cc1 else 0, ccm,
               frames / ccw if ccw else 0)
        control_line(path, rows[selected if 0 <= selected < len(rows) else 0],
                     ctl, f1, fm, frames)

        clean &= check_ladder(path, rows)
        best = rows[0]
        print(f"\nWidest rung against libavcodec: **{best[1] / f1:.2f}x** its "
              f"single-threaded CPU time, **{best[3] / fm:.2f}x** with every "
              f"thread; SIMD is worth **{rows[-1][1] / best[1]:.1f}x** over the "
              f"scalar reference.\n")
    if not clean:
        print("At least one table failed the ladder check; exiting non-zero "
              "so a regeneration cannot publish it unnoticed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
