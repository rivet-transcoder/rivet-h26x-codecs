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


def run_once(cmd, env):
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
        cpu, wall = run_once(cmd, env)
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
    return [("AVX2", {}), ("AVX (VEX-128)", {"H26X_MAX_SIMD": "avx"}),
            ("SSE4.1", {"H26X_MAX_SIMD": "sse41"}),
            ("SSSE3", {"H26X_MAX_SIMD": "ssse3"}),
            ("SSE2", {"H26X_MAX_SIMD": "sse2"}),
            ("scalar", {"H26X_NO_SIMD": "1"})]


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

    print(f"**{cpu_name()}**, {threads} hardware threads, "
          f"selecting **{picked or 'unknown'}**. Best of {args.runs}. "
          f"Cost is CPU seconds, throughput is frames per wall second.\n")

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
        best = rows[0]
        print(f"\nWidest rung against libavcodec: **{best[1] / f1:.2f}x** its "
              f"single-threaded CPU time, **{best[3] / fm:.2f}x** with every "
              f"thread; SIMD is worth **{rows[-1][1] / best[1]:.1f}x** over the "
              f"scalar reference.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
