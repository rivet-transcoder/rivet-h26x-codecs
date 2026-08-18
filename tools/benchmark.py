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


def process_cpu_seconds(handle):
    """CPU time of a finished process."""
    if WINDOWS:
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        ctypes.windll.kernel32.GetProcessTimes(
            wintypes.HANDLE(handle), ctypes.byref(c), ctypes.byref(e),
            ctypes.byref(k), ctypes.byref(u))
        ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
        return ft(k) + ft(u)
    r = os.wait4(handle, 0)[2]
    return r.ru_utime + r.ru_stime


def run_best(cmd, env_extra, runs):
    """(best CPU seconds, best wall seconds) over `runs` runs."""
    env = dict(os.environ)
    env.update(env_extra)
    best_cpu, best_wall = float("inf"), float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        kw = {"creationflags": HIGH_PRIORITY} if WINDOWS else {}
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **kw)
        p.wait()
        wall = time.perf_counter() - t
        cpu = process_cpu_seconds(p._handle if WINDOWS else p.pid)
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
            ("scalar", {"H26X_NO_SIMD": "1"})]


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

    print(f"**{cpu_name()}**, {threads} hardware threads. Best of {args.runs}. "
          f"Cost is CPU seconds, throughput is frames per wall second.\n")

    for path in streams:
        frames, w, h = stream_info(args.dec, path)
        codec = "H.265" if path.endswith((".265", ".hevc")) else "H.264"
        print(f"### `{path}` — {codec}, {w}x{h}, {frames} frames\n")
        print("| instructions | 1 thread: CPU s | fps | all threads: CPU s | fps |")
        print("|---|---:|---:|---:|---:|")
        rows = []
        for name, extra in available_tiers():
            e1 = dict(extra, H26X_THREADS="1", H26XDEC_NOMD5="1")
            em = dict(extra, H26XDEC_NOMD5="1")
            c1, _ = run_best([args.dec, path], e1, args.runs)
            cm, wm = run_best([args.dec, path], em, args.runs)
            rows.append((name, c1, frames / c1 if c1 else 0, cm,
                         frames / wm if wm else 0))
        f1, _ = run_best([args.ffmpeg, "-threads", "1", "-i", path, "-f",
                          "rawvideo", "-y", null], {}, args.runs)
        fm, fmw = run_best([args.ffmpeg, "-i", path, "-f", "rawvideo", "-y",
                            null], {}, args.runs)
        for name, c1, fps1, cm, fpsm in rows:
            print(f"| {name} | {c1:.3f} | {fps1:.0f} | {cm:.3f} | {fpsm:.0f} |")
        print(f"| libavcodec | {f1:.3f} | {frames / f1 if f1 else 0:.0f} | "
              f"{fm:.3f} | {frames / fmw if fmw else 0:.0f} |")
        best = rows[0]
        print(f"\nWidest rung against libavcodec: **{best[1] / f1:.2f}x** its "
              f"single-threaded CPU time, **{best[3] / fm:.2f}x** with every "
              f"thread; SIMD is worth **{rows[-1][1] / best[1]:.1f}x** over the "
              f"scalar reference.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
