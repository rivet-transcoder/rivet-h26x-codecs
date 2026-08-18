# bench.py <stream[,stream...]> [runs]: best-of-N wall and CPU (user+kernel) time,
# ours (1 thread / all threads) vs ffmpeg (1 thread / default threads), output discarded.
import subprocess, sys, time, os, ctypes
from ctypes import wintypes

DEC = os.environ.get("DEC", "../release/examples/h26xdec.exe")
FF = os.environ.get("FFMPEG", os.path.expanduser("~/scoop/apps/ffmpeg/current/bin/ffmpeg.exe"))
kernel32 = ctypes.windll.kernel32
HIGH_PRIORITY_CLASS = 0x00000080


def cpu_seconds(handle):
    c, e, k, u = (wintypes.FILETIME(),) * 4
    c, e, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    kernel32.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
    ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return ft(k) + ft(u)


def best(cmd, env=None, n=3):
    e = dict(os.environ)
    e.update(env or {})
    bw = bc = 1e9
    for _ in range(n):
        t = time.perf_counter()
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=HIGH_PRIORITY_CLASS)
        p.wait()
        w = time.perf_counter() - t
        c = cpu_seconds(p._handle)
        bw, bc = min(bw, w), min(bc, c)
    return bw, bc


n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
mode = os.environ.get("MODE", "all")  # all | ours | st
for f in sys.argv[1].split(","):
    o1w, o1c = best([DEC, f], {"H26X_THREADS": "1", "H26XDEC_NOMD5": "1"}, n)
    line = f"{f:20s} ours 1t wall={o1w:.3f} cpu={o1c:.3f}"
    if mode == "all":
        omw, omc = best([DEC, f], {"H26XDEC_NOMD5": "1"}, n)
        f1w, f1c = best([FF, "-threads", "1", "-i", f, "-f", "rawvideo", "-y", "NUL"], None, n)
        fmw, fmc = best([FF, "-i", f, "-f", "rawvideo", "-y", "NUL"], None, n)
        line += f"  mt wall={omw:.3f} cpu={omc:.3f} | ffmpeg 1t wall={f1w:.3f} cpu={f1c:.3f}  mt wall={fmw:.3f} cpu={fmc:.3f} | ratio 1t={o1w / f1w:.2f} (cpu {o1c / f1c:.2f}) mt={omw / fmw:.2f}"
    elif mode == "st":
        f1w, f1c = best([FF, "-threads", "1", "-i", f, "-f", "rawvideo", "-y", "NUL"], None, n)
        line += f" | ffmpeg 1t wall={f1w:.3f} cpu={f1c:.3f} | ratio {o1w / f1w:.2f} (cpu {o1c / f1c:.2f})"
    print(line)
