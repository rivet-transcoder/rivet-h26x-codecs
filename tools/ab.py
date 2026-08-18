# ab.py <exeA> <exeB> <stream> [runs]: interleaved A/B runs (1 thread, no md5), min and median CPU seconds.
import subprocess, sys, time, os, ctypes, statistics
from ctypes import wintypes

kernel32 = ctypes.windll.kernel32
HIGH_PRIORITY_CLASS = 0x00000080


def cpu_seconds(handle):
    c, e, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    kernel32.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
    ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return ft(k) + ft(u)


def run(exe, f):
    e = dict(os.environ)
    e.update({"H26XDEC_NOMD5": "1"})
    if os.environ.get("MT") != "1":
        e["H26X_THREADS"] = "1"
    t = time.perf_counter()
    p = subprocess.Popen([exe, f], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=HIGH_PRIORITY_CLASS)
    if os.environ.get("MT") != "1":
        kernel32.SetProcessAffinityMask(wintypes.HANDLE(p._handle), ctypes.c_size_t(int(os.environ.get('AFFINITY', '4'), 0)))
    p.wait()
    return time.perf_counter() - t, cpu_seconds(p._handle)


a, b, f = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 7
ra, rb = [], []
for i in range(n):
    ra.append(run(a, f))
    rb.append(run(b, f))
ca = [c for _, c in ra]
cb = [c for _, c in rb]
wa = [w for w, _ in ra]
wb = [w for w, _ in rb]
print(f"A {os.path.basename(a)}: cpu min={min(ca):.3f} med={statistics.median(ca):.3f}  wall min={min(wa):.3f}")
print(f"B {os.path.basename(b)}: cpu min={min(cb):.3f} med={statistics.median(cb):.3f}  wall min={min(wb):.3f}")
print(f"B/A cpu min ratio = {min(cb) / min(ca):.3f}, median ratio = {statistics.median(cb) / statistics.median(ca):.3f}")
