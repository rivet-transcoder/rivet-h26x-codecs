#!/usr/bin/env python3
"""bd_rate.py — Bjøntegaard rate difference between two encoder binaries.

  python tools/bd_rate.py <enc_A> <enc_B> <clip.yuv> <WxH> <fmt> [encoder flags...]

Encodes the clip with both binaries at QP 22, 27, 32, 37 (the JCT-VC common
test points), takes the size in bits and the luma PSNR of each binary's own
reconstruction, and reports B's rate against A's at equal PSNR by the usual
cubic fit in log-rate, plus the per-QP size and PSNR so a reader can see
where a difference came from. Negative is B smaller. Env A_ENV / B_ENV
(space-separated VAR=VAL) apply to the respective side.

This is the honest measure of a decision change: a change that saves time
by choosing differently can only be judged against what it cost in rate at
equal quality, and one QP cannot say that.
"""
import math
import os
import subprocess
import sys
import tempfile

QPS = [22, 27, 32, 37]


def psnr_luma(src, rec, w, h, fmt):
    ysize = w * h
    csize = {"420": ysize // 2, "422": ysize, "444": 2 * ysize, "400": 0}[fmt]
    frame = ysize + csize
    a = open(src, "rb").read()
    b = open(rec, "rb").read()
    n = min(len(a), len(b)) // frame
    se = 0
    for f in range(n):
        o = f * frame
        ya = a[o:o + ysize]
        yb = b[o:o + ysize]
        se += sum((x - y) * (x - y) for x, y in zip(ya, yb))
    mse = se / (n * ysize)
    return float("inf") if mse == 0 else 10 * math.log10(255 * 255 / mse), n


def run(enc, env, src, w, h, fmt, qp, flags, out):
    e = dict(os.environ)
    for item in env.split():
        k, _, v = item.partition("=")
        e[k] = v
    bs = out + ".bin"
    rec = out + ".rec.yuv"
    subprocess.run([enc, "--input", src, "--size", f"{w}x{h}", "--format", fmt, "--qp", str(qp),
                    "--output", bs, "--recon", rec] + flags,
                   env=e, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bits = os.path.getsize(bs) * 8
    p, n = psnr_luma(src, rec, w, h, fmt)
    return bits, p, n


def bd_rate(ra, pa, rb, pb):
    """Bjøntegaard: integrate the cubic fit of log10(rate) against PSNR over
    the overlapping PSNR range and take the difference."""
    la = [math.log10(r) for r in ra]
    lb = [math.log10(r) for r in rb]

    def fit(p, l):
        # Least squares cubic via normal equations, 4 points -> exact.
        import itertools
        n = len(p)
        X = [[pi ** k for k in range(4)] for pi in p]
        XtX = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(4)] for r in range(4)]
        Xty = [sum(X[i][r] * l[i] for i in range(n)) for r in range(4)]
        # Gaussian elimination.
        M = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
        for c in range(4):
            piv = max(range(c, 4), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            for r in range(4):
                if r != c:
                    f = M[r][c] / M[c][c]
                    for k in range(c, 5):
                        M[r][k] -= f * M[c][k]
        return [M[i][4] / M[i][i] for i in range(4)]

    ca, cb = fit(pa, la), fit(pb, lb)
    lo, hi = max(min(pa), min(pb)), min(max(pa), max(pb))
    if hi <= lo:
        return float("nan")

    def integ(c):
        return sum(c[k] / (k + 1) * (hi ** (k + 1) - lo ** (k + 1)) for k in range(4))

    avg = (integ(cb) - integ(ca)) / (hi - lo)
    return (10 ** avg - 1) * 100


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    enc_a, enc_b, src, geom, fmt = sys.argv[1:6]
    flags = sys.argv[6:]
    w, h = (int(v) for v in geom.split("x"))
    a_env, b_env = os.environ.get("A_ENV", ""), os.environ.get("B_ENV", "")
    with tempfile.TemporaryDirectory() as d:
        ra, pa, rb, pb = [], [], [], []
        print(f"{os.path.basename(src)} {' '.join(flags)}")
        print(f"{'QP':>3} {'A bytes':>9} {'A PSNR':>7}   {'B bytes':>9} {'B PSNR':>7}   {'size':>7}")
        for qp in QPS:
            ba, qa, n = run(enc_a, a_env, src, w, h, fmt, qp, flags, os.path.join(d, f"a{qp}"))
            bb, qb, _ = run(enc_b, b_env, src, w, h, fmt, qp, flags, os.path.join(d, f"b{qp}"))
            ra.append(ba); pa.append(qa); rb.append(bb); pb.append(qb)
            print(f"{qp:>3} {ba // 8:>9} {qa:>7.3f}   {bb // 8:>9} {qb:>7.3f}   {100 * (bb / ba - 1):>+6.2f}%")
        if all(math.isfinite(p) for p in pa + pb):
            print(f"BD-rate B vs A: {bd_rate(ra, pa, rb, pb):+.3f}%  ({n} frames)")
        else:
            print("BD-rate: not defined (a lossless point)")


if __name__ == "__main__":
    main()
