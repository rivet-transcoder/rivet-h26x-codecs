"""Would multiple reference pictures help, on this corpus, at this block size?

Multi-reference prediction pays only where an OLDER reference predicts a block
better than the newest one — occlusion, periodic motion, or noise that averages
out differently. Ask this BEFORE building it: a feature whose gate rows are all
zeros costs bits to signal and buys nothing.

Method: for each frame t, for each BLK x BLK luma block, motion-search it
against frame t-1 and against frame t-2 over the same window and compare the
best SADs. The SAD is subsampled for speed, which biases both references
identically and so cannot skew the comparison between them. A block only counts
as choosing the older reference when it wins by more than MARGIN.

    python tools/multiref_opportunity.py <dir-with-src_*.yuv>

WHAT IT FOUND, 2026-08-20, and the lesson in it. At BLK=16 this corpus gave
6.9% of blocks preferring the older reference, with large local gains, and on
that basis two-reference prediction was built. It LOST 0.80% BD-rate on every
clip. Re-run at BLK=32 — the size the H.265 encoder actually codes, since its
CUs are whole CTUs — the answer is ZERO blocks out of 140.

So the probe was right and the reading was wrong: it was run at a block size
the encoder does not use. Multi-reference's value is gated on partition size,
because a large block averages over enough content that one reference always
serves it. Set BLK to the block size the decision will really be made at, or
the number means nothing.
"""

import sys, os

RANGE = 4          # +/- full samples, both references, same window
MARGIN = 0.05      # t-2 must beat t-1 by this fraction to count as a choice
BLK = 16
STEP = 2           # SAD subsampling


def planes(path, w, h):
    n = w * h * 3 // 2
    data = open(path, 'rb').read()
    return [data[i * n:i * n + w * h] for i in range(len(data) // n)]


def best_sad(cur, ref, w, h, bx, by):
    best = None
    for dy in range(-RANGE, RANGE + 1):
        for dx in range(-RANGE, RANGE + 1):
            s = 0
            for y in range(0, BLK, STEP):
                cy = by + y
                ry = cy + dy
                if ry < 0 or ry >= h:
                    s += 1 << 20
                    continue
                crow = cy * w
                rrow = ry * w
                for x in range(0, BLK, STEP):
                    cx = bx + x
                    rx = cx + dx
                    if rx < 0 or rx >= w:
                        s += 255
                        continue
                    d = cur[crow + cx] - ref[rrow + rx]
                    s += d if d >= 0 else -d
            if best is None or s < best:
                best = s
    return best


def probe(path, w, h, max_frames):
    fr = planes(path, w, h)
    if len(fr) < 3:
        return None
    idx = list(range(2, len(fr)))
    if len(idx) > max_frames:
        stride = len(idx) / float(max_frames)
        idx = [idx[int(i * stride)] for i in range(max_frames)]
    total = 0
    older_wins = 0
    gains = []
    for t in idx:
        cur, r1, r2 = fr[t], fr[t - 1], fr[t - 2]
        for by in range(0, h - BLK + 1, BLK):
            for bx in range(0, w - BLK + 1, BLK):
                s1 = best_sad(cur, r1, w, h, bx, by)
                s2 = best_sad(cur, r2, w, h, bx, by)
                total += 1
                if s1 > 0 and s2 < s1 * (1.0 - MARGIN):
                    older_wins += 1
                    gains.append((s1 - s2) / float(s1))
    med = 0.0
    if gains:
        gains.sort()
        med = gains[len(gains) // 2]
    return total, older_wins, med, len(idx)


if __name__ == '__main__':
    work = sys.argv[1]
    rows = []
    for f in sorted(os.listdir(work)):
        if not (f.startswith('src_') and f.endswith('.yuv') and '_420' in f):
            continue
        base = f[:-4]
        geom = base.split('_')[-2]
        w, h = (int(v) for v in geom.split('x'))
        r = probe(os.path.join(work, f), w, h, 8)
        if r is None:
            continue
        total, wins, med, nf = r
        rows.append((base, total, wins, med, nf))
        print("%-26s frames %2d  blocks %4d  older-ref wins %4d (%5.1f%%)  median gain %5.1f%%"
              % (base, nf, total, wins, 100.0 * wins / total, 100.0 * med))
    if rows:
        t = sum(r[1] for r in rows)
        wnum = sum(r[2] for r in rows)
        print()
        print("CORPUS TOTAL: %d blocks, %d would pick the older reference (%.1f%%)"
              % (t, wnum, 100.0 * wnum / t))
