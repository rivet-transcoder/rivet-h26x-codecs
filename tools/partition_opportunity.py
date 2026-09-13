"""Would inter prediction-unit partitions pay, on this corpus, at the CU size
this encoder codes?

The H.265 encoder here codes one 2Nx2N prediction unit per coding tree block
(32x32 on the 64x64 clips, 16x16 on the odd-sized one). The standard offers
seven more shapes: the symmetric halves (2NxN, Nx2N), the four asymmetric
motion partitions (2NxnU, 2NxnD, nLx2N, nRx2N — a quarter and three
quarters, `amp_enabled_flag`), and NxN. Each costs a second vector and a
`part_mode` to signal, and pays only where the two halves of a block move
differently. Ask that BEFORE building it, the way `multiref_opportunity.py`
asked about references: a shape no block on the corpus takes is `part_mode`
bins paid on every unit for a choice that never has a better answer.

Method: for each frame t and each BLK x BLK luma block, motion-search it
whole against frame t-1 over a +/-RANGE full-sample window, then search each
half of every split independently over the same window and sum the halves.
A split counts as winning when its sum beats the whole block's SAD by more
than MARGIN, which stands in for the bits the extra vector and the
`part_mode` cost. The SAD is subsampled, which biases every shape identically
and so cannot skew the comparison between them.

    python tools/partition_opportunity.py <dir-with-src_*.yuv>

Reports, per clip: how many blocks a symmetric split wins, how many an AMP
split wins BEYOND the best symmetric one (the asymmetric shapes' own value,
since an encoder with AMP has the symmetric shapes too), and the median gain
of the wins. Clips that hold one frame (src_static) or fade without moving
(src_fade) must come out at zero: nothing in them moves at all, let alone
differently in two halves.

WHAT IT FOUND, 2026-09-13, at BLK 32 (16 on the odd clip), RANGE 4, MARGIN 5%,
over the seven 8-bit 4:2:0 clips (210 blocks):

    cut     18.8% symmetric, 40.6% AMP beyond it, median gain 28.8%
    detail  17.9%            21.4%                             25.8%
    fade     6.2%             6.2%                              6.3%
    grad    75.0%            25.0%                             18.2%
    motion   0.0%             0.0%
    odd      0.0%             0.0%
    static   0.0%             0.0%
    TOTAL   16.2% symmetric (34 of 210), 13.3% AMP beyond it (28 of 210)

Read it with two biases in mind, both of which inflate it. An integer-sample
search over a SMOOTH clip (grad: gradients under a slow pan) lets two halves
approximate a fractional shift by taking different integer offsets, which
the encoder's quarter-sample refinement does without any split — grad's 75%
is that, not two motions. And a strip of 32x8 samples searched over 81
positions finds a coincidental low SAD in FRACTAL texture more easily than
a 32x32 block does — cut's mandelbrot half is where its AMP count comes
from. Outside grad the symmetric figure is 7.1% (13 of 182); outside grad
and cut it is 4.7%. The clips with real, uniform motion (motion, odd) and
the held frame come out at zero, which is right.

So: a few percent of blocks with a median gain around a fifth of their SAD,
against a second vector plus `part_mode` on every one of them and, for any
non-2Nx2N shape under this SPS's `max_transform_hierarchy_depth_inter`,
an inferred transform split (`interSplitFlag`) the inter writer would have
to learn to spell. The symmetric shapes are the prerequisite for AMP, and
AMP's own value beyond them sits inside the overfit. Not built; the
encoder's docs carry this number so the next reader can decide against a
larger corpus rather than re-derive it.
"""

import sys, os

RANGE = 4          # +/- full samples, every shape, same window
MARGIN = 0.05      # a split must beat the whole block by this fraction
STEP = 2           # SAD subsampling


def planes(path, w, h):
    n = w * h * 3 // 2
    data = open(path, 'rb').read()
    return [data[i * n:i * n + w * h] for i in range(len(data) // n)]


def best_sad(cur, ref, w, h, bx, by, bw, bh):
    """Best subsampled SAD of the bw x bh block at (bx, by) over the window."""
    best = None
    for dy in range(-RANGE, RANGE + 1):
        for dx in range(-RANGE, RANGE + 1):
            s = 0
            for y in range(0, bh, STEP):
                cy = by + y
                ry = cy + dy
                if ry < 0 or ry >= h:
                    s += 1 << 20
                    continue
                crow = cy * w
                rrow = ry * w
                for x in range(0, bw, STEP):
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


def shapes(blk):
    """The split shapes as lists of (dx, dy, w, h) parts, by name."""
    q = blk // 4
    hh = blk // 2
    return {
        '2NxN': [(0, 0, blk, hh), (0, hh, blk, hh)],
        'Nx2N': [(0, 0, hh, blk), (hh, 0, hh, blk)],
        '2NxnU': [(0, 0, blk, q), (0, q, blk, blk - q)],
        '2NxnD': [(0, 0, blk, blk - q), (0, blk - q, blk, q)],
        'nLx2N': [(0, 0, q, blk), (q, 0, blk - q, blk)],
        'nRx2N': [(0, 0, blk - q, blk), (blk - q, 0, q, blk)],
    }


def probe(path, w, h, blk, max_frames):
    fr = planes(path, w, h)
    if len(fr) < 2:
        return None
    idx = list(range(1, len(fr)))
    if len(idx) > max_frames:
        stride = len(idx) / float(max_frames)
        idx = [idx[int(i * stride)] for i in range(max_frames)]
    total = 0
    sym_wins = 0
    amp_wins = 0
    gains = []
    sh = shapes(blk)
    for t in idx:
        cur, ref = fr[t], fr[t - 1]
        for by in range(0, h - blk + 1, blk):
            for bx in range(0, w - blk + 1, blk):
                whole = best_sad(cur, ref, w, h, bx, by, blk, blk)
                total += 1
                cost = {}
                for name, parts in sh.items():
                    cost[name] = sum(best_sad(cur, ref, w, h, bx + dx, by + dy, pw, ph) for (dx, dy, pw, ph) in parts)
                best_sym = min(cost['2NxN'], cost['Nx2N'])
                best_amp = min(cost['2NxnU'], cost['2NxnD'], cost['nLx2N'], cost['nRx2N'])
                if whole > 0 and best_sym < whole * (1.0 - MARGIN):
                    sym_wins += 1
                    gains.append((whole - best_sym) / float(whole))
                # AMP's own value: beyond the best symmetric split, by the same margin.
                if best_sym > 0 and best_amp < best_sym * (1.0 - MARGIN) and best_amp < whole * (1.0 - MARGIN):
                    amp_wins += 1
    med = 0.0
    if gains:
        gains.sort()
        med = gains[len(gains) // 2]
    return total, sym_wins, amp_wins, med, len(idx)


if __name__ == '__main__':
    work = sys.argv[1]
    rows = []
    for f in sorted(os.listdir(work)):
        # 8-bit 4:2:0 clips only: the format token is exactly `420`. A
        # `420p10` clip read as bytes is two half-samples per sample and
        # scores as noise — the first run of this probe reported the deep
        # clips at 56..94% split wins for exactly that reason.
        if not (f.startswith('src_') and f.endswith('_420.yuv')):
            continue
        base = f[:-4]
        geom = base.split('_')[-2]
        w, h = (int(v) for v in geom.split('x'))
        # The CTB the encoder picks for this geometry: 32 where the picture
        # is a whole number of 32s, else 16 (h265_syntax::Geometry).
        blk = 32 if w % 32 == 0 and h % 32 == 0 else 16
        r = probe(os.path.join(work, f), w, h, blk, 8)
        if r is None:
            continue
        total, sym, amp, med, nf = r
        rows.append((base, total, sym, amp, med, nf))
        print("%-26s blk %2d frames %2d  blocks %4d  symmetric split wins %4d (%5.1f%%)  AMP beyond it %4d (%5.1f%%)  median gain %5.1f%%"
              % (base, blk, nf, total, sym, 100.0 * sym / total, amp, 100.0 * amp / total, 100.0 * med))
    if rows:
        t = sum(r[1] for r in rows)
        s = sum(r[2] for r in rows)
        a = sum(r[3] for r in rows)
        print()
        print("CORPUS TOTAL: %d blocks, %d would take a symmetric split (%.1f%%), %d an AMP shape beyond it (%.1f%%)"
              % (t, s, 100.0 * s / t, a, 100.0 * a / t))
