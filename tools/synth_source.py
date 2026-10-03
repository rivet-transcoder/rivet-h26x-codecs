#!/usr/bin/env python3
"""synth_source.py — deterministic synthetic video, written as raw planar YUV.

    synth_source.py RECIPE --size WxH --frames N --format FMT [--rate R] OUT

FMT is a planar layout: `400`, `420`, `422` or `444`, with an optional depth
suffix (`420p10`, `444p12`; no suffix is 8 bits). Samples are bytes at 8
bits and little-endian 16-bit words above, plane after plane (Y, Cb, Cr) and
frame after frame — the layout h26xdec writes, h26xenc reads and x264 / x265
take with `--input-csp iNNN --input-depth D --input-res WxH`. `400` writes
the luma plane only.

The clips the decoder fixtures (make_fixtures.sh), the encoder gate's source
corpus (make_encode_sources.sh) and the vendored test streams are made from
come from here. Nothing in it takes a seed or reads a clock: the same
arguments give the same bytes on every run (the one transcendental recipe,
`wsine`, depends on the platform's sin / cos, which IEEE leaves to the libm;
every other recipe is arithmetic only).

Recipes — each is content chosen for what it makes an encoder do:

  detail     colour ramps, a scrolling band of glyph-like high-frequency
             detail, a bouncing block and a sliding bar: moving detail with
             smooth areas beside it
  zoom       a Mandelbrot zoom: dense detail with motion everywhere
  motion     the detail picture under fast motion (a global pan plus
             blocks moving several samples a frame)
  grad       two-colour linear gradients, drifting slowly
  bars       static colour bars over a ramp: hard vertical edges, nothing
             moving
  grey       flat mid grey
  fade       `detail` fading in from black over ten frames
  static     frame 0 of `detail`, held
  tff / bff  interlaced: fields woven from consecutive frames of `detail`
             at twice the rate, the top (tff) or bottom (bff) field first
  cut        `detail` at 2.5x its speed for 51 frames, then a hard cut to
             a slower `zoom`
  half       left half `detail`, right half flat grey
  hfade      `half` with its luma at gain
             (1 - n/16): a pure-gain fade for weighted prediction
  wpoff      `hfade` with an offset too: luma p * (1 - n/16) - 3n
  settle     24 frames of `detail` scrolling sideways 5% of its width a
             frame, then its frame 24 held
  big        quarters of four unrelated contents: detail, zoom, grad, bars
  interlace  left half `bars` scrolling sideways 7% of its width a source
             frame, right half one `detail` picture held, woven into fields
             from a source at twice the rate (top field first)
  wsine      a drifting sinusoidal texture computed at full precision (its
             low bits carry the texture, not noise)

`--noise BITS` adds BITS of deterministic noise to every sample after the
conversion to the output depth, so a deep clip's low bits are not all zero
(an 8-bit picture shifted up hides any bug that only touches them).
`--luma-offset-per-frame K` and `--fade16` are the `wpoff` / `fdeep` fade
applied at the output depth (luma p * (1 - n/16) - K*n, clipped at 0).
"""
import argparse
import sys

import numpy as np

# ------------------------------------------------------------------ helpers
# Pictures are (Y, Cb, Cr) float64 arrays at full resolution in 8-bit
# nominal units: luma 16..235, chroma 16..240 about 128.


def _grid(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return x.astype(np.float64), y.astype(np.float64)


def _hash(*arrs):
    """A 32-bit integer hash of integer arrays (wrapping arithmetic, so the
    same on every platform)."""
    h = np.uint64(0x9E3779B97F4A7C15)
    acc = np.zeros(np.broadcast(*arrs).shape, dtype=np.uint64) + h
    for a in arrs:
        acc ^= np.asarray(a).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) + (acc << np.uint64(6)) + (acc >> np.uint64(2))
        acc *= np.uint64(0xBF58476D1CE4E5B9)
        acc ^= acc >> np.uint64(31)
    return (acc >> np.uint64(16)) & np.uint64(0xFFFFFFFF)


def _tri(v):
    """Triangle wave of period 1, range 0..1."""
    f = v - np.floor(v)
    return 1.0 - np.abs(2.0 * f - 1.0)


def _rgb_to_yuv(r, g, b):
    """BT.601 limited range from 0..1 RGB."""
    y = 16 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128 + 112.0 * r - 93.786 * g - 18.214 * b
    return y, cb, cr


# ------------------------------------------------------------------ sources
def detail(w, h, n, speed=1.0):
    x, y = _grid(w, h)
    t = n * speed
    # Background: a hue ramp across, a luma ramp down.
    r = _tri(x / max(w, 1) * 0.9 + 0.05)
    g = _tri(y / max(h, 1) * 0.7 + 0.30)
    b = _tri((x + y) / max(w + h, 1) * 1.3 + 0.60)
    Y, Cb, Cr = _rgb_to_yuv(r, g, b)
    # A band of glyph-like detail scrolling left one sample a frame: 4x6
    # cells, each a random 3x5 bitmap of a pseudo-random character.
    band0, band1 = h // 4, h // 4 + max(12, h // 5)
    xs = (x + t).astype(np.int64)
    cell = xs // 4
    gx, gy = xs % 4, (y.astype(np.int64) - band0) % 6
    row = (y.astype(np.int64) - band0) // 6
    bits = _hash(cell, row) % np.uint64(1 << 15)
    on = (gx < 3) & (gy < 5) & (((bits >> (gy * 3 + gx).astype(np.uint64)) & np.uint64(1)) == 1)
    inband = (y >= band0) & (y < band1)
    Y = np.where(inband, np.where(on, 230.0, 30.0), Y)
    Cb = np.where(inband, 128.0, Cb)
    Cr = np.where(inband, 128.0, Cr)
    # A block bouncing around the lower half, and a bar sliding across.
    bw, bh = max(8, w // 6), max(8, h // 6)
    span_x, span_y = max(1, w - bw), max(1, (h - bh) // 2)
    px = int(_tri(t * 3.0 / (2 * span_x)) * span_x)
    py = h // 2 + int(_tri(t * 2.0 / (2 * span_y)) * span_y) - 1
    blk = (x >= px) & (x < px + bw) & (y >= py) & (y < py + bh)
    Y = np.where(blk, 200.0 - 2 * ((x - px) % 7), Y)
    Cb = np.where(blk, 60.0, Cb)
    Cr = np.where(blk, 200.0, Cr)
    bar = ((x - 2 * t) % w < max(3, w // 20)) & (y >= h - h // 8)
    Y = np.where(bar, 235.0, Y)
    # A counter: a small block whose shade steps every frame.
    cnt = (x < max(6, w // 10)) & (y < max(6, h // 10))
    Y = np.where(cnt, 16.0 + (n * 37) % 219, Y)
    return Y, Cb, Cr


def zoom(w, h, n, rate=0.92):
    x, y = _grid(w, h)
    scale = 0.45 * (rate ** n)
    cx, cy = -0.743643887037151, 0.131825904205330
    cre = cx + (x - w / 2) * scale / max(w, h)
    cim = cy + (y - h / 2) * scale / max(w, h)
    zr = np.zeros_like(cre)
    zi = np.zeros_like(cim)
    it = np.zeros(cre.shape, dtype=np.float64)
    alive = np.ones(cre.shape, dtype=bool)
    maxit = 160
    for _ in range(maxit):
        zr2, zi2 = zr * zr, zi * zi
        alive &= (zr2 + zi2) <= 4.0
        it += alive
        zi = np.where(alive, 2 * zr * zi + cim, zi)
        zr = np.where(alive, zr2 - zi2 + cre, zr)
    f = it / maxit
    Y = 16 + 219 * np.sqrt(f)
    Cb = 128 + 100 * (_tri(f * 3.0) - 0.5)
    Cr = 128 + 100 * (_tri(f * 5.0 + 0.25) - 0.5)
    return Y, Cb, Cr


def motion(w, h, n):
    Y, Cb, Cr = detail(w, h, n, speed=5.0)
    # A global pan: the whole picture shifted three samples a frame.
    s = (3 * n) % w
    Y, Cb, Cr = (np.roll(p, -s, axis=1) for p in (Y, Cb, Cr))
    x, y = _grid(w, h)
    for k in range(3):
        bw, bh = max(6, w // 8), max(6, h // 8)
        px = (n * (7 + 2 * k) + k * w // 3) % max(1, w - bw)
        py = (n * (5 + k) + k * h // 4) % max(1, h - bh)
        m = (x >= px) & (x < px + bw) & (y >= py) & (y < py + bh)
        Y = np.where(m, 40.0 + 60 * k + ((x + y) % 5) * 8, Y)
        Cb = np.where(m, 200.0 - 60 * k, Cb)
    return Y, Cb, Cr


def grad(w, h, n):
    x, y = _grid(w, h)
    d = (x / max(w - 1, 1) + y / max(h - 1, 1)) / 2 + 0.01 * n
    f = _tri(d * 0.5)
    c0 = np.array(_rgb_to_yuv(0x20 / 255, 0x50 / 255, 0xa0 / 255))
    c1 = np.array(_rgb_to_yuv(0xe0 / 255, 0xb0 / 255, 0x40 / 255))
    return tuple(c0[i] + (c1[i] - c0[i]) * f for i in range(3))


def bars(w, h, n):
    x, y = _grid(w, h)
    cols = [(.75, .75, .75), (.75, .75, 0), (0, .75, .75), (0, .75, 0), (.75, 0, .75), (.75, 0, 0), (0, 0, .75)]
    idx = np.minimum((x * 7 / w).astype(int), 6)
    r = np.choose(idx, [c[0] for c in cols])
    g = np.choose(idx, [c[1] for c in cols])
    b = np.choose(idx, [c[2] for c in cols])
    Y, Cb, Cr = _rgb_to_yuv(r, g, b)
    ramp = y >= h * 3 // 4
    Y = np.where(ramp, 16 + 219 * x / max(w - 1, 1), Y)
    Cb = np.where(ramp, 128.0, Cb)
    Cr = np.where(ramp, 128.0, Cr)
    return Y, Cb, Cr


def grey(w, h, n):
    z = np.zeros((h, w))
    return z + 128.0, z + 128.0, z + 128.0


def wsine(w, h, n):
    """Computed in 10-bit units, returned in 8-bit nominal units."""
    x, y = _grid(w, h)
    Y = (512 + 300 * np.sin(x / 5 + n / 4) * np.cos(y / 7)) * (1 - n / 16) - 12 * n
    Cb = 512 + 120 * np.cos(x / 9 - n / 5)
    Cr = 512 + 120 * np.sin(y / 8 + n / 6)
    return np.maximum(Y, 0) / 4, Cb / 4, Cr / 4


# --------------------------------------------------------------- composites
def hstack(*pics):
    return tuple(np.hstack([p[i] for p in pics]) for i in range(3))


def vstack(*pics):
    return tuple(np.vstack([p[i] for p in pics]) for i in range(3))


def weave(a, b, top_first):
    """Two pictures as the fields of one frame: the first in time on the
    top lines (top_first) or the bottom ones."""
    first, second = (a, b)
    out = []
    for i in range(3):
        f = first[i].copy()
        if top_first:
            f[1::2] = second[i][1::2]
        else:
            f[0::2] = second[i][0::2]
            f[1::2] = first[i][1::2]
        out.append(f)
    return tuple(out)


def scroll(pic, n, frac):
    w = pic[0].shape[1]
    s = int(round(n * frac * w)) % w
    return tuple(np.roll(p, -s, axis=1) for p in pic)


def hfade(w, h, n):
    hw = w // 2
    p = hstack(detail(hw, h, n), grey(w - hw, h, n))
    return (p[0] * (1 - n / 16), p[1], p[2])


RECIPES = {
    'detail': lambda w, h, n: detail(w, h, n),
    'zoom': zoom,
    'motion': motion,
    'grad': grad,
    'bars': bars,
    'grey': grey,
    'fade': lambda w, h, n: (lambda p: (16 + (p[0] - 16) * min(1.0, n / 10), 128 + (p[1] - 128) * min(1.0, n / 10), 128 + (p[2] - 128) * min(1.0, n / 10)))(detail(w, h, n)),
    'static': lambda w, h, n: detail(w, h, 0),
    'tff': lambda w, h, n: weave(detail(w, h, 2 * n), detail(w, h, 2 * n + 1), True),
    'bff': lambda w, h, n: weave(detail(w, h, 2 * n), detail(w, h, 2 * n + 1), False),
    'cut': lambda w, h, n: detail(w, h, n, speed=2.5) if n < 51 else zoom(w, h, n - 51, rate=0.985),
    'hfade': hfade,
    'half': lambda w, h, n: hstack(detail(w // 2, h, n), grey(w - w // 2, h, n)),
    'wpoff': lambda w, h, n: (lambda p: (np.maximum(0, p[0] - 3 * n), p[1], p[2]))(hfade(w, h, n)),
    'settle': lambda w, h, n: scroll(detail(w, h, min(n, 24)), min(n, 24), 0.05),
    'big': lambda w, h, n: vstack(hstack(detail(w // 2, h // 2, n), zoom(w - w // 2, h // 2, n)),
                                  hstack(grad(w // 2, h - h // 2, n), bars(w - w // 2, h - h // 2, n))),
    'interlace': lambda w, h, n: weave(*[hstack(scroll(bars(w // 2, h, 0), 2 * n + k, 0.07),
                                                detail(w - w // 2, h, 0)) for k in (0, 1)], True),
    'wsine': wsine,
}


# ------------------------------------------------------------------ output
def parse_format(fmt):
    chroma, _, depth = fmt.partition('p')
    if chroma not in ('400', '420', '422', '444'):
        raise SystemExit(f'synth_source: unknown format {fmt}')
    return chroma, int(depth) if depth else 8


def to_planes(pic, chroma, depth, n, noise, fade16, offset_per_frame):
    scale = 1 << (depth - 8)
    maxv = (1 << depth) - 1
    Y, Cb, Cr = pic
    if chroma == '420':
        Cb = (Cb[0::2, 0::2] + Cb[1::2, 0::2] + Cb[0::2, 1::2] + Cb[1::2, 1::2]) / 4
        Cr = (Cr[0::2, 0::2] + Cr[1::2, 0::2] + Cr[0::2, 1::2] + Cr[1::2, 1::2]) / 4
    elif chroma == '422':
        Cb = (Cb[:, 0::2] + Cb[:, 1::2]) / 2
        Cr = (Cr[:, 0::2] + Cr[:, 1::2]) / 2
    planes = [Y] if chroma == '400' else [Y, Cb, Cr]
    out = []
    for i, p in enumerate(planes):
        v = np.floor(p * scale + 0.5)
        if i == 0 and fade16:
            v = np.floor(v * (1 - n / 16))
        if i == 0 and offset_per_frame:
            v = np.maximum(0, v - offset_per_frame * n)
        if noise:
            hh, ww = p.shape
            yy, xx = np.mgrid[0:hh, 0:ww]
            v = v + (_hash(xx, yy, np.int64(n), np.int64(i)) % np.uint64(1 << noise)).astype(np.float64)
        v = np.clip(v, 0, maxv)
        out.append(v.astype('<u2' if depth > 8 else np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recipe', choices=sorted(RECIPES))
    ap.add_argument('--size', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='420')
    ap.add_argument('--noise', type=int, default=0)
    ap.add_argument('--fade16', action='store_true')
    ap.add_argument('--luma-offset-per-frame', type=int, default=0)
    ap.add_argument('out')
    a = ap.parse_args()
    w, h = map(int, a.size.lower().split('x'))
    if w % 2 or h % 2:
        raise SystemExit('synth_source: width and height must be even')
    chroma, depth = parse_format(a.format)
    gen = RECIPES[a.recipe]
    with open(a.out, 'wb') as f:
        for n in range(a.frames):
            for p in to_planes(gen(w, h, n), chroma, depth, n, a.noise, a.fade16, a.luma_offset_per_frame):
                f.write(p.tobytes())


if __name__ == '__main__':
    sys.exit(main())
