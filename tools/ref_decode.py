#!/usr/bin/env python3
"""ref_decode.py — the ITU-T reference decoders as black boxes, and the
comparison of this crate's decoder against them.

    ref_decode.py decode STREAM OUT.yuv [--luma-only] [--rgb]
    ref_decode.py framemd5 STREAM FRAMES [--rgb]
    ref_decode.py check  [--dec H26XDEC] [--rgb] STREAM...

The reference for H.264 is JM's `ldecod`, for HEVC HM's `TAppDecoder`: the
decoders the ITU-T / ISO/IEC joint teams publish beside the standards, used
here only as programs (their source is not read). Which one is decided by
the extension (`.264 .h264 .jsv .26l .avc .jvt` H.264; `.265 .h265 .hevc .bit .bin`
HEVC) or `--codec`.

`decode` writes the reference decoder's pictures in the layout h26xdec
writes: Y, Cb, Cr planes, bytes at 8 bits and little-endian 16-bit words
above, cropped. An H.264 4:0:0 stream comes out with grey 4:2:0 chroma, as
h26xdec pads it (JM's `WriteUV=1`), unless `--luma-only` asks for the luma
plane alone (h26xdec's `H26XDEC_NO_CHROMA_PAD=1`); HEVC 4:0:0 is luma only
from both. `--rgb` is for 4:4:4 H.264 streams whose VUI says
matrix_coefficients 0: JM writes those planes as R, G, B — Cr, Y, Cb —
which is undone here so the decoded planes are compared, not JM's file
layout.

`framemd5` decodes STREAM and prints `<i> <md5>` for each of its FRAMES
pictures (the output split into FRAMES equal pictures; it fails if it does
not split evenly) — the per-frame form the conformance runners compare.

`check` decodes each STREAM with h26xdec (`--dec`, default `$DEC` or
h26xdec on PATH) and with the reference, and compares the two outputs frame
by frame: one line per stream, `frames ref=N mine=M match=K mismatch=J
first_bad=F`, and exit status 1 if any stream differs anywhere.

    LDECOD=path       JM's ldecod (default: `ldecod` on PATH)
    TAPPDECODER=path  HM's TAppDecoder (default: `TAppDecoder` on PATH)

Building them (cmake, no other dependency):

    git clone https://vcgit.hhi.fraunhofer.de/jvet/JM.git && cd JM
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
    git clone -b HM-18.0 https://vcgit.hhi.fraunhofer.de/jvet/HM.git && cd HM
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --target TAppDecoder

The binaries land under each tree's `bin/`.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

H264_EXT = ('.264', '.h264', '.jsv', '.26l', '.avc', '.jvt')
HEVC_EXT = ('.265', '.h265', '.hevc', '.bit', '.bin')


def codec_of(path, forced=None):
    if forced:
        return forced
    ext = os.path.splitext(path)[1].lower()
    if ext in H264_EXT:
        return 'h264'
    if ext in HEVC_EXT:
        return 'h265'
    raise SystemExit(f'ref_decode: cannot tell the codec of {path}; pass --codec')


def tool(env, name):
    p = os.environ.get(env) or shutil.which(name)
    if not p:
        raise SystemExit(f'ref_decode: no {name} (set {env} or put it on PATH)')
    return p


def reference_decode(stream, out, codec, luma_only=False, rgb=False, plane_bytes=None):
    """Decode STREAM with the reference decoder into OUT. Returns the
    decoder's log (stdout and stderr) on success; raises on failure."""
    stream = os.path.abspath(stream)
    out = os.path.abspath(out)
    with tempfile.TemporaryDirectory() as tmp:
        if codec == 'h264':
            # Run in an empty directory: ldecod reads `decoder.cfg` from the
            # working directory when there is one, and its defaults are what
            # is wanted here.
            cmd = [tool('LDECOD', 'ldecod'), '-p', f'InputFile={stream}', '-p', f'OutputFile={out}',
                   '-p', 'Silent=1', '-p', f'WriteUV={0 if luma_only else 1}', '-p', 'RefFile=none.yuv']
        else:
            cmd = [tool('TAPPDECODER', 'TAppDecoder'), '-b', stream, '-o', out, '-d', '0']
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, errors='replace')
        log = r.stdout + r.stderr
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f'{os.path.basename(cmd[0])} failed ({r.returncode}): {log.strip()[-300:]}')
    if rgb:
        if not plane_bytes:
            raise SystemExit('ref_decode: --rgb needs the plane size (use check, or --plane-bytes)')
        data = open(out, 'rb').read()
        fs = 3 * plane_bytes
        fixed = bytearray()
        for i in range(len(data) // fs):
            f = data[i * fs:(i + 1) * fs]
            # JM wrote R, G, B = Cr, Y, Cb.
            fixed += f[plane_bytes:2 * plane_bytes] + f[2 * plane_bytes:] + f[:plane_bytes]
        open(out, 'wb').write(fixed)
    return log


def run_ours(dec, stream, out, luma_only):
    env = dict(os.environ)
    if luma_only:
        env['H26XDEC_NO_CHROMA_PAD'] = '1'
    r = subprocess.run([dec, stream, out], capture_output=True, text=True, errors='replace', env=env)
    frames = [ln.split(',') for ln in r.stdout.splitlines() if ln.count(',') >= 4]
    return r.returncode, frames, r.stderr.strip()


def check(stream, dec, codec, rgb):
    with tempfile.TemporaryDirectory() as tmp:
        mine = os.path.join(tmp, 'mine.yuv')
        ref = os.path.join(tmp, 'ref.yuv')
        st, frames, err = run_ours(dec, stream, mine, False)
        if st != 0:
            return False, f'our decoder failed: {err.splitlines()[-1] if err else st}'
        mine_data = open(mine, 'rb').read()
        sizes = []
        if frames:
            per = len(mine_data) // len(frames) if len(mine_data) % len(frames) == 0 else None
            sizes = [per] * len(frames) if per else []
        plane = None
        if rgb and frames:

            plane = sizes[0] // 3 if sizes else None
        try:
            reference_decode(stream, ref, codec, rgb=rgb, plane_bytes=plane)
        except RuntimeError as e:
            return False, f'the reference decoder failed: {e}'
        ref_data = open(ref, 'rb').read()
        if not sizes:
            same = ref_data == mine_data
            return same, f'whole output {"identical" if same else "differs"} ({len(ref_data)} / {len(mine_data)} bytes; frame sizes vary)'
        fs = sizes[0]
        n_ref = len(ref_data) // fs if fs else 0
        ok = bad = 0
        first = -1
        for i in range(max(n_ref, len(frames))):
            a = ref_data[i * fs:(i + 1) * fs]
            b = mine_data[i * fs:(i + 1) * fs]
            if a and a == b:
                ok += 1
            else:
                bad += 1
                first = i if first < 0 else first
        good = bad == 0 and n_ref == len(frames) and len(ref_data) == len(mine_data) and n_ref > 0
        return good, f'frames ref={n_ref} mine={len(frames)}  match={ok} mismatch={bad} first_bad={first}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('decode')
    d.add_argument('stream')
    d.add_argument('out')
    d.add_argument('--codec', choices=('h264', 'h265'))
    d.add_argument('--luma-only', action='store_true')
    d.add_argument('--rgb', action='store_true')
    d.add_argument('--plane-bytes', type=int)
    f = sub.add_parser('framemd5')
    f.add_argument('stream')
    f.add_argument('frames', type=int)
    f.add_argument('--codec', choices=('h264', 'h265'))
    f.add_argument('--rgb', action='store_true')
    c = sub.add_parser('check')
    c.add_argument('streams', nargs='+')
    c.add_argument('--dec', default=os.environ.get('DEC') or shutil.which('h26xdec'))
    c.add_argument('--codec', choices=('h264', 'h265'))
    c.add_argument('--rgb', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'decode':
        try:
            reference_decode(a.stream, a.out, codec_of(a.stream, a.codec), a.luma_only, a.rgb, a.plane_bytes)
        except RuntimeError as e:
            print(f'ref_decode: {e}', file=sys.stderr)
            return 1
        return 0
    if a.cmd == 'framemd5':
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'ref.yuv')
            try:
                reference_decode(a.stream, out, codec_of(a.stream, a.codec))
            except RuntimeError as e:
                print(f'ref_decode: {e}', file=sys.stderr)
                return 1
            data = open(out, 'rb').read()
        if a.frames <= 0 or len(data) % a.frames:
            print(f'ref_decode: {len(data)} bytes do not split into {a.frames} pictures', file=sys.stderr)
            return 1
        fs = len(data) // a.frames
        for i in range(a.frames):
            fr = data[i * fs:(i + 1) * fs]
            if a.rgb:
                ps = fs // 3
                fr = fr[ps:2 * ps] + fr[2 * ps:] + fr[:ps]
            print(i, hashlib.md5(fr).hexdigest())
        return 0
    if not a.dec:
        raise SystemExit('ref_decode: no h26xdec (pass --dec or set DEC)')
    fail = 0
    for s in a.streams:
        good, msg = check(s, a.dec, codec_of(s, a.codec), a.rgb)
        print(f'{s:<40} {msg}')
        fail |= not good
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
