# Instruction-address histogram of the leaf samples inside given functions of the
# profiled executable, plus the function start addresses (RVA), so the hot spots
# can be disassembled: symaddr.py profile.json.gz load.log 'fn substring' ...
import gzip, json, sys, re, urllib.request, collections

prof = json.load(gzip.open(sys.argv[1]))
log = open(sys.argv[2]).read()
m = re.search(r'127\.0\.0\.1%3A(\d+)%2F([a-z0-9]+)', log)
base = 'http://127.0.0.1:%s/%s' % (m.group(1), m.group(2))
pats = sys.argv[3:]
t = prof['threads'][0]
libs = prof['libs']
ft, fut, st, rt = t['frameTable'], t['funcTable'], t['stackTable'], t['resourceTable']


def frame_lib(fi):
    fn = ft['func'][fi]
    r = fut['resource'][fn]
    return None if r is None or r < 0 else rt['lib'][r]


li_dec = next(i for i, l in enumerate(libs) if l['name'].startswith('h26xdec'))
lib = libs[li_dec]
hist = collections.Counter()
total = 0
for si in t['samples']['stack']:
    if si is None:
        continue
    total += 1
    f = st['frame'][si]
    if frame_lib(f) == li_dec:
        hist[ft['address'][f]] += 1
addrs = sorted(hist)
body = json.dumps({"jobs": [{"memoryMap": [[lib['debugName'], lib['breakpadId']]],
                             "stacks": [[[0, a] for a in addrs]]}]}).encode()
res = json.load(urllib.request.urlopen(urllib.request.Request(
    base + '/symbolicate/v5', data=body, headers={'Content-Type': 'application/json'})))
fr = res['results'][0]['stacks'][0]
funcs = collections.defaultdict(list)
for a, f in zip(addrs, fr):
    funcs[f.get('function')].append((a, f.get('function_offset'), hist[a], f.get('line'), f.get('inlines') or []))
print('exe:', lib['path'], 'total samples', total)
for pat in pats:
    for name, entries in funcs.items():
        if not name or pat not in name:
            continue
        starts = set()
        for a, o, _, _, _ in entries:
            if o is None:
                continue
            o = int(o, 16) if isinstance(o, str) else o
            starts.add(a - o)
        n = sum(c for _, _, c, _, _ in entries)
        print('\n%s: start %s, %d samples (%.2f%%), addr range %s..%s' % (
            name, [hex(s) for s in starts], n, 100.0 * n / total, hex(min(a for a, _, _, _, _ in entries)), hex(max(a for a, _, _, _, _ in entries))))
        top = sorted(entries, key=lambda e: -e[2])[:40]
        for a, o, c, ln, inl in top:
            chain = ' <- '.join('%s:%s' % ((x.get('function') or '?').split('::')[-1][:28], x.get('line')) for x in inl)
            print('  %s  %4d  line %s  %s' % (hex(a), c, ln, chain))
