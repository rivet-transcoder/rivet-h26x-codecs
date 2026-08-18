# Symbolicate a samply/gecko profile through samply's local symbol server and print
# self-time / total-time per function.
#   usage: symprof.py profile.json.gz http://127.0.0.1:PORT/TOKEN [thread-index] [top] [caller-patterns...]
#   env:   LINES=pat1,pat2  -> line-level self-time breakdown inside functions matching pat
#          LINES_TOP=30
import gzip, json, sys, os, urllib.request, collections

prof = json.load(gzip.open(sys.argv[1]))
base = sys.argv[2].rstrip('/')
ti = int(sys.argv[3]) if len(sys.argv) > 3 else 0
top = int(sys.argv[4]) if len(sys.argv) > 4 else 40
t = prof['threads'][ti]
libs = prof['libs']
sa = t['stringArray']
ft, fut, st, rt = t['frameTable'], t['funcTable'], t['stackTable'], t['resourceTable']


def frame_lib(fi):
    fn = ft['func'][fi]
    r = fut['resource'][fn]
    if r is None or r < 0:
        return None
    return rt['lib'][r]


req = collections.defaultdict(set)
for fi in range(ft['length']):
    li = frame_lib(fi)
    a = ft['address'][fi]
    if li is not None and a is not None and a >= 0:
        req[li].add(a)

symtab = {}
for li, addrs in req.items():
    lib = libs[li]
    addrs = sorted(addrs)
    for i in range(0, len(addrs), 5000):
        chunk = addrs[i:i + 5000]
        body = json.dumps({"jobs": [{"memoryMap": [[lib['debugName'], lib['breakpadId']]],
                                     "stacks": [[[0, a] for a in chunk]]}]}).encode()
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                base + '/symbolicate/v5', data=body, headers={'Content-Type': 'application/json'}), timeout=600)
            res = json.load(r)
        except Exception as e:
            print('symbolicate failed for', lib['name'], e, file=sys.stderr)
            continue
        for a, fr in zip(chunk, res['results'][0]['stacks'][0]):
            name = fr.get('function') or ('0x%x' % a)
            inl = fr.get('inlines') or []
            symtab[(li, a)] = (name, [x.get('function') for x in inl], fr.get('line'),
                               [(x.get('function'), x.get('line')) for x in inl])


def frame_name(fi):
    li = frame_lib(fi)
    a = ft['address'][fi]
    if li is None or a is None or a < 0:
        return sa[fut['name'][ft['func'][fi]]]
    e = symtab.get((li, a))
    if e is None:
        return sa[fut['name'][ft['func'][fi]]]
    return e[0]


samples = t['samples']
stacks = samples['stack']
weights = samples.get('weight') or [1] * len(stacks)


def stack_frames(si):
    out = []
    while si is not None:
        out.append(st['frame'][si])
        si = st['prefix'][si]
    return out  # leaf first


selfc = collections.Counter()
totc = collections.Counter()
inlc = collections.Counter()
n = 0
for si, w in zip(stacks, weights):
    if si is None:
        continue
    n += w
    frs = stack_frames(si)
    leaf = frame_name(frs[0])
    selfc[leaf] += w
    li = frame_lib(frs[0])
    a = ft['address'][frs[0]]
    e = symtab.get((li, a))
    if e and e[1]:
        inlc[e[1][0] or leaf] += w
    else:
        inlc[leaf] += w
    seen = set()
    for f in frs:
        nm = frame_name(f)
        if nm in seen:
            continue
        seen.add(nm)
        totc[nm] += w

print("samples: %d" % n)

for pat in [x for x in os.environ.get('LINES', '').split(',') if x]:
    linec = collections.Counter()
    tot = 0
    for si, w in zip(stacks, weights):
        if si is None:
            continue
        f0 = stack_frames(si)[0]
        li = frame_lib(f0)
        a = ft['address'][f0]
        e = symtab.get((li, a))
        if not e or pat not in e[0]:
            continue
        tot += w
        chain = list(e[3]) + [(e[0], e[2])]
        key = ' <- '.join("%s:%s" % ((fn or '?').split('::')[-1], ln) for fn, ln in chain[:3])
        linec[key] += w
    print("")
    print("== lines of *%s* self (%d samples) ==" % (pat, tot))
    for k, v in linec.most_common(int(os.environ.get('LINES_TOP', '30'))):
        print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))


for pat in [x for x in os.environ.get('CALLER_LINES', '').split(',') if x]:
    linec = collections.Counter()
    tot = 0
    for si, w in zip(stacks, weights):
        if si is None:
            continue
        frs = stack_frames(si)
        if pat not in frame_name(frs[0]):
            continue
        # first frame above that is not the same function
        for f in frs[1:]:
            if pat in frame_name(f):
                continue
            li = frame_lib(f)
            a = ft['address'][f]
            e = symtab.get((li, a))
            if e:
                chain = list(e[3]) + [(e[0], e[2])]
                key = ' <- '.join("%s:%s" % ((fn or '?').split('::')[-1], ln) for fn, ln in chain[:4])
            else:
                key = frame_name(f)
            linec[key] += w
            tot += w
            break
    print("")
    print("== caller lines of *%s* (%d samples) ==" % (pat, tot))
    for k, v in linec.most_common(int(os.environ.get('LINES_TOP', '30'))):
        print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))

for pat in sys.argv[5:]:
    callers = collections.Counter()
    callers2 = collections.Counter()
    tot = 0
    for si, w in zip(stacks, weights):
        if si is None:
            continue
        frs = stack_frames(si)
        names = []
        for f in frs:
            nm = frame_name(f)
            if not names or names[-1] != nm:
                names.append(nm)
        idx = next((i for i, nm in enumerate(names) if pat in nm), None)
        if idx is None:
            continue
        tot += w
        if idx + 1 < len(names):
            callers[names[idx + 1]] += w
        if idx + 2 < len(names):
            callers2[names[idx + 1] + '  <-  ' + names[idx + 2]] += w
    print("")
    print("== callers of *%s* (%d samples incl.) ==" % (pat, tot))
    for k, v in callers.most_common(12):
        print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))
    for k, v in callers2.most_common(12):
        print("%6.2f%%  %6d      %s" % (100 * v / n, v, k))

print("")
print("== self time (outermost non-inlined function) ==")
for k, v in selfc.most_common(top):
    print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))
print("")
print("== self time (innermost inlined function) ==")
for k, v in inlc.most_common(top):
    print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))
print("")
print("== inclusive ==")
for k, v in totc.most_common(top):
    print("%6.2f%%  %6d  %s" % (100 * v / n, v, k))
