#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer (ans ignored).

Instance: n transmission lines, k reserved configs.  A deployment is a set of
substation configurations in F_3^n.  A RESONANCE CASCADE occurs whenever three
DISTINCT deployed configs a,b,c satisfy a+b+c == 0 (mod 3) on every line -- i.e.
they form a line in F_3^n.  A valid deployment is resonance-free (a cap set),
uses no reserved config, and every coordinate is a phase in {0,1,2}.

Objective: maximise the number of deployed substations.

Baseline B (built here, mirrored by solutions/trivial.py): the resonance-free
audit grid {phase in {0,1} on lines 0..n-2, phase 0 on the last line} minus any
reserved config.  size(B) = 2^(n-1) (reserved configs are phase-2 on the last
line, hence disjoint).  Maximisation normalisation, trivial -> ~0.1.
"""
import sys


def third(a, b, n):
    return tuple((-(a[i] + b[i])) % 3 for i in range(n))


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    n = int(next(it)); k = int(next(it))
    blocked = set()
    for _ in range(k):
        blocked.add(tuple(int(next(it)) for _ in range(n)))
    return n, k, blocked


def baseline(n, blocked):
    S = set()
    for mask in range(1 << (n - 1)):
        v = tuple((mask >> i) & 1 for i in range(n - 1)) + (0,)
        if v not in blocked:
            S.add(v)
    return S


def main():
    inp, out = sys.argv[1], sys.argv[2]
    n, k, blocked = read_instance(inp)
    B = len(baseline(n, blocked))

    try:
        toks = open(out).read().split()
    except Exception:
        print("Ratio: 0.0 (no output)"); return
    if not toks:
        print("Ratio: 0.0 (empty output)"); return

    it = iter(toks)
    try:
        m = int(next(it))
    except Exception:
        print("Ratio: 0.0 (bad count)"); return
    if m < 0 or m > 3 ** n:
        print("Ratio: 0.0 (count out of range)"); return

    S = []
    try:
        for _ in range(m):
            v = tuple(int(next(it)) for _ in range(n))
            S.append(v)
    except StopIteration:
        print("Ratio: 0.0 (truncated deployment)"); return

    for v in S:
        for x in v:
            if x < 0 or x > 2:
                print("Ratio: 0.0 (phase out of range)"); return

    Sset = set(S)
    if len(Sset) != len(S):
        print("Ratio: 0.0 (duplicate config)"); return
    for v in S:
        if v in blocked:
            print("Ratio: 0.0 (uses reserved config)"); return

    L = list(S)
    for i in range(len(L)):
        ai = L[i]
        for j in range(i + 1, len(L)):
            if third(ai, L[j], n) in Sset:
                print("Ratio: 0.0 (resonance cascade)"); return

    F = len(S)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
