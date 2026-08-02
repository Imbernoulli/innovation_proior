#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the thermostability
mutation-design problem. Prints 'Ratio: <float in [0,1]>' as the LAST line.
"""
import sys
import math


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return
    inpath, outpath = sys.argv[1], sys.argv[2]

    with open(inpath) as f:
        toks = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); K = int(nxt()); C = int(nxt()); R = int(nxt())
    A0 = float(nxt()); ActMin = float(nxt()); alpha = float(nxt())
    dstab = [0.0] * n
    dact = [0.0] * n
    dist = [0] * n
    for i in range(n):
        dstab[i] = float(nxt())
        dact[i] = float(nxt())
        dist[i] = int(nxt())
    m_epi = int(nxt())
    epi_s = {}
    epi_a = {}
    for _ in range(m_epi):
        i = int(nxt()); j = int(nxt()); es = float(nxt()); ea = float(nxt())
        a, b = (i, j) if i < j else (j, i)
        epi_s[(a, b)] = es
        epi_a[(a, b)] = ea

    # ---- checker's own internal baseline B: single best individually-feasible mutation ----
    B = 0.0
    for i in range(n):
        cc = 1 if dist[i] <= R else 0
        crowd = alpha * max(0, cc - C) ** 2
        if (A0 + dact[i] - crowd) >= ActMin - 1e-9 and dstab[i] > B:
            B = dstab[i]
    if B <= 1e-9:
        B = 1e-6  # degenerate fallback; should not occur by construction

    # ---- read participant artifact ----
    try:
        with open(outpath) as f:
            otext = f.read()
    except Exception:
        print("Ratio: 0.0 (cannot read output)")
        return
    otoks = otext.split()
    if not otoks:
        print("Ratio: 0.0 (empty output)")
        return
    vals = []
    for t in otoks:
        try:
            v = int(t)
        except ValueError:
            print("Ratio: 0.0 (non-integer token %r)" % t)
            return
        vals.append(v)

    m = vals[0]
    if m < 0 or m > K:
        print("Ratio: 0.0 (budget violated: m=%d, K=%d)" % (m, K))
        return
    if len(vals) != 1 + m:
        print("Ratio: 0.0 (token count mismatch: expected %d got %d)" % (1 + m, len(vals)))
        return
    S = vals[1:]
    if len(set(S)) != len(S):
        print("Ratio: 0.0 (duplicate mutation index)")
        return
    for x in S:
        if x < 0 or x >= n:
            print("Ratio: 0.0 (index %d out of range [0,%d))" % (x, n))
            return

    Sset = set(S)
    stab = sum(dstab[i] for i in S)
    act = A0 + sum(dact[i] for i in S)
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            i, j = S[a], S[b]
            key = (i, j) if i < j else (j, i)
            if key in epi_s:
                stab += epi_s[key]
                act += epi_a[key]
    cc = sum(1 for i in S if dist[i] <= R)
    act -= alpha * max(0, cc - C) ** 2

    if not (math.isfinite(stab) and math.isfinite(act)):
        print("Ratio: 0.0 (non-finite objective)")
        return
    if act < ActMin - 1e-6:
        print("Ratio: 0.0 (activity floor violated: act=%.4f < ActMin=%.4f)" % (act, ActMin))
        return

    F = max(0.0, stab)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.4f B=%.4f act=%.4f m=%d Ratio: %.6f" % (F, B, act, m, sc / 1000.0))


if __name__ == "__main__":
    main()
