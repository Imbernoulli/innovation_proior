#!/usr/bin/env python3
# Deterministic checker for "Resistance-Portrait Fitting" (format C, minimize error).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
#
# Instance (stdin of the solver):
#   n m wmax P
#   then P lines:  i j t w      (target effective resistance t>0, importance w>0)
# Solver output:
#   E
#   then E lines:  u v c        (undirected edge, conductance 0<c<=wmax)
#
# Score: F = weighted RMS RELATIVE error of the realized effective resistances vs
# targets, computed exactly from the Laplacian pseudoinverse. Baseline B = a uniform
# conductance-1 backbone path over the active terminals (built by the checker).
# minimization:  Ratio = min(1.0, 0.1 * B / F).
import sys, math

TOL = 1e-6
RELCAP = 3.0  # per-pair relative error is clipped so no single tiny target dominates


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_ints_floats(path):
    with open(path) as f:
        return f.read().split()


def eff_res_all(n, edges, pairs):
    """Effective resistance for each (i,j) in pairs, via Laplacian pseudoinverse."""
    import numpy as np
    L = np.zeros((n, n), dtype=float)
    for (u, v, c) in edges:
        L[u, u] += c
        L[v, v] += c
        L[u, v] -= c
        L[v, u] -= c
    Lp = np.linalg.pinv(L, hermitian=True)
    res = []
    for (i, j) in pairs:
        r = Lp[i, i] + Lp[j, j] - 2.0 * Lp[i, j]
        res.append(float(r))
    return res


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    try:
        it = read_ints_floats(sys.argv[1])
        pos = 0
        n = int(it[pos]); pos += 1
        m = int(it[pos]); pos += 1
        wmax = float(it[pos]); pos += 1
        P = int(it[pos]); pos += 1
        pt = []  # (i,j,t,w)
        for _ in range(P):
            i = int(it[pos]); j = int(it[pos + 1])
            t = float(it[pos + 2]); w = float(it[pos + 3]); pos += 4
            pt.append((i, j, t, w))
    except Exception:
        fail("bad instance")

    try:
        ot = read_ints_floats(sys.argv[2])
    except Exception:
        fail("no output")
    if not ot:
        fail("empty output")

    try:
        E = int(ot[0])
    except Exception:
        fail("bad E")
    if E < 0 or E > m:
        fail("E out of budget")
    if len(ot) < 1 + 3 * E:
        fail("truncated edges")

    edges = []
    eset = set()
    for k in range(E):
        try:
            u = int(ot[1 + 3 * k]); v = int(ot[2 + 3 * k]); c = float(ot[3 + 3 * k])
        except Exception:
            fail("bad edge %d" % k)
        if not math.isfinite(c):
            fail("non-finite conductance %d" % k)
        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            fail("bad endpoints %d" % k)
        if c <= 0.0 or c > wmax + 1e-6:
            fail("conductance out of range %d" % k)
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in eset:
            fail("duplicate edge %d" % k)
        eset.add((a, b))
        edges.append((a, b, c))

    # connectivity of every specified pair
    dsu = DSU(n)
    for (a, b, c) in edges:
        dsu.union(a, b)
    for (i, j, t, w) in pt:
        if dsu.find(i) != dsu.find(j):
            fail("specified pair (%d,%d) disconnected" % (i, j))

    pairs = [(i, j) for (i, j, t, w) in pt]
    R = eff_res_all(n, edges, pairs)
    for r in R:
        if not math.isfinite(r) or r <= 0.0:
            fail("degenerate resistance")

    def wrms(Rvals):
        num = 0.0
        den = 0.0
        for (r, (i, j, t, w)) in zip(Rvals, pt):
            rel = (r - t) / t
            if rel > RELCAP:
                rel = RELCAP
            elif rel < -RELCAP:
                rel = -RELCAP
            num += w * rel * rel
            den += w
        return math.sqrt(num / max(1e-12, den))

    F = wrms(R)

    # ---- internal baseline B: uniform conductance-1 backbone path over active verts
    active = sorted(set([i for (i, j, t, w) in pt] + [j for (i, j, t, w) in pt]))
    bedges = []
    for k in range(len(active) - 1):
        bedges.append((active[k], active[k + 1], 1.0))
    Rb = eff_res_all(n, bedges, pairs)
    B = wrms(Rb)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
