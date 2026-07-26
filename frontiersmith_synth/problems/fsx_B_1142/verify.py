#!/usr/bin/env python3
"""Deterministic checker for "Radiator Network: Multi-Scale Cooling-Curve Match"
(format C, minimize max log-error of the heat-kernel trace).
CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
"""
import sys
import math
import numpy as np


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
        p = 0
        n = int(toks[p]); p += 1
        m = int(toks[p]); p += 1
        T = int(toks[p]); p += 1
        edges = []
        for _ in range(m):
            u = int(toks[p]); v = int(toks[p + 1]); cap = int(toks[p + 2]); p += 3
            edges.append((u, v, cap))
        ts = [float(toks[p + j]) for j in range(T)]; p += T
        targets = [float(toks[p + j]) for j in range(T)]; p += T
        return n, edges, ts, targets
    except Exception:
        return None


def trace_values(n, edges, weights, ts):
    L = np.zeros((n, n))
    for (u, v, cap), w in zip(edges, weights):
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    eigs = np.linalg.eigvalsh(L)
    return [float(np.sum(np.exp(-t * eigs))) for t in ts]


def max_log_err(n, edges, weights, ts, targets):
    tv = trace_values(n, edges, weights, ts)
    errs = [abs(math.log(max(v, 1e-12)) - math.log(max(g, 1e-12)))
            for v, g in zip(tv, targets)]
    return max(errs)


def main():
    inst = read_instance(sys.argv[1])
    if inst is None:
        fail("unreadable/corrupt instance")
    n, edges, ts, targets = inst
    m, T = len(edges), len(ts)

    if n <= 0 or m <= 0 or T <= 0:
        fail("bad instance dimensions")
    for (u, v, cap) in edges:
        if not (0 <= u < n and 0 <= v < n) or cap < 1:
            fail("bad instance edge/cap")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != m:
        fail("expected exactly %d integer edge weights, got %d" % (m, len(otoks)))

    weights = []
    for k, (u, v, cap) in enumerate(edges):
        tok = otoks[k]
        try:
            w = int(tok)
        except ValueError:
            fail("weight %d ('%s') is not an integer" % (k, tok))
        if not math.isfinite(w):
            fail("weight %d is not finite" % k)
        if w < 1 or w > cap:
            fail("weight %d = %d out of range [1,%d]" % (k, w, cap))
        weights.append(w)

    F = max_log_err(n, edges, weights, ts, targets)
    if not math.isfinite(F):
        fail("non-finite objective")

    # internal baseline: the checker's own "half the cap, ignore structure" construction
    triv_weights = [max(1, cap // 2) for (u, v, cap) in edges]
    B = max_log_err(n, edges, triv_weights, ts, targets)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("radiator-fit max-log-error F=%.6f baseline B=%.6f Ratio: %.6f"
          % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
