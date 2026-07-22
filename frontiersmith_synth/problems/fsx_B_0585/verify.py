#!/usr/bin/env python3
# Deterministic checker for quantile-hedged-cargo-manifest (format C, MAXIMIZE
# the 20th-percentile scenario total).  CLI: python3 verify.py <in> <out> <ans>
# (ans ignored).  Prints "... Ratio: <r>", r in [0,1]; any feasibility breach or
# non-finite / malformed output -> Ratio: 0.0.
import sys
import math


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    it = iter(toks)
    N = int(next(it)); S = int(next(it)); W = int(next(it)); V = int(next(it)); k = int(next(it))
    w = [0] * N; v = [0] * N; vals = [None] * N
    for i in range(N):
        w[i] = int(next(it)); v[i] = int(next(it))
        vals[i] = [int(next(it)) for _ in range(S)]
    return N, S, W, V, k, w, v, vals


def quantile_of(sel, vals, S, k):
    tot = [0] * S
    for i in sel:
        row = vals[i]
        for s in range(S):
            tot[s] += row[s]
    tot.sort()
    return tot[k - 1]  # k-th smallest (1-indexed)


def baseline_selection(N, W, V, w, v):
    # value-BLIND small feasible fill: lightest items first, stop at ~1/3 cap.
    order = sorted(range(N), key=lambda i: (w[i] + v[i], i))
    sel = []
    cw = cv = 0
    for i in order:
        if cw + w[i] <= W and cv + v[i] <= V:
            sel.append(i); cw += w[i]; cv += v[i]
            if cw * 3 >= W or cv * 3 >= V:
                break
    return sel


def main():
    if len(sys.argv) < 3:
        fail("usage")
    N, S, W, V, k, w, v, vals = read_instance(sys.argv[1])

    # ---- parse participant output: whitespace-separated item indices -------
    try:
        with open(sys.argv[2]) as fh:
            raw = fh.read().split()
    except Exception:
        fail("no output")
    sel = []
    seen = set()
    for t in raw:
        # strict integer token; reject nan/inf/floats/garbage
        try:
            x = int(t)
        except Exception:
            fail("non-integer token %r" % t[:16])
        if not math.isfinite(x):
            fail("non-finite")
        if x < 0 or x >= N:
            fail("index out of range")
        if x in seen:
            fail("duplicate index")
        seen.add(x); sel.append(x)

    # empty manifest is feasible but scores 0 (quantile of all-zero = 0)
    cw = sum(w[i] for i in sel)
    cv = sum(v[i] for i in sel)
    if cw > W:
        fail("weight capacity exceeded")
    if cv > V:
        fail("volume capacity exceeded")

    F = quantile_of(sel, vals, S, k) if sel else 0

    B = quantile_of(baseline_selection(N, W, V, w, v), vals, S, k)
    B = max(1e-9, float(B))

    sc = min(1000.0, 100.0 * float(F) / B)
    print("F=%d B=%.3f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
