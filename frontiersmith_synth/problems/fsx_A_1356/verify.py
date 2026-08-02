#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  (deterministic checker; <ans> is an unused placeholder)

Scores a mixed row-strategy p against the true (unbounded) adversary: the column
that minimizes the row player's payoff given p. This is the strategy's guaranteed,
worst-case-optimal value -- i.e. the negative of its exploitability.
"""
import math
import sys


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    m = int(nxt())
    n = int(nxt())
    A = [[int(nxt()) for _ in range(n)] for _ in range(m)]
    _N = int(nxt())
    H = [int(nxt()) for _ in range(n)]
    return m, n, A, H


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    m, n, A, H = read_instance(in_path)

    try:
        with open(out_path, "r") as f:
            out_toks = f.read().split()
    except OSError:
        fail("cannot read output")

    if len(out_toks) != m:
        fail(f"expected {m} numbers, got {len(out_toks)}")

    p = []
    for tok in out_toks:
        try:
            v = float(tok)
        except ValueError:
            fail(f"non-numeric token {tok!r}")
        if not math.isfinite(v):
            fail("non-finite value in output")
        if v < -1e-6:
            fail("negative probability")
        p.append(max(0.0, v))

    s = sum(p)
    if abs(s - 1.0) > 1e-4:
        fail(f"probabilities sum to {s}, not 1")

    # renormalize the tiny clamping/rounding slack away (does not change the
    # feasibility verdict above, only removes float noise before scoring)
    p = [x / s for x in p]

    def worst_case_value(strategy):
        best = None
        for j in range(n):
            val = 0.0
            for i in range(m):
                val += strategy[i] * A[i][j]
            if best is None or val < best:
                best = val
        return best

    F = worst_case_value(p)

    # Internal baseline B: the checker's own trivial reference strategy -- the
    # single PURE row that looks best if the opponent's column were assumed
    # uniformly random (i.e. ignoring the historical log entirely). This is
    # always a valid, positive-value strategy, and is exactly what solutions/
    # trivial.py reproduces.
    col_avg = [sum(A[i][j] for j in range(n)) / n for i in range(m)]
    i_star = 0
    for i in range(1, m):
        if col_avg[i] > col_avg[i_star]:
            i_star = i
    baseline_strategy = [0.0] * m
    baseline_strategy[i_star] = 1.0
    B = worst_case_value(baseline_strategy)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print(f"F={F:.6f} B={B:.6f}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
