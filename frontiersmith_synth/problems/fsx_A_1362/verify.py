#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the envy-free cake problem.

Feasibility gate (must ALL hold, else Ratio: 0.0):
  - output has exactly m lines, each with exactly n finite, non-negative numbers
  - every item's fractions sum to 1 (tol 1e-6)
  - the allocation is envy-free: for every ordered pair (i,k),
    val(i, bundle_i) >= val(i, bundle_k) - 1e-6

Objective (maximize): F = total social welfare = sum_i val(i, bundle_i).
Baseline B (checker's own trivial construction): give every agent an identical
1/n fraction of every item -- always feasible, always envy-free by symmetry.

sc = min(1000.0, 100.0 * F / max(1e-9, B));  print "Ratio: %.6f" % (sc/1000.0)
"""
import math
import sys

EPS = 1e-6


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    v = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    return n, m, v


def read_output(path, n, m):
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except Exception as e:
        fail(f"cannot read output: {e}")

    # strip blank trailing lines but require at least m content lines
    content = [ln for ln in lines if ln.strip() != ""]
    if len(content) != m:
        fail(f"expected {m} lines, got {len(content)}")

    x = [[0.0] * n for _ in range(n)]  # x[i][j] = fraction of item j given to agent i
    for j in range(m):
        toks = content[j].split()
        if len(toks) != n:
            fail(f"item {j}: expected {n} numbers, got {len(toks)}")
        row = []
        for t in toks:
            try:
                val = float(t)
            except ValueError:
                fail(f"item {j}: non-numeric token '{t}'")
            if not math.isfinite(val):
                fail(f"item {j}: non-finite value '{t}'")
            if val < -1e-6:
                fail(f"item {j}: negative fraction {val}")
            row.append(max(0.0, val))
        s = sum(row)
        if abs(s - 1.0) > 1e-6:
            fail(f"item {j}: fractions sum to {s}, expected 1")
        for i in range(n):
            x[i][j] = row[i]
    return x


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, v = read_instance(in_path)
    x = read_output(out_path, n, m)

    # val_matrix[i][k] = value agent i places on bundle k (agent k's allocation)
    val = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            s = 0.0
            for j in range(m):
                s += x[k][j] * v[i][j]
            val[i][k] = s

    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            if val[i][i] < val[i][k] - EPS:
                fail(f"agent {i} envies agent {k}: {val[i][i]:.6f} < {val[i][k]:.6f}")

    F = sum(val[i][i] for i in range(n))

    total_v = sum(sum(row) for row in v)
    B = total_v / n

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F: %.6f B: %.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
