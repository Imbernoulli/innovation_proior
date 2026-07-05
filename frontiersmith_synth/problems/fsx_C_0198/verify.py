#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the watchtower
channel-assignment (jigsaw / gerechte Latin-square completion) problem.

Reads the instance (n, district map, partial channel grid) from <in> and the
participant's completed grid from <out>. Validates the gerechte rules exactly;
any violation prints `Ratio: 0.0` and exits 0. Otherwise:

    F = number of tuned (non-empty) towers in the feasible output (givens incl.)
    B = number of pre-tuned givens (the "tune nothing new" baseline)
    sc = min(1000, 100 * F / B);  Ratio = sc / 1000

so echoing only the givens scores 0.1 and tuning ~10x the givens caps at 1.0.
All-integer arithmetic; bit-for-bit reproducible.
"""
import sys


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    tin = read_tokens(in_path)
    it = iter(tin)
    n = int(next(it))
    reg = [[int(next(it)) for _ in range(n)] for _ in range(n)]
    givens = [[None] * n for _ in range(n)]
    B = 0
    for i in range(n):
        for j in range(n):
            tok = next(it)
            if tok == "." or tok == "-1":
                givens[i][j] = None
            else:
                givens[i][j] = int(tok)
                B += 1
    if B <= 0:
        fail("no givens in instance")

    # ---- parse participant output: exactly n*n tokens ----
    tout = read_tokens(out_path)
    if len(tout) != n * n:
        fail("expected %d tokens, got %d" % (n * n, len(tout)))

    grid = [[None] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(n):
            tok = tout[k]; k += 1
            if tok == "." or tok == "-1":
                grid[i][j] = None
            else:
                try:
                    v = int(tok)
                except ValueError:
                    fail("non-integer token %r" % tok)
                if v < 0 or v >= n:
                    fail("channel %d out of range [0,%d]" % (v, n - 1))
                grid[i][j] = v

    # ---- givens must be preserved exactly ----
    for i in range(n):
        for j in range(n):
            if givens[i][j] is not None and grid[i][j] != givens[i][j]:
                fail("given at (%d,%d) altered/emptied" % (i, j))

    # ---- gerechte constraints: no repeat in any row / col / district ----
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    regs = [set() for _ in range(n)]
    F = 0
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v is None:
                continue
            F += 1
            if v in rows[i]:
                fail("channel %d repeats in row %d" % (v, i))
            rows[i].add(v)
            if v in cols[j]:
                fail("channel %d repeats in col %d" % (v, j))
            cols[j].add(v)
            d = reg[i][j]
            if v in regs[d]:
                fail("channel %d repeats in district %d" % (v, d))
            regs[d].add(v)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
