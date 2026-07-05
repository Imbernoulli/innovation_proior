#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the wind-farm
partial-Latin-square completion problem.

Feasibility (any violation -> Ratio: 0.0):
  * output has exactly n*n tokens;
  * every filled value is an integer in [0, n-1] ('.' or '-1' means empty);
  * every pre-installed given slot is preserved unchanged;
  * no turbine model repeats within any row or any column (among filled slots).

Objective F = number of filled slots (givens count).
Baseline  B = number of pre-installed givens (the trivial feasible layout:
              install nothing new).  B > 0 always.
Score (maximization):  sc = min(1000, 100 * F / B);  print Ratio: sc/1000.
So installing nothing new -> 0.1; filling 10x the givens caps at 1.0.
"""
import sys


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    itok = read_tokens(inf)
    if not itok:
        fail("empty input")
    n = int(itok[0])
    vals = itok[1:1 + n * n]
    if len(vals) < n * n:
        fail("truncated input")

    G = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            t = vals[idx]; idx += 1
            G[i][j] = -1 if t == '.' else int(t)
    givens = sum(1 for i in range(n) for j in range(n) if G[i][j] != -1)
    if givens <= 0:
        fail("degenerate instance (no givens)")

    otok = read_tokens(outf)
    if len(otok) != n * n:
        fail("output must have exactly n*n=%d tokens, got %d" % (n * n, len(otok)))

    S = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            t = otok[idx]; idx += 1
            if t == '.' or t == '-1':
                S[i][j] = -1
            else:
                try:
                    v = int(t)
                except ValueError:
                    fail("non-integer token %r" % t)
                if v < 0 or v >= n:
                    fail("value %d out of range [0,%d]" % (v, n - 1))
                S[i][j] = v

    # givens preserved
    for i in range(n):
        for j in range(n):
            if G[i][j] != -1 and S[i][j] != G[i][j]:
                fail("given slot (%d,%d) was altered/removed" % (i, j))

    # Latin constraints among filled slots
    for i in range(n):
        seen = set()
        for j in range(n):
            v = S[i][j]
            if v != -1:
                if v in seen:
                    fail("row %d has duplicate model %d" % (i, v))
                seen.add(v)
    for j in range(n):
        seen = set()
        for i in range(n):
            v = S[i][j]
            if v != -1:
                if v in seen:
                    fail("col %d has duplicate model %d" % (j, v))
                seen.add(v)

    F = sum(1 for i in range(n) for j in range(n) if S[i][j] != -1)
    B = givens
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d givens=%d B=%d  Ratio: %.6f" % (F, givens, B, sc / 1000.0))


if __name__ == "__main__":
    main()
