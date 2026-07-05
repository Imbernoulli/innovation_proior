#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  --  deterministic scorer for the depot-routing
partial-Latin-square completion problem.

Feasibility (any violation -> Ratio: 0.0):
  * output must contain at least N*N integers (an N x N grid, row-major)
  * every cell value in {0..N}   (0 = left empty)
  * every prefilled cell preserved exactly
  * no depot symbol repeats within a row (a truck visits each depot <= once)
  * no depot symbol repeats within a column (a depot serves <= one truck per slot)

Objective F = number of scheduled (non-zero) cells.
Internal baseline B = number of prefilled cells (the "schedule nothing new"
construction the checker builds itself; always positive by generation).
Maximization: sc = min(1000, 100 * F / max(1e-9, B)); print Ratio = sc/1000.
"""
import sys


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    toks = read_ints(inf)
    idx = 0
    N = int(toks[idx]); idx += 1
    given = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            given[i][j] = int(toks[idx]); idx += 1

    B = sum(1 for i in range(N) for j in range(N) if given[i][j] != 0)

    try:
        otoks = read_ints(outf)
        vals = [int(x) for x in otoks[:N * N]]
    except Exception:
        print("Ratio: 0.0 (unparseable output)")
        return

    if len(vals) < N * N:
        print("Ratio: 0.0 (output has fewer than N*N integers)")
        return

    grid = [[vals[i * N + j] for j in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(N):
            v = grid[i][j]
            if v < 0 or v > N:
                print("Ratio: 0.0 (symbol %d out of range 0..%d)" % (v, N))
                return
            if given[i][j] != 0 and v != given[i][j]:
                print("Ratio: 0.0 (prefilled cell (%d,%d) altered)" % (i, j))
                return

    for i in range(N):
        seen = set()
        for j in range(N):
            v = grid[i][j]
            if v != 0:
                if v in seen:
                    print("Ratio: 0.0 (row %d repeats depot %d)" % (i, v))
                    return
                seen.add(v)
    for j in range(N):
        seen = set()
        for i in range(N):
            v = grid[i][j]
            if v != 0:
                if v in seen:
                    print("Ratio: 0.0 (column %d repeats depot %d)" % (j, v))
                    return
                seen.add(v)

    F = sum(1 for i in range(N) for j in range(N) if grid[i][j] != 0)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("Ratio: %.6f (F=%d B=%d N=%d)" % (sc / 1000.0, F, B, N))


if __name__ == "__main__":
    main()
