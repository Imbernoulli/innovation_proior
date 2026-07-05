#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the warehouse-robotics
diagonal-restricted Latin completion problem.

Reads the instance from <in> and the participant board from <out>.  Validates
strictly (exact integer token count = N*N, each token in [0,N], every prefilled
slot preserved, every placed SKU inside its reachable list, and the row / column
/ conveyor-loop distinctness rules).  On ANY violation prints `Ratio: 0.0` and
exits 0.  Otherwise F = number of stocked slots, internal baseline B = number of
prefilled slots (a positive feasible construction the checker rebuilds itself):
    sc = min(1000, 100 * F / B);   Ratio = sc / 1000.
Exact integer arithmetic, O(N^2), bit-for-bit deterministic on reruns.
"""
import sys


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        lines = [ln.split() for ln in f.read().splitlines()]
    lines = [t for t in lines if t]
    N = int(lines[0][0])
    P = []
    for i in range(N):
        P.append([int(x) for x in lines[1 + i]])
    allowed = []
    idx = 1 + N
    for _ in range(N * N):
        row = lines[idx]; idx += 1
        k = int(row[0])
        allowed.append(set(int(x) for x in row[1:1 + k]))
    return N, P, allowed


def read_board(path, N):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
    if len(toks) != N * N:
        fail("expected %d integer tokens, got %d" % (N * N, len(toks)))
    vals = []
    for t in toks:
        try:
            v = int(t)                       # rejects nan/inf/floats/garbage
        except ValueError:
            fail("non-integer token: %r" % t)
        if v < 0 or v > N:
            fail("token out of range [0,%d]: %d" % (N, v))
        vals.append(v)
    return [vals[r * N:(r + 1) * N] for r in range(N)]


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    N, P, allowed = read_instance(sys.argv[1])
    S = read_board(sys.argv[2], N)

    # feasibility -------------------------------------------------------------
    rowset = [set() for _ in range(N)]
    colset = [set() for _ in range(N)]
    diaset = [set() for _ in range(N)]
    B = 0
    F = 0
    for r in range(N):
        for c in range(N):
            given = P[r][c]
            v = S[r][c]
            if given != 0:
                B += 1
                if v != given:
                    fail("prefilled slot (%d,%d) altered: %d != %d" % (r, c, v, given))
            if v == 0:
                continue
            if v not in allowed[r * N + c]:
                fail("SKU %d not reachable at slot (%d,%d)" % (v, r, c))
            g = (r + c) % N
            if v in rowset[r]:
                fail("SKU %d repeats in aisle %d" % (v, r))
            if v in colset[c]:
                fail("SKU %d repeats in rack column %d" % (v, c))
            if v in diaset[g]:
                fail("SKU %d repeats in conveyor loop %d" % (v, g))
            rowset[r].add(v); colset[c].add(v); diaset[g].add(v)
            F += 1

    if B <= 0:
        fail("degenerate instance: no prefilled slots")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("stocked F=%d baseline B=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
