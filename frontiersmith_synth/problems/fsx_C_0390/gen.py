#!/usr/bin/env python3
"""gen.py <testId>  -- prints ONE instance of the warehouse-robotics
diagonal-restricted Latin completion problem to stdout.

testId 1..10 is a difficulty ladder (grid grows, lists stay tight -> the
maximum completion stays genuinely unknown / no full completion exists).
All randomness is seeded by testId only, so generation is reproducible.

Instance format (stdin of a solver):
    N
    N lines: the prefilled grid, N integers each (0 = empty slot)
    N*N lines (row-major, r=0..N-1 then c=0..N-1):
        k  a_1 a_2 ... a_k     = the SKUs the robot at slot (r,c) may place
Zones (conveyor loops) are IMPLICIT: loop(r,c) = (r + c) mod N.
Each SKU may appear at most once per aisle (row), per rack column, and per
conveyor loop.  A prefilled slot always respects all three rules and lists.
"""
import sys, random


def build(test_id):
    rnd = random.Random(20260390 + test_id)
    N = 8 + test_id                      # 9 .. 18  (medium scale)
    d = 3                                # per-slot reachable SKU count (tight lists)

    allowed = [[sorted(rnd.sample(range(1, N + 1), d)) for _ in range(N)]
               for _ in range(N)]

    P = [[0] * N for _ in range(N)]
    rowset = [set() for _ in range(N)]
    colset = [set() for _ in range(N)]
    diaset = [set() for _ in range(N)]   # index (r+c) % N

    cells = [(r, c) for r in range(N) for c in range(N)]
    rnd.shuffle(cells)
    target = max(N, round(0.14 * N * N))
    placed = 0
    for (r, c) in cells:
        if placed >= target:
            break
        g = (r + c) % N
        opts = [s for s in allowed[r][c]
                if s not in rowset[r] and s not in colset[c] and s not in diaset[g]]
        if not opts:
            continue
        s = rnd.choice(opts)
        P[r][c] = s
        rowset[r].add(s); colset[c].add(s); diaset[g].add(s)
        placed += 1
    return N, allowed, P


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N, allowed, P = build(test_id)
    out = [str(N)]
    for r in range(N):
        out.append(" ".join(str(P[r][c]) for c in range(N)))
    for r in range(N):
        for c in range(N):
            a = allowed[r][c]
            out.append(str(len(a)) + " " + " ".join(str(x) for x in a))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
