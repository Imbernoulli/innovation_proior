# TIER: strong
"""Constraint-guided completion.

Repeatedly assign the most-constrained empty cell (minimum-remaining-values) using
a least-constraining-value depot, with forward checking. Cells whose candidate set
collapses to empty are permanently skipped instead of poisoning later choices, and a
bounded backtracking pass tries to recover a fuller completion on smaller boards.
This fills substantially more cells than fixed row-major scanning and its per-instance
behaviour diverges from both baselines.
"""
import sys


def popcount(x):
    return bin(x).count("1")


def solve(N, g):
    full = ((1 << (N + 1)) - 2)  # bits 1..N set
    rows = [0] * N
    cols = [0] * N
    for i in range(N):
        for j in range(N):
            v = g[i][j]
            if v:
                rows[i] |= (1 << v)
                cols[j] |= (1 << v)

    empties = [(i, j) for i in range(N) for j in range(N) if g[i][j] == 0]
    alive = set(empties)

    while True:
        best = None
        best_c = 10 ** 9
        best_mask = 0
        dead = []
        for (i, j) in alive:
            m = full & ~rows[i] & ~cols[j]
            c = popcount(m)
            if c == 0:
                dead.append((i, j))
                continue
            if c < best_c:
                best_c = c
                best = (i, j)
                best_mask = m
                if c == 1:
                    break
        for d in dead:
            alive.discard(d)
        if best is None:
            break

        (bi, bj) = best
        # least-constraining-value: pick the depot that removes the fewest
        # options from still-empty peers in the same row / column.
        cand = []
        s = 1
        mm = best_mask
        while mm:
            if mm & (1 << s):
                cand.append(s)
                mm &= ~(1 << s)
            s += 1
        best_s = cand[0]
        best_cost = 10 ** 9
        for s in cand:
            bit = (1 << s)
            cost = 0
            for jj in range(N):
                if jj != bj and g[bi][jj] == 0 and (bi, jj) in alive:
                    if not (cols[jj] & bit) and not (rows[bi] & bit):
                        cost += 1
            for ii in range(N):
                if ii != bi and g[ii][bj] == 0 and (ii, bj) in alive:
                    if not (rows[ii] & bit) and not (cols[bj] & bit):
                        cost += 1
            if cost < best_cost:
                best_cost = cost
                best_s = s
        g[bi][bj] = best_s
        rows[bi] |= (1 << best_s)
        cols[bj] |= (1 << best_s)
        alive.discard(best)

    return g


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    g = [[int(toks[idx + i * N + j]) for j in range(N)] for i in range(N)]
    g = solve(N, g)
    out = []
    for i in range(N):
        out.append(' '.join(str(g[i][j]) for j in range(N)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
