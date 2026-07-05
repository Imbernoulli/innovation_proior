#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE partial-Latin-square instance to stdout.

Difficulty ladder (testId 1..10): grid side N grows small->large; the reveal
fraction is chosen so a full completion always EXISTS (the revealed cells are a
subset of a hidden valid Latin square) yet finding a large completion is hard
(completing a partial Latin square is NP-complete). All randomness is seeded by
testId only, so generation is bit-for-bit deterministic.
"""
import sys
import random


def make_latin(N, rng):
    """A shuffled cyclic Latin square (valid full solution the reveal is drawn from)."""
    base = [[(i + j) % N + 1 for j in range(N)] for i in range(N)]
    rows = list(range(N)); rng.shuffle(rows)
    cols = list(range(N)); rng.shuffle(cols)
    syms = list(range(1, N + 1)); rng.shuffle(syms)  # syms[old-1] = new symbol
    return [[syms[base[rows[i]][cols[j]] - 1] for j in range(N)] for i in range(N)]


def main():
    tid = int(sys.argv[1])
    Ns = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18]
    N = Ns[(tid - 1) % len(Ns)]
    rng = random.Random(20250701 + 7919 * tid)

    L = make_latin(N, rng)

    # reveal fraction: dense enough that a naive row-major greedy gets trapped,
    # sparse enough that the score ratio has headroom.
    frac = 0.22
    k = max(N, int(round(frac * N * N)))
    cells = [(i, j) for i in range(N) for j in range(N)]
    rng.shuffle(cells)

    given = [[0] * N for _ in range(N)]
    for (i, j) in cells[:k]:
        given[i][j] = L[i][j]

    out = [str(N)]
    for i in range(N):
        out.append(' '.join(str(given[i][j]) for j in range(N)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
