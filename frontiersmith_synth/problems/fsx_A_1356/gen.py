#!/usr/bin/env python3
"""
gen.py <testId> -> prints one zero-sum-matrix-solve instance to stdout.

Instance = a positive-integer payoff matrix A (m x n, row player maximizes) plus a
historical column-play log H (n non-negative integers). The matrix always contains:
  - an r-action "core" (rows core_rows[k], cols core_cols[k]) built as a cyclic
    win/tie/lose block with NO pure saddle point (a genuine mixed equilibrium is
    required to guarantee a good worst-case value).
  - D "decoy" rows, each of which scores very well against the columns the
    historical log favors (H), but is crushed by one specific shared "punish"
    column that the log almost never used.
Row/column identities are permuted with a per-test PRNG so no positional shortcut
(e.g. "core rows are always first") works.
Deterministic: all randomness is seeded from testId only.
"""
import random
import sys

NEUTRAL = 500

TESTS = {
    1: (2, 0), 2: (2, 1), 3: (3, 2), 4: (3, 4), 5: (3, 6),
    6: (4, 8), 7: (4, 11), 8: (4, 14), 9: (5, 17), 10: (5, 20),
}


def build(test_id):
    r, D = TESTS[test_id]
    rng = random.Random(1_000_003 * test_id + 7919)
    m = n = r + D

    row_idx = list(range(m))
    rng.shuffle(row_idx)
    col_idx = list(range(n))
    rng.shuffle(col_idx)

    core_rows = row_idx[:r]
    decoy_rows = row_idx[r:]
    core_cols = col_idx[:r]
    rest_cols = col_idx[r:]

    if D >= 2:
        punish_col = rest_cols[0]
        generic_cols = rest_cols[1:]
    elif D == 1:
        punish_col = None
        generic_cols = rest_cols
    else:
        punish_col = None
        generic_cols = []

    WIN = rng.randint(850, 950)
    LOSS = rng.randint(80, 150)

    def jit(base, spread):
        return max(1, base + rng.randint(-spread, spread))

    A = [[None] * n for _ in range(m)]

    # core-vs-core cyclic block: row k beats col (k+1)%r, loses to col (k-1)%r,
    # ties everywhere else (r==2 has no tie cell: diagonal win, off-diagonal loss).
    for k in range(r):
        for l in range(r):
            rp, cp = core_rows[k], core_cols[l]
            if r == 2:
                val = WIN if k == l else LOSS
            else:
                if l == (k + 1) % r:
                    val = WIN
                elif l == (k - 1) % r:
                    val = LOSS
                else:
                    val = NEUTRAL
            A[rp][cp] = jit(val, 6)

    # core rows vs every non-core column: neutral (decoys/punish do not help or
    # hurt a core row).
    for rp in core_rows:
        for cp in rest_cols:
            A[rp][cp] = jit(NEUTRAL, 10)

    # decoy rows vs core columns: neutral.
    for rp in decoy_rows:
        for cp in core_cols:
            A[rp][cp] = jit(NEUTRAL, 10)

    # decoy rows vs generic/punish columns.
    for k, rp in enumerate(decoy_rows):
        HIGH_k = 780 - 6 * k
        LOW_k = 150 + 6 * k
        for cp in generic_cols:
            A[rp][cp] = jit(HIGH_k, 10)
        if punish_col is not None:
            A[rp][punish_col] = jit(LOW_k, 8)
        elif D == 1:
            A[rp][rest_cols[0]] = jit(NEUTRAL, 10)

    for i in range(m):
        for j in range(n):
            assert A[i][j] is not None
            A[i][j] = max(1, min(999, A[i][j]))

    # historical opponent play log: heavy on generic columns, negligible on the
    # punish column, light on core columns.
    H = [0] * n
    for cp in core_cols:
        H[cp] = rng.randint(1, 3)
    for cp in generic_cols:
        H[cp] = rng.randint(80, 120)
    if punish_col is not None:
        H[punish_col] = rng.randint(0, 2)
    elif D == 1:
        H[rest_cols[0]] = rng.randint(1, 3)
    if sum(H) == 0:
        H[0] = 1

    return m, n, A, H


def main():
    test_id = int(sys.argv[1])
    m, n, A, H = build(test_id)
    out = [f"{m} {n}"]
    for row in A:
        out.append(" ".join(map(str, row)))
    out.append(str(sum(H)))
    out.append(" ".join(map(str, H)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
