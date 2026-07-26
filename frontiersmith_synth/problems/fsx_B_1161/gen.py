#!/usr/bin/env python3
"""gen.py <testId> -- masked bilinear-rank-completion instance ("torn tapestry").

Builds an m x m matrix over F_p as L = U V^T (a bilinear pattern of rank
r_true) plus a handful of sparse "stain" spikes at isolated (row,col)
positions (always revealed, so they must be reproduced verbatim but also
force a rank floor above r_true). A block of R_REF rows is always left fully
intact (fully revealed, never stained) -- the "surviving reference threads"
of the tapestry. Regular rows are revealed independently with probability Q.

r_true VARIES from test to test and is never told to the solver -- discovering
it (rather than assuming a fixed textbook value) is part of the task.
Everything is seeded deterministically from testId alone.
"""
import sys, random

P = 1000003  # prime modulus, fixed for every test case

# per-test ladder: (m, r_true)
CONFIGS = [
    (14, 3), (16, 4), (18, 3), (20, 2), (22, 3),
    (26, 5), (30, 3), (34, 2), (38, 3), (42, 4),
]
R_REF = 6      # number of always-fully-revealed, always-spike-free reference rows
Q_REVEAL = 0.55


def build(testId):
    m, r_true = CONFIGS[testId - 1]
    S = max(3, m // 10)
    rng = random.Random(1161000 + testId)

    U = [[rng.randrange(P) for _ in range(r_true)] for _ in range(m)]
    V = [[rng.randrange(P) for _ in range(r_true)] for _ in range(m)]
    M = [[sum(U[i][k] * V[j][k] for k in range(r_true)) % P for j in range(m)]
         for i in range(m)]

    regular_rows = list(range(0, m - R_REF))
    ref_rows = list(range(m - R_REF, m))

    spike_rows = rng.sample(regular_rows, min(S, len(regular_rows)))
    spike_cols = rng.sample(range(m), len(spike_rows))
    spikes = list(zip(spike_rows, spike_cols))
    for (i, j) in spikes:
        delta = rng.randrange(1, P)
        M[i][j] = (M[i][j] + delta) % P

    revealed = [[False] * m for _ in range(m)]
    for i in ref_rows:
        for j in range(m):
            revealed[i][j] = True
    for i in regular_rows:
        for j in range(m):
            revealed[i][j] = rng.random() < Q_REVEAL
    for (i, j) in spikes:
        revealed[i][j] = True

    return m, M, revealed


def main():
    testId = int(sys.argv[1])
    m, M, revealed = build(testId)
    out = [f"{m} {P}"]
    for i in range(m):
        row = []
        for j in range(m):
            row.append(str(M[i][j]) if revealed[i][j] else "?")
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
