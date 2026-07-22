#!/usr/bin/env python3
"""gen.py <testId> -- prints one 'nodal-line-gardening' instance to stdout.

Instance format (all integers, whitespace separated across lines):
  line 1: N                      (grid is N x N, cells indexed row-major 0..N*N-1)
  line 2: k cap budget           (target mode rank k [1-indexed], per-cell mass cap, total mass budget)
  line 3: t                      (number of target cells)
  line 4: t integers             (0-indexed target cell ids, row-major = row*N+col)

Deterministic: every random/search choice below is a pure function of testId and N
(no wall-clock, no external entropy). The tiny per-cell mass epsilon (1e-6 * cell
index) that appears in both gen.py and verify.py breaks exact spectral ties
generically so the eigen-decomposition is always well-defined.
"""
import sys
import numpy as np

EPS = 1e-6
TAU = 0.2


def build_K(N):
    Nc = N * N
    K = np.zeros((Nc, Nc))
    for r in range(N):
        for c in range(N):
            i = r * N + c
            K[i, i] = 4.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    K[i, rr * N + cc] = -1.0
    return K


def base_mass(N):
    Nc = N * N
    return np.array([1.0 + i * EPS for i in range(Nc)])


def spectrum(K, m):
    s = 1.0 / np.sqrt(m)
    L = (s[:, None] * K) * s[None, :]
    w, U = np.linalg.eigh(L)
    return w, U, s


def physical_hat(U_col, s):
    v = s * U_col
    ma = np.max(np.abs(v))
    return v / ma


# Per-testId plan: (N, k, trap) -- k is the 1-indexed rank of the SMALLER member
# of a numerically-verified, well-isolated near-degenerate adjacent eigenvalue
# pair of the *unloaded* (zero mass-loading) clamped-membrane spectrum.  Every
# entry below was checked offline to satisfy: |w[k]-w[k+1]| < 1e-3 and both
# neighbouring gaps > 0.03 (isolation from unrelated modes).
PLAN = {
    1: (3, 2, True),
    2: (4, 2, True),
    3: (4, 5, False),
    4: (5, 5, True),
    5: (5, 9, False),
    6: (6, 7, True),
    7: (6, 12, True),
    8: (7, 9, False),
    9: (7, 14, True),
    10: (7, 18, True),
}


def choose_targets(N, k, trap, w, U, s):
    Nc = N * N
    idxA, idxB = k - 1, k  # 0-indexed adjacent tied pair
    A = physical_hat(U[:, idxA], s)
    B = physical_hat(U[:, idxB], s)
    t = max(4, min(8, N))
    if trap:
        # cells that are ANTINODES of the partner mode B but already NODAL in
        # the current rank-k mode A: loading them directly (the obvious move)
        # preferentially lowers omega^2 of B, risking a rank swap that makes
        # the target cells become ANTINODES of whatever ends up at rank k.
        score = np.abs(B) - np.abs(A)
    else:
        # cells that are simply antinodes of the current rank-k mode itself.
        score = np.abs(A)
    order = np.argsort(-score)
    targets = sorted(int(c) for c in order[:t])
    return targets


def main():
    testId = int(sys.argv[1])
    N, k, trap = PLAN[testId]
    K = build_K(N)
    m0 = base_mass(N)
    w, U, s = spectrum(K, m0)
    targets = choose_targets(N, k, trap, w, U, s)
    t = len(targets)

    cap = 6
    budget = cap * t
    assert budget // (N * N) <= cap  # keeps the checker's uniform baseline feasible

    out = []
    out.append(str(N))
    out.append(f"{k} {cap} {budget}")
    out.append(str(t))
    out.append(" ".join(str(c) for c in targets))
    print("\n".join(out))


if __name__ == "__main__":
    main()
