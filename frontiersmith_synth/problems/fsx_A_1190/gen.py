#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE spectral-unmixing instance to stdout.

A hidden scene is a linear mixture of K=3 pure material spectra ("endmembers") m_1,m_2,m_3
in R=8 bands: pixel j's spectrum is y_j = sum_k a_kj * m_k + noise, where the abundance
vector a_j is nonnegative and sums to 1 (a point in the standard 2-simplex). For several
test ids the abundances are additionally capped (max_k a_kj <= cap < 1): NO pixel is ever a
pure material, but pixels near each EDGE of the simplex (two-material mixtures) are still
densely sampled. The hidden endmembers, abundances and cap are NEVER printed; they are
reconstructed inside verify.py from the testId using the exact same seeded formula (see
`hidden_instance` below, duplicated verbatim in verify.py).
"""
import sys, math, random

R = 8
K = 3

# per-testId ceiling on the largest abundance component (None = uncapped); every test id here
# is at least mildly capped so no pixel is ever pure, and cases 1,3,4,6,7,8,10 use a cap tight
# enough that a single-most-extreme-pixel estimate is measurably short of the true vertex.
CAP_BY_TEST = {1: 0.95, 2: 0.92, 3: 0.85, 4: 0.80, 5: 0.90,
               6: 0.88, 7: 0.75, 8: 0.85, 9: 0.72, 10: 0.90}
SIGMA = 0.01  # additive measurement-noise std


def hidden_instance(t):
    """Deterministic hidden scene for this test id (lives in gen AND verify, never printed)."""
    rng = random.Random(31337 + 101 * t)
    N = 24 + 8 * t
    cap = CAP_BY_TEST.get(t, None)

    while True:
        M = [[round(rng.uniform(0.1, 1.0), 6) for _ in range(K)] for _ in range(R)]
        ok = True
        for a_ in range(K):
            for b_ in range(a_ + 1, K):
                d = sum((M[r][a_] - M[r][b_]) ** 2 for r in range(R)) ** 0.5
                if d < 0.8:
                    ok = False
        if ok:
            break

    A, Y = [], []
    for _j in range(N):
        while True:
            expo = [-math.log(rng.random()) for _ in range(K)]
            s = sum(expo)
            u = [e / s for e in expo]
            if cap is None or max(u) <= cap:
                break
        a = u
        y = [sum(M[r][k] * a[k] for k in range(K)) + rng.gauss(0.0, SIGMA) for r in range(R)]
        y = [max(0.0, v) for v in y]
        A.append(a)
        Y.append(y)
    return N, cap, M, A, Y


def main():
    t = int(sys.argv[1])
    N, _cap, _M, _A, Y = hidden_instance(t)

    out = [str(t), "%d %d %d" % (R, K, N)]
    for y in Y:
        out.append(" ".join("%.6f" % v for v in y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
