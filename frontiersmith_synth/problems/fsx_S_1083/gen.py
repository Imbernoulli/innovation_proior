#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of 'shared-basis signal probes' to stdout.

Deterministic: all randomness is seeded from testId only.

Instance: K signal families in R^n. Family k is sparse (sparsity s) in a planted
basis B_k = [U | V_k] (n x d_k, columns orthonormal), where U (n x c) is a subspace
SHARED by all families and V_k (n x r_k) is family-specific and orthogonal to U.
The solver designs m integer probe vectors (entries in [-pmax, pmax]); the checker
measures hidden test signals through the probes and decodes with a FIXED OMP solver.
"""
import sys
import numpy as np

# (n, K, c, r_list, s)  -- difficulty ladder: small -> large/adversarial.
# Later cases raise the sparsity s and the union dimension c + sum(r) past the
# probe budget m (=40): generic (random) probes hit the OMP phase transition
# there, while structure-aware probes do not.
CASES = [
    (64, 3, 12, [6, 6, 6], 5),
    (80, 3, 10, [8, 8, 8], 6),
    (96, 4, 10, [7, 7, 7, 7], 7),
    (112, 4, 6, [4, 4, 6, 20], 7),
    (96, 4, 12, [8, 8, 8, 8], 8),
    (128, 5, 8, [7, 7, 7, 7, 7], 8),
    (112, 4, 8, [9, 9, 9, 9], 8),
    (128, 5, 12, [5, 5, 5, 5, 14], 9),
    (128, 5, 10, [9, 9, 9, 9, 9], 9),
    (128, 4, 8, [12, 12, 12, 12], 9),
]

M = 40      # probe budget
PMAX = 7    # entry bound


def main():
    tid = int(sys.argv[1])
    assert 1 <= tid <= len(CASES), "testId out of range"
    n, K, c, r_list, s = CASES[tid - 1]
    rng = np.random.default_rng(tid * 7919 + 13)

    # shared subspace U (n x c), orthonormal columns
    A = rng.standard_normal((n, c))
    U, _ = np.linalg.qr(A)

    bases = []
    for k in range(K):
        r = r_list[k]
        V = rng.standard_normal((n, r))
        V -= U @ (U.T @ V)
        Vk, _ = np.linalg.qr(V)
        Vk = Vk[:, :r]
        B = np.concatenate([U, Vk], axis=1)  # n x (c + r), orthonormal columns
        bases.append(B)

    out = []
    out.append(f"{n} {M} {K} {s} {PMAX}")
    for B in bases:
        d = B.shape[1]
        out.append(str(d))
        for i in range(n):
            out.append(" ".join("%.10f" % B[i, j] for j in range(d)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
