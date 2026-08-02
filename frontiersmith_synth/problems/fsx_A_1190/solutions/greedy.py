# TIER: greedy
"""Textbook vertex-selection unmixing (N-FINDR / pixel-purity style): project the pixels
onto their top (K-1) principal axes, greedily pick the K OBSERVED pixels that maximize the
simplex volume they span, declare those pixels themselves as the endmembers, then fit each
pixel's abundance by nonnegative least squares (renormalized to sum to 1) against those
endmembers. This is the "pick an extreme observed pixel" trap: when no pixel is pure the
true vertices lie outside the data cloud and this method can never reach them."""
import sys
import numpy as np
from scipy.optimize import nnls


def pick_extreme(Z, K):
    N = Z.shape[0]
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    i0, j0 = np.unravel_index(np.argmax(d), d.shape)
    chosen = [int(i0), int(j0)]
    while len(chosen) < K:
        best_idx, best_vol = None, -1.0
        for cand in range(N):
            if cand in chosen:
                continue
            pts = Z[chosen + [cand]]
            base = pts[0]
            edges = pts[1:] - base
            if edges.shape[0] == edges.shape[1]:
                vol = abs(np.linalg.det(edges))
            else:
                vol = float(np.linalg.norm(edges))
            if vol > best_vol:
                best_vol = vol
                best_idx = cand
        chosen.append(best_idx)
    return chosen


def nnls_abundance(Y, M_hat):
    K = M_hat.shape[0]
    N = Y.shape[0]
    A = np.zeros((N, K))
    BIG = 100.0
    Maug = np.vstack([M_hat.T, BIG * np.ones((1, K))])
    for j in range(N):
        yaug = np.concatenate([Y[j], [BIG]])
        sol, _ = nnls(Maug, yaug)
        s = sol.sum()
        A[j] = sol / s if s > 1e-9 else np.full(K, 1.0 / K)
    return A


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    _t = int(next(it))
    R = int(next(it)); K = int(next(it)); N = int(next(it))
    Y = np.array([[float(next(it)) for _ in range(R)] for _ in range(N)])

    mean_y = Y.mean(axis=0)
    Yc = Y - mean_y
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    Z = Yc @ Vt[:K - 1].T

    chosen = pick_extreme(Z, K)
    M_hat = Y[chosen]
    A_hat = nnls_abundance(Y, M_hat)

    out = []
    for k in range(K):
        out.append(" ".join("%.6f" % v for v in M_hat[k]))
    for j in range(N):
        out.append(" ".join("%.6f" % v for v in A_hat[j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
