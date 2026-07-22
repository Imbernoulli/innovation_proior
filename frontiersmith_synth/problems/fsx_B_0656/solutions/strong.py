# TIER: strong
"""Insight: matching a single tristimulus point under one illuminant is
under-determined (many metameric spectra project to the same point), but
matching the FULL reflectance-spectrum SHAPE is illuminant-invariant -- it
automatically stays close under every illuminant simultaneously.

Reformulation: Kubelka-Munk mixing is LINEAR in (K, S): K_mix = sum c_i K_i,
S_mix = sum c_i S_i. We want the mixture's ratio K_mix/S_mix to match the
target's ratio at every band. Multiplying through, this is equivalent to
minimizing, over the concentration simplex,

    sum_band ( sum_i c_i * (K_i(band) - ratio_target(band) * S_i(band)) )^2

which is a per-band LINEAR residual in the concentrations -- solved with the
same Frank-Wolfe machinery as the naive 3-point match, just lifted from a
3-dimensional tristimulus space to the full N-dimensional spectral space.
A recipe that drives this residual to ~0 reproduces the target spectrum
everywhere (not just under one light), so it is automatically consistent
across all given illuminants -- no illuminant-specific reasoning needed.
"""
import sys
import math


def main():
    toks = sys.stdin.read().split()
    idx = 0

    def nextf():
        nonlocal idx
        v = float(toks[idx]); idx += 1
        return v

    def nexti():
        nonlocal idx
        v = int(toks[idx]); idx += 1
        return v

    N = nexti(); M = nexti(); K_ill = nexti()
    xbar = [nextf() for _ in range(N)]
    ybar = [nextf() for _ in range(N)]
    zbar = [nextf() for _ in range(N)]
    illuminants = [[nextf() for _ in range(N)] for _ in range(K_ill)]
    Rtarget = [nextf() for _ in range(N)]
    cost_weight = nextf()
    all_K, all_S = [], []
    for _ in range(M):
        all_K.append([nextf() for _ in range(N)])
        all_S.append([nextf() for _ in range(N)])
    costs = [nextf() for _ in range(M)]

    def inv_km(Rv):
        Rv = max(1e-4, min(1 - 1e-4, Rv))
        return (1.0 - Rv) ** 2 / (2.0 * Rv)

    ratio_target = [inv_km(Rtarget[b]) for b in range(N)]

    # A_i(band) = K_i(band) - ratio_target(band) * S_i(band); we want
    # sum_i c_i A_i(band) ~= 0 for every band simultaneously.
    A = []
    for i in range(M):
        A.append([all_K[i][b] - ratio_target[b] * all_S[i][b] for b in range(N)])

    # Frank-Wolfe on the simplex minimizing || sum c_i A_i ||^2 over R^N,
    # with a light cost-aware tie-break folded into the linear step.
    w = [1.0 / M] * M
    n_iters = 250
    for t in range(n_iters):
        cur = [sum(w[i] * A[i][b] for i in range(M)) for b in range(N)]
        best_i, best_val = 0, float("inf")
        for i in range(M):
            g = 2.0 * sum(cur[b] * A[i][b] for b in range(N))
            g += 0.02 * cost_weight * costs[i]  # mild preference for cheaper pigments
            if g < best_val:
                best_val = g
                best_i = i
        gamma = 2.0 / (t + 2.0)
        for i in range(M):
            w[i] *= (1.0 - gamma)
        w[best_i] += gamma

    # prune negligible weights, then renormalize onto the simplex exactly
    w = [x if x > 5e-4 else 0.0 for x in w]
    s = sum(w)
    if s <= 0:
        w = [1.0 / M] * M
        s = 1.0
    w = [x / s for x in w]
    print(" ".join(f"{x:.8f}" for x in w))


if __name__ == "__main__":
    main()
