# TIER: greedy
"""Naive recipe: match the target's tristimulus point under the FIRST
illuminant only, treating each pigment's own (undiluted) reflectance as if
reflectances mixed linearly (they don't -- Kubelka-Munk mixes K and S, then
takes a nonlinear ratio). This is the "solve 3 colour equations" shortcut an
average coder reaches for first. It ignores every other illuminant and the
overall spectral shape, so it is exactly the kind of solution that a
metameric decoy pigment (matches under illuminant 1, diverges elsewhere)
will fool.
"""
import sys
import math


def km_reflectance(ratio):
    r = max(0.0, ratio)
    return 1.0 + r - math.sqrt(r * r + 2.0 * r)


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

    L0 = illuminants[0]
    Ywhite = max(sum(L0[b] * ybar[b] for b in range(N)), 1e-9)
    k0 = 100.0 / Ywhite

    def tristimulus(R):
        X = k0 * sum(L0[b] * xbar[b] * R[b] for b in range(N))
        Y = k0 * sum(L0[b] * ybar[b] * R[b] for b in range(N))
        Z = k0 * sum(L0[b] * zbar[b] * R[b] for b in range(N))
        return (X, Y, Z)

    # each pigment's own (undiluted) reflectance -> its tristimulus vertex
    vertices = []
    for i in range(M):
        ratio = [all_K[i][b] / max(all_S[i][b], 1e-6) for b in range(N)]
        Ri = [km_reflectance(r) for r in ratio]
        vertices.append(tristimulus(Ri))

    Tt = tristimulus(Rtarget)

    # Frank-Wolfe on the simplex minimizing || sum w_i v_i - target ||^2,
    # WRONGLY assuming reflectances (hence tristimulus) mix linearly.
    w = [1.0 / M] * M
    for t in range(60):
        cur = [sum(w[i] * vertices[i][d] for i in range(M)) for d in range(3)]
        diff = [cur[d] - Tt[d] for d in range(3)]
        best_i, best_val = 0, float("inf")
        for i in range(M):
            g = 2.0 * sum(diff[d] * vertices[i][d] for d in range(3))
            if g < best_val:
                best_val = g
                best_i = i
        gamma = 2.0 / (t + 2.0)
        for i in range(M):
            w[i] *= (1.0 - gamma)
        w[best_i] += gamma

    s = sum(w)
    w = [x / s for x in w]
    print(" ".join(f"{x:.8f}" for x in w))


if __name__ == "__main__":
    main()
