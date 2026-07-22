#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>

Scores a pigment-recipe artifact against a pigment-metamerism-matching
instance. Prints "Ratio: <float in [0,1]>" as its last line and exits 0.
"""
import sys
import math


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def km_reflectance(ratio):
    r = max(0.0, ratio)
    return 1.0 + r - math.sqrt(r * r + 2.0 * r)


def lab_f(t):
    d = 6.0 / 29.0
    if t > d ** 3:
        return t ** (1.0 / 3.0)
    return t / (3 * d * d) + 4.0 / 29.0


def xyz_to_lab(X, Y, Z, Xn, Yn, Zn):
    fx = lab_f(X / Xn)
    fy = lab_f(Y / Yn)
    fz = lab_f(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def reflectance_to_lab(R, illum, xbar, ybar, zbar, N):
    Ywhite = sum(illum[b] * ybar[b] for b in range(N))
    Ywhite = max(Ywhite, 1e-9)
    k = 100.0 / Ywhite
    X = k * sum(illum[b] * xbar[b] * R[b] for b in range(N))
    Y = k * sum(illum[b] * ybar[b] * R[b] for b in range(N))
    Z = k * sum(illum[b] * zbar[b] * R[b] for b in range(N))
    Xn = k * sum(illum[b] * xbar[b] for b in range(N))
    Yn = 100.0
    Zn = k * sum(illum[b] * zbar[b] for b in range(N))
    Xn = max(Xn, 1e-9)
    Zn = max(Zn, 1e-9)
    return xyz_to_lab(X, Y, Z, Xn, Yn, Zn)


def total_objective(weights, all_K, all_S, costs, Rtarget, illuminants, xbar, ybar, zbar, N, cost_weight):
    R = []
    for b in range(N):
        kmix = sum(w * K[b] for w, K in zip(weights, all_K))
        smix = sum(w * S[b] for w, S in zip(weights, all_S))
        smix = max(smix, 1e-6)
        R.append(km_reflectance(kmix / smix))

    total_de = 0.0
    for illum in illuminants:
        Lr, ar, br = reflectance_to_lab(R, illum, xbar, ybar, zbar, N)
        Lt, at, bt = reflectance_to_lab(Rtarget, illum, xbar, ybar, zbar, N)
        total_de += math.sqrt((Lr - Lt) ** 2 + (ar - at) ** 2 + (br - bt) ** 2)

    n_used = sum(1 for w, c in zip(weights, costs) if w > 1e-4)
    used_cost = sum(c for w, c in zip(weights, costs) if w > 1e-4)
    penalty = cost_weight * used_cost
    return total_de + penalty, n_used


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    idx = 0

    def nextf():
        nonlocal idx
        v = float(toks[idx])
        idx += 1
        return v

    def nexti():
        nonlocal idx
        v = int(toks[idx])
        idx += 1
        return v

    N = nexti()
    M = nexti()
    K_ill = nexti()

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

    # ---- parse participant output ----
    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("missing output file")

    if len(out_toks) != M:
        fail(f"expected {M} concentrations, got {len(out_toks)}")

    weights = []
    for t in out_toks:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric token in output")
        if not math.isfinite(v):
            fail("non-finite value in output")
        weights.append(v)

    for w in weights:
        if w < -1e-6:
            fail("negative concentration")
    weights = [max(0.0, w) for w in weights]

    s = sum(weights)
    if abs(s - 1.0) > 1e-3:
        fail(f"concentrations must sum to 1 (got {s:.6f})")
    # renormalize tiny floating slack away
    weights = [w / s for w in weights]

    F, n_used = total_objective(weights, all_K, all_S, costs, Rtarget, illuminants,
                                 xbar, ybar, zbar, N, cost_weight)

    # ---- internal baseline: uniform mixture over all pigments ----
    base_w = [1.0 / M] * M
    B, _ = total_objective(base_w, all_K, all_S, costs, Rtarget, illuminants,
                            xbar, ybar, zbar, N, cost_weight)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"objective={F:.6f} baseline={B:.6f} n_used={n_used}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
