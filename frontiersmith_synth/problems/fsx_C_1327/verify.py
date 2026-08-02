#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the anti-reflection
worst-case coating-stack problem.

Reads the instance from <in> and the participant stack from <out>. Validates
strictly (token count, integer ranges, finite positive thicknesses). On ANY
violation prints `Ratio: 0.0` and exits 0.

Otherwise computes the stack's WORST-CASE reflectance over an exact grid of
5 incidence angles in [0, theta_max_deg] times all 3^L per-layer thickness
perturbations in {-tol_j, 0, +tol_j} (transfer-matrix method, s-polarised,
lossless dielectrics). Suppression score F = -10*log10(max(worst_R, 1e-6))
(decibels). The checker also builds its own single-layer baseline (the
FIRST provided material, nominal normal-incidence quarter-wave thickness --
a genuine do-nothing-clever construction) scored the SAME way -> B.
sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000.

Pure function of (in,out): no randomness, no wall-time. O(3^L * L) per case
with L <= 5 -> a few thousand transfer-matrix evaluations, well under a
second.
"""
import sys
import math
import itertools

D_MAX = 1200.0


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_instance(path):
    with open(path) as f:
        toks = f.read().split()
    p = 0
    N_max = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    materials = []  # (n_i, tol_i)
    for _ in range(K):
        ni = float(toks[p]); p += 1
        ti = float(toks[p]); p += 1
        materials.append((ni, ti))
    n0 = float(toks[p]); p += 1
    n_sub = float(toks[p]); p += 1
    lam0 = float(toks[p]); p += 1
    theta_max = float(toks[p]); p += 1
    return N_max, K, materials, n0, n_sub, lam0, theta_max


def parse_output(path, N_max, K):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) < 1:
        return None
    try:
        Lraw = toks[0]
        L = int(Lraw)
        if str(L) != Lraw:
            return None
    except ValueError:
        return None
    if L < 1 or L > N_max:
        return None
    if len(toks) != 1 + 2 * L:
        return None
    layers = []  # (mat_idx0, thickness)
    p = 1
    for _ in range(L):
        mtok = toks[p]; p += 1
        dtok = toks[p]; p += 1
        try:
            m = int(mtok)
            if str(m) != mtok:
                return None
        except ValueError:
            return None
        if m < 1 or m > K:
            return None
        try:
            d = float(dtok)
        except ValueError:
            return None
        if not math.isfinite(d):
            return None
        if d <= 0.0 or d > D_MAX:
            return None
        layers.append((m - 1, d))
    return layers


def matmul2(A, B):
    return (
        A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
        A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
    )


def reflectance(n0, theta0_deg, layers, n_sub, lam):
    """layers: list of (n_j, d_j). s-polarised transfer-matrix reflectance."""
    theta0 = math.radians(theta0_deg)
    sin0 = n0 * math.sin(theta0)
    M = (1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j)
    for nj, dj in layers:
        s = sin0 / nj
        c2 = 1.0 - s * s
        if c2 < 0.0:
            c2 = 0.0
        c = math.sqrt(c2)
        eta = nj * c
        delta = 2.0 * math.pi * nj * dj * c / lam
        cd, sd = math.cos(delta), math.sin(delta)
        Mj = (cd, 1j * sd / eta, 1j * eta * sd, cd)
        M = matmul2(M, Mj)
    s_sub = sin0 / n_sub
    c2s = 1.0 - s_sub * s_sub
    if c2s < 0.0:
        c2s = 0.0
    c_sub = math.sqrt(c2s)
    eta_sub = n_sub * c_sub
    eta0 = n0 * math.cos(theta0)
    Bc = M[0] + M[1] * eta_sub
    Cc = M[2] + M[3] * eta_sub
    num = eta0 * Bc - Cc
    den = eta0 * Bc + Cc
    r = num / den
    return r.real * r.real + r.imag * r.imag


def angle_grid(theta_max_deg):
    if theta_max_deg <= 0:
        return [0.0]
    return [theta_max_deg * i / 4.0 for i in range(5)]


def worst_case_R(layer_specs, theta_max_deg, n0, n_sub, lam):
    """layer_specs: list of (n_j, d_j, tol_j). Exact worst case over the
    angle grid x all 3^L independent per-layer thickness perturbations."""
    L = len(layer_specs)
    angles = angle_grid(theta_max_deg)
    worst = 0.0
    for combo in itertools.product((-1, 0, 1), repeat=L):
        pert = [(nj, dj * (1.0 + k * tj)) for (nj, dj, tj), k in zip(layer_specs, combo)]
        for th in angles:
            R = reflectance(n0, th, pert, n_sub, lam)
            if R > worst:
                worst = R
    return worst


def suppression_db(worst_R):
    return -10.0 * math.log10(max(worst_R, 1e-6))


def pick_baseline_index(materials, n0, n_sub):
    """Trivial baseline: always the FIRST provided material (no search for
    the best index match) -- a genuine do-nothing-clever construction."""
    return 0


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        N_max, K, materials, n0, n_sub, lam0, theta_max = parse_instance(in_path)
    except Exception as e:
        fail("bad instance: %r" % (e,))

    layers = parse_output(out_path, N_max, K)
    if layers is None:
        fail("malformed output (token count / range / non-finite thickness)")

    layer_specs = [(materials[m][0], d, materials[m][1]) for (m, d) in layers]
    worst_R = worst_case_R(layer_specs, theta_max, n0, n_sub, lam0)
    F = suppression_db(worst_R)

    bi = pick_baseline_index(materials, n0, n_sub)
    nb, tb = materials[bi]
    db0 = lam0 / (4.0 * nb)
    if db0 <= 0.0 or db0 > D_MAX:
        db0 = min(max(db0, 1e-3), D_MAX)
    base_worst_R = worst_case_R([(nb, db0, tb)], theta_max, n0, n_sub, lam0)
    B = suppression_db(base_worst_R)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("L=%d worst_R=%.8f F=%.4fdB base_worst_R=%.8f B=%.4fdB" %
          (len(layers), worst_R, F, base_worst_R, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
