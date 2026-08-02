# TIER: strong
"""The insight: a razor-sharp nominal (theta0=0, zero thickness error) null
is the WRONG target -- the score is a worst case over an angle range and
independent per-layer deposition drift. This solver (1) phase-matches each
candidate stack to the CENTER of the angle range instead of normal
incidence (so angle error is symmetric rather than one-sided), and (2)
ranks and selects candidates using the checker's own worst-case objective
(the max reflectance over the angle grid x all 3^L tolerance-perturbation
combinations), not the nominal value alone. That systematically prefers a
broader, shallower interference minimum over a deeper but brittle one --
exactly the trade-off a nominal-only optimizer cannot see.

Search stays cheap: a fast APPROXIMATE worst-case (few representative
angles/perturbations) ranks many (material-sequence, thickness-scale)
candidates; only the best handful are re-scored and locally polished with
the EXACT worst-case objective (same combinatorial grid the checker uses).
"""
import sys
import math
import itertools

D_MAX = 1200.0


def matmul2(A, B):
    return (
        A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
        A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
    )


def reflectance(n0, theta0_deg, layers, n_sub, lam):
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


def exact_worst_R(layer_specs, theta_max_deg, n0, n_sub, lam):
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


def approx_worst_R(layer_specs, theta_max_deg, n0, n_sub, lam):
    """Cheap proxy: a handful of representative angles/perturbations, used
    only to RANK many candidates fast. Final selection re-scores with
    exact_worst_R."""
    L = len(layer_specs)
    if theta_max_deg <= 0:
        angles = [0.0]
    else:
        angles = [0.0, theta_max_deg / 2.0, theta_max_deg]
    combos = [tuple([0] * L), tuple([1] * L), tuple([-1] * L)]
    if L >= 2:
        combos.append(tuple(1 if i % 2 == 0 else -1 for i in range(L)))
        combos.append(tuple(-1 if i % 2 == 0 else 1 for i in range(L)))
    worst = 0.0
    for combo in combos:
        pert = [(nj, dj * (1.0 + k * tj)) for (nj, dj, tj), k in zip(layer_specs, combo)]
        for th in angles:
            R = reflectance(n0, th, pert, n_sub, lam)
            if R > worst:
                worst = R
    return worst


def gen_architectures(K, N_max):
    idxs = list(range(K))
    archs = [[i] for i in idxs]
    if N_max >= 2:
        for i in idxs:
            for j in idxs:
                if i != j:
                    archs.append([i, j])
        if K == 1:
            archs.append([0, 0])
    if N_max >= 3:
        for i in idxs:
            for j in idxs:
                if i != j:
                    archs.append([i, j, i])
        if K >= 3:
            for combo in itertools.combinations(idxs, 3):
                archs.append(list(combo))
                archs.append(list(reversed(combo)))
    if N_max >= 4:
        for i in idxs:
            for j in idxs:
                if i != j:
                    archs.append([i, j, i, j])
    if N_max >= 5:
        for i in idxs:
            for j in idxs:
                if i != j:
                    archs.append([i, j, i, j, i])
        if K >= 5:
            combo = tuple(sorted(idxs))[:5]
            archs.append(list(combo))
            archs.append(list(reversed(combo)))
    return archs


def mid_angle_thickness(nj, theta_mid_deg, n0, lam0):
    theta_mid = math.radians(theta_mid_deg)
    sin0 = n0 * math.sin(theta_mid)
    s = sin0 / nj
    c2 = 1.0 - s * s
    if c2 < 0.0:
        c2 = 0.0
    c = math.sqrt(c2)
    d = lam0 / (4.0 * nj * c) if c > 1e-9 else lam0 / (4.0 * nj)
    return min(max(d, 1e-3), D_MAX)


def main():
    toks = sys.stdin.read().split()
    p = 0
    N_max = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    materials = []
    for _ in range(K):
        ni = float(toks[p]); p += 1
        ti = float(toks[p]); p += 1
        materials.append((ni, ti))
    n0 = float(toks[p]); p += 1
    n_sub = float(toks[p]); p += 1
    lam0 = float(toks[p]); p += 1
    theta_max = float(toks[p]); p += 1

    theta_mid = theta_max / 2.0
    archs = gen_architectures(K, N_max)
    SCALES = [0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25]

    candidates = []  # (approxR, arch, thicknesses)
    for arch in archs:
        d0 = [mid_angle_thickness(materials[m][0], theta_mid, n0, lam0) for m in arch]
        for s in SCALES:
            ds = [min(max(dj * s, 1e-3), D_MAX) for dj in d0]
            specs = [(materials[m][0], ds[j], materials[m][1]) for j, m in enumerate(arch)]
            Ra = approx_worst_R(specs, theta_max, n0, n_sub, lam0)
            candidates.append((Ra, arch, ds))

    candidates.sort(key=lambda t: t[0])
    top = candidates[:15]

    best_R = None
    best_arch = None
    best_ds = None
    for Ra, arch, ds in top:
        specs = [(materials[m][0], ds[j], materials[m][1]) for j, m in enumerate(arch)]
        Re = exact_worst_R(specs, theta_max, n0, n_sub, lam0)
        if best_R is None or Re < best_R:
            best_R, best_arch, best_ds = Re, arch, list(ds)

    # local polish of the winner using the EXACT worst-case objective.
    L = len(best_arch)
    OFFS = [0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10]
    for _round in range(3):
        improved = False
        for j in range(L):
            base_d = best_ds[j]
            for mul in OFFS:
                trial_d = min(max(base_d * mul, 1e-3), D_MAX)
                trial_ds = list(best_ds)
                trial_ds[j] = trial_d
                specs = [(materials[m][0], trial_ds[jj], materials[m][1])
                         for jj, m in enumerate(best_arch)]
                Re = exact_worst_R(specs, theta_max, n0, n_sub, lam0)
                if Re < best_R - 1e-15:
                    best_R, best_ds = Re, trial_ds
                    improved = True
        if not improved:
            break

    out = [str(L)]
    for j, m in enumerate(best_arch):
        out.append("%d %.6f" % (m + 1, best_ds[j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
