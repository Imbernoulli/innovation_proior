# TIER: greedy
"""The obvious approach: design the stack to minimize reflectance AT THE
NOMINAL DESIGN POINT ONLY -- normal incidence (theta0=0), thickness exactly
as designed (no deposition error). This is the textbook quarter-wave /
multilayer AR-coating optimization: search material sequences and thickness
multipliers, always scoring with a single reflectance() call at theta0=0
with zero perturbation.

It reliably finds a near-zero nominal reflectance -- but that null can be
razor-sharp: nothing here ever looks at theta_max or at any material's
tolerance, so a stack built from a high-index, high-tolerance material tuned
to a hair's-width nominal zero looks best to this search even though it is
the most fragile choice once the checker's worst-case grid (angle sweep +
independent per-layer thickness drift) is applied.
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
    # theta_max_deg intentionally not read: nominal-only search.

    archs = gen_architectures(K, N_max)
    MULTS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    best_R = None
    best_layers = None
    for arch in archs:
        L = len(arch)
        ds = [min(max(lam0 / (4.0 * materials[m][0]), 1e-3), D_MAX) for m in arch]
        cur = [(materials[m][0], ds[j]) for j, m in enumerate(arch)]
        curR = reflectance(n0, 0.0, cur, n_sub, lam0)
        for _round in range(4):
            improved = False
            for j in range(L):
                base_d = ds[j]
                bestj_d, bestj_R = base_d, curR
                for mul in MULTS:
                    trial_d = min(max(base_d * mul, 1e-3), D_MAX)
                    trial_layers = list(cur)
                    trial_layers[j] = (materials[arch[j]][0], trial_d)
                    R = reflectance(n0, 0.0, trial_layers, n_sub, lam0)
                    if R < bestj_R - 1e-15:
                        bestj_R, bestj_d = R, trial_d
                if bestj_d != base_d:
                    ds[j] = bestj_d
                    cur[j] = (materials[arch[j]][0], bestj_d)
                    curR = bestj_R
                    improved = True
            if not improved:
                break
        if best_R is None or curR < best_R:
            best_R = curR
            best_layers = [(arch[j], ds[j]) for j in range(L)]

    out = [str(len(best_layers))]
    for m, d in best_layers:
        out.append("%d %.6f" % (m + 1, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
