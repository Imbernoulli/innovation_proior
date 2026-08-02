# TIER: strong
"""The insight: cap the interfacial stress concentration by keeping only the
EDGES of the bond line compliant, while leaving the interior as stiff as
possible -- instead of either (a) maximizing stiffness everywhere (greedy,
wins in the sub-critical regime but collapses past the size threshold) or
(b) retreating to uniform-softest everywhere (safe but gives up the
interior's advantage).

Re-implements the checker's own deterministic stress model (it is fully
specified in the statement) and searches a small parametrized family of
SYMMETRIC graded ramps: a soft-edge width `w` and an interior stiffness
level `mid`, with a smooth linear step through library ranks over the first
`w` segments on each side (a sharp 2-level step creates its own stress
spike right at the step -- ramping through several intermediate ranks
avoids that). Evaluates every (w, mid) template plus every plain uniform
choice directly via the exact checker objective and keeps the best.

Deliberately NOT explored: fully free per-segment (non-templated)
optimization, asymmetric grading, or multi-scale ramps -- that headroom is
left open above this solution."""
import sys
import math


def sim_H(N, Csub, k_list):
    H = [0.0]*(N+1); slip = [0.0]*(N+1); shear = [0.0]*(N+1)
    slip[0] = 1.0
    for i in range(1, N+1):
        shear[i] = k_list[i-1]*slip[i-1]
        H[i] = H[i-1] + shear[i]
        slip[i] = slip[i-1] + Csub*H[i]
    return H, shear


def sim_T(N, Csub, dAlpha, k_list):
    T = [0.0]*(N+1); tslip = [0.0]*(N+1); tshear = [0.0]*(N+1)
    for i in range(1, N+1):
        tshear[i] = k_list[i-1]*tslip[i-1]
        T[i] = T[i-1] + tshear[i]
        tslip[i] = tslip[i-1] + Csub*T[i] + dAlpha
    return T, tshear


def cycling_life(N, Csub, dAlpha, k_list, s_list, dTs, p=3):
    H, shear = sim_H(N, Csub, k_list)
    if H[N] <= 0 or not math.isfinite(H[N]):
        return 0.0
    T, tshear = sim_T(N, Csub, dAlpha, k_list)
    if not math.isfinite(T[N]):
        return 0.0
    d0 = -T[N]/H[N]
    R = 0.0
    for i in range(1, N+1):
        th = tshear[i] + d0*shear[i]
        if not math.isfinite(th):
            return 0.0
        r = abs(th)/s_list[i-1]
        if r > R:
            R = r
    if R <= 0:
        return 0.0
    Q = sum(abs(dt)**p for dt in dTs)
    try:
        denom = Q*(R**p)
    except OverflowError:
        return 0.0
    if not math.isfinite(denom) or denom <= 0:
        return 0.0
    val = 1.0/denom
    return val if math.isfinite(val) else 0.0


def design_ramp(N, w, mid):
    w = min(w, N//2)
    d = [mid]*N
    for i in range(w):
        idx = int(round(i/(w-1)*mid)) if w > 1 else mid
        d[i] = idx
        d[N-1-i] = idx
    return d


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    Csub = float(next(it)); dAlpha = float(next(it))
    C = int(next(it))
    dTs = [float(next(it)) for _ in range(C)]
    lib = []
    for _ in range(M):
        k = float(next(it)); s = float(next(it))
        lib.append((k, s))
    k_lib = [t[0] for t in lib]
    s_lib = [t[1] for t in lib]

    best_val = -1.0
    best_d = [0]*N

    cand_w = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35,
                          40, 45, 50, 55, 60, N // 2]))
    for w in cand_w:
        if w > N // 2 or w < 1:
            continue
        for mid in range(M):
            d = design_ramp(N, w, mid)
            v = cycling_life(N, Csub, dAlpha, [k_lib[a] for a in d],
                              [s_lib[a] for a in d], dTs)
            if v > best_val:
                best_val, best_d = v, d

    for j in range(M):
        d = [j]*N
        v = cycling_life(N, Csub, dAlpha, [k_lib[a] for a in d],
                          [s_lib[a] for a in d], dTs)
        if v > best_val:
            best_val, best_d = v, d

    print(" ".join(str(a) for a in best_d))


if __name__ == "__main__":
    main()
