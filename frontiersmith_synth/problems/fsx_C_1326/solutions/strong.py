# TIER: strong
"""The insight: decouple the electrochemical-window requirement from the bulk
solvent choice with a small sacrificial SEI-forming additive, instead of
retreating to a native-safe (but slow) solvent.

For every candidate primary solvent i (single-solvent blend, x_i=1):
  - branch A: use it bare, if its native threshold already clears V_target.
  - branch B: for every non-empty subset of additives, dose them (in order of
    SEI-strength-per-combined-penalty) just far enough to reach the required
    coverage, and use that dosing IF it fits inside the additive budget.
Evaluate every reachable (solvent, dosing) pair under the checker's own
objective and keep the best. This lets a low-viscosity/high-conductivity
solvent that would otherwise decompose get "rescued" by a cheap additive,
instead of the whole formulation retreating to the safe-but-slow solvent.

Multi-solvent blending and joint (non-single-primary) additive tuning are
deliberately NOT explored here -- that headroom is left open above this
solution."""
import sys
from itertools import combinations


def compute_F(x, a, Kconst, V_target, cov_target, solv, add):
    N, M = len(solv), len(add)
    used_solv = [i for i in range(N) if x[i] > 1e-9]
    min_thr_used = min(solv[i][2] for i in used_solv)
    used_add = [min(a[j], add[j][3]) for j in range(M)]
    coverage = sum(used_add[j] * add[j][0] for j in range(M))
    window_ok = (coverage >= cov_target - 1e-6) or (min_thr_used >= V_target - 1e-6)
    if not window_ok:
        return 0.0
    numer = sum(x[i] * solv[i][1] for i in range(N)) - sum(a[j] * add[j][2] for j in range(M))
    numer = max(numer, 0.0)
    denom = sum(x[i] * solv[i][0] for i in range(N)) + sum(a[j] * add[j][1] for j in range(M))
    if denom <= 0:
        return 0.0
    return Kconst * numer / denom


def dose_for_coverage(cov_target, add, subset):
    """Greedily dose additives in `subset`, in order of SEI-strength per
    combined (viscosity+conductivity) penalty, just far enough to reach
    cov_target. Returns (dosing dict j->amount, coverage achieved)."""
    # dose to a hair over cov_target (not exactly on the boundary): the
    # checker reads doses back from printed text, so landing exactly on the
    # threshold risks a spurious fail after print/parse round-off.
    target = cov_target * (1.0 + 1e-4) + 1e-6
    order = sorted(subset, key=lambda j: -add[j][0] / (add[j][1] + add[j][2]))
    dosing = {}
    cov = 0.0
    for j in order:
        p, etapen, kappapen, cap = add[j]
        if cov >= target:
            break
        remaining = target - cov
        dose = min(cap, remaining / p)
        dosing[j] = dose
        cov += dose * p
    return dosing, cov


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    A_max = float(next(it)); V_target = float(next(it))
    cov_target = float(next(it)); Kconst = float(next(it))
    solv = []
    for _ in range(N):
        eta = float(next(it)); kappa = float(next(it)); thr = float(next(it))
        solv.append((eta, kappa, thr))
    add = []
    for _ in range(M):
        p = float(next(it)); etapen = float(next(it))
        kappapen = float(next(it)); cap = float(next(it))
        add.append((p, etapen, kappapen, cap))

    best_val = -1.0
    best_x = [0.0] * N
    best_a = [0.0] * M

    for i in range(N):
        eta_i, kappa_i, thr_i = solv[i]
        x = [0.0] * N
        x[i] = 1.0

        # branch A: bare, if natively safe
        if thr_i >= V_target - 1e-9:
            a0 = [0.0] * M
            v = compute_F(x, a0, Kconst, V_target, cov_target, solv, add)
            if v > best_val:
                best_val, best_x, best_a = v, x[:], a0

        # branch B: rescue via additive subset
        for r in range(1, M + 1):
            for subset in combinations(range(M), r):
                dosing, cov = dose_for_coverage(cov_target, add, subset)
                total_dose = sum(dosing.values())
                if cov < cov_target - 1e-6 or total_dose > A_max + 1e-9:
                    continue
                a_vec = [0.0] * M
                for j, d in dosing.items():
                    a_vec[j] = d
                v = compute_F(x, a_vec, Kconst, V_target, cov_target, solv, add)
                if v > best_val:
                    best_val, best_x, best_a = v, x[:], a_vec

    if best_val < 0:
        # should not happen (branch A on the native-safest solvent always works)
        i_star = max(range(N), key=lambda i: (solv[i][2], -i))
        best_x = [0.0] * N
        best_x[i_star] = 1.0
        best_a = [0.0] * M

    print(" ".join(f"{v:.6f}" for v in best_x))
    print(" ".join(f"{v:.6f}" for v in best_a))


if __name__ == "__main__":
    main()
