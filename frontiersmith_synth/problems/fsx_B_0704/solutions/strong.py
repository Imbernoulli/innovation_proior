# TIER: strong
"""Insight: replicate the stated kinetics in-process (it's fully deterministic and
fully specified from the input) and use it as an internal simulator to identify
the actual nucleation knee, rather than trusting one global summary statistic.

For every candidate constant plateau anchored at (or just under) each DISTINCT
threshold value actually present in the billet -- i.e. every place the regime
could plausibly flip between "protect this grain" and "sacrifice it for speed" --
simulate the whole affordable trajectory and track the BEST (minimum-F) prefix,
not just the final step (early stopping: once the protectable majority is fully
healed, further steps at that plateau may only feed a fragile minority's runaway
nucleation, so continuing is not free). Keep the best (temperature, stop-step)
pair found by this internal system identification. This is a genuine reformulation
(search + trajectory-argmin over the same dynamics) rather than a fixed recipe.
"""
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def simulate_best(d0, theta, T, max_steps):
    """Return (best_F, best_step) along the constant-T trajectory of length
    max_steps, tracking the running minimum total defect count."""
    if np is not None:
        d = d0.astype(np.int64).copy()
        th = theta
        best_F = int(d.sum())
        best_step = 0
        mob = T - 2 if T > 2 else 0
        for s in range(1, max_steps + 1):
            heal = np.minimum(d, mob)
            nuc = np.maximum(0, T - th)
            d = d - heal + nuc
            F = int(d.sum())
            if F < best_F:
                best_F = F
                best_step = s
        return best_F, best_step
    else:
        d = list(d0)
        L = len(d)
        best_F = sum(d)
        best_step = 0
        mob = T - 2 if T > 2 else 0
        for s in range(1, max_steps + 1):
            for i in range(L):
                di = d[i]
                heal = di if di < mob else mob
                nuc = T - theta[i] if T > theta[i] else 0
                d[i] = di - heal + nuc
            F = sum(d)
            if F < best_F:
                best_F = F
                best_step = s
        return best_F, best_step


def main():
    data = sys.stdin.read().split()
    p = 0
    L = int(data[p]); p += 1
    Tmax = int(data[p]); p += 1
    C0 = int(data[p]); p += 1
    n_max = int(data[p]); p += 1
    B = int(data[p]); p += 1
    d0_list = [int(data[p + i]) for i in range(L)]; p += L
    theta_list = [int(data[p + i]) for i in range(L)]; p += L

    if np is not None:
        d0 = np.array(d0_list, dtype=np.int64)
        theta = np.array(theta_list, dtype=np.int64)
    else:
        d0 = d0_list
        theta = theta_list

    D_init = sum(d0_list)

    candidates = set(range(0, Tmax + 1))
    for th in set(theta_list):
        candidates.add(max(0, th - 1))
        candidates.add(min(Tmax, th))

    best_F = D_init
    best_n = 0
    best_T = 0

    for T in sorted(candidates):
        cost_per_step = C0 + T
        if cost_per_step <= 0:
            max_afford = n_max
        else:
            max_afford = min(n_max, B // cost_per_step)
        if max_afford <= 0:
            continue
        F, step = simulate_best(d0, theta, T, max_afford)
        if F < best_F:
            best_F = F
            best_n = step
            best_T = T

    print(best_n)
    if best_n > 0:
        print(" ".join([str(best_T)] * best_n))


main()
