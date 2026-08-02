# TIER: strong
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); F = int(next(it))
    L = [int(next(it)) for _ in range(T)]
    y = [int(next(it)) for _ in range(T)]
    Base = [int(next(it)) for _ in range(T)]
    hn = int(next(it)); hd = int(next(it))
    next(it); next(it)  # S, window_len (hidden windows aren't needed by the hedge)

    # Insight: ANY date could turn out to be the stressed one, so plan against the
    # worst case at every date simultaneously -- cap each date's exposure at what the
    # market could still absorb even under the haircut. Start from the natural
    # cashflow-matched ladder (zero prefunding slack) and, wherever a date's natural
    # maturity would exceed its own worst-case safe capacity, shift the excess to
    # EARLIER dates that still have spare safe capacity (moving a unit earlier can
    # only help the cumulative prefunding requirement, never hurt it), preferring the
    # cheapest available yield among the safe slots. This smooths the rollover profile
    # away from the clustering trap at a modest yield cost.
    safe_cap = [(Base[t] * hn) // hd for t in range(T)]
    p = L[:]

    for t in range(T):
        if p[t] <= safe_cap[t]:
            continue
        excess = p[t] - safe_cap[t]
        candidates = [tp for tp in range(t) if p[tp] < safe_cap[tp]]
        candidates.sort(key=lambda tp: y[tp])
        for tp in candidates:
            if excess <= 0:
                break
            room = safe_cap[tp] - p[tp]
            take = min(room, excess)
            p[tp] += take
            p[t] -= take
            excess -= take
        # if no earlier room remains, the residual is left in place: pushing it LATER
        # would break the prefunding requirement, which the checker penalizes harder.

    assert sum(p) == F
    print(" ".join(map(str, p)))


main()
