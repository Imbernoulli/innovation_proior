# TIER: strong
"""Envelope insight: track the running MAX-over-scenarios prefix of cumulative inflow
(the true overflow threat) and release the bare minimum needed to stay under it -- this
keeps storage as high as feasible for as long as feasible in EVERY scenario at once,
which is what a concave, storage-driven power curve rewards. No single scenario's own
optimum has this shape; it only appears once you take the max across all K trajectories
at every prefix."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); K = int(next(it)); C = int(next(it)); D = int(next(it))
    S0 = int(next(it)); Rmax = int(next(it))
    next(it); next(it)  # HMIN, HMAX unused by this strategy
    scenarios = [[int(next(it)) for _ in range(T)] for _ in range(K)]

    cum = [[0] * (T + 1) for _ in range(K)]
    for k in range(K):
        for t in range(1, T + 1):
            cum[k][t] = cum[k][t - 1] + scenarios[k][t - 1]
    max_in = [max(cum[k][t] for k in range(K)) for t in range(T + 1)]
    min_in = [min(cum[k][t] for k in range(K)) for t in range(T + 1)]

    lo = [0] * (T + 1)
    hi = [0] * (T + 1)
    for t in range(1, T + 1):
        lo[t] = max(lo[t - 1], max(0, S0 + max_in[t] - C))
        hi[t] = max(hi[t - 1], S0 + min_in[t] - D)

    Rel = [0] * (T + 1)
    for t in range(1, T + 1):
        floor_ = max(Rel[t - 1], lo[t])
        ceil_ = min(Rel[t - 1] + Rmax, hi[t])
        Rel[t] = min(max(lo[t], floor_), ceil_)

    r = [Rel[t] - Rel[t - 1] for t in range(1, T + 1)]
    print(" ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
