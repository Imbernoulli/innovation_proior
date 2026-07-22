# TIER: trivial
"""Safety-first default: compute the true worst-case feasibility corridor [lo(t), hi(t)]
(so it never violates a hard constraint) but, having no view on the concave power curve,
picks a point 90% of the way toward the "drain fast, keep storage low" edge of that
corridor every week. It survives every scenario but leaves almost all of the head-related
value on the table."""
import sys

ALPHA = 0.9


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); K = int(next(it)); C = int(next(it)); D = int(next(it))
    S0 = int(next(it)); Rmax = int(next(it))
    next(it); next(it)
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
        want = lo[t] + ALPHA * (hi[t] - lo[t])
        floor_ = max(Rel[t - 1], lo[t])
        ceil_ = min(Rel[t - 1] + Rmax, hi[t])
        Rel[t] = min(max(want, floor_), ceil_)

    r = [int(round(Rel[t] - Rel[t - 1])) for t in range(1, T + 1)]
    print(" ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
