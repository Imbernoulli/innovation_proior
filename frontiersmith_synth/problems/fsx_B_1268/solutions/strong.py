# TIER: strong
# Insight: reprice against the POST-selection pool, not today's pool. Tiers are
# independent under the cap, so for each tier simulate, at every feasible integer
# price in the rate-change-cap band, exactly which risk buckets survive (linear
# elasticity ramp keyed to each bucket's own departure thresholds) and score the
# REALIZED profit of the pool that remains -- then take the tier's true argmax.
# This is an exact per-tier search over a reformulated (post-selection) objective,
# not "greedy plus more iterations": greedy never evaluates departures at all.
import sys


def depart_frac(gap, tlo, thi):
    if gap <= tlo:
        return 0.0
    if gap >= thi:
        return 1.0
    if thi <= tlo:
        return 1.0
    return (gap - tlo) / float(thi - tlo)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it))
    prices = []
    for _ in range(K):
        p0 = int(next(it)); c = int(next(it)); cap = int(next(it)); B = int(next(it))
        buckets = []
        for _ in range(B):
            n = int(next(it)); m = int(next(it))
            vs = []; prs = []
            for _ in range(m):
                vs.append(int(next(it))); prs.append(int(next(it)))
            tlo = int(next(it)); thi = int(next(it))
            eloss = sum(v * p for v, p in zip(vs, prs)) / 1000.0
            buckets.append((n, eloss, tlo, thi))

        delta = (p0 * cap) // 100
        lo, hi = max(0, p0 - delta), p0 + delta

        best_p, best_profit = p0, None
        for p in range(lo, hi + 1):
            gap = p - c
            claims = 0.0
            remain_total = 0.0
            for n, eloss, tlo, thi in buckets:
                frac = depart_frac(gap, tlo, thi)
                remaining = n * (1.0 - frac)
                claims += remaining * eloss
                remain_total += remaining
            profit = p * remain_total - claims
            if best_profit is None or profit > best_profit:
                best_profit = profit
                best_p = p
        prices.append(best_p)
    sys.stdout.write("\n".join(str(p) for p in prices) + "\n")


if __name__ == "__main__":
    main()
