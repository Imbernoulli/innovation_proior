# TIER: trivial
"""
Non-adaptive 2-look reference: one interim look at N_max/2 that spends only
5% of the alpha budget (so it barely dents the final look's power), and the
final look at N_max spends the remaining 95%. Both boundaries are exact
closed-form normal quantiles (Bonferroni-safe: 5%+95%=100% <= alpha_total by
the union bound, regardless of the correlation between looks), so no
Monte-Carlo calibration is needed at all. No onset-awareness, no futility.
This is exactly the construction the checker uses as its own baseline B, so
this solution reproduces Ratio ~= 0.1 by design.
"""
import sys
import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    toks = sys.stdin.read().split()
    N_max = int(toks[0]); K_max = int(toks[1])
    alpha_total = float(toks[2])

    n1 = max(1, round(N_max / 2))
    n2 = N_max
    a1, a2 = 0.05 * alpha_total, 0.95 * alpha_total
    z1, z2 = norm_ppf(1 - a1), norm_ppf(1 - a2)

    out = [str(2), str(n1), "%.6f" % z1, "-50.000000",
           str(n2), "%.6f" % z2, "-50.000000"]
    print(" ".join(out))


if __name__ == "__main__":
    main()
