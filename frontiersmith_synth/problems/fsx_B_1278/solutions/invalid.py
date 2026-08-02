# TIER: invalid
"""
The classic trap named in the brief: "just test at the usual significance
threshold at every interim look" -- i.e. repeated testing with NO alpha
adjustment at all. z_eff_i = Phi^{-1}(1 - alpha_total) at every one of the
K_max looks, equally spaced, no futility. This has the best *nominal* power
of any plan here, but peeking K_max times at the uncorrected threshold
inflates the true family-wise type-I error far above alpha_total (well past
the checker's Monte-Carlo cap), so it must score 0 on every instance.
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

    K = K_max
    n_list = [max(1, round(i * N_max / K)) for i in range(1, K + 1)]
    for i in range(1, len(n_list)):
        if n_list[i] <= n_list[i - 1]:
            n_list[i] = n_list[i - 1] + 1
    n_list[-1] = N_max
    z = norm_ppf(1 - alpha_total)

    parts = [str(K)]
    for n_i in n_list:
        parts += [str(n_i), "%.6f" % z, "-50.000000"]
    print(" ".join(parts))


if __name__ == "__main__":
    main()
