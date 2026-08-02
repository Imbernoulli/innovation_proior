# TIER: greedy
"""The obvious first instinct: 'smaller pores separate similarly-sized
molecules better, so shrink the pore as far as the throughput requirement
allows.' A single pore radius (K=1), alpha always 0 -- the solubility /
chemistry channel never even considered. Binary-search the SMALLEST feasible
radius that still clears the target throughput P_min (a naive, single-lever
size-sieving recipe).

This wins decently when the target and twin sizes are genuinely different
(shrinking the pore does buy real size-selectivity there). But on the trap
cases -- where the twin is almost the SAME size as the target -- no radius
gives any real separation (both molecules see nearly the same steric
resistance at every r), and P_min itself is set above what any alpha=0
design can reach at any radius, so this recipe cannot even find a
radius that clears throughput and just falls back to the widest pore --
i.e. it reproduces the baseline there."""
import sys
import math

BETA = 4.0


def sigmoid_D(lam):
    x = BETA * (lam - 1.0)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def main():
    toks = sys.stdin.read().split()
    (d_T, d_C, chi_T, chi_C, base_sol_T, base_sol_C,
     K_max, r_min, r_max, alpha_max, delta_coat, P_min) = [float(x) for x in toks[:12]]

    def P_T_at(r):
        lam_T = d_T / (2.0 * r)
        return sigmoid_D(lam_T) * base_sol_T

    if P_T_at(r_max) < P_min:
        # cannot meet throughput anywhere in the size-only search space;
        # best effort is to maximize permeability (widest pore)
        r = r_max
    else:
        lo, hi = r_min, r_max
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if P_T_at(mid) >= P_min:
                hi = mid
            else:
                lo = mid
        r = hi

    print(1)
    print(0.0)
    print("%.6f %.6f" % (r, 1.0))


if __name__ == "__main__":
    main()
