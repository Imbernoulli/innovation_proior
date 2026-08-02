#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE membrane-design instance to stdout.

Family: membrane-selectivity-design. Two molecules -- a TARGET (T) that must
permeate and a TWIN competitor (C) that must be excluded -- are nearly the
same kinetic size but differ in chemical affinity. The designer picks a
pore-size DISTRIBUTION (mechanism 1) and a chemical-functionalization loading
alpha (mechanism 2, the solubility channel). Both channels ultimately obey a
smooth permeability-vs-selectivity trade-off curve (mechanism 3): pushing
either channel too far starves the required throughput of the target.

Difficulty ladder (testId 1..10): the pore-family budget K_max grows
2 -> 6 with testId. Four of the ten cases (3, 6, 9, 10) are TRAP cases: the
target and twin diameters are set almost identical (<=3% apart), so NO choice
of pore radius can size-sieve them apart, and the required throughput is set
just above what an alpha=0 (chemistry-free) design can ever deliver at any
radius -- the size-only "shrink the pore" playbook is provably capped there.
All randomness is seeded by testId only (bit-for-bit deterministic).
"""
import sys
import random
import math


def sigmoid_D(lam, beta):
    """Smooth steric-passage fraction: 1 near lam=0 (wide open), ~0 as
    lam -> large (solute much bigger than the effective pore)."""
    x = beta * (lam - 1.0)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


BETA = 4.0
TRAP_IDS = {3, 6, 9, 10}


def main():
    tid = int(sys.argv[1])
    rng = random.Random(900001 + 104729 * tid)

    d_T = round(0.40 * rng.uniform(0.9, 1.1), 6)

    is_trap = tid in TRAP_IDS
    if is_trap:
        eps = rng.uniform(0.01, 0.03)
    else:
        eps = rng.uniform(0.14, 0.42)
    d_C = round(d_T * (1.0 + eps), 6)

    chi_T = round(rng.uniform(0.25, 0.40), 4)
    chi_C = round(rng.uniform(-0.40, -0.25), 4)

    base_sol_T = round(rng.uniform(0.85, 1.15), 4)
    base_sol_C = round(rng.uniform(0.85, 1.15), 4)

    K_max = min(6, 2 + (tid - 1) // 2)

    r_min = round(0.30 * d_T, 6)
    r_max = round(d_T * (2.6 + 0.4 * rng.random()), 6)

    alpha_max = 1.0
    delta_coat = round(0.05 * d_T, 6)

    # supremum of P_T reachable with alpha=0 (chemistry off), any radius in
    # [r_min, r_max]: D_size is monotone increasing as r grows, so the sup is
    # attained at r_max.
    lam_T_rmax = d_T / (2.0 * r_max)
    D_T_rmax = sigmoid_D(lam_T_rmax, BETA)
    P_T_alpha0_sup = D_T_rmax * base_sol_T

    if is_trap:
        P_min = round(1.05 * P_T_alpha0_sup, 6)
    else:
        P_min = round(rng.uniform(0.35, 0.55) * P_T_alpha0_sup, 6)

    toks = [d_T, d_C, chi_T, chi_C, base_sol_T, base_sol_C,
            K_max, r_min, r_max, alpha_max, delta_coat, P_min]
    sys.stdout.write(' '.join(str(x) for x in toks) + '\n')


if __name__ == "__main__":
    main()
