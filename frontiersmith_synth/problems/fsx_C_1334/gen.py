#!/usr/bin/env python3
"""gen.py <testId> -> prints one reaction-network-pruning instance to stdout.

Deterministic: seeded ONLY from testId. Ladder testId=1..10, small -> large.
Trap testIds ({3,6,9}) plant a slow "bridge" reaction that is the SOLE path from a
precursor species to the target for a subset of the held-out conditions, while the
first (index-0) condition never needs it -- a rate-constant-only pruning heuristic
that validates against that one condition will remove it and then fail the other
held-out conditions during grading.
"""
import sys
import random

TRAP_IDS = {3, 6, 9}
T_HORIZON = 15.0
N_STEPS = 80
EPS = 0.05
P_CONDITIONS = 5


def build(test_id):
    rng = random.Random(1000003 * test_id + 7)
    trap = test_id in TRAP_IDS

    M = 3 + (test_id - 1) // 3     # main-chain reaction count (grows with testId)
    Df = 3 + (test_id - 1) // 2    # fast-decoy count
    Ds = (M + Df) - (1 if trap else 0)   # slow-decoy count: keeps m even and keeps
                                          # {slow decoys (+bridge)} == exactly the bottom
                                          # half of reactions by rate constant

    target = 0
    main_species = list(range(1, 1 + M))
    nxt = 1 + M
    p_alt = None
    if trap:
        p_alt = nxt
        nxt += 1
    decoy_species = []
    for _ in range(Ds + Df):
        a, b = nxt, nxt + 1
        nxt += 2
        decoy_species.append((a, b))
    n = nxt

    reactions = []  # (r, p, rate)
    for i in range(M):
        r = main_species[i]
        p = main_species[i + 1] if i + 1 < M else target
        rate = round(rng.uniform(3.0, 9.0), 6)
        reactions.append((r, p, rate))
    if trap:
        rate = round(rng.uniform(0.05, 0.15), 6)
        reactions.append((p_alt, target, rate))
    for i in range(Ds):
        a, b = decoy_species[i]
        rate = round(rng.uniform(0.05, 0.9), 6)
        reactions.append((a, b, rate))
    for i in range(Ds, Ds + Df):
        a, b = decoy_species[i]
        rate = round(rng.uniform(3.0, 9.0), 6)
        reactions.append((a, b, rate))
    m = len(reactions)

    # obscure positional shortcuts: permute reaction order (rate-based sorting used by
    # any solver is unaffected, since it re-sorts by the rate VALUE, not position).
    order = list(range(m))
    rng.shuffle(order)
    reactions = [reactions[i] for i in order]

    conditions = []
    bridge_dep = set()
    if trap:
        others = list(range(1, P_CONDITIONS))
        rng.shuffle(others)
        bridge_dep = set(others[:2])
    for p_idx in range(P_CONDITIONS):
        c = [0.0] * n
        c[main_species[0]] = round(rng.uniform(0.8, 2.0), 6)
        if trap and p_idx in bridge_dep:
            c[p_alt] = round(rng.uniform(1.2, 2.5), 6)
        for (a, b) in decoy_species:
            c[a] = round(rng.uniform(0.1, 1.0), 6)
        conditions.append(c)

    return n, m, target, P_CONDITIONS, T_HORIZON, N_STEPS, EPS, reactions, conditions


def main():
    test_id = int(sys.argv[1])
    n, m, target, P, T, N, eps, reactions, conditions = build(test_id)
    out = []
    out.append(f"{n} {m} {target} {P} {T:.6f} {N} {eps:.6f}")
    for (r, p, rate) in reactions:
        out.append(f"{r} {p} {rate:.6f}")
    for c in conditions:
        out.append(" ".join(f"{x:.6f}" for x in c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
