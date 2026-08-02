#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE substituent-property-tune instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large/adversarial.

A ring scaffold has N substitution positions (cyclic adjacency i <-> i+1 mod N).
A library of K candidate substituents is given, each with an electronic term,
a steric-bulk term, and a synthesis cost (steps). The true property surrogate
is additive EXCEPT for an adjacency correction: two substituents that are both
"bulky" (steric bulk above a threshold) and sit on ring-adjacent positions
clash, and the clash penalty coefficient exceeds the per-group steric bonus,
so the combined pair's net steric contribution flips from positive to
negative. A naive additive-model optimizer (rank substituents by a per-item
value density and pack the best one everywhere it fits) cannot see this and
happily piles the single best-density substituent onto consecutive ring
positions -- exactly where the clash bites hardest.

Plants (>=3, in fact 6 of 10) "star substituent" trap cases: one library
entry is engineered to have both the best naive value-density AND a steric
bulk safely above the clash threshold, so the density-greedy heuristic packs
several adjacent copies of it and detonates the clash correction.

Also computes, internally, the TRUE optimum (via the same adjacency-aware
ring DP shipped in solutions/strong.py) so the target property can be placed
just out of reach -- the objective normalizes closeness-to-target, so this
keeps the ceiling open (no submission can ever hit distance 0 by construction
the checker can verify from the input alone).
"""
import sys
import random

NEG = float('-inf')


def solve_ring(N, K, budget, lib, alpha, beta, s_thresh):
    """Exact adjacency-aware optimum of S = sum(e+alpha*s over occupied
    positions) - sum(beta*(s_i+s_j) over ring-adjacent bulky-bulky pairs),
    subject to sum(cost) <= budget. Cyclic DP: fix the first position's
    choice, DP forward with state = (budget_used, previous choice), then
    close the ring against the fixed first choice."""
    bulky = [s > s_thresh for (e, s, c) in lib]

    def item_value(t):
        e, s, c = lib[t]
        return e + alpha * s

    def edge_penalty(t1, t2):
        if t1 < 0 or t2 < 0:
            return 0.0
        if bulky[t1] and bulky[t2]:
            return beta * (lib[t1][1] + lib[t2][1])
        return 0.0

    choices = list(range(-1, K))
    best_overall = NEG

    for first in choices:
        first_cost = 0 if first < 0 else lib[first][2]
        if first_cost > budget:
            continue
        layer = [[NEG] * (K + 1) for _ in range(budget + 1)]
        layer[first_cost][first + 1] = 0.0 if first < 0 else item_value(first)

        for i in range(1, N):
            ndp = [[NEG] * (K + 1) for _ in range(budget + 1)]
            for b in range(budget + 1):
                row = layer[b]
                for pidx in range(K + 1):
                    val = row[pidx]
                    if val == NEG:
                        continue
                    prev = pidx - 1
                    for t in choices:
                        cost_t = 0 if t < 0 else lib[t][2]
                        nb = b + cost_t
                        if nb > budget:
                            continue
                        add_val = 0.0 if t < 0 else item_value(t)
                        nv = val + add_val - edge_penalty(prev, t)
                        tidx = t + 1
                        if nv > ndp[nb][tidx]:
                            ndp[nb][tidx] = nv
            layer = ndp

        for b in range(budget + 1):
            row = layer[b]
            for pidx in range(K + 1):
                val = row[pidx]
                if val == NEG:
                    continue
                prev = pidx - 1
                total = val - edge_penalty(prev, first)
                if total > best_overall:
                    best_overall = total
    return best_overall


def build(tid):
    rng = random.Random(20000 + 97 * tid)
    sizes = {1: (6, 4), 2: (8, 5), 3: (8, 5), 4: (10, 6), 5: (10, 6),
             6: (12, 6), 7: (12, 7), 8: (14, 7), 9: (14, 8), 10: (16, 8)}
    trap_ids = {3, 5, 7, 9, 10}
    N, K = sizes[tid]
    is_trap = tid in trap_ids

    P0 = round(rng.uniform(30.0, 50.0), 2)
    alpha = round(rng.uniform(0.4, 0.7), 3)
    beta = round(rng.uniform(1.3, 2.0), 3)
    s_thresh = round(rng.uniform(4.5, 5.5), 2)

    lib = []
    for _ in range(K):
        e = round(rng.uniform(-3.5, 3.5), 2)
        s = round(rng.uniform(0.5, 4.5), 2)
        c = rng.randint(1, 4)
        lib.append([e, s, c])

    if is_trap:
        star_e = round(rng.uniform(3.0, 4.5), 2)
        star_s = round(rng.uniform(s_thresh + 1.5, s_thresh + 4.0), 2)
        star_c = rng.randint(2, 3)
        lib[0] = [star_e, star_s, star_c]
        # a second, milder decoy substituent close behind in density but
        # NOT bulky, so an adjacency-aware search still has a good escape
        decoy_e = round(rng.uniform(1.0, 2.2), 2)
        decoy_s = round(rng.uniform(0.5, 3.0), 2)
        decoy_c = rng.randint(1, 3)
        if K > 1:
            lib[1] = [decoy_e, decoy_s, decoy_c]

    avg_cost = sum(c for (e, s, c) in lib) / K
    budget = max(4, round(N * avg_cost * rng.uniform(0.5, 0.7)))

    lib_t = [tuple(x) for x in lib]
    s_max = solve_ring(N, K, budget, lib_t, alpha, beta, s_thresh)
    if s_max <= 0.5:
        s_max = 2.0
    D = s_max * rng.uniform(1.08, 1.18)
    window = round(0.5 * D, 4)
    target = round(P0 + D, 4)

    lines = [f"{N} {K} {budget}",
             f"{P0} {alpha} {beta} {s_thresh}",
             f"{target} {window}"]
    for (e, s, c) in lib:
        lines.append(f"{e} {s} {c}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
