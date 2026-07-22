#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE 'Trunk Pricing' instance to stdout.

Each commodity k gets a fixed 3-node DAG (source=0, hub=1, sink=2):
  edge0: (0,1) TRUNK   -- shared=1, capped, cheap per unit, draws the global backbone
  edge1: (1,2) toll    -- shared=0, completes the trunk route, generous cap
  edge2: (0,2) BYPASS  -- shared=0, generous cap (always >= d_k), pricier per unit
  edge3: (0,2) decoy   -- (larger tests only) shared=0, strictly dominated by bypass

Every commodity individually prefers its trunk edge (trunk route always net
cheaper than bypass by construction). The ONLY thing that ever stops a
commodity from using its trunk edge fully is the single GLOBAL backbone budget
C shared across all commodities -- that is the sole coupling constraint.

For "adversarial" test cases the commodities are emitted in ASCENDING order of
their marginal value (cost saved per backbone unit if routed via trunk instead
of bypass): the first commodities in the input benefit LEAST from the trunk,
the last commodities benefit MOST. A naive "patch commodities in input order
until the budget runs out" repair therefore burns the scarce backbone on the
least valuable commodities first and starves the most valuable ones -- exactly
backwards from the optimum. Seeded ONLY by testId -> fully deterministic.
"""
import sys
import random


def build_case(testid):
    rnd = random.Random(1000003 * testid + 7)

    k_by_test = {1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: 7, 7: 7, 8: 8, 9: 8, 10: 9}
    K = k_by_test.get(testid, 5)
    adversarial = True
    extra_edge = testid >= 7
    alpha = 0.40 + 0.02 * testid  # 0.42 .. 0.60, tight enough to bind hard

    commodities = []
    for _ in range(K):
        d = rnd.randint(4, 10) + (2 if testid >= 8 else 0)
        w = rnd.randint(1, 6)
        cap_trunk = max(1, min(d, round(d * rnd.uniform(0.65, 0.99))))
        cost_trunk = rnd.randint(1, 2)
        toll = 0
        margin = rnd.randint(20, 60)
        trunk_total = cost_trunk + toll
        cost_bypass = trunk_total + margin
        cap_bypass = d + rnd.randint(1, 3)

        edges = [
            (0, 1, cap_trunk, cost_trunk, 1, w),
            (1, 2, d, toll, 0, 0),
            (0, 2, cap_bypass, cost_bypass, 0, 0),
        ]
        if extra_edge:
            deco_cap = max(1, d // 3)
            deco_cost = cost_bypass + rnd.randint(3, 8)
            edges.append((0, 2, deco_cap, deco_cost, 0, 0))

        mv = (cost_bypass - trunk_total) / w
        commodities.append({"n": 3, "s": 0, "t": 2, "d": d, "edges": edges, "mv": mv})

    if adversarial:
        commodities.sort(key=lambda c: c["mv"])
    else:
        rnd.shuffle(commodities)

    total_cap_weight = sum(
        sum(e[2] * e[5] for e in c["edges"] if e[4] == 1) for c in commodities
    )
    C = max(1, round(alpha * total_cap_weight))
    return commodities, C


def main():
    testid = int(sys.argv[1])
    commodities, C = build_case(testid)
    lines = [str(len(commodities))]
    for c in commodities:
        lines.append(f"{c['n']} {len(c['edges'])} {c['s']} {c['t']} {c['d']}")
        for (u, v, cap, cost, shared, weight) in c["edges"]:
            lines.append(f"{u} {v} {cap} {cost} {shared} {weight}")
    lines.append(str(C))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
