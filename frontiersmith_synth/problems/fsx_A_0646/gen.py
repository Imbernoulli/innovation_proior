#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE caravan-insurance instance to stdout.
Deterministic: all randomness is seeded from testId only.
"""
import random
import sys

PLAN = {
    1: dict(N=200, kind='easy'),
    2: dict(N=320, kind='easy'),
    3: dict(N=450, kind='mild'),
    4: dict(N=600, kind='mild'),
    5: dict(N=800, kind='medium'),
    6: dict(N=1000, kind='medium'),
    7: dict(N=1200, kind='hard'),
    8: dict(N=1600, kind='hard'),
    9: dict(N=2000, kind='hardest'),
    10: dict(N=2500, kind='hardest'),
}


def gen(test_id):
    rng = random.Random(1000 + test_id * 7919)
    plan = PLAN[test_id]
    n = plan['N']
    kind = plan['kind']
    agents = []

    if kind == 'easy':
        # single narrow-variance population: no cluster structure, generous margin.
        for _ in range(n):
            p = rng.uniform(0.27, 0.33)
            a = rng.uniform(0.003, 0.010)
            L = rng.randint(950, 1050)
            h = rng.uniform(0.35, 0.65)
            u = -(1.0 + h) * p * L
            agents.append((p, a, L, u))
        return agents

    sev = {'mild': 0.35, 'medium': 0.65, 'hard': 0.90, 'hardest': 1.0}[kind]
    f_low = 0.45 + 0.15 * sev
    f_tox = 0.05 + 0.20 * sev
    f_high = 1.0 - f_low - f_tox
    for _ in range(n):
        r = rng.random()
        if r < f_low:
            typ = 'L'
        elif r < f_low + f_high:
            typ = 'H'
        else:
            typ = 'T'
        if typ == 'L':
            # cheap-route caravans: low raid risk, risk-tolerant, a fair private-escort
            # alternative -- only a thin margin is ever available from this type.
            p = rng.uniform(0.02, 0.08)
            a = rng.uniform(0.0, 0.0008)
            L = rng.randint(400, 1200)
            g = rng.uniform(0.02, 0.20)
            u = -(1.0 + g) * p * L
        elif typ == 'H':
            # dangerous-route caravans: high raid risk, highly risk-averse, no good
            # alternative -- large genuine margin is available if priced correctly.
            p = rng.uniform(0.25, 0.45)
            a = rng.uniform(0.003, 0.012)
            L = rng.randint(800, 2000)
            h = rng.uniform(0.30, 0.90)
            u = -(1.0 + h) * p * L
        else:
            # toxic type: high raid risk BUT a private escort market that is often
            # as cheap as (or cheaper than) actuarially-fair insurance -- no price
            # profitably serves most of this type; they must be let walk.
            p = rng.uniform(0.30, 0.50)
            a = rng.uniform(0.003, 0.012)
            L = rng.randint(800, 2000)
            g = rng.uniform(-0.15, 0.05)
            u = -(1.0 + g) * p * L
        agents.append((p, a, L, u))
    rng.shuffle(agents)
    return agents


def main():
    test_id = int(sys.argv[1])
    agents = gen(test_id)
    out = [str(len(agents))]
    for (p, a, L, u) in agents:
        out.append("%.6f %.8f %d %.6f" % (p, a, L, u))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
