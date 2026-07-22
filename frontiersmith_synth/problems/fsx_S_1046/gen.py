#!/usr/bin/env python3
# Generator for "Bell-Sounding: Worst-Case Crack Probes" (fsx_S_1046, format C).
# `python3 gen.py <testId>` prints ONE instance to stdout. testId 1..10 is a fixed
# size/difficulty ladder; ALL randomness is seeded from testId only (deterministic).
import sys, random

LADDER = [
    dict(b=6,  L=3, m=4,  q=2, I_max=4),
    dict(b=6,  L=3, m=5,  q=3, I_max=4),
    dict(b=8,  L=3, m=5,  q=3, I_max=5),
    dict(b=8,  L=4, m=6,  q=3, I_max=5),
    dict(b=8,  L=4, m=6,  q=4, I_max=6),
    dict(b=8,  L=4, m=7,  q=4, I_max=6),
    dict(b=10, L=4, m=8,  q=4, I_max=6),
    dict(b=10, L=4, m=8,  q=5, I_max=7),
    dict(b=10, L=5, m=9,  q=5, I_max=7),
    dict(b=12, L=5, m=10, q=5, I_max=8),
]


def gen(testId):
    cfg = LADDER[testId - 1]
    b, L, m, q, I_max = cfg['b'], cfg['L'], cfg['m'], cfg['q'], cfg['I_max']
    rng = random.Random(1000 + testId)
    interior_layers = list(range(1, L - 1))          # layers 1..L-2 (strictly interior)
    all_slots = [(p, l) for l in interior_layers for p in range(b)]
    rng.shuffle(all_slots)
    defects = all_slots[:m]
    alpha = round(rng.uniform(0.15, 0.45), 3)
    g_r = g_c = g_core = 1.0

    out = []
    out.append(f"{b} {L} {m} {q} {I_max}")
    out.append(f"{g_r} {g_c} {g_core}")
    out.append(f"{alpha}")
    for pos, layer in defects:
        out.append(f"{pos} {layer}")
    print("\n".join(out))


if __name__ == "__main__":
    testId = int(sys.argv[1])
    gen(testId)
