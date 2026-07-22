#!/usr/bin/env python3
"""gen.py <testId> -- prints one min-cut-redundancy-reliability instance to stdout.

Topology: a chain of k terminal "hubs" 1..k.  Between consecutive hubs i and i+1
sits a redundancy GROUP i built as:
    hub_i --(bridge edge)--> mid_i --(width_i parallel "fan" edges)--> hub_{i+1}
The bridge edge is a true single point of failure (no redundancy: it alone is a
min-cut of size 1).  The fan is a set of parallel edges between mid_i and
hub_{i+1} (a min-cut of size width_i -- ALL of them must fail simultaneously to
break that link, so it starts already fairly reliable once width_i >= 2).

Deterministic: every random draw is seeded purely from testId.
"""
import sys
import random

# testId -> (k terminals, [width_1 .. width_{k-1}])
WIDTH_TABLE = {
    1:  (2, [1]),
    2:  (2, [2]),
    3:  (2, [4]),
    4:  (3, [1, 3]),
    5:  (3, [3, 1]),
    6:  (3, [4, 4]),
    7:  (4, [2, 1, 3]),
    8:  (4, [4, 2, 4]),
    9:  (5, [3, 1, 4, 2]),
    10: (5, [4, 3, 1, 4]),
}

# fraction of the "max out everything" cost handed out as budget
FRAC_TABLE = {
    1: 0.45, 2: 0.45, 3: 0.40,
    4: 0.40, 5: 0.40, 6: 0.35,
    7: 0.35, 8: 0.32, 9: 0.30, 10: 0.28,
}


def main():
    tid = int(sys.argv[1])
    k, widths = WIDTH_TABLE[tid]
    frac = FRAC_TABLE[tid]
    rng = random.Random(20260711 + 977 * tid)

    terminals = list(range(1, k + 1))
    n = k + (k - 1)  # k terminals + (k-1) mid nodes

    edges = []  # (u, v, p0_per_mille, cost, maxu)
    for i in range(1, k):
        hub_a, mid, hub_b = i, k + i, i + 1
        # bridge edge: weak-ish base reliability, relatively pricey per level
        p0 = rng.randint(380, 480)
        cost = rng.randint(7, 9)
        edges.append((hub_a, mid, p0, cost, 3))
        # fan edges: moderate base reliability, cheap per level
        w = widths[i - 1]
        for _ in range(w):
            p0b = rng.randint(300, 400)
            costb = rng.randint(2, 3)
            edges.append((mid, hub_b, p0b, costb, 2))

    full_cost = sum(c * mu for (_, _, _, c, mu) in edges)
    budget = max(1, round(full_cost * frac))

    m = len(edges)
    lines = [f"{n} {m} {k} {budget}", " ".join(str(t) for t in terminals)]
    for (u, v, p0, c, mu) in edges:
        lines.append(f"{u} {v} {p0} {c} {mu}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
