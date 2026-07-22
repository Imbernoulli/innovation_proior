#!/usr/bin/env python3
"""gen.py <testId>  ->  one instance of the crosscountry-jacobian-order problem.

We plant an HOURGLASS-chained computational DAG: wide input layer -> (wide cloud ->
narrow waist) repeated -> wide cloud -> output layer.  All paths between two wide
clouds must funnel through a NARROW WAIST (a genuine graph cut).  Edges carry local
partial derivatives (elements of GF(P)).  Vertices are numbered layer-by-layer so
that ascending id == a topological order.

Difficulty grows with testId: more hourglass segments + sharper width contrast, so
that pure forward / reverse / mode-selection land ever farther from a cut-aware
cross-country elimination order.

Determinism: everything is seeded from testId only.
"""
import sys, random

P = 1_000_000_007


def build(testId):
    rng = random.Random(1000 + 7919 * testId)

    # width contrast + segment count both grow with testId (1..10)
    t = testId
    m = 14 + t                     # inputs  (wide)
    n = 5 + (t % 3)                # outputs (narrow-er than inputs so reverse != forward)
    W = 10                         # interior cloud width
    k = 2 if t % 2 else 3          # waist width (narrow cut)
    S = 2 + (t + 1) // 3           # number of hourglass segments (waists)
    fan = 4                        # sparse fan-in between layers

    # layer widths:  m , (W , k) * S , W , n
    widths = [m]
    for _ in range(S):
        widths += [W, k]
    widths += [W, n]

    # assign vertex ids layer by layer (ascending id == topological order)
    layers = []
    nxt = 0
    for w in widths:
        layers.append(list(range(nxt, nxt + w)))
        nxt += w
    V = nxt
    M = m
    N = n

    edges = []  # (u, v, val)

    def connect(prev, cur, fanin):
        # each cur vertex draws up to fanin distinct parents from prev
        for c in cur:
            k2 = min(fanin, len(prev))
            for p in rng.sample(prev, k2):
                edges.append((p, c, rng.randint(1, P - 1)))
        # coverage: every prev vertex must have >=1 child (keeps the DAG connected
        # and forces all signal through the waist)
        haschild = set(u for (u, _, _) in edges)
        for p in prev:
            if p not in haschild:
                c = rng.choice(cur)
                edges.append((p, c, rng.randint(1, P - 1)))

    for li in range(len(layers) - 1):
        prev, cur = layers[li], layers[li + 1]
        wprev, wcur = len(prev), len(cur)
        # funnel densely into / out of a narrow waist; sparse between wide layers
        if wcur <= 4 or wprev <= 4:
            fanin = max(fan, min(wprev, 8))   # dense into/out of waist
        else:
            fanin = fan
        connect(prev, cur, fanin)

    # dedup parallel edges (sum values); keep it a simple DAG
    agg = {}
    for u, v, val in edges:
        agg[(u, v)] = (agg.get((u, v), 0) + val) % P
        if agg[(u, v)] == 0:
            agg[(u, v)] = 1
    edges = [(u, v, w) for (u, v), w in agg.items()]
    edges.sort()

    return V, M, N, edges


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    V, M, N, edges = build(testId)
    out = [f"{V} {len(edges)} {M} {N}"]
    for u, v, w in edges:
        out.append(f"{u} {v} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
