import sys, random

# Difficulty ladder: (B backbone nodes, H hub clusters, F filler nodes, K sensors == D decoys)
# testId 1..10, small -> large. D (decoy count) is set equal to K by construction, and
# B (backbone count) is also set equal to K: a naive "cover the backbone only" plan uses
# exactly the whole budget on the weak backbone tier, which is the checker's internal baseline.
LADDER = [
    (5, 1, 2, 5),
    (6, 1, 3, 6),
    (7, 2, 3, 7),
    (8, 2, 4, 8),
    (9, 2, 4, 9),
    (10, 3, 5, 10),
    (11, 3, 5, 11),
    (12, 4, 6, 12),
    (13, 4, 6, 13),
    (14, 5, 7, 14),
]

R = 3            # reach threshold radius (fixed across the instance)
BIG = R + 5       # isolating edge weight (>> R so far-side clusters are unreachable)

HUB_W, HUB_TAU, HUB_BETA = 4, 3, 16  # critical hub: modest solo weight, big double-cover bonus
DECOY_W, DECOY_TAU = 5, 1            # decoy: attractive, fast-saturating solo value
FILLER_W, FILLER_TAU = 1, 2          # filler: weak solo value (third tier)


def main():
    t = int(sys.argv[1])
    idx = min(max(t, 1), len(LADDER)) - 1
    B, H, F, K = LADDER[idx]
    rng = random.Random(1000 + t)

    N = B + 3 * H + K + F     # backbone + hub-triples + decoys(=K) + fillers
    D = K

    # node id ranges
    hub_base = B
    decoy_base = hub_base + 3 * H
    filler_base = decoy_base + D

    w = [0] * N
    crit = [0] * N
    tau = [1] * N
    beta = [0] * N

    # backbone: mild positive importance, mild saturation scale (deterministic small variation)
    for v in range(B):
        w[v] = 1 + (rng.randint(0, 1))
        tau[v] = 2
        crit[v] = 0
        beta[v] = 0

    # hub clusters: hub node is critical with a double-cover bonus; twins are plain helper sites
    for h in range(H):
        hub = hub_base + 3 * h
        twinA = hub + 1
        twinB = hub + 2
        w[hub], tau[hub], crit[hub], beta[hub] = HUB_W, HUB_TAU, 1, HUB_BETA
        w[twinA] = w[twinB] = 0
        tau[twinA] = tau[twinB] = 1
        crit[twinA] = crit[twinB] = 0
        beta[twinA] = beta[twinB] = 0

    # decoys: attractive solo value, saturate fast with a single sensor
    for d in range(D):
        v = decoy_base + d
        w[v], tau[v] = DECOY_W, DECOY_TAU
        crit[v] = 0
        beta[v] = 0

    # fillers: weak solo value, third tier
    for f in range(F):
        v = filler_base + f
        w[v], tau[v] = FILLER_W, FILLER_TAU
        crit[v] = 0
        beta[v] = 0

    edges = []  # (u, v, weight)

    # backbone: random tree over [0,B) with small weights (1 or 2)
    for i in range(1, B):
        j = rng.randint(0, i - 1)
        c = rng.randint(1, 2)
        edges.append((i, j, c))
    if B == 0:
        pass

    # hub cluster edges + isolating attach edge to the backbone
    for h in range(H):
        hub = hub_base + 3 * h
        twinA = hub + 1
        twinB = hub + 2
        edges.append((hub, twinA, 1))
        edges.append((hub, twinB, 1))
        anchor = (h * 5) % max(B, 1)
        edges.append((hub, anchor, BIG))

    # decoy isolating attach edges
    for d in range(D):
        v = decoy_base + d
        anchor = (d * 3 + 1) % max(B, 1)
        edges.append((v, anchor, BIG))

    # filler isolating attach edges
    for f in range(F):
        v = filler_base + f
        anchor = (f * 7 + 2) % max(B, 1)
        edges.append((v, anchor, BIG))

    M = len(edges)
    out = []
    out.append("%d %d %d %d" % (N, M, K, R))
    for v in range(N):
        out.append("%d %d %d %d" % (w[v], crit[v], tau[v], beta[v]))
    for (u, v, c) in edges:
        out.append("%d %d %d" % (u, v, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
