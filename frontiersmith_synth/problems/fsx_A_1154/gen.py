#!/usr/bin/env python3
"""
gen.py <testId> -- Spectrum-Preserving Sparsifier Duel (fsx_A_1154)

Produces ONE test instance for the "pocket-size map that answers like the atlas" family.

Graph layout (fully deterministic, seeded ONLY from testId):
  - a long DECOY path of `decoy_len` edges (nodes 1..decoy_len+1), every decoy edge is a
    genuine bridge (removing it disconnects the path).
  - BLOCKS pendant clusters, each of BSIZE nodes, each cluster attached to the decoy path by
    exactly ONE boundary edge (also a bridge).
  - inside each cluster: a Hamiltonian path (guarantees connectivity) plus extra random chords
    (extra parallel routes -> LOW effective resistance per edge, unlike the decoy bridges).

The published test-vector family (K = 2*BLOCKS vectors, printed in the instance) consists of,
per cluster: (a) a 0/1 indicator vector of that cluster (a "near-cut" probe -- its quadratic
form equals exactly the cluster's single boundary-edge weight) and (b) a "smooth" parity-split
vector (0/1 alternating by local node order) whose quadratic form sums the weight of every
intra-cluster edge crossing the parity split -- probing intra-cluster fidelity with a
UNIFORM per-edge contribution (no single edge dominates), so partial edge coverage yields
proportionally graduated credit instead of an all-or-nothing outcome.
Every test vector is IDENTICALLY ZERO on the decoy path, so decoy edges contribute NOTHING to
any published test vector's quadratic form -- the general "preserve all of R^n" theorem would
still rank decoy bridges at the top of any effective-resistance leverage score (bridges always
have leverage score exactly 1), while an approach that restricts itself to the published
family's actual spectral support ignores them entirely.

decoy_len ramps with testId; once it exceeds the edge quota S_QUOTA, textbook leverage-score
sampling burns its whole budget on the (irrelevant) decoy bridges.
"""
import sys
import random

BLOCKS = 2
BSIZE = 10
CHORD_PROB = 0.3
S_QUOTA = 19

# only 2 clusters -> the objective's max-over-K aggregation cannot be dominated by an
# unlucky "loser" cluster nearly as easily as with many small clusters. Every cluster has a
# guaranteed Hamiltonian path of BSIZE-1=9 edges plus 1 boundary edge, so the GUARANTEED
# minimum block-relevant edge count is BLOCKS*(1+9) = 20 > S_QUOTA=19 -- even with the
# luckiest possible (zero-chord) draw, no solution can ever fully reconstruct every cluster,
# so headroom never saturates. The leftover budget after bridges (S_QUOTA - BLOCKS) = 16
# exceeds a single cluster's worst-case all-bridge demand (BSIZE-1=9), so even a resistance
# ranking that lets one lucky/unlucky cluster dominate the top tier still leaves a genuine
# remainder for the other cluster. decoy_len ramp: testId 1..5 comfortable (well below
# quota), 6..10 severe trap (decoy alone meets/exceeds the whole quota).
_DECOY_LEN = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 50, 9: 80, 10: 120}


def decoy_len_for(test_id: int) -> int:
    if test_id in _DECOY_LEN:
        return _DECOY_LEN[test_id]
    # fallback ramp for any testId outside the curated ladder (kept deterministic)
    return 6 + 10 * (test_id - 1)


def main():
    test_id = int(sys.argv[1])
    rng = random.Random(1000003 * test_id + 7919)

    decoy_len = decoy_len_for(test_id)
    decoy_nodes = decoy_len + 1
    block_total = BLOCKS * BSIZE
    n = decoy_nodes + block_total

    edges = []  # (u, v, w) 1-indexed, printed in THIS order (decoy first, then boundary, then intra)

    # ---- decoy path: nodes 1..decoy_nodes ----
    for i in range(1, decoy_nodes):
        w = round(rng.uniform(0.05, 0.30), 3)
        edges.append((i, i + 1, w))

    # ---- block layout ----
    block_start = []
    cur = decoy_nodes + 1
    for _k in range(BLOCKS):
        block_start.append(cur)
        cur += BSIZE

    anchors = []
    for k in range(BLOCKS):
        if decoy_nodes > 1:
            a = 1 + (k * (decoy_nodes - 1)) // BLOCKS
        else:
            a = 1
        a = max(1, min(decoy_nodes, a))
        anchors.append(a)

    # ---- boundary edges (one bridge per block) ----
    for k in range(BLOCKS):
        w = round(rng.uniform(1.0, 3.0), 3)
        edges.append((anchors[k], block_start[k], w))

    # ---- intra-block edges: Hamiltonian path + random chords ----
    for k in range(BLOCKS):
        bs = block_start[k]
        for i in range(BSIZE - 1):
            w = round(rng.uniform(1.0, 3.0), 3)
            edges.append((bs + i, bs + i + 1, w))
        for i in range(BSIZE):
            for j in range(i + 2, BSIZE):
                if rng.random() < CHORD_PROB:
                    w = round(rng.uniform(1.0, 3.0), 3)
                    edges.append((bs + i, bs + j, w))

    m = len(edges)

    # ---- published test-vector family ----
    K = 2 * BLOCKS
    vecs = []
    for k in range(BLOCKS):
        x = [0.0] * n
        for i in range(BSIZE):
            x[block_start[k] - 1 + i] = 1.0
        vecs.append(x)
    for k in range(BLOCKS):
        x = [0.0] * n
        for i in range(BSIZE):
            x[block_start[k] - 1 + i] = float(i % 2)  # parity split: uniform per-edge weight
        vecs.append(x)

    out = [f"{n} {m} {S_QUOTA} {K}"]
    for (u, v, w) in edges:
        out.append(f"{u} {v} {w:.3f}")
    for x in vecs:
        out.append(" ".join(f"{v:.3f}" for v in x))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
