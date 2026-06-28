#!/usr/bin/env python3
"""Instance generator for "Knapsack with Synergies" (Quadratic Knapsack, ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n W
    w_0 v_0
    w_1 v_1
    ...
    w_{n-1} v_{n-1}
    p
    i_0 j_0 b_0
    i_1 j_1 b_1
    ...
    i_{p-1} j_{p-1} b_{p-1}

where
  * n      = number of items (deterministic from seed, in [400, 900]),
  * W      = the weight budget (a fraction of total weight),
  * w_i    = positive integer weight of item i,
  * v_i    = non-negative integer linear value of item i,
  * p      = number of synergy pairs,
  * i j b  = an UNORDERED pair {i, j} (0 <= i < j < n) with a positive integer
             synergy bonus b earned only if BOTH i and j are selected.

This is the Quadratic Knapsack Problem (QKP): maximize the sum of selected
linear values plus the sum of synergy bonuses over pairs whose both endpoints
are selected, subject to the total selected weight being at most W.

To make the *quadratic* (synergy) term the decisive lever -- the whole point of
the problem -- the synergy graph is built with COMMUNITY STRUCTURE: items are
partitioned into a few latent "clusters", and synergy pairs are drawn far more
often WITHIN a cluster than across clusters, with larger within-cluster bonuses.
A value/weight-ratio greedy that ignores synergies leaves most of that bonus on
the table, so a synergy-aware search can clearly beat it.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x9D2C_5680 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # ---- size ----
    n = rng.randint(400, 900)

    # ---- latent clusters that drive the synergy community structure ----
    num_clusters = rng.randint(4, 10)
    cluster_of = [rng.randrange(num_clusters) for _ in range(n)]
    members = [[] for _ in range(num_clusters)]
    for i in range(n):
        members[cluster_of[i]].append(i)

    # ---- item weights and (modest) linear values ----
    # Weights in [1, 100]. Linear values are deliberately MODEST relative to the
    # synergy bonuses so the quadratic term dominates the decision.
    w = [rng.randint(1, 100) for _ in range(n)]
    v = [rng.randint(0, 40) for _ in range(n)]

    total_w = sum(w)
    # budget: a tight fraction of total weight so selection really matters.
    frac = rng.uniform(0.18, 0.32)
    W = max(1, int(round(frac * total_w)))

    # ---- synergy pairs with community structure ----
    # Target density: average synergy degree per item ~ d_avg.
    d_avg = rng.uniform(6.0, 14.0)
    target_pairs = int(round(d_avg * n / 2.0))

    seen = set()
    pairs = []
    p_within = 0.82          # fraction of pairs drawn within a cluster
    attempts = 0
    max_attempts = 40 * target_pairs + 1000
    while len(pairs) < target_pairs and attempts < max_attempts:
        attempts += 1
        within = (rng.random() < p_within)
        if within:
            c = rng.randrange(num_clusters)
            if len(members[c]) < 2:
                continue
            a = rng.choice(members[c])
            b = rng.choice(members[c])
            if a == b:
                continue
            bonus = rng.randint(10, 80)   # large within-cluster synergy
        else:
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b or cluster_of[a] == cluster_of[b]:
                continue
            bonus = rng.randint(1, 15)    # small cross-cluster synergy
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b, bonus))

    out = [f"{n} {W}"]
    out.extend(f"{w[i]} {v[i]}" for i in range(n))
    out.append(str(len(pairs)))
    out.extend(f"{a} {b} {bonus}" for (a, b, bonus) in pairs)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
