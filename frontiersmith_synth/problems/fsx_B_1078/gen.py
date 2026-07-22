#!/usr/bin/env python3
"""
gen.py <testId> -- Foundry Line Batching instance generator.

Builds a DAG made of disjoint "blocks":
  - GROUP blocks: a linear precedence chain of length ~H (sometimes longer, to force
    trimming), moderate per-task profit, non-monotonic along the chain (so "take the
    first H" is wrong; you must pick the top-profit H while preserving order).
  - POISON blocks: singleton tasks (no precedence relation to anything), individually
    HIGH profit but mutually incomparable -- each one that gets scheduled permanently
    occupies an entire production line (since a line can hold only one task from a
    pairwise-incomparable set), wasting that line's remaining capacity. This is the
    trap: naive highest-profit-first packing grabs these first.
  - NOISE blocks: small chains (length 2-4) with random profit, for structural variety.

All block-local ids are then relabeled through a random permutation of 1..N so no
solution can exploit index order, and edges/profit lines follow the permuted ids.
Deterministic: everything is seeded from testId only.
"""
import random
import sys


def build_blocks(rnd, cfg):
    """Returns list of blocks; each block is a list of profits IN CHAIN ORDER
    (task i in the list must precede task i+1 in the list)."""
    blocks = []

    for _ in range(cfg["n_groups"]):
        L = cfg["group_len"] + rnd.choice(cfg["group_len_jitter"])
        L = max(2, L)
        lo, hi = cfg["group_profit"]
        profits = [rnd.randint(lo, hi) for _ in range(L)]
        # Bias profit toward the chain HEAD (descending), with the low-value TAIL
        # locally scrambled. This lets an append-only greedy that starts a line at
        # this chain's head actually grow it (so groups aren't a total loss even on
        # the free lines left over after the poison trap), while still forcing a
        # real "pick the top-H by profit, not the first H" decision whenever the
        # chain is longer than H (the scrambled tail is where that decision bites).
        profits.sort(reverse=True)
        tail = min(4, L)
        if tail >= 2:
            window = profits[L - tail:]
            rnd.shuffle(window)
            profits[L - tail:] = window
        blocks.append(profits)

    for _ in range(cfg["n_poison"]):
        lo, hi = cfg["poison_profit"]
        blocks.append([rnd.randint(lo, hi)])  # singleton chain

    for _ in range(cfg["n_noise"]):
        L = rnd.randint(2, 4)
        lo, hi = cfg["noise_profit"]
        profits = [rnd.randint(lo, hi) for _ in range(L)]
        profits.sort(reverse=True)  # same head-biased shape as groups
        blocks.append(profits)

    return blocks


# testId -> (k, H, block config). Sizes grow with testId (small -> large/adversarial).
# trap=True cases plant an oversized poison antichain against deep parallel groups
# (>=3 of these are required so the obvious highest-profit-first greedy lands far
# from the achievable width-aware optimum).
CONFIGS = {
    1: dict(k=3, H=5, n_groups=3, group_len=5, group_len_jitter=[0, 0, 1],
            group_profit=(20, 40), n_poison=1, poison_profit=(30, 45),
            n_noise=4, noise_profit=(5, 60), trap=False),
    2: dict(k=3, H=6, n_groups=3, group_len=6, group_len_jitter=[-1, 0, 1],
            group_profit=(20, 45), n_poison=3, poison_profit=(35, 55),
            n_noise=6, noise_profit=(5, 60), trap=False),
    3: dict(k=4, H=6, n_groups=4, group_len=6, group_len_jitter=[0, 0, 1],
            group_profit=(25, 40), n_poison=3, poison_profit=(70, 95),
            n_noise=5, noise_profit=(5, 40), trap=True),
    4: dict(k=4, H=7, n_groups=4, group_len=7, group_len_jitter=[0, 1, 2],
            group_profit=(25, 42), n_poison=2, poison_profit=(75, 100),
            n_noise=6, noise_profit=(5, 40), trap=True),
    5: dict(k=4, H=8, n_groups=4, group_len=8, group_len_jitter=[0, 0, 2],
            group_profit=(28, 45), n_poison=3, poison_profit=(80, 105),
            n_noise=8, noise_profit=(5, 45), trap=True),
    6: dict(k=5, H=8, n_groups=5, group_len=8, group_len_jitter=[0, 1, 3],
            group_profit=(28, 46), n_poison=3, poison_profit=(85, 110),
            n_noise=10, noise_profit=(5, 45), trap=True),
    7: dict(k=5, H=9, n_groups=5, group_len=9, group_len_jitter=[0, 0, 2],
            group_profit=(30, 48), n_poison=4, poison_profit=(90, 115),
            n_noise=10, noise_profit=(5, 50), trap=True),
    8: dict(k=5, H=9, n_groups=6, group_len=9, group_len_jitter=[-2, 0, 1],
            group_profit=(20, 60), n_poison=6, poison_profit=(40, 70),
            n_noise=16, noise_profit=(5, 65), trap=False),
    9: dict(k=6, H=10, n_groups=6, group_len=10, group_len_jitter=[0, 1, 3],
            group_profit=(30, 50), n_poison=5, poison_profit=(95, 120),
            n_noise=14, noise_profit=(5, 50), trap=True),
    10: dict(k=6, H=11, n_groups=7, group_len=11, group_len_jitter=[0, 2, 4],
             group_profit=(30, 52), n_poison=4, poison_profit=(100, 130),
             n_noise=18, noise_profit=(5, 55), trap=True),
}


def main():
    test_id = int(sys.argv[1])
    cfg = CONFIGS[test_id]
    rnd = random.Random(20260 + 17 * test_id)

    blocks = build_blocks(rnd, cfg)

    # Assemble construction-order profits and chain (cover) edges.
    profits_c = []
    edges_c = []
    for blk in blocks:
        base = len(profits_c)
        profits_c.extend(blk)
        for i in range(len(blk) - 1):
            edges_c.append((base + i, base + i + 1))

    N = len(profits_c)
    M = len(edges_c)
    k = cfg["k"]
    H = cfg["H"]

    # Random relabeling: construction-index -> final 1..N id.
    perm = list(range(N))
    rnd.shuffle(perm)
    profits = [0] * N
    for i in range(N):
        profits[perm[i]] = profits_c[i]
    edges = [(perm[u] + 1, perm[v] + 1) for (u, v) in edges_c]
    rnd.shuffle(edges)

    out = [f"{N} {M} {k} {H}"]
    out.append(" ".join(map(str, profits)))
    for u, v in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
