#!/usr/bin/env python3
"""gen.py <testId> -- prints one Row-Placement / Channel-Congestion / Timing instance.

Deterministic: all randomness is seeded from testId only.

Layout: n_cells standard cells must be placed into n_cells distinct row slots
0..n_cells-1 (a permutation). Between consecutive slots g and g+1 there is a
routing channel with a fixed integer capacity[g]. Each net (a set of >=2 cell
indices) is routed by the checker as the *contiguous slot interval* spanned by
its terminals' assigned slots (a bounding-interval route -- the standard HPWL
model) and it consumes one unit of capacity on every channel strictly inside
that interval. Some nets are flagged timing-critical and carry an explicit
slack bound on their span.

testId 1..6  : sparse warm-ups (small, generous capacity/slack margins).
testId 7..10 : dense trap cases (multi-fan-in "hub" nets + tight margins) --
               a placement that minimizes each net's span in isolation packs
               everything into one small block and overflows the channels
               that block's internal gaps, exactly where hub nets pile up.

The instance always guarantees the IDENTITY placement (cell i -> slot i) is
feasible: capacities/slacks are generated as identity's own usage/span plus a
per-test margin, so a trivial feasible reference always exists.
"""
import sys
import random


def build(test_id: int):
    rng = random.Random(20000 + 17 * test_id)

    n_cells = 8 + 2 * (test_id - 1)          # 8, 10, ..., 26
    dense = test_id >= 7

    all_cells = list(range(n_cells))
    nets = []                                 # each: list[int] terminal cell indices

    n_regular = n_cells
    for _ in range(n_regular):
        a, b = rng.sample(all_cells, 2)
        nets.append([a, b])

    if dense:
        n_hub = rng.randint(3, 5)
        lo = max(4, (n_cells * 2) // 5)
        hi = max(lo + 1, (n_cells * 3) // 5)
        hi = min(hi, n_cells)
        lo = min(lo, hi)
        for _ in range(n_hub):
            hub_size = rng.randint(lo, hi)
            hub_size = max(2, min(hub_size, n_cells))
            terms = rng.sample(all_cells, hub_size)
            nets.append(terms)

    n_nets = len(nets)

    # pick critical nets only among the plain 2-terminal regular nets
    regular_idx = [i for i, net in enumerate(nets) if len(net) == 2]
    n_crit = max(1, n_cells // 5)
    n_crit = min(n_crit, len(regular_idx))
    crit_idx = set(rng.sample(regular_idx, n_crit)) if regular_idx and n_crit > 0 else set()

    # identity usage/span (baseline: pos[i] = i)
    span_identity = [0] * n_nets
    usage_identity = [0] * max(n_cells - 1, 0)
    for i, net in enumerate(nets):
        lo_c, hi_c = min(net), max(net)
        span_identity[i] = hi_c - lo_c
        for g in range(lo_c, hi_c):
            usage_identity[g] += 1

    capacity = []
    for g in range(n_cells - 1):
        margin = rng.choice([0, 0, 0, 1]) if dense else rng.randint(3, 6)
        capacity.append(usage_identity[g] + margin)

    slack = {}
    for i in crit_idx:
        margin = rng.choice([0, 0, 1]) if dense else rng.randint(2, 4)
        slack[i] = span_identity[i] + margin

    lines = [f"{n_cells} {n_nets}"]
    lines.append(" ".join(map(str, capacity)) if capacity else "")
    for i, net in enumerate(nets):
        k = len(net)
        crit = 1 if i in crit_idx else 0
        s = slack[i] if crit else -1
        lines.append(f"{k} {crit} {s} " + " ".join(map(str, net)))
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(build(test_id))


if __name__ == "__main__":
    main()
