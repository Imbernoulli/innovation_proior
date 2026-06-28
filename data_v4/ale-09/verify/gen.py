#!/usr/bin/env python3
"""Instance generator for "Dynamic Bin Packing with Rebalancing"
(ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    N C
    a_0 d_0 s_0
    a_1 d_1 s_1
    ...
    a_{N-1} d_{N-1} s_{N-1}

Each line describes one item: it is "alive" during the half-open time interval
[a_i, d_i) (0 <= a_i < d_i) and consumes an integer size s_i (1 <= s_i <= C)
while alive.  C is the per-bin capacity.  An item must be placed into exactly one
bin for its whole lifetime; at every instant the sizes of the items alive in a bin
must not exceed C.  The goal is to minimize the number of bins ever used.

Design notes
------------
This is *temporal* / dynamic bin packing (a.k.a. interval bin packing or dynamic
storage allocation): items arrive AND depart, so a bin that is full now can become
reusable later.  The instance regime is chosen so the trivial first-fit-by-arrival
baseline leaves real slack: a mixture of long-lived "background" items and many
short-lived "burst" items, with sizes drawn so several items can co-reside in a
bin but the packing is non-trivial (NP-hard in general).  N is in [400, 1200];
the time horizon and capacity scale with N so that the maximum instantaneous load
is comfortably larger than C (forcing many bins) while leaving reassignment room.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xB1A5_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # number of items, deterministic from the seed
    N = rng.randint(400, 1200)

    # per-bin capacity
    C = rng.randint(20, 60)

    # time horizon: long enough that lifetimes overlap in waves
    T = rng.randint(max(50, N // 4), max(120, N // 2))

    # fraction of "long-lived background" items (span much of the horizon)
    bg_frac = rng.uniform(0.10, 0.30)

    items = []
    for _ in range(N):
        if rng.random() < bg_frac:
            # background item: long lifetime, often biggish
            a = rng.randint(0, max(0, T // 4))
            length = rng.randint(max(1, T // 2), T)
            s = rng.randint(max(1, C // 4), max(1, (3 * C) // 4))
        else:
            # burst item: short lifetime, small-to-medium size
            a = rng.randint(0, T - 1)
            length = rng.randint(1, max(1, T // 6))
            s = rng.randint(1, max(1, C // 2))
        d = a + length
        if d > T:
            d = T
        if d <= a:
            d = a + 1
        if s < 1:
            s = 1
        if s > C:
            s = C
        items.append((a, d, s))

    out = [f"{N} {C}"]
    out.extend(f"{a} {d} {s}" for (a, d, s) in items)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
