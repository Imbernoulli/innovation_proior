#!/usr/bin/env python3
"""Instance generator for "Interval Scheduling on Few Rooms" (ale-25).

Usage:
    python3 gen.py SEED      # prints one instance to stdout

Instance format (stdout):
    n K
    then n lines: start end weight

A realistic, hard instance has many intervals (jobs) competing for a small
number of identical rooms (machines). Intervals are clustered in time so that
contention is real (a pure-greedy first-fit leaves weight on the table), and
weights are heavy-tailed so that a few high-value jobs must be protected even
when they conflict with several cheaper ones.

All randomness is derived from SEED, so the instance is reproducible.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed * 1_000_003 + 12345)

    # Problem size: n intervals, K rooms. K is deliberately small ("few rooms").
    n = rng.randint(400, 600)
    K = rng.randint(2, 5)

    T = 100_000  # time horizon [0, T]

    # Build a small number of temporal "hot spots" (clusters) so that contention
    # is concentrated -- this is what makes the room-as-resource packing nontrivial.
    n_clusters = rng.randint(4, 8)
    centers = [rng.randint(5_000, T - 5_000) for _ in range(n_clusters)]
    spreads = [rng.randint(1_500, 8_000) for _ in range(n_clusters)]

    intervals = []
    for _ in range(n):
        c = rng.randrange(n_clusters)
        mu, sd = centers[c], spreads[c]
        # start drawn around the cluster center
        s = int(rng.gauss(mu, sd))
        # duration: heavy-ish tail, mostly short, occasionally long
        if rng.random() < 0.15:
            dur = rng.randint(4_000, 12_000)   # long, hard-to-pack jobs
        else:
            dur = rng.randint(300, 3_000)      # short jobs
        s = max(0, min(T - 1, s))
        e = s + dur
        if e > T:
            e = T
        if e <= s:
            e = s + 1
        # weight: heavy-tailed. Most jobs modest, a few very valuable.
        r = rng.random()
        if r < 0.05:
            w = rng.randint(800, 1000)         # rare high-value jobs
        elif r < 0.25:
            w = rng.randint(200, 800)
        else:
            w = rng.randint(1, 200)
        intervals.append((s, e, w))

    out = [f"{n} {K}"]
    for (s, e, w) in intervals:
        out.append(f"{s} {e} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
