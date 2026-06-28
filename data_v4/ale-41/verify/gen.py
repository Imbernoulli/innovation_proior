#!/usr/bin/env python3
"""
Instance generator for ale-41: Online Bin Assignment (sequential, partial info).

Usage: python3 gen.py SEED  ->  writes an instance to stdout.

Instance format (this is the *full* stream, used by the interactive scorer; the
solver itself only ever sees one item at a time, fed by score.py):

    Line 1:  K N
    Line 2:  c_1 c_2 ... c_K        (integer bin capacities)
    Next N lines: s_i v_i           (item size and value, in arrival order)

Design notes (why instances look like this):
  * K in [4, 12] bins; each capacity drawn so the total capacity is a fraction
    of total item size -> the assignment is genuinely contested (you cannot pack
    everything, so *which* value you keep matters; that is what makes an adaptive
    acceptance threshold pay off).
  * Item sizes are moderate vs capacity (no single item fills a bin), values are
    drawn from a *non-stationary* distribution: the value level drifts over the
    stream and occasionally spikes. A fixed threshold is wrong because the
    "good" cutoff early differs from late; this is exactly the regime the
    adaptive empirical-quantile threshold is built for.
  * N is large (a few thousand) so the empirical value distribution seen so far
    is informative well before the stream ends.
"""
import sys
import random


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(seed * 1_000_003 + 12345)

    K = rng.randint(4, 12)
    N = rng.randint(2000, 4000)

    # Capacities: each bin capacity in a band; total capacity deliberately
    # smaller than total item size so bins fill and selectivity matters.
    caps = [rng.randint(800, 1600) for _ in range(K)]

    # Item sizes: small relative to a single capacity (1..120), so many items
    # fit per bin and packing is about value selection, not bin-packing tetris.
    # Values: non-stationary. We move a slowly drifting baseline mean and inject
    # occasional high-value spikes. Sizes and values are mildly correlated
    # (bigger items tend to carry more value) but with heavy noise, so pure
    # "value density" greedy is not obviously right.
    items = []
    base = rng.uniform(30.0, 60.0)
    for i in range(N):
        # drift the value baseline
        base += rng.uniform(-1.5, 1.5)
        base = max(15.0, min(120.0, base))
        size = rng.randint(1, 120)
        spike = (rng.random() < 0.06)
        mean_val = base + 0.4 * size + (rng.uniform(150, 400) if spike else 0.0)
        val = int(max(1, round(rng.gauss(mean_val, 0.30 * mean_val))))
        items.append((size, val))

    out = []
    out.append(f"{K} {N}")
    out.append(" ".join(str(c) for c in caps))
    for s, v in items:
        out.append(f"{s} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
