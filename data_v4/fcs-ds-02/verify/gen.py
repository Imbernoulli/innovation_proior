#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Emits a valid instance: n, q on line 1; n values; then q valid (l, r, k) queries
# with 1 <= l <= r <= n and 1 <= k <= r - l + 1.
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 12)
    q = rng.randint(1, 12)

    # Mix of value regimes: tiny range (forces duplicates), wide range, negatives.
    regime = rng.randint(0, 3)
    if regime == 0:
        lo, hi = 0, 3            # heavy duplicates
    elif regime == 1:
        lo, hi = -5, 5
    elif regime == 2:
        lo, hi = -10**9, 10**9   # extreme magnitudes
    else:
        lo, hi = 7, 7            # all equal

    a = [rng.randint(lo, hi) for _ in range(n)]

    lines = ["{} {}".format(n, q)]
    lines.append(" ".join(map(str, a)))
    for _ in range(q):
        l = rng.randint(1, n)
        r = rng.randint(l, n)
        k = rng.randint(1, r - l + 1)
        lines.append("{} {} {}".format(l, r, k))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
