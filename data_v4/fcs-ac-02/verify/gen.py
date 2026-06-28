#!/usr/bin/env python3
"""Random small-case generator. Param: int seed.

Prints a single integer k on stdout. Most seeds produce a small k so the differential
test exercises many easy-to-check cases; some seeds produce values near binomial
boundaries C(c,3)/C(c,2)-1, which is exactly where the greedy clique-stacking is most
likely to mis-decompose (the adversarial region)."""
import sys
import random
from math import comb


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = rng.randint(0, 5)
    if mode == 0:
        k = rng.randint(0, 30)                       # tiny
    elif mode == 1:
        k = rng.randint(0, 2000)                     # small
    elif mode == 2:
        k = rng.randint(0, 200000)                   # medium
    elif mode == 3:
        # near a C(c,3) boundary (just below the next complete-clique level)
        c = rng.randint(3, 120)
        k = comb(c, 3) - rng.randint(0, 5)
    elif mode == 4:
        # near a C(c,2) boundary
        c = rng.randint(2, 200)
        k = comb(c, 2) - rng.randint(0, 3)
    else:
        k = rng.randint(0, 100000000)                # full range (up to 1e8)

    if k < 0:
        k = 0
    print(k)


if __name__ == "__main__":
    main()
