#!/usr/bin/env python3
import sys, random

MOD = 998244353

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the O(N*k) brute force is feasible.
    k = rng.randint(1, 8)
    # N can be < k (returns seed) or up to a few thousand.
    mode = rng.randint(0, 3)
    if mode == 0:
        N = rng.randint(0, k - 1)          # N < k -> seed branch
    elif mode == 1:
        N = rng.randint(0, 60)             # small
    else:
        N = rng.randint(0, 3000)           # medium

    # Coefficients and seeds: sometimes full range, sometimes small / signed.
    style = rng.randint(0, 2)
    def rv():
        if style == 0:
            return rng.randint(0, MOD - 1)
        elif style == 1:
            return rng.randint(-5, 5)
        else:
            return rng.randint(0, 10)

    c = [rv() for _ in range(k)]
    a = [rv() for _ in range(k)]

    out = []
    out.append(str(k))
    out.append(" ".join(str(x) for x in c))
    out.append(" ".join(str(x) for x in a))
    out.append(str(N))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
