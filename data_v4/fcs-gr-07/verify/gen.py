#!/usr/bin/env python3
# Random small-case generator for Functional-Graph Cycle Arithmetic.
# Usage: python3 gen.py <seed>
# Emits a valid instance to stdout. Mixes small t (so the pure-simulation brute
# is fast) with some large t (exercising the cycle modular arithmetic path).
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small n so the brute is fast; occasionally force tiny structures (single
    # self-loop, full cycle, long single chain) to hit edge cases.
    mode = rng.randint(0, 6)
    n = rng.randint(1, 12)
    f = [0] * n
    if mode == 0:
        # everything points to a single self-loop tail chain
        f = [max(0, i - 1) for i in range(n)]  # 0 is self-loop, rest a chain
    elif mode == 1:
        # one big cycle
        f = [(i + 1) % n for i in range(n)]
    elif mode == 2:
        # all self loops
        f = [i for i in range(n)]
    elif mode == 3:
        # two halves: a cycle plus a chain feeding into it
        c = max(1, n // 2)
        for i in range(c):
            f[i] = (i + 1) % c
        for i in range(c, n):
            f[i] = i - 1  # chain into node c-1 ... actually into i-1
    else:
        for i in range(n):
            f[i] = rng.randint(0, n - 1)

    q = rng.randint(1, 12)
    out = []
    out.append(str(n))
    out.append(" ".join(map(str, f)))
    out.append(str(q))
    for _ in range(q):
        s = rng.randint(0, n - 1)
        r = rng.random()
        if r < 0.45:
            t = rng.randint(0, 30)            # tiny: exact step walk
        elif r < 0.8:
            t = rng.randint(0, 2_000_000)     # within brute's direct loop
        else:
            t = rng.randint(0, 10**18)        # huge: cycle modular path
        out.append(f"{s} {t}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
