#!/usr/bin/env python3
# Random SMALL-case generator: python3 gen.py <seed>
# Keeps n small (so brute's 2^n is fast) and W small (so sol's table is small),
# while exercising: equal masses, equal phases, M=1, q at the boundary, mass
# larger than W (items that cannot fit), etc.
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 12)
    # mix of regimes
    mode = rng.randint(0, 4)
    M = rng.randint(1, 6)
    # weights small so that hitting exactly W is plausible
    maxw = rng.choice([1, 2, 3, 5, 8])
    W = rng.randint(0, 14)

    w = []
    p = []
    for _ in range(n):
        if mode == 0:
            wi = rng.randint(1, maxw)
            pi = rng.randint(0, M - 1)
        elif mode == 1:
            # all equal mass -> tempting "binomial" closed form
            wi = maxw
            pi = rng.randint(0, M - 1)
        elif mode == 2:
            # all equal phase -> tempting "phase = k*p" shortcut
            wi = rng.randint(1, maxw)
            pi = rng.randint(0, M - 1)
            pi = 0 if M == 1 else (seed % M)
        elif mode == 3:
            # some masses exceed W (cannot fit)
            wi = rng.randint(1, max(1, W + 4))
            pi = rng.randint(0, M - 1)
        else:
            wi = rng.randint(1, maxw)
            # phases can be >= M (sol must reduce mod M)
            pi = rng.randint(0, 3 * M)
        w.append(wi)
        p.append(pi)

    q = rng.randint(0, M - 1)

    out = [f"{n} {W} {M} {q}"]
    for i in range(n):
        out.append(f"{w[i]} {p[i]}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
