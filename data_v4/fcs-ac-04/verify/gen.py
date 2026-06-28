#!/usr/bin/env python3
"""
Random small-case generator for "Divisor Nim".

Usage: gen.py <seed>

Emits small instances (few piles, small values) so the full-minimax brute oracle
stays tractable. Mixes in values with rich factor structure (highly composite,
prime powers, primes) to exercise the Omega-based Grundy computation.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # number of piles: small so the combined minimax is feasible
    n = rng.randint(0, 4)

    # value cap kept small for the brute; occasionally allow a single larger value
    # to exercise factorization while keeping the state space tractable.
    vmax = rng.choice([6, 8, 12, 16, 20, 24])

    # a pool biased toward composite numbers with interesting factorizations
    interesting = [1, 2, 3, 4, 6, 8, 12, 16, 18, 24]
    interesting = [v for v in interesting if v <= max(vmax, 24)]

    piles = []
    for _ in range(n):
        if rng.random() < 0.4 and interesting:
            piles.append(rng.choice(interesting))
        else:
            piles.append(rng.randint(1, vmax))

    out = [str(n)]
    if piles:
        out.append(" ".join(map(str, piles)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
