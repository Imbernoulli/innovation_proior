#!/usr/bin/env python3
"""
Random small-case generator for q1(N) (Renyi-Ulam, one lie).

Usage: gen.py <seed>

The brute oracle is an exponential game-tree search, so N must stay small for it
to terminate. We emit T queries with small N (heavy on the boundary values where
the parity correction bites). Format:  first line T, then T lines each one N.
"""
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    T = rng.randint(1, 6)
    Ns = []
    for _ in range(T):
        r = rng.random()
        if r < 0.25:
            N = rng.randint(1, 5)        # tiny / edge (1,2,3 boundary cases)
        elif r < 0.7:
            N = rng.randint(1, 16)       # small, where parity correction matters
        else:
            N = rng.randint(1, 40)       # still brute-feasible
        Ns.append(N)
    print(T)
    for N in Ns:
        print(N)

if __name__ == "__main__":
    main()
