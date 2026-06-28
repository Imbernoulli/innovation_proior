#!/usr/bin/env python3
"""Random + edge-case generator for the stair-climbing composition problem.

Usage: gen.py <seed> [mode]

Emits to stdout one test instance in the format:
    N k p
    s_1 ... s_k

For differential testing against the brute oracle we keep N small (the brute DP
is O(N*|S|)), but we exercise a wide variety of step sets, moduli and corners.
The hidden-test regime (N up to 1e9) is covered separately by the matrix path;
here we only need correctness of f(N) mod p for N where brute is feasible.
"""
import random
import sys

PRIMES = [2, 3, 5, 7, 11, 13, 97, 101, 998244353, 1000000007, 19260817,
          167772161, 1999999973, 1999999927]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "random"
    rng = random.Random(seed)

    if mode == "edge":
        # A curated set of corner cases, selected by seed.
        cases = []
        # N = 0 with various step sets
        cases.append((0, [1], 1000000007))
        cases.append((0, [1, 2], 2))
        # N = 1
        cases.append((1, [1], 1000000007))
        cases.append((1, [2], 1000000007))      # cannot reach 1 with only step 2 -> 0
        cases.append((1, [1, 2], 7))
        # single step set, only multiples reachable
        cases.append((10, [3], 5))               # 10 not multiple of 3 -> 0
        cases.append((9, [3], 5))                # 9 = 3+3+3 -> exactly 1 way
        # max step equals N (recurrence boundary N == m)
        cases.append((5, [5], 13))               # exactly 1 way
        cases.append((5, [1, 5], 13))
        # m = 1: every n has exactly 1 way
        cases.append((20, [1], 97))
        # mod 1 -> everything is 0
        cases.append((7, [1, 2, 3], 1))
        # mod 2 small prime, fibonacci-ish parity
        cases.append((15, [1, 2], 2))
        # duplicate step sizes in input
        cases.append((8, [2, 2, 3, 3, 3], 1000000007))
        # large-ish max step, small N below the max
        cases.append((4, [7], 11))               # N < m, unreachable -> 0
        cases.append((7, [7], 11))               # N == m -> 1
        cases.append((6, [2, 4, 6], 101))
        # step set not containing 1, classic stairs {2,3}
        cases.append((11, [2, 3], 1000000007))
        idx = seed % len(cases)
        N, S, p = cases[idx]
        emit(N, S, p)
        return

    # random mode
    p = rng.choice(PRIMES)
    # Maximum step m: keep modest so N can stay >= m sometimes and brute is fast.
    m = rng.randint(1, 12)
    # Choose a nonempty subset of {1..m} that includes m (so the max really is m).
    others = list(range(1, m))
    rng.shuffle(others)
    take = rng.randint(0, len(others))
    S = sorted(others[:take] + [m])
    # Possibly add duplicates to stress input dedup.
    if rng.random() < 0.3 and S:
        dup = rng.choice(S)
        S = sorted(S + [dup] * rng.randint(1, 2))
    # N range: include values below, equal to, and above m.
    N = rng.randint(0, 60)
    emit(N, S, p)


def emit(N, S, p):
    out = []
    out.append(f"{N} {len(S)} {p}")
    out.append(" ".join(str(x) for x in S))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
