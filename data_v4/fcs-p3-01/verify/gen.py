#!/usr/bin/env python3
"""Random + edge-case test generator for the tribonacci-mod problem.

Usage: gen.py <seed> [mode]
Emits a full test file to stdout:
  T
  then T lines: n p f0 f1 f2

The brute oracle iterates O(n) per query, so generated n must stay small
here (so brute terminates quickly). The solution itself handles n up to 1e18;
those huge-n cases are covered separately by a closed-form cross-check in the
self-verify script, not by the O(n) brute.

modes:
  small  - tiny n (0..12), tiny p, small initials  (default)
  mid    - n up to a few thousand, varied p
  edge   - hand-picked corner cases
"""
import random
import sys


def gen_small(rng):
    queries = []
    Q = rng.randint(1, 30)
    for _ in range(Q):
        n = rng.randint(0, 12)
        p = rng.choice([1, 2, 3, 5, 7, 10, 13, 100])
        f0 = rng.randint(0, 50)
        f1 = rng.randint(0, 50)
        f2 = rng.randint(0, 50)
        queries.append((n, p, f0, f1, f2))
    return queries


def gen_mid(rng):
    queries = []
    Q = rng.randint(1, 20)
    for _ in range(Q):
        n = rng.randint(0, 4000)
        p = rng.choice([1, 2, 7, 999, 1000, 1009, 1_000_000_007, 998_244_353])
        f0 = rng.randint(0, 10**9)
        f1 = rng.randint(0, 10**9)
        f2 = rng.randint(0, 10**9)
        queries.append((n, p, f0, f1, f2))
    return queries


def gen_edge(rng):
    big = 4_611_686_018_427_387_847  # a prime < 2^62
    queries = [
        (0, 1, 0, 0, 0),
        (1, 1, 5, 9, 13),
        (2, 1, 5, 9, 13),
        (3, 1000000007, 0, 0, 0),
        (3, 1000000007, 1, 1, 1),     # -> 3
        (4, 1000000007, 1, 1, 1),     # f3=3, f4=5
        (5, 1000000007, 1, 1, 1),     # f5=9
        (6, 1000000007, 1, 1, 1),     # f6=17
        (0, 2, 1, 1, 1),
        (1, 2, 1, 1, 1),
        (2, 2, 1, 1, 1),
        (3, 2, 1, 1, 1),
        (10, 7, 0, 1, 1),
        (100, 13, 2, 3, 5),
        (1000, 1000000007, 123456789, 987654321, 555555555),
        (2, big, big - 1, big - 2, big - 3),
        (3, big, big - 1, big - 2, big - 3),
        (50, big, 10**18, 10**18, 10**18),
    ]
    return queries


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "small"
    rng = random.Random(seed)
    if mode == "small":
        q = gen_small(rng)
    elif mode == "mid":
        q = gen_mid(rng)
    elif mode == "edge":
        q = gen_edge(rng)
    else:
        raise SystemExit("unknown mode")
    out = [str(len(q))]
    for (n, p, f0, f1, f2) in q:
        out.append(f"{n} {p} {f0} {f1} {f2}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
