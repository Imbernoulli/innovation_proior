#!/usr/bin/env python3
"""
Random + edge-case generator for the bracket-sequence problem.

Emits a test file on stdout. Because the brute oracle uses exact big-integer
Catalan numbers (O(max_n^2) plus huge integers), n is kept modest here so the
differential test stays fast; correctness of the closed form does not depend
on n being large.

Constraint enforced (matches the statement): p is prime and p > 2n, so every
factor 1..2n is invertible mod p.

Usage: gen.py <seed> [mode]
  mode "edge" emits a fixed battery of corner cases.
"""
import sys
import random


def is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    i = 3
    while i * i <= m:
        if m % i == 0:
            return False
        i += 2
    return True


def next_prime(x):
    if x <= 2:
        return 2
    if x % 2 == 0:
        x += 1
    while not is_prime(x):
        x += 2
    return x


def rand_prime_gt(lo):
    # smallest-ish prime strictly greater than lo, with a little jitter
    start = lo + 1 + random.randint(0, 50)
    return next_prime(start)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "random"
    random.seed(seed)

    lines = []
    if mode == "edge":
        cases = []
        # n = 0 (empty sequence, Catalan(0) = 1)
        cases.append((0, rand_prime_gt(0)))
        cases.append((0, 2))
        # n = 1 (Catalan = 1)
        cases.append((1, rand_prime_gt(2)))
        cases.append((1, 3))
        # small n with the tightest legal prime p just above 2n
        for n in [2, 3, 4, 5, 6, 7, 8, 10]:
            cases.append((n, next_prime(2 * n + 1)))
        # p just barely bigger than 2n so the answer wraps a lot
        cases.append((12, next_prime(2 * 12 + 1)))
        cases.append((20, next_prime(2 * 20 + 1)))
        # a couple where p is far larger than the Catalan number (no wrap)
        cases.append((5, 1000000007))
        cases.append((9, 1000000007))
        lines.append(str(len(cases)))
        for n, p in cases:
            lines.append(f"{n} {p}")
    else:
        q = random.randint(1, 12)
        lines.append(str(q))
        for _ in range(q):
            n = random.randint(0, 140)
            # mix: sometimes tight prime just over 2n, sometimes a big prime
            if random.random() < 0.5:
                p = rand_prime_gt(2 * n)
            else:
                p = random.choice([1000000007, 998244353, 1000003])
                # ensure p > 2n (always true here since 2n <= 280)
            lines.append(f"{n} {p}")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
