#!/usr/bin/env python3
"""Random + edge-case generator for the Pell-mod-p problem.

Usage: gen.py SEED [MODE]
Emits a valid input file on stdout:
  T
  N p          (T lines)
with 0 <= N <= 1e18 and p a prime in [2, 1e18].

MODE controls the distribution so we hit the tempting small-N region as well
as the huge-N region where any hardcoded lookup table would be exposed.
"""
import random
import sys


def is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def rand_prime(lo, hi, rng):
    for _ in range(10000):
        c = rng.randint(lo, hi)
        c |= 1
        if c < 2:
            c = 2
        if is_prime(c):
            return c
    return 2


SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
BIG = 10**18


def gen_case(rng, mode):
    if mode == "small":
        N = rng.randint(0, 30)
        p = rng.choice(SMALL_PRIMES)
    elif mode == "tiny_mod":
        N = rng.randint(0, BIG)
        p = rng.choice([2, 3, 5, 7])
    elif mode == "big_n":
        N = rng.randint(BIG // 2, BIG)
        p = rand_prime(10**17, BIG, rng)
    elif mode == "edge":
        N = rng.choice([0, 1, 2, 3, BIG, BIG - 1, 10**17, 999999999999999989])
        p = rng.choice([2, 3, 999999999999999989, rand_prime(10**9, 10**10, rng)])
    else:  # mixed
        N = rng.choice([
            rng.randint(0, 40),
            rng.randint(0, 10**6),
            rng.randint(0, BIG),
            rng.choice([0, 1, 2, BIG, BIG - 1]),
        ])
        p = rng.choice([
            rng.choice(SMALL_PRIMES),
            rand_prime(10**3, 10**6, rng),
            rand_prime(10**9, 10**12, rng),
            rand_prime(10**17, BIG, rng),
        ])
    return N, p


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    rng = random.Random(seed)
    T = rng.randint(1, 30)
    lines = [str(T)]
    for _ in range(T):
        N, p = gen_case(rng, mode)
        lines.append(f"{N} {p}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
