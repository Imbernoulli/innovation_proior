#!/usr/bin/env python3
"""
Random + structured test generator for the digit-string decoding count problem.

Usage: python3 gen.py SEED [MODE]
Emits to stdout:
  line 1: prime p
  line 2: digit string s  (1 <= len(s) <= cap)

MODE controls the flavor so the differential test covers tricky regions:
  short      - very short strings (length 1..6), full exhaustive territory
  zeros      - strings biased toward '0' so leading-zero / "10"/"20" cases dominate
  boundary   - digits biased to 1,2 so many "10..26" two-digit groups appear
  random     - uniform digits, moderate length
  highdigit  - digits biased to 3..9 so few two-digit groups are valid
  big        - long string (up to ~2000 here) for the bigint-dp oracle path
"""
import random
import sys

PRIMES = [
    2, 3, 5, 7, 11, 13, 97, 101, 998244353, 1000000007, 1000000009,
    167772161, 469762049, 754974721, 2147483647,
]


def gen_string(rng, mode):
    if mode == "short":
        L = rng.randint(1, 6)
        digits = [str(rng.randint(0, 9)) for _ in range(L)]
    elif mode == "zeros":
        L = rng.randint(1, 16)
        pool = list("0001122")  # heavy on 0, and 1/2 so '10'/'20' arise
        digits = [rng.choice(pool) for _ in range(L)]
    elif mode == "boundary":
        L = rng.randint(1, 16)
        pool = list("11223456")  # many valid 10..26 two-digit groups
        digits = [rng.choice(pool) for _ in range(L)]
    elif mode == "highdigit":
        L = rng.randint(1, 16)
        pool = list("3456789")
        digits = [rng.choice(pool) for _ in range(L)]
    elif mode == "big":
        L = rng.randint(200, 2000)
        digits = [str(rng.randint(0, 9)) for _ in range(L)]
    else:  # random
        L = rng.randint(1, 16)
        digits = [str(rng.randint(0, 9)) for _ in range(L)]
    return "".join(digits)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "random"
    rng = random.Random(seed * 7919 + hash(mode) % 100000)
    p = rng.choice(PRIMES)
    s = gen_string(rng, mode)
    print(p)
    print(s)


if __name__ == "__main__":
    main()
