#!/usr/bin/env python3
"""Random + edge-case generator for the coin-multiset counting problem.

Usage: gen.py SEED
Prints one test case to stdout in the format:
    n S p
    c[0] ... c[n-1]

Targets are kept modest so the independent recursive/memoized oracle stays fast,
while still exercising the structures that break the tempting wrong approaches:
  - small dense coin sets where the count is large (so order-vs-no-order matters),
  - sets with and without a 1 coin,
  - impossible / zero targets,
  - duplicate denominations in the input (must collapse to one type),
  - coins larger than S,
  - a mix of prime moduli, including very small p (p = 2, 3) so off-by-one in the
    "1 % p" base case would show up, and large primes near 1e9.
"""
import random
import sys

PRIMES = [2, 3, 5, 7, 13, 97, 101, 1009, 998244353, 1000000007]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 12
    p = rng.choice(PRIMES)

    if mode == 0:
        # Small dense coin set, modest target: count is large => overcounting
        # (ordered compositions) would differ wildly from the multiset count.
        coins = rng.sample(range(1, 8), rng.randint(2, 5))
        S = rng.randint(0, 25)
    elif mode == 1:
        # No coin of value 1; many targets unreachable (count 0).
        base = rng.choice([2, 3, 4, 5])
        coins = [base * rng.randint(1, 5) for _ in range(rng.randint(1, 4))]
        S = rng.randint(0, 30)
    elif mode == 2:
        # Includes 1 so always >= 1 way; mixed sizes.
        coins = [1] + rng.sample(range(2, 15), rng.randint(0, 4))
        S = rng.randint(0, 30)
    elif mode == 3:
        # Single denomination: count is 1 iff value divides S, else 0.
        coins = [rng.randint(1, 12)]
        S = rng.randint(0, 40)
    elif mode == 4:
        # Duplicate denominations in input; must collapse to one coin type.
        pool = list(range(1, 7))
        coins = [rng.choice(pool) for _ in range(rng.randint(2, 7))]
        S = rng.randint(0, 25)
    elif mode == 5:
        # Coins possibly larger than the target.
        coins = [rng.randint(1, 30) for _ in range(rng.randint(1, 5))]
        S = rng.randint(0, 15)
    elif mode == 6:
        # S = 0 edge: exactly one way (the empty multiset), i.e. 1 % p.
        coins = [rng.randint(1, 20) for _ in range(rng.randint(1, 5))]
        S = 0
    elif mode == 7:
        # Tiny modulus stress: p in {2,3} so the answer wraps a lot.
        p = rng.choice([2, 3])
        coins = rng.sample(range(1, 9), rng.randint(2, 5))
        S = rng.randint(0, 25)
    elif mode == 8:
        # Slightly larger target to stretch the DP / oracle.
        coins = rng.sample(range(1, 20), rng.randint(2, 6))
        S = rng.randint(0, 60)
    elif mode == 9:
        # All coins > S => zero ways unless S == 0.
        coins = [rng.randint(20, 40) for _ in range(rng.randint(1, 4))]
        S = rng.randint(1, 18)
    elif mode == 10:
        # Classic small set {1,2,5}-style, where counts are familiar.
        coins = [1, 2, 5]
        S = rng.randint(0, 30)
    else:
        # Fully random small instance.
        k = rng.randint(1, 8)
        coins = [rng.randint(1, 25) for _ in range(k)]
        S = rng.randint(0, 40)

    if not coins:
        coins = [rng.randint(1, 5)]

    rng.shuffle(coins)
    print(len(coins), S, p)
    print(" ".join(map(str, coins)))


if __name__ == "__main__":
    main()
