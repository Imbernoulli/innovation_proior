#!/usr/bin/env python3
"""Random + edge-case generator for the minimum-coins problem.

Usage: gen.py SEED
Prints one test case to stdout in the format:
    n S
    c[0] ... c[n-1]

Mixes regimes so the differential test exercises:
  - small targets with small dense coin sets (greedy-killer territory),
  - cases that include / exclude coin value 1,
  - impossible targets,
  - duplicate denominations,
  - larger targets to stress the DP a bit.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 12

    if mode == 0:
        # Classic greedy-killer family: small dense coins, small target.
        coins = rng.sample(range(1, 12), rng.randint(2, 6))
        S = rng.randint(0, 30)
    elif mode == 1:
        # Coins that often make S impossible (no 1, common factor).
        base = rng.choice([2, 3, 4, 5])
        coins = [base * rng.randint(1, 8) for _ in range(rng.randint(1, 5))]
        S = rng.randint(0, 40)
    elif mode == 2:
        # Includes 1 so always solvable; mix of sizes.
        coins = [1] + rng.sample(range(2, 25), rng.randint(0, 5))
        S = rng.randint(0, 60)
    elif mode == 3:
        # Single coin denomination.
        coins = [rng.randint(1, 20)]
        S = rng.randint(0, 50)
    elif mode == 4:
        # Duplicates allowed in the input list.
        pool = list(range(1, 10))
        coins = [rng.choice(pool) for _ in range(rng.randint(1, 8))]
        S = rng.randint(0, 40)
    elif mode == 5:
        # Coins possibly larger than target.
        coins = [rng.randint(1, 50) for _ in range(rng.randint(1, 6))]
        S = rng.randint(0, 20)
    elif mode == 6:
        # S = 0 edge.
        coins = [rng.randint(1, 30) for _ in range(rng.randint(1, 6))]
        S = 0
    elif mode == 7:
        # Wide spread of denominations.
        coins = rng.sample(range(1, 200), rng.randint(1, 10))
        S = rng.randint(0, 300)
    elif mode == 8:
        # Larger target to stress DP loop a little.
        coins = rng.sample(range(1, 60), rng.randint(2, 8))
        S = rng.randint(0, 1500)
    elif mode == 9:
        # All coins > S => impossible unless S == 0.
        coins = [rng.randint(50, 100) for _ in range(rng.randint(1, 5))]
        S = rng.randint(1, 40)
    elif mode == 10:
        # The canonical {1,3,4} greedy counterexample neighborhood.
        coins = [1, 3, 4]
        S = rng.randint(0, 25)
    else:
        # Fully random.
        n = rng.randint(1, 12)
        coins = [rng.randint(1, 100) for _ in range(n)]
        S = rng.randint(0, 200)

    if not coins:
        coins = [rng.randint(1, 5)]

    rng.shuffle(coins)
    print(len(coins), S)
    print(" ".join(map(str, coins)))


if __name__ == "__main__":
    main()
