#!/usr/bin/env python3
"""Differential tester for sol.cpp against independent_oracle.py."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from independent_oracle import solve as oracle_solve


PRIMES = [2, 3, 5, 7, 11, 97, 1009, 998244353, 1000000007]


def format_case(coins: list[int], target: int, mod: int) -> str:
    return f"{len(coins)} {target} {mod}\n{' '.join(map(str, coins))}\n"


def edge_cases() -> list[tuple[list[int], int, int]]:
    return [
        ([1, 2, 5], 5, 1000000007),
        ([2, 2], 4, 1000000007),
        ([3, 3, 3], 9, 97),
        ([2, 4], 7, 101),
        ([4], 12, 13),
        ([4], 13, 13),
        ([99, 100], 3, 1009),
        ([5, 7, 11], 0, 2),
        ([1, 2, 5], 5, 2),
        ([6, 10, 15], 30, 3),
        ([1], 0, 1000000007),
        ([1], 25, 5),
        ([8, 8, 9, 20, 21], 16, 7),
        ([2, 3, 6], 60, 998244353),
    ]


def random_case(rng: random.Random, idx: int) -> tuple[list[int], int, int]:
    mode = idx % 14
    mod = rng.choice(PRIMES)

    if mode == 0:
        coins = rng.sample(range(1, 9), rng.randint(2, 6))
        target = rng.randint(0, 35)
    elif mode == 1:
        base = rng.choice([2, 3, 4, 5])
        coins = [base * rng.randint(1, 7) for _ in range(rng.randint(1, 7))]
        target = rng.randint(0, 45)
    elif mode == 2:
        coins = [1] + [rng.randint(2, 18) for _ in range(rng.randint(0, 6))]
        target = rng.randint(0, 45)
    elif mode == 3:
        coins = [rng.randint(1, 20)]
        target = rng.randint(0, 80)
    elif mode == 4:
        coins = [rng.choice(range(1, 9)) for _ in range(rng.randint(2, 10))]
        target = rng.randint(0, 40)
    elif mode == 5:
        target = rng.randint(1, 20)
        coins = [rng.randint(target + 1, target + 50) for _ in range(rng.randint(1, 8))]
    elif mode == 6:
        coins = [rng.randint(1, 50) for _ in range(rng.randint(1, 8))]
        target = 0
    elif mode == 7:
        mod = rng.choice([2, 3])
        coins = rng.sample(range(1, 12), rng.randint(2, 7))
        target = rng.randint(0, 50)
    elif mode == 8:
        coins = rng.sample(range(1, 30), rng.randint(2, 8))
        target = rng.randint(0, 90)
    elif mode == 9:
        coins = [rng.randint(2, 12) * 2 for _ in range(rng.randint(1, 7))]
        target = rng.randrange(1, 60, 2)
    elif mode == 10:
        coins = [1, 2, 5]
        target = rng.randint(0, 50)
    elif mode == 11:
        coins = [1, 3, 4, 7, 11]
        target = rng.randint(0, 70)
    elif mode == 12:
        coins = [rng.randint(1, 25) for _ in range(12)]
        target = rng.randint(0, 55)
    else:
        coins = [rng.randint(1, 35) for _ in range(rng.randint(1, 10))]
        target = rng.randint(0, 65)

    rng.shuffle(coins)
    return coins, target, mod


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: diff_test_independent.py SOL_BINARY", file=sys.stderr)
        return 2

    sol = Path(sys.argv[1])
    cases = edge_cases()
    rng = random.Random(1729)
    cases.extend(random_case(rng, i) for i in range(500))

    for idx, (coins, target, mod) in enumerate(cases):
        inp = format_case(coins, target, mod)
        expected = str(oracle_solve(inp))
        got = subprocess.check_output([str(sol)], input=inp, text=True).strip()
        if got != expected:
            print(f"mismatch on case {idx}", file=sys.stderr)
            print(inp, file=sys.stderr, end="")
            print(f"expected {expected}, got {got}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
