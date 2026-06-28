#!/usr/bin/env python3
"""Independent differential harness for fcs-p3-02.

The small-N oracle enumerates domino placements on the 2 x N board, rather than
using the matrix recurrence from sol.cpp. A separate fast-doubling Fibonacci
check is included only for large-N smoke coverage.
"""

from __future__ import annotations

import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


PRIME_MODS = [
    2,
    3,
    5,
    7,
    11,
    97,
    101,
    1009,
    998244353,
    999999937,
    1000000007,
]


@lru_cache(maxsize=None)
def brute_tilings(n: int, mask: int = 0) -> int:
    """Count tilings by recursive placement over a 2-by-n bitmask board."""
    cells = 2 * n
    if mask == (1 << cells) - 1:
        return 1

    first = 0
    while mask & (1 << first):
        first += 1

    row, col = divmod(first, n)
    total = 0

    # Horizontal domino in the same row.
    if col + 1 < n:
        other = first + 1
        if not (mask & (1 << other)):
            total += brute_tilings(n, mask | (1 << first) | (1 << other))

    # Vertical domino in the same column.
    other = (1 - row) * n + col
    if not (mask & (1 << other)):
        total += brute_tilings(n, mask | (1 << first) | (1 << other))

    return total


def fib_pair(n: int, mod: int) -> tuple[int, int]:
    if n == 0:
        return 0, 1 % mod
    a, b = fib_pair(n >> 1, mod)
    c = (a * ((2 * b - a) % mod)) % mod
    d = (a * a + b * b) % mod
    if n & 1:
        return d, (c + d) % mod
    return c, d


def large_oracle(n: int, mod: int) -> int:
    return fib_pair(n + 1, mod)[0]


def run_solution(exe: Path, cases: list[tuple[int, int]]) -> list[int]:
    payload = str(len(cases)) + "\n" + "\n".join(f"{n} {p}" for n, p in cases) + "\n"
    proc = subprocess.run(
        [str(exe)],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}:\n{proc.stderr}")
    return [int(x) for x in proc.stdout.split()]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/compiled/sol", file=sys.stderr)
        return 2

    exe = Path(sys.argv[1])
    rng = random.Random(20260628)

    brute_cases: list[tuple[int, int]] = []
    for n in range(19):
        for p in PRIME_MODS:
            brute_cases.append((n, p))

    for _ in range(500):
        n = rng.randrange(0, 19)
        p = rng.choice(PRIME_MODS)
        brute_cases.append((n, p))

    actual = run_solution(exe, brute_cases)
    expected = [brute_tilings(n) % p for n, p in brute_cases]
    for i, ((n, p), got, want) in enumerate(zip(brute_cases, actual, expected), 1):
        if got != want:
            print(f"BRUTE MISMATCH case #{i}: N={n} p={p} got={got} want={want}")
            return 1

    large_cases = [
        (10**18, 2),
        (10**18, 998244353),
        (10**18, 1000000007),
        (999999999999999999, 999999937),
        ((1 << 60) - 1, 1000000007),
        ((1 << 59) - 1, 998244353),
    ]
    actual = run_solution(exe, large_cases)
    expected = [large_oracle(n, p) for n, p in large_cases]
    for i, ((n, p), got, want) in enumerate(zip(large_cases, actual, expected), 1):
        if got != want:
            print(f"LARGE MISMATCH case #{i}: N={n} p={p} got={got} want={want}")
            return 1

    print(f"PASS brute_cases={len(brute_cases)} large_cases={len(large_cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
