#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle here is intentionally not a profile DP and not a matrix/recurrence:
for small N it recursively fills the concrete 3 x N grid by taking the first
empty cell and trying the two legal domino placements from that cell.
"""

from functools import lru_cache
import os
import random
import subprocess
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = os.path.join(ROOT, "verify", "sol.cpp")


@lru_cache(maxsize=None)
def exact_tilings(n: int) -> int:
    total = 3 * n
    full = (1 << total) - 1

    def bit(row: int, col: int) -> int:
        return 1 << (row * n + col)

    @lru_cache(maxsize=None)
    def dfs(mask: int) -> int:
        if mask == full:
            return 1

        first = 0
        while mask & (1 << first):
            first += 1
        row, col = divmod(first, n)

        ways = 0
        here = 1 << first

        if col + 1 < n:
            right = bit(row, col + 1)
            if not (mask & right):
                ways += dfs(mask | here | right)

        if row + 1 < 3:
            down = bit(row + 1, col)
            if not (mask & down):
                ways += dfs(mask | here | down)

        return ways

    return dfs(0)


def run_solution(binary: str, n: int, m: int) -> int:
    proc = subprocess.run(
        [binary],
        input=f"{n} {m}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"solution exited {proc.returncode} for N={n}, m={m}\n{proc.stderr}"
        )
    return int(proc.stdout.strip())


def build_solution() -> str:
    binary = os.path.join(tempfile.gettempdir(), "fcs_p3_09_sol")
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", SOL, "-o", binary],
        check=True,
    )
    return binary


def cases():
    fixed_n = list(range(0, 19)) + [20, 22, 24]
    fixed_m = [
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        10,
        97,
        1000,
        99991,
        998244353,
        1000000000,
        1000000007,
    ]
    for n in fixed_n:
        for m in fixed_m:
            yield n, m

    rng = random.Random(20260628)
    modulus_pool = [
        1,
        2,
        6,
        9,
        16,
        25,
        36,
        101,
        9973,
        1000000007,
        998244353,
    ]
    for _ in range(700):
        n = rng.randrange(0, 25)
        if rng.random() < 0.45:
            m = rng.choice(modulus_pool)
        else:
            m = rng.randrange(1, 1000000001)
        yield n, m


def main() -> int:
    binary = build_solution()
    tested = 0
    for n, m in cases():
        expected = exact_tilings(n) % m
        got = run_solution(binary, n, m)
        tested += 1
        if got != expected:
            print(
                f"MISMATCH after {tested} cases: N={n}, m={m}, "
                f"expected={expected}, got={got}",
                file=sys.stderr,
            )
            return 1
    print(f"PASS {tested} brute-force differential cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
