#!/usr/bin/env python3
"""Independent differential verifier for sol.cpp.

The oracle here deliberately enumerates every non-empty contiguous subarray.
It shares no recurrence or state with the submitted O(n) min/max DP.
"""

from __future__ import annotations

import itertools
import random
import re
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
VERIFY = BASE / "verify"
SOL_CPP = VERIFY / "sol.cpp"
SOL_BIN = VERIFY / "_independent_sol"


def brute_max_product(a: list[int]) -> int:
    best: int | None = None
    for left in range(len(a)):
        product = 1
        for right in range(left, len(a)):
            product *= a[right]
            if best is None or product > best:
                best = product
    assert best is not None
    return best


def run_solution(a: list[int]) -> int:
    data = f"{len(a)}\n{' '.join(map(str, a))}\n"
    completed = subprocess.run(
        [str(SOL_BIN)],
        input=data,
        text=True,
        capture_output=True,
        check=True,
    )
    return int(completed.stdout.strip())


def check_markdown_blocks() -> None:
    sol = SOL_CPP.read_text()
    for name in ("train_answer.md", "reasoning.md"):
        text = (BASE / name).read_text()
        blocks = re.findall(r"```cpp\n(.*?)```", text, flags=re.S)
        if not blocks:
            raise AssertionError(f"{name}: no cpp block found")
        if blocks[-1] != sol:
            raise AssertionError(f"{name}: final cpp block differs from verify/sol.cpp")


def compile_solution() -> None:
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL_CPP), "-o", str(SOL_BIN)],
        check=True,
    )


def edge_cases() -> list[list[int]]:
    return [
        [0],
        [9],
        [-9],
        [0, 0],
        [-9, -9],
        [9, 9],
        [-1, -1, -1],
        [9] * 18,
        [-9] * 18,
        [-9, 0, -9, 0, -9],
        [0, -1, 0, -1],
        [-2, -3, -2, -3, -2, -3],
        [-1, 2, -3, 4, -5, 6, -7],
        [2, 3, -2, 4],
        [-1, -2, -3, -4, -5],
        [-2, 3, -4],
        [-3, -4, -5],
        [2, -5, -2, -4, 3, -1],
        [0, 9, -9, -9, 0, -9, 9],
        [1, -1] * 9,
    ]


def random_case(rng: random.Random, case_id: int) -> list[int]:
    mode = case_id % 8
    if mode == 0:
        n = rng.randint(1, 18)
        return [rng.randint(-9, 9) for _ in range(n)]
    if mode == 1:
        n = rng.randint(1, 18)
        return [rng.choice([-2, -1, 0, 1, 2]) for _ in range(n)]
    if mode == 2:
        n = rng.randint(1, 18)
        return [rng.randint(1, 9) for _ in range(n)]
    if mode == 3:
        n = rng.randint(1, 18)
        return [rng.randint(-9, -1) for _ in range(n)]
    if mode == 4:
        n = rng.randint(1, 18)
        return [0 if rng.random() < 0.45 else rng.randint(-9, 9) for _ in range(n)]
    if mode == 5:
        n = rng.randint(1, 3)
        return [rng.randint(-9, 9) for _ in range(n)]
    if mode == 6:
        return [rng.choice([-9, 9]) for _ in range(18)]
    n = rng.randint(1, 18)
    return [rng.choice([-9, -8, -1, 0, 1, 8, 9]) for _ in range(n)]


def check_case(a: list[int], label: str) -> None:
    got = run_solution(a)
    expected = brute_max_product(a)
    if got != expected:
        raise AssertionError(f"{label}: input={a} sol={got} brute={expected}")


def main() -> None:
    check_markdown_blocks()
    compile_solution()

    total = 0
    for i, a in enumerate(edge_cases(), start=1):
        check_case(a, f"edge#{i}")
        total += 1

    rng = random.Random(20260628)
    for i in range(1000):
        check_case(random_case(rng, i), f"random#{i}")
        total += 1

    for n in range(1, 5):
        for a in itertools.product(range(-3, 4), repeat=n):
            check_case(list(a), f"exhaustive-n{n}")
            total += 1

    SOL_BIN.unlink(missing_ok=True)
    print(f"PASS independent differential tests: {total} cases")


if __name__ == "__main__":
    main()
