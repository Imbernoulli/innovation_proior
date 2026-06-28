#!/usr/bin/env python3
import random
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"
BIN = ROOT / "verify" / "sol_independent_test_bin"


def parse_cpp_blocks(path):
    text = path.read_text()
    return re.findall(r"```cpp\n(.*?)\n```", text, flags=re.S)


def assert_markdown_code_matches_solution():
    sol = SOL.read_text().strip()
    for name in ("train_answer.md", "reasoning.md"):
        blocks = parse_cpp_blocks(ROOT / name)
        if not blocks:
            raise AssertionError(f"{name}: no cpp block found")
        final_block = blocks[-1].strip()
        if final_block != sol:
            raise AssertionError(f"{name}: final cpp block differs from verify/sol.cpp")


def oracle(grid):
    n = len(grid)
    if n == 0:
        return 0

    best = None

    def dfs(i, j, total):
        nonlocal best
        total += grid[i][j]
        if i == n - 1:
            if best is None or total < best:
                best = total
            return
        for nj in (j - 1, j, j + 1):
            if 0 <= nj < n:
                dfs(i + 1, nj, total)

    for start in range(n):
        dfs(0, start, 0)
    return best


def run_solution(grid):
    n = len(grid)
    data = [str(n)]
    for row in grid:
        data.extend(str(x) for x in row)
    proc = subprocess.run(
        [str(BIN)],
        input=" ".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    if not re.fullmatch(r"-?\d+", out):
        raise RuntimeError(f"non-integer output: {out!r}, stderr={proc.stderr!r}")
    return int(out)


def compile_solution():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL), "-o", str(BIN)],
        check=True,
    )


def fixed_cases():
    return [
        [],
        [[0]],
        [[-7]],
        [[10**9]],
        [[-10**9]],
        [[2, 1, 3], [6, 5, 4], [7, 8, 9]],
        [[5, 5, 5], [1, 9, 9], [50, 50, 1]],
        [[1, 100], [100, 1]],
        [[-1, -2], [-3, -4]],
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
        [
            [10**9, -10**9, 10**9],
            [-10**9, 10**9, -10**9],
            [10**9, -10**9, 10**9],
        ],
        [
            [0, 99, 99, 99],
            [99, 0, 99, 99],
            [99, 99, 0, 99],
            [99, 99, 99, -50],
        ],
    ]


def random_grid(rng):
    mode = rng.randrange(8)
    n = rng.randint(0, 8)
    if mode == 0:
        n = rng.randint(0, 3)
        lo, hi = -3, 3
    elif mode == 1:
        lo, hi = -20, 20
    elif mode == 2:
        lo, hi = -10**9, 10**9
    elif mode == 3:
        lo, hi = -50, -1
    elif mode == 4:
        lo, hi = 0, 50
    elif mode == 5:
        value = rng.randint(-30, 30)
        return [[value for _ in range(n)] for _ in range(n)]
    elif mode == 6:
        lo, hi = -1, 1
    else:
        lo, hi = -1000, 1000
    return [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]


def main():
    assert_markdown_code_matches_solution()
    compile_solution()

    cases = fixed_cases()
    rng = random.Random(20260628)
    cases.extend(random_grid(rng) for _ in range(400))

    for idx, grid in enumerate(cases):
        expected = oracle(grid)
        actual = run_solution(grid)
        if actual != expected:
            print(f"Mismatch on case {idx}", file=sys.stderr)
            print(f"n={len(grid)}", file=sys.stderr)
            for row in grid:
                print(" ".join(map(str, row)), file=sys.stderr)
            print(f"expected={expected} actual={actual}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
