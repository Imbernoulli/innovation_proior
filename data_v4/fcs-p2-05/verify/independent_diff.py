#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle here enumerates permutations directly. It deliberately does not use
the bitmask-DP recurrence from sol.cpp.
"""

from __future__ import annotations

import itertools
import random
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"


def brute(cost: list[list[int]]) -> int:
    n = len(cost)
    if n == 0:
        return 0
    best: int | None = None
    for perm in itertools.permutations(range(n)):
        total = sum(cost[i][perm[i]] for i in range(n))
        if best is None or total < best:
            best = total
    assert best is not None
    return best


def render(cost: list[list[int]]) -> str:
    out = [str(len(cost))]
    out.extend(" ".join(str(x) for x in row) for row in cost)
    return "\n".join(out) + "\n"


def run_solver(exe: Path, cost: list[list[int]]) -> int:
    proc = subprocess.run(
        [str(exe)],
        input=render(cost),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver exited {proc.returncode}: {proc.stderr}")
    return int(proc.stdout.strip() or "0")


def edge_cases() -> list[list[list[int]]]:
    cases: list[list[list[int]]] = [
        [],
        [[0]],
        [[-10**9]],
        [[10**9]],
        [[0, 6, 3], [6, 0, 8], [3, 7, 7]],
        [[5, 3], [4, 9]],
        [[-5, 2], [3, -1]],
    ]
    for n in range(2, 8):
        cases.append([[7 for _ in range(n)] for _ in range(n)])
        cases.append([[0 if i == j else 10**6 for j in range(n)] for i in range(n)])
        cases.append([[-10**6 if i + j == n - 1 else 10**6 for j in range(n)] for i in range(n)])
        cases.append([[((i * 37 + j * 11) % 9) - 4 for j in range(n)] for i in range(n)])
    return cases


def random_case(rng: random.Random) -> list[list[int]]:
    n = rng.randint(0, 8)
    mode = rng.randrange(6)
    if mode == 0:
        lo, hi = -20, 20
    elif mode == 1:
        lo, hi = -10**9, 10**9
    elif mode == 2:
        lo, hi = -3, 3
    elif mode == 3:
        value = rng.randint(-100, 100)
        return [[value for _ in range(n)] for _ in range(n)]
    elif mode == 4:
        return [[rng.choice([-10**9, -1, 0, 1, 10**9]) for _ in range(n)] for _ in range(n)]
    else:
        lo, hi = -1000, 1000
    return [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]


def main() -> int:
    rng = random.Random(20260628)
    cases = edge_cases()
    while len(cases) < 360:
        cases.append(random_case(rng))

    with tempfile.TemporaryDirectory(prefix="fcs_p2_05_") as td:
        exe = Path(td) / "sol"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", str(SOL), "-o", str(exe)],
            check=True,
        )
        for idx, cost in enumerate(cases, 1):
            want = brute(cost)
            got = run_solver(exe, cost)
            if got != want:
                print(f"mismatch on case {idx}", file=sys.stderr)
                print(render(cost), file=sys.stderr)
                print(f"want {want}, got {got}", file=sys.stderr)
                return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
