#!/usr/bin/env python3
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL = ROOT / "sol_under_test"


def oracle(values):
    """Definition-level minimax: return first player's total.

    The recursion returns (mover_total, other_total) for the interval.  The mover
    chooses the branch that maximizes only their own final total.
    """
    n = len(values)

    @lru_cache(None)
    def play(l, r):
        if l > r:
            return (0, 0)

        next_mover, next_other = play(l + 1, r)
        left = (values[l] + next_other, next_mover)

        next_mover, next_other = play(l, r - 1)
        right = (values[r] + next_other, next_mover)

        return left if left[0] >= right[0] else right

    return play(0, n - 1)[0]


def run_sol(values):
    data = str(len(values)) + "\n"
    if values:
        data += " ".join(map(str, values)) + "\n"
    proc = subprocess.run(
        [str(SOL)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return int(proc.stdout.strip() or "0")


def edge_cases():
    m = 10**9
    cases = [
        [],
        [0],
        [-7],
        [m],
        [-m],
        [1, 5, 233, 7],
        [1, 1, 3, 2],
        [1, 1],
        [-3, -1, -4],
        [m, -m, m, -m, m],
        [-m, m, -m, m, -m],
        [5, -100, 6, -100, 7],
        [0, 0, 0, 0, 0, 0],
        [9, -8, 7, -6, 5, -4],
    ]
    for n in range(1, 13):
        cases.append([m if i % 2 == 0 else -m for i in range(n)])
        cases.append([-(i + 1) for i in range(n)])
        cases.append([i % 3 - 1 for i in range(n)])
    return cases


def random_cases(seed=20260628, count=420):
    rng = random.Random(seed)
    cases = []
    palettes = [
        [-3, -2, -1, 0, 1, 2, 3],
        [-10, -1, 0, 1, 10],
        [-(10**9), -999_999_937, -5, 0, 5, 999_999_937, 10**9],
    ]
    for _ in range(count):
        n = rng.randint(0, 12)
        mode = rng.randrange(5)
        if mode == 0:
            values = [rng.randint(-20, 20) for _ in range(n)]
        elif mode == 1:
            values = [rng.choice(rng.choice(palettes)) for _ in range(n)]
        elif mode == 2:
            values = [
                (rng.choice([-1, 1]) * rng.randint(0, 10**9))
                for _ in range(n)
            ]
        elif mode == 3:
            big = rng.randint(50, 200)
            small = rng.randint(-5, 5)
            values = [big if i % 2 else small for i in range(n)]
        else:
            values = [rng.randint(-3, 3) for _ in range(n)]
        cases.append(values)
    return cases


def main():
    all_cases = edge_cases() + random_cases()
    for idx, values in enumerate(all_cases, 1):
        got = run_sol(values)
        want = oracle(tuple(values))
        if got != want:
            print("MISMATCH", file=sys.stderr)
            print(f"case #{idx}: n={len(values)} values={values}", file=sys.stderr)
            print(f"solution={got} oracle={want}", file=sys.stderr)
            return 1
    print(f"PASS {len(all_cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
