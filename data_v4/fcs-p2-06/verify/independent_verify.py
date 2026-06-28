#!/usr/bin/env python3
"""Independent differential verifier for verify/sol.cpp.

The oracle is intentionally exponential and only targets small chains: it
enumerates every full parenthesization recursively without memoization, which
is the direct specification rather than the submitted bottom-up DP.
"""
import argparse
import random
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL_CPP = ROOT / "verify" / "sol.cpp"


def brute_cost(p):
    n = len(p) - 1
    if n <= 1:
        return 0

    def enum(i, j):
        if i == j:
            return [0]
        out = []
        for k in range(i, j):
            for left in enum(i, k):
                for right in enum(k + 1, j):
                    out.append(left + right + p[i - 1] * p[k] * p[j])
        return out

    return min(enum(1, n))


def run_case(exe, p):
    n = len(p) - 1
    inp = f"{n}\n{' '.join(map(str, p))}\n"
    got = subprocess.check_output([exe], input=inp, text=True).strip()
    return int(got)


def edge_cases():
    return [
        [1],
        [7, 9],
        [2, 3, 4],
        [1, 1, 1],
        [10, 1, 100, 10],
        [40, 20, 30, 10],
        [5, 4, 6, 2, 7],
        [30, 35, 15, 5, 10, 20],
        [1000, 1000, 1000],
        [1, 1000, 1, 1000],
        [999, 1, 999, 1, 999],
        [8, 2, 9, 3, 7, 4],
    ]


def random_case(rng):
    n = rng.randint(0, 9)
    families = [
        lambda: rng.randint(1, 8),
        lambda: rng.randint(1, 60),
        lambda: rng.choice([1, 2, 3, 97, 251, 1000]),
        lambda: rng.randint(900, 1000),
    ]
    pick = rng.choice(families)
    return [pick() for _ in range(n + 1)]


def trap_case(rng):
    n = rng.randint(3, 9)
    p = []
    for i in range(n + 1):
        if i % 2 == rng.randint(0, 1):
            p.append(rng.randint(1, 3))
        else:
            p.append(rng.randint(80, 1000))
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = edge_cases()
    while len(cases) < args.cases:
        if rng.random() < 0.25:
            cases.append(trap_case(rng))
        else:
            cases.append(random_case(rng))

    with tempfile.TemporaryDirectory() as td:
        exe = str(Path(td) / "sol")
        subprocess.check_call(
            ["g++", "-std=c++17", "-O2", "-pipe", str(SOL_CPP), "-o", exe]
        )
        for idx, p in enumerate(cases, 1):
            want = brute_cost(p)
            got = run_case(exe, p)
            if got != want:
                print(f"Mismatch on case {idx}:")
                print(len(p) - 1)
                print(" ".join(map(str, p)))
                print(f"got {got}, want {want}")
                raise SystemExit(1)

    print(f"PASS {len(cases)} cases")


if __name__ == "__main__":
    main()
