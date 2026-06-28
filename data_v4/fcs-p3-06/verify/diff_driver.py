#!/usr/bin/env python3
"""
Differential test verify/sol.cpp against brute.py's independent oracle.

The generated n values stay within ORACLE_CAP because the oracle computes exact
inclusion-exclusion integers, and literal permutation enumeration is used as a
self-check for tiny n.
"""

import random
import subprocess
import sys
from pathlib import Path

from brute import oracle


ROOT = Path(__file__).resolve().parent
SOL_CPP = ROOT / "sol.cpp"
SOL_BIN = ROOT / "sol_test_bin"
ORACLE_CAP = 220
PRIMES = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    97,
    101,
    9973,
    99991,
    999983,
    998244353,
    1000000007,
    1000000009,
    2147483647,
]


def compile_solution():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL_CPP), "-o", str(SOL_BIN)],
        check=True,
    )


def edge_cases():
    return [
        (1000000007, [0]),
        (1000000007, [1]),
        (1000000007, [0, 1, 2, 3, 4, 5, 6, 7]),
        (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (3, [0, 1, 2, 3, 7, 8, 12, 20, ORACLE_CAP]),
        (5, [ORACLE_CAP, ORACLE_CAP, ORACLE_CAP]),
        (2147483647, [0, 1, 2, 3, 7, 8, 20, 75, 150, ORACLE_CAP]),
        (998244353, list(range(0, 31))),
        (99991, [0, ORACLE_CAP, 1, ORACLE_CAP - 1, 2, ORACLE_CAP - 2]),
    ]


def random_case(rng):
    p = rng.choice(PRIMES)
    count = rng.randint(1, 18)
    ns = []
    for _ in range(count):
        bucket = rng.random()
        if bucket < 0.35:
            ns.append(rng.randint(0, 8))
        elif bucket < 0.70:
            ns.append(rng.randint(0, 40))
        elif bucket < 0.92:
            ns.append(rng.randint(0, ORACLE_CAP))
        else:
            ns.append(rng.choice([0, 1, 2, 7, 8, ORACLE_CAP - 1, ORACLE_CAP]))
    return p, ns


def run_case(p, ns):
    inp = f"{len(ns)} {p}\n{' '.join(map(str, ns))}\n"
    proc = subprocess.run(
        [str(SOL_BIN)],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
    )
    got = proc.stdout.split()
    expected = [str(oracle(n, p)) for n in ns]
    return got, expected, inp


def main():
    compile_solution()
    rng = random.Random(20260628)
    cases = edge_cases()
    cases.extend(random_case(rng) for _ in range(420))

    for index, (p, ns) in enumerate(cases, 1):
        got, expected, inp = run_case(p, ns)
        if got != expected:
            print(f"MISMATCH case={index} p={p} ns={ns}")
            print("--- input ---")
            print(inp, end="")
            print("--- sol ---")
            print("\n".join(got))
            print("--- oracle ---")
            print("\n".join(expected))
            return 1

    print(f"PASS cases={len(cases)} random={len(cases) - len(edge_cases())} edge={len(edge_cases())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
