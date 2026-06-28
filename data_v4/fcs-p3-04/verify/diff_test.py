#!/usr/bin/env python3
"""Differential test sol.cpp against the local independent oracle."""

import random
import subprocess
import sys
from pathlib import Path

import brute


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "verify" / "sol_test_bin"


def run_solution(cases):
    payload = [str(len(cases))]
    payload.extend(f"{n} {p}" for n, p in cases)
    raw = subprocess.check_output(
        [str(BIN)],
        input=("\n".join(payload) + "\n").encode(),
    )
    return [int(x) for x in raw.split()]


def build_cases(seed, random_count):
    rng = random.Random(seed)
    cases = [
        (0, 1),
        (0, 2),
        (0, 10**18),
        (1, 1),
        (1, 2),
        (1, 10**18),
        (2, 1),
        (2, 7),
        (3, 2),
        (4, 1000000007),
        (5, 13),
        (22, 999983),
        (24, 10**18 - 3),
        (25, 2),
        (100, 97),
        (1000, 998244353),
        (10000, 1000000007),
        (10**18, 1),
        (10**18, 2),
        (10**18, 1000000007),
        (10**18, 10**18),
        (10**18 - 1, 999999999999999989),
        ((1 << 60), 1000000000000000000),
    ]

    fixed_mods = [
        1,
        2,
        3,
        5,
        7,
        11,
        13,
        97,
        998244353,
        1000000007,
        999999999999999989,
        1000000000000000000,
    ]

    for _ in range(random_count):
        bucket = rng.random()
        if bucket < 0.35:
            n = rng.randint(0, brute.ENUM_LIMIT)
        elif bucket < 0.70:
            n = rng.randint(brute.ENUM_LIMIT + 1, brute.DP_LIMIT)
        else:
            n = rng.choice(
                [
                    rng.randint(0, 10**18),
                    10**18 - rng.randint(0, 1000000),
                    (1 << rng.randint(0, 60)) + rng.randint(0, 1000),
                ]
            )

        mod_bucket = rng.random()
        if mod_bucket < 0.45:
            p = rng.choice(fixed_mods)
        elif mod_bucket < 0.70:
            p = rng.randint(1, 1000)
        elif mod_bucket < 0.90:
            p = rng.randint(1, 10**9)
        else:
            p = rng.randint(1, 10**18)
        cases.append((n, p))

    return cases


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260628
    random_count = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    cases = build_cases(seed, random_count)
    got = run_solution(cases)
    want = [brute.oracle(n, p) for n, p in cases]

    if got != want:
        for i, ((n, p), g, w) in enumerate(zip(cases, got, want)):
            if g != w:
                print(f"Mismatch at case {i}: N={n} p={p} sol={g} oracle={w}")
                return 1
        print(f"Output length mismatch: sol={len(got)} oracle={len(want)}")
        return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
