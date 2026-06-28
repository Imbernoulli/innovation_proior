#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle below follows the problem statement literally: keep the live labels
in a circle, count k live people from the current position, remove the k-th,
then continue from the next live position. It intentionally does not use the
Josephus recurrence or any of the repository's existing verifier artifacts.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys


def brute_survivor(n: int, k: int) -> int:
    live = list(range(1, n + 1))
    pos = 0
    while len(live) > 1:
        pos = (pos + k - 1) % len(live)
        del live[pos]
    return live[0]


def build_cases(seed: int) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []

    for n in range(1, 101):
        for k in range(1, 51):
            cases.append((n, k))

    for n in [1, 2, 3, 4, 5, 10, 31, 32, 33, 41, 63, 64, 65, 99, 100, 127, 128, 129, 251, 500]:
        for k in [1, 2, 3, 4, 5, 7, 11, 17, 31, 49, 50]:
            cases.append((n, k))

    rng = random.Random(seed)
    for _ in range(1000):
        n = rng.randint(1, 800)
        k = rng.randint(1, 50)
        cases.append((n, k))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", help="path to compiled solver executable")
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    cases = build_cases(args.seed)
    expected = [brute_survivor(n, k) for n, k in cases]
    payload = str(len(cases)) + "\n" + "".join(f"{n} {k}\n" for n, k in cases)

    run = subprocess.run(
        [args.solver],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        print(f"solver exited with status {run.returncode}", file=sys.stderr)
        return 2

    got = [int(x) for x in run.stdout.split()]
    if len(got) != len(cases):
        print(f"wrong output length: got {len(got)}, expected {len(cases)}", file=sys.stderr)
        return 2

    for i, ((n, k), want, have) in enumerate(zip(cases, expected, got), start=1):
        if want != have:
            print(
                f"mismatch at case {i}: n={n} k={k} expected={want} got={have}",
                file=sys.stderr,
            )
            return 1

    print(f"PASS {len(cases)} cases seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
