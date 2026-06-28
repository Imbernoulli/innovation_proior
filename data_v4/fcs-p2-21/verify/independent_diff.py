#!/usr/bin/env python3
import os
import random
import subprocess
import sys
from functools import lru_cache


ROOT = os.path.dirname(__file__)
SOL_CPP = os.path.join(ROOT, "sol.cpp")
SOL_BIN = os.path.join(ROOT, "sol_independent_check")


def orientations(boxes):
    out = []
    for x, y, z in boxes:
        dims = (x, y, z)
        for k in range(3):
            h = dims[k]
            base = [dims[(k + 1) % 3], dims[(k + 2) % 3]]
            base.sort()
            out.append((base[0], base[1], h))
    return out


def oracle(boxes):
    oriented = orientations(boxes)

    @lru_cache(None)
    def best_above(limit_w, limit_d):
        best = 0
        for w, d, h in oriented:
            if w < limit_w and d < limit_d:
                best = max(best, h + best_above(w, d))
        return best

    return best_above(10**18, 10**18)


def run_solution(boxes):
    data = [str(len(boxes))]
    data.extend(f"{x} {y} {z}" for x, y, z in boxes)
    proc = subprocess.run(
        [SOL_BIN],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return int(proc.stdout.strip())


def edge_cases():
    cases = [
        [],
        [(2, 2, 2)],
        [(2, 3, 4)],
        [(1, 1, 1)],
        [(1, 1, 2)],
        [(6, 6, 10), (5, 9, 9), (4, 8, 8)],
        [(4, 6, 7), (1, 2, 3), (4, 5, 6), (10, 12, 32)],
        [(8, 8, 4), (4, 8, 10)],
        [(1000000, 999999, 999998)],
        [(1000000, 1, 1000000), (999999, 1, 999999)],
        [(1000000 - i, 999000 - i, 998000 - i) for i in range(200)],
    ]
    for side in range(1, 8):
        cases.append([(side, side, side)])
    for n in range(1, 8):
        cases.append([(i + 1, i + 2, i + 3) for i in range(n)])
        cases.append([(10 - i, 10 - i, i + 1) for i in range(n)])
    return cases


def random_case(rng):
    mode = rng.randrange(8)
    if mode == 0:
        n = rng.randrange(0, 8)
        return [(rng.randrange(1, 8), rng.randrange(1, 8), rng.randrange(1, 8)) for _ in range(n)]
    if mode == 1:
        n = rng.randrange(1, 10)
        return [(rng.randrange(1, 5), rng.randrange(1, 5), rng.randrange(1, 5)) for _ in range(n)]
    if mode == 2:
        n = rng.randrange(1, 12)
        return [(rng.choice([2, 3, 5]), rng.choice([2, 3, 5]), rng.randrange(1, 12)) for _ in range(n)]
    if mode == 3:
        n = rng.randrange(1, 9)
        return [(rng.randrange(1, 15), rng.randrange(1, 15), rng.randrange(1, 30)) for _ in range(n)]
    if mode == 4:
        n = rng.randrange(2, 9)
        base = rng.randrange(8, 20)
        return [(base - i, base + i, rng.randrange(1, 25)) for i in range(n)]
    if mode == 5:
        n = rng.randrange(1, 12)
        return [(rng.randrange(1, 20), rng.randrange(1, 20), rng.randrange(900000, 1000001)) for _ in range(n)]
    if mode == 6:
        n = rng.randrange(1, 12)
        return [(rng.randrange(900000, 1000001), rng.randrange(900000, 1000001), rng.randrange(1, 20)) for _ in range(n)]
    n = rng.randrange(1, 14)
    return [(rng.randrange(1, 50), rng.randrange(1, 50), rng.randrange(1, 50)) for _ in range(n)]


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", SOL_CPP, "-o", SOL_BIN],
        check=True,
    )

    rng = random.Random(20260628)
    cases = edge_cases()
    cases.extend(random_case(rng) for _ in range(500))

    for idx, boxes in enumerate(cases):
        expected = oracle(boxes)
        actual = run_solution(boxes)
        if actual != expected:
            print("MISMATCH", file=sys.stderr)
            print(f"case #{idx}", file=sys.stderr)
            print(len(boxes), file=sys.stderr)
            for box in boxes:
                print(*box, file=sys.stderr)
            print(f"expected {expected}, got {actual}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
