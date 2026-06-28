#!/usr/bin/env python3
import os
import random
import subprocess
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = os.path.join(ROOT, "verify", "sol.cpp")


def compile_solution():
    exe = os.path.join(tempfile.gettempdir(), "fcs_gx_04_sol")
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", SOL, "-o", exe],
        check=True,
    )
    return exe


def run_solution(exe, points):
    data = [str(len(points))]
    data.extend(f"{x} {y} {w}" for x, y, w in points)
    proc = subprocess.run(
        [exe],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return int(proc.stdout.strip() or "0")


def brute_oracle(points):
    if not points:
        return 0
    min_x = min(x for x, _, _ in points)
    max_x = max(x for x, _, _ in points)
    min_y = min(y for _, y, _ in points)
    max_y = max(y for _, y, _ in points)
    best = None
    for x0 in range(min_x, max_x + 1):
        for y0 in range(min_y, max_y + 1):
            cost = 0
            for x, y, w in points:
                cost += w * (abs(x0 - x) + abs(y0 - y))
            if best is None or cost < best:
                best = cost
    return best


def big_expected_two_clusters(count_each):
    # count_each groups at (-1e9,-1e9), count_each at (1e9,1e9), all weight 1e9.
    return count_each * 2 * 10**9 * 10**9 * 2


def check_case(exe, points, expected):
    got = run_solution(exe, points)
    if got != expected:
        raise AssertionError(
            f"mismatch\npoints={points[:20]}{'...' if len(points) > 20 else ''}\n"
            f"expected={expected}\ngot={got}"
        )


def main():
    exe = compile_solution()
    rng = random.Random(20260628)
    checked = 0

    adversarial = [
        [],
        [(0, 0, 1)],
        [(5, -7, 100)],
        [(2, 3, 1), (2, 3, 9), (2, 3, 4)],
        [(-3, 0, 1), (3, 0, 1)],
        [(0, -3, 1), (0, 3, 1)],
        [(-2, -2, 1), (2, -2, 1), (-2, 2, 1), (2, 2, 1)],
        [(0, 0, 10), (5, 5, 1), (-5, -5, 1)],
        [(-4, 1, 1), (-1, 1, 2), (3, 1, 4), (4, 1, 8)],
        [(-3, -3, 2), (-3, 3, 2), (3, -3, 2), (3, 3, 2)],
        [(-5, 0, 4), (-1, 0, 1), (1, 0, 1), (5, 0, 4)],
        [(-8, -8, 7), (8, 8, 7), (0, 0, 1)],
    ]
    for points in adversarial:
        check_case(exe, points, brute_oracle(points))
        checked += 1

    for _ in range(600):
        n = rng.randint(0, 12)
        points = []
        for _ in range(n):
            points.append(
                (
                    rng.randint(-8, 8),
                    rng.randint(-8, 8),
                    rng.randint(1, 9),
                )
            )
        check_case(exe, points, brute_oracle(points))
        checked += 1

    for _ in range(200):
        n = rng.randint(1, 14)
        points = []
        for _ in range(n):
            points.append(
                (
                    rng.choice([-3, -1, 0, 1, 3]),
                    rng.choice([-3, 0, 3]),
                    rng.choice([1, 2, 2, 4, 8]),
                )
            )
        check_case(exe, points, brute_oracle(points))
        checked += 1

    big = [(-10**9, -10**9, 10**9), (10**9, 10**9, 10**9)]
    check_case(exe, big, 4 * 10**18)
    checked += 1

    medium_big = [(-10**9, -10**9, 10**9)] * 9 + [(10**9, 10**9, 10**9)] * 9
    check_case(exe, medium_big, big_expected_two_clusters(9))
    checked += 1

    print(f"PASS {checked} cases")


if __name__ == "__main__":
    main()
