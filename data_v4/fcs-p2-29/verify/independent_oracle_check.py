#!/usr/bin/env python3
import random
import subprocess
import sys


def brute(a):
    best = None
    n = len(a)
    for l in range(n):
        total = 0
        for r in range(l, n):
            total += a[r]
            if best is None or total > best:
                best = total
            if r > l:
                for k in range(l, r + 1):
                    candidate = total - a[k]
                    if candidate > best:
                        best = candidate
    return best


def run_solution(exe, a):
    payload = str(len(a)) + "\n" + " ".join(map(str, a)) + "\n"
    got = subprocess.check_output([exe], input=payload.encode(), timeout=2)
    return int(got.decode().strip())


def cases():
    edge = [
        [0],
        [7],
        [-7],
        [10, -100, 10],
        [5, 6, -3, 3, -5],
        [-3, -1, -4],
        [1, 2, 3, 4],
        [0, 0, 0],
        [1_000_000_000, 1_000_000_000, -1_000_000_000, 1_000_000_000],
        [-1_000_000_000, -999_999_999],
        [4, -1],
        [-1, 4],
        [4, -10, 4, 4],
        [-5, 0, -2],
        [0, -5, 0],
    ]
    for a in edge:
        yield a

    rng = random.Random(20260628)
    for _ in range(700):
        n = rng.randint(1, 35)
        mode = rng.randrange(7)
        if mode == 0:
            a = [rng.randint(-10, 10) for _ in range(n)]
        elif mode == 1:
            a = [rng.randint(-20, -1) for _ in range(n)]
        elif mode == 2:
            a = [rng.randint(0, 20) for _ in range(n)]
        elif mode == 3:
            a = [rng.choice([-1, 0, 1]) * rng.randint(0, 8) for _ in range(n)]
        elif mode == 4:
            a = [((-1) ** i) * rng.randint(0, 25) for i in range(n)]
        elif mode == 5:
            a = [rng.choice([-1_000_000_000, -999_999_999, 0, 999_999_999, 1_000_000_000]) for _ in range(n)]
        else:
            a = [rng.randint(-1000, 1000) for _ in range(n)]
        yield a


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: independent_oracle_check.py ./solution_exe")
    exe = sys.argv[1]
    checked = 0
    for a in cases():
        expected = brute(a)
        actual = run_solution(exe, a)
        checked += 1
        if actual != expected:
            print("MISMATCH")
            print("a =", a)
            print("expected =", expected)
            print("actual =", actual)
            return 1
    print(f"PASS {checked} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
