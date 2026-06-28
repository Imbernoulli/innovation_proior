#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL = ROOT / "sol.cpp"
BIN = ROOT / "sol_own_diff"


def brute(a):
    n = len(a)
    best = 0
    for mask in range(1 << n):
        ok = True
        if n >= 2:
            for i in range(n):
                if (mask >> i) & 1 and (mask >> ((i + 1) % n)) & 1:
                    ok = False
                    break
        if ok:
            total = 0
            for i, x in enumerate(a):
                if (mask >> i) & 1:
                    total += x
            if total > best:
                best = total
    return best


def run_sol(a):
    data = str(len(a)) + "\n" + " ".join(map(str, a)) + "\n"
    proc = subprocess.run(
        [str(BIN)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return int(proc.stdout.strip() or "0")


def cases():
    fixed = [
        [],
        [0],
        [5],
        [-5],
        [3, 4],
        [-3, -4],
        [0, 0],
        [5, 1, 1, 5, 1, 5],
        [8, 9, 2, 9, 9, -2, 8],
        [10**9],
        [-10**9],
        [10**9, 10**9],
        [10**9, -10**9, 10**9],
        [10**9, -10**9, 10**9, -10**9],
        [7, 7, 7, 7, 7],
        [-7, -7, -7, -7, -7],
        [0, 5, 0, 5, 0],
        [5, 0, 5, 0, 5],
        [1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, 1],
    ]
    for a in fixed:
        yield a

    rng = random.Random(20260628)
    palettes = [
        lambda n: [rng.randint(-10, 10) for _ in range(n)],
        lambda n: [rng.randint(-10**9, 10**9) for _ in range(n)],
        lambda n: [rng.choice([-10**9, -1, 0, 1, 10**9]) for _ in range(n)],
        lambda n: [rng.randint(0, 25) for _ in range(n)],
        lambda n: [rng.randint(-25, 0) for _ in range(n)],
    ]
    for _ in range(900):
        n = rng.randint(0, 16)
        yield rng.choice(palettes)(n)

    for n in range(0, 17):
        yield [1] * n
        yield [(-1) ** i * (i % 5) for i in range(n)]
        yield [10**9 if i % 2 == 0 else -10**9 for i in range(n)]


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra", str(SOL), "-o", str(BIN)],
        check=True,
    )

    total = 0
    for total, a in enumerate(cases(), 1):
        expected = brute(a)
        actual = run_sol(a)
        if expected != actual:
            print("MISMATCH", file=sys.stderr)
            print(f"case #{total}: n={len(a)} a={a}", file=sys.stderr)
            print(f"expected={expected} actual={actual}", file=sys.stderr)
            return 1
    print(f"PASS {total} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
