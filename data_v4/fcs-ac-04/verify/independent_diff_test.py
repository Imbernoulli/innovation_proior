#!/usr/bin/env python3
import functools
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL_CPP = ROOT / "verify" / "sol.cpp"
SOL_BIN = ROOT / "verify" / "sol_independent_check"


@functools.lru_cache(maxsize=None)
def divisors_below(x):
    out = []
    d = 1
    while d * d <= x:
        if x % d == 0:
            if d < x:
                out.append(d)
            q = x // d
            if q != d and q < x:
                out.append(q)
        d += 1
    return out


@functools.lru_cache(maxsize=None)
def winning(state):
    if not state:
        return False
    items = list(state)
    for i, x in enumerate(items):
        prefix = items[:i] + items[i + 1 :]
        for y in divisors_below(x):
            nxt = prefix[:]
            if y > 1:
                nxt.append(y)
            nxt.sort()
            if not winning(tuple(nxt)):
                return True
    return False


def oracle(piles):
    live = tuple(sorted(x for x in piles if x > 1))
    return "First" if winning(live) else "Second"


def run_solution(piles):
    data = str(len(piles)) + "\n" + " ".join(map(str, piles)) + "\n"
    result = subprocess.run(
        [str(SOL_BIN)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def adversarial_cases():
    cases = [
        [],
        [1],
        [1, 1, 1],
        [2],
        [3],
        [4],
        [8],
        [16],
        [32],
        [64],
        [9],
        [27],
        [81],
        [6],
        [12],
        [18],
        [36],
        [48],
        [60],
        [72],
        [96],
        [2, 2],
        [2, 3],
        [4, 9],
        [8, 12, 6],
        [1, 2, 4, 8, 16],
        [6, 10, 15],
        [12, 18, 20],
        [24, 36, 48],
        [30, 42, 66],
        [64, 81],
        [72, 72],
        [1, 1, 64, 81, 72],
        [999_983],
        [999_999_937],
        [1_000_000_000],
        [31_622 * 31_622],
        [73513440],
        [999_983, 1],
        [1_000_000_000, 2],
        [31_622 * 31_622, 4],
    ]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    cases.append(primes[:])
    cases.append(primes[:10] + [1, 1])
    for p in primes[:8]:
        cases.append([p, p * p, p * p * p])
    return cases


def random_cases(count, seed=20260628):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        n = rng.randint(0, 7)
        cap = rng.choice([12, 20, 36, 60, 84])
        cases.append([rng.randint(1, cap) for _ in range(n)])
    return cases


def main():
    compile_cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-Wall",
        "-Wextra",
        str(SOL_CPP),
        "-o",
        str(SOL_BIN),
    ]
    subprocess.run(compile_cmd, check=True)

    cases = adversarial_cases() + random_cases(500)
    for idx, piles in enumerate(cases, 1):
        expected = oracle(piles)
        actual = run_solution(piles)
        if actual != expected:
            print("MISMATCH")
            print(f"case_index={idx}")
            print(f"piles={piles}")
            print(f"expected={expected}")
            print(f"actual={actual}")
            return 1
    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
