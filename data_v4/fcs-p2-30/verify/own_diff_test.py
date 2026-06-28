#!/usr/bin/env python3
import itertools
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL_CPP = ROOT / "sol.cpp"
SOL_BIN = ROOT / "sol_own_diff"


def brute_count(s: str) -> int:
    """Enumerate every legal cut of s into 1- or 2-digit codes."""
    n = len(s)
    total = 0
    stack = [0]
    while stack:
        i = stack.pop()
        if i == n:
            total += 1
            continue

        if s[i] != "0":
            stack.append(i + 1)

        if i + 1 < n:
            value = (ord(s[i]) - 48) * 10 + (ord(s[i + 1]) - 48)
            if 10 <= value <= 26:
                stack.append(i + 2)

    return total


def run_sol(p: int, s: str) -> int:
    proc = subprocess.run(
        [str(SOL_BIN)],
        input=f"{p}\n{s}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver failed on p={p}, s={s!r}: {proc.stderr}")
    return int(proc.stdout.strip())


def case_set() -> list[tuple[int, str]]:
    primes = [2, 3, 5, 7, 97, 1_000_000_007, 2_147_483_647]
    strings = {
        "0",
        "00",
        "01",
        "06",
        "10",
        "20",
        "30",
        "100",
        "101",
        "110",
        "2101",
        "226",
        "26",
        "27",
        "111111111111111111",
        "222222222222222222",
        "121212121212121212",
        "909090",
        "120120",
    }

    for n in range(1, 7):
        for tup in itertools.product("0123456789", repeat=n):
            strings.add("".join(tup))
            if len(strings) > 260:
                break
        if len(strings) > 260:
            break

    rng = random.Random(20260628)
    alphabets = ["0123456789", "012", "120", "123456789", "7890"]
    while len(strings) < 420:
        alphabet = rng.choice(alphabets)
        n = rng.randint(1, 18)
        strings.add("".join(rng.choice(alphabet) for _ in range(n)))

    cases = []
    for idx, s in enumerate(sorted(strings)):
        cases.append((primes[idx % len(primes)], s))
    rng.shuffle(cases)
    return cases


def main() -> int:
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra", str(SOL_CPP), "-o", str(SOL_BIN)],
        check=True,
    )

    cases = case_set()
    for p, s in cases:
        expected = brute_count(s) % p
        actual = run_sol(p, s)
        if actual != expected:
            print(f"MISMATCH p={p} s={s!r} expected={expected} actual={actual}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
