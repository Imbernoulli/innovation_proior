#!/usr/bin/env python3
import itertools
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"
BIN = Path("/tmp/fcs_p2_18_sol")


def brute_min_insertions(s: str) -> int:
    """Brute force all subsequences; answer is n - longest palindromic subsequence."""
    n = len(s)
    best = 0
    for mask in range(1 << n):
        t = []
        for i, ch in enumerate(s):
            if mask >> i & 1:
                t.append(ch)
        if len(t) > best and t == t[::-1]:
            best = len(t)
    return n - best


def run_solution(s: str) -> int:
    data = s.encode()
    out = subprocess.check_output([str(BIN)], input=data)
    return int(out.decode().strip())


def add_cases(cases: list[str]) -> None:
    edges = [
        "",
        "a",
        "z",
        "aa",
        "ab",
        "aba",
        "abc",
        "race",
        "abcba",
        "leetcode",
        "mbadm",
        "zzzzzzzzzz",
        "abababab",
        "abcddcba",
        "abcdedcba",
        "abcdef",
        "aabbcc",
        "abacabad",
    ]
    cases.extend(edges)

    for n in range(0, 9):
        for p in itertools.product("abc", repeat=n):
            cases.append("".join(p))

    rng = random.Random(20260628)
    alphabets = ["ab", "abc", "abcd", "abcdef", "abcdefghijklmnopqrstuvwxyz"]
    for _ in range(500):
        alpha = rng.choice(alphabets)
        n = rng.randint(0, 14)
        cases.append("".join(rng.choice(alpha) for _ in range(n)))

    for _ in range(100):
        half = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 7)))
        mid = rng.choice(["", rng.choice("abcd")])
        pal = half + mid + half[::-1]
        if pal and rng.random() < 0.6:
            pos = rng.randrange(len(pal))
            repl = rng.choice("abcd")
            pal = pal[:pos] + repl + pal[pos + 1 :]
        cases.append(pal)


def main() -> None:
    subprocess.check_call(
        ["g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra", "-pedantic", str(SOL), "-o", str(BIN)]
    )

    cases: list[str] = []
    add_cases(cases)

    for idx, s in enumerate(cases, 1):
        expected = brute_min_insertions(s)
        actual = run_solution(s)
        if actual != expected:
            raise SystemExit(
                f"Mismatch on case {idx}: {s!r}: solution={actual}, brute={expected}"
            )

    print(f"PASS {len(cases)} cases")


if __name__ == "__main__":
    main()
