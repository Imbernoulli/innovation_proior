#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle here is deliberately direct: for each small test string, enumerate
every subsequence by bitmask and keep the longest palindromic one. It does not
use the interval DP recurrence from sol.cpp.
"""

import random
import subprocess
import sys


def brute_lps(s: str) -> int:
    n = len(s)
    best = 0
    for mask in range(1 << n):
        t = []
        for i, ch in enumerate(s):
            if (mask >> i) & 1:
                t.append(ch)
        if len(t) > best and t == t[::-1]:
            best = len(t)
    return best


def run_solution(exe: str, s: str) -> int:
    input_data = "" if s == "" else s + "\n"
    proc = subprocess.run(
        [exe],
        input=input_data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode} on {s!r}: {proc.stderr}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"solution produced empty output on {s!r}")
    return int(out)


def make_cases() -> list[str]:
    cases = [
        "",
        "a",
        "aa",
        "ab",
        "aba",
        "abba",
        "abcba",
        "abca",
        "bbbab",
        "character",
        "abcdef",
        "fedcba",
        "aaaaaa",
        "abababab",
        "abcabcabc",
        "zzxyyxzz",
        "a" * 14,
        "abcdefghijklmnopqrstuvwxyz"[:14],
        "abcdefghijklmn",
        "nmlkjihgfedcba",
    ]

    rng = random.Random(20260628)
    alphabets = ["ab", "abc", "abcd", "abcde", "abcdefghijklmnopqrstuvwxyz"]
    while len(cases) < 420:
        n = rng.randint(0, 14)
        alphabet = rng.choice(alphabets)
        s = "".join(rng.choice(alphabet) for _ in range(n))
        cases.append(s)

    for n in range(15):
        cases.append("a" * n)
        cases.append("".join(chr(ord("a") + i) for i in range(n)))
        half = "".join(chr(ord("a") + (i % 4)) for i in range(n // 2))
        middle = "z" if n % 2 else ""
        cases.append(half + middle + half[::-1])

    return cases


def main() -> None:
    exe = sys.argv[1] if len(sys.argv) > 1 else "./sol"
    cases = make_cases()
    for idx, s in enumerate(cases, 1):
        expected = brute_lps(s)
        got = run_solution(exe, s)
        if got != expected:
            print(
                f"Mismatch on case {idx}: s={s!r}, expected={expected}, got={got}",
                file=sys.stderr,
            )
            sys.exit(1)
    print(f"PASS {len(cases)} cases")


if __name__ == "__main__":
    main()
