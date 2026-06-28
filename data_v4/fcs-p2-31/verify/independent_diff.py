#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle below follows the problem statement directly: at each '*' it tries
every possible split length, while literals and '?' consume exactly one
character. It deliberately does not share the rolling-prefix DP used by the
C++ solution.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from functools import lru_cache


ALPHABET = "abc"
PCHARS = ALPHABET + "?*"


def encode_token(x: str) -> str:
    return x if x else "-"


def oracle(pattern: str, text: str) -> str:
    n = len(pattern)
    m = len(text)
    sys.setrecursionlimit(max(10000, (n + m + 10) * 4))

    @lru_cache(maxsize=None)
    def rec(i: int, j: int) -> bool:
        if i == n:
            return j == m
        c = pattern[i]
        if c == "*":
            for end in range(j, m + 1):
                if rec(i + 1, end):
                    return True
            return False
        if j == m:
            return False
        if c == "?" or c == text[j]:
            return rec(i + 1, j + 1)
        return False

    return "YES" if rec(0, 0) else "NO"


def run_solution(exe: str, pattern: str, text: str) -> str:
    payload = f"{encode_token(pattern)} {encode_token(text)}\n"
    got = subprocess.run(
        [exe],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if got.returncode != 0:
        raise RuntimeError(f"solution exited {got.returncode}: {got.stderr}")
    return got.stdout.strip()


def derived_yes_case(rng: random.Random) -> tuple[str, str]:
    text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 18)))
    parts: list[str] = []
    i = 0
    while i < len(text):
        roll = rng.random()
        if roll < 0.18:
            parts.append("*")
        elif roll < 0.40:
            parts.append("?")
            i += 1
        else:
            parts.append(text[i])
            i += 1
    if rng.random() < 0.45:
        parts.append("*")
    if not parts and rng.random() < 0.5:
        parts.append("*")
    return "".join(parts), text


def random_case(rng: random.Random) -> tuple[str, str]:
    p_len = rng.randint(0, 16)
    s_len = rng.randint(0, 16)
    return (
        "".join(rng.choice(PCHARS) for _ in range(p_len)),
        "".join(rng.choice(ALPHABET) for _ in range(s_len)),
    )


def star_heavy_case(rng: random.Random) -> tuple[str, str]:
    p_len = rng.randint(0, 22)
    s_len = rng.randint(0, 22)
    weighted = ALPHABET + "??" + "******"
    return (
        "".join(rng.choice(weighted) for _ in range(p_len)),
        "".join(rng.choice(ALPHABET) for _ in range(s_len)),
    )


def build_cases(seed: int, random_count: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    cases: list[tuple[str, str]] = [
        ("", ""),
        ("", "a"),
        ("a", ""),
        ("?", ""),
        ("*", ""),
        ("**", ""),
        ("*?*", ""),
        ("a", "a"),
        ("a", "b"),
        ("?", "b"),
        ("a*b", "axxxb"),
        ("a*b", "axxxc"),
        ("*a*a*a*", "aaa"),
        ("*a*a*a*", "aa"),
        ("*a", "ba"),
        ("*a", "b"),
        ("a*", "a"),
        ("a**b", "ab"),
        ("**a**b**", "xxayybzz"),
        ("*a*a*a*b", "aaaaaa"),
        ("*b", "aaaaaaaaaa"),
        ("?*?", "a"),
        ("?*?", "ab"),
        ("***", "abc"),
    ]

    modes = [random_case, star_heavy_case, derived_yes_case]
    for i in range(random_count):
        cases.append(modes[i % len(modes)](rng))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exe")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--random-count", type=int, default=900)
    args = parser.parse_args()

    cases = build_cases(args.seed, args.random_count)
    yes = 0
    for idx, (pattern, text) in enumerate(cases, 1):
        expected = oracle(pattern, text)
        actual = run_solution(args.exe, pattern, text)
        if expected == "YES":
            yes += 1
        if actual != expected:
            print("MISMATCH")
            print(f"case={idx}")
            print(f"pattern={pattern!r}")
            print(f"text={text!r}")
            print(f"expected={expected}")
            print(f"actual={actual}")
            return 1

    print(
        f"PASS cases={len(cases)} random={args.random_count} "
        f"seed={args.seed} yes={yes} no={len(cases) - yes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
