#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle intentionally uses a different formulation from sol.cpp:
minimum number of palindrome pieces in suffix recursion, with palindrome
checks done by Python slicing/reversal. The final answer is pieces - 1.
"""

from functools import lru_cache
import itertools
import os
import random
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sol.cpp")


def oracle(s: str) -> int:
    if not s:
        return 0

    @lru_cache(maxsize=None)
    def best(i: int) -> int:
        if i == len(s):
            return 0
        ans = len(s) - i
        for j in range(i + 1, len(s) + 1):
            part = s[i:j]
            if part == part[::-1]:
                ans = min(ans, 1 + best(j))
        return ans

    return best(0) - 1


def compile_solution(exe: str) -> None:
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", SRC, "-o", exe],
        check=True,
    )


def run_solution(exe: str, s: str) -> int:
    proc = subprocess.run(
        [exe],
        input=s + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"empty output for {s!r}")
    return int(out)


def random_case(rng: random.Random) -> str:
    mode = rng.randrange(8)
    if mode == 0:
        return "".join(rng.choice("ab") for _ in range(rng.randint(1, 18)))
    if mode == 1:
        return "".join(rng.choice("abc") for _ in range(rng.randint(1, 24)))
    if mode == 2:
        return rng.choice("abcd") * rng.randint(1, 50)
    if mode == 3:
        pieces = []
        for _ in range(rng.randint(1, 7)):
            half = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 5)))
            mid = rng.choice("abcd") if rng.randrange(2) else ""
            pieces.append(half + mid + half[::-1])
        return "".join(pieces) or rng.choice("abcd")
    if mode == 4:
        n = rng.randint(2, 50)
        half = [rng.choice("abc") for _ in range(n // 2)]
        chars = half + ([rng.choice("abc")] if n % 2 else []) + half[::-1]
        for _ in range(rng.randint(1, 5)):
            chars[rng.randrange(n)] = rng.choice("abc")
        return "".join(chars)
    if mode == 5:
        return "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 35)))
    if mode == 6:
        base = "ab" * rng.randint(1, 25)
        return base[: rng.randint(1, len(base))]
    return rng.choice(["aaba", "abaa", "banana", "civicduty", "aabbaa"])


def cases(count: int):
    edge_cases = [
        "",
        "a",
        "aa",
        "ab",
        "aba",
        "aab",
        "aaba",
        "abaa",
        "abc",
        "racecar",
        "abacaba",
        "abcdefgh",
        "aaaabaaaa",
        "abababab",
        "aabbccddeeff",
        "a" * 80,
        "ab" * 40,
        "abc" * 25,
        "b" + "a" * 50 + "b",
    ]
    for s in edge_cases:
        yield "edge", s

    for n in range(1, 9):
        for tup in itertools.product("ab", repeat=n):
            yield f"exhaustive-binary-{n}", "".join(tup)

    rng = random.Random(90210)
    for i in range(count):
        yield f"random-{i}", random_case(rng)


def main() -> int:
    random_count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "sol")
        compile_solution(exe)
        for label, s in cases(random_count):
            total += 1
            want = oracle(s)
            got = run_solution(exe, s)
            if got != want:
                print(f"MISMATCH case={label} s={s!r} got={got} want={want}")
                return 1
    print(f"PASS {total} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
