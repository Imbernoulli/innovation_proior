#!/usr/bin/env python3
import random
import subprocess
import sys


def oracle(patterns, text):
    weight_by_pattern = {}
    for pattern, weight in patterns:
        weight_by_pattern[pattern] = weight_by_pattern.get(pattern, 0) + weight

    total = 0
    for pattern, weight in weight_by_pattern.items():
        count = 0
        limit = len(text) - len(pattern)
        for start in range(limit + 1):
            if text[start:start + len(pattern)] == pattern:
                count += 1
        total += weight * count
    return total


def render_case(patterns, text):
    lines = [str(len(patterns))]
    lines.extend(f"{pattern} {weight}" for pattern, weight in patterns)
    if text:
        lines.append(text)
    return "\n".join(lines) + "\n"


def run_solution(binary, patterns, text):
    proc = subprocess.run(
        [binary],
        input=render_case(patterns, text),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return proc.stdout.strip()


def adversarial_cases():
    cases = [
        ([], ""),
        ([], "abc"),
        ([("a", 7)], ""),
        ([("abcdef", 9)], "abc"),
        ([("abc", 4)], "abc"),
        ([("aa", 3), ("aa", -1)], "aaaa"),
        ([("a", 1), ("aa", 2), ("aaa", 3)], "aaaaa"),
        ([("b", -5)], "bbb"),
        ([("ab", 5), ("bc", 3), ("ab", 2)], "ababcab"),
        ([("a", 0), ("aa", -4), ("aaa", 10), ("aaaa", -3)], "aaaaaa"),
        ([("abab", 8), ("bab", -2), ("ab", 5), ("b", 1)], "abababab"),
    ]

    overflow_patterns = [("a", 10**9)] * 100000
    overflow_text = "a" * 900000
    cases.append((overflow_patterns, overflow_text))
    return cases


def random_cases():
    rng = random.Random(20260628)
    cases = []

    for _ in range(450):
        alphabet = "abcd"[:rng.randint(1, 4)]
        m = rng.randint(0, 14)
        patterns = []
        for _ in range(m):
            length = rng.randint(1, 8)
            pattern = "".join(rng.choice(alphabet) for _ in range(length))
            weight = rng.randint(-20, 20)
            patterns.append((pattern, weight))
        text_len = rng.randint(0, 35)
        text = "".join(rng.choice(alphabet) for _ in range(text_len))
        cases.append((patterns, text))

    for _ in range(150):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        m = rng.randint(0, 12)
        patterns = []
        for _ in range(m):
            length = rng.randint(1, 10)
            pattern = "".join(rng.choice(alphabet) for _ in range(length))
            weight = rng.randint(-1000, 1000)
            patterns.append((pattern, weight))
        text_len = rng.randint(0, 60)
        text = "".join(rng.choice(alphabet) for _ in range(text_len))
        cases.append((patterns, text))

    for n in range(1, 31):
        patterns = [("a" * k, (-1) ** k * k) for k in range(1, n + 1)]
        cases.append((patterns, "a" * (n + 5)))

    return cases


def main():
    if len(sys.argv) != 2:
        print("usage: independent_diff.py /path/to/compiled/solution", file=sys.stderr)
        return 2

    binary = sys.argv[1]
    cases = adversarial_cases() + random_cases()
    for idx, (patterns, text) in enumerate(cases, 1):
        expected = str(oracle(patterns, text))
        actual = run_solution(binary, patterns, text)
        if actual != expected:
            print(f"Mismatch on case {idx}", file=sys.stderr)
            print(f"m={len(patterns)} text_len={len(text)}", file=sys.stderr)
            if len(patterns) <= 20 and len(text) <= 120:
                print(render_case(patterns, text), file=sys.stderr)
            print(f"expected={expected}", file=sys.stderr)
            print(f"actual={actual}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
