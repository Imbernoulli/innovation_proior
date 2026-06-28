#!/usr/bin/env python3
"""Differential tester: sol (C++) vs brute.py over generated + edge cases."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "sol")

sys.path.insert(0, HERE)
import gen as gmod
import brute as bmod


def run_sol(s: str) -> str:
    p = subprocess.run([SOL], input=s + "\n", capture_output=True, text=True)
    return p.stdout.strip()


def check(s, label):
    expected = str(bmod.solve(s))
    got = run_sol(s)
    if got != expected:
        print(f"MISMATCH [{label}] s={s!r} sol={got} brute={expected}")
        return False
    return True


def main():
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    mism = 0
    total = 0

    # Edge / structured cases
    edges = [
        "", "a", "aa", "ab", "aba", "abc", "aaaa", "aab", "racecar",
        "abacaba", "abacdc", "noon", "aabaa", "bb", "cdd", "abccba",
        "abcba", "aabbaa", "abababab", "zzzzzzzz", "abcdefgh",
        "aaaabaaaa", "ababa", "baab", "aabbccddeeff",
        "a" * 60, "ab" * 30, "abc" * 20,
    ]
    for s in edges:
        total += 1
        if not check(s, "edge"):
            mism += 1

    # Random cases via generator
    for seed in range(1, n_rand + 1):
        s = gmod.gen(seed)
        total += 1
        if not check(s, f"rand{seed}"):
            mism += 1
            if mism >= 15:
                break

    print(f"TOTAL: {total} cases, {mism} mismatches")
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
