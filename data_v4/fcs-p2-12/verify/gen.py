#!/usr/bin/env python3
"""Random + structured test generator for palindrome partitioning min-cuts.

Usage: python3 gen.py <seed>
Emits a single line: the string s (1..~40 chars for brute feasibility),
drawn from small alphabets to make palindromic structure dense.
"""
import random
import sys


def gen(seed: int) -> str:
    rng = random.Random(seed)
    mode = rng.randint(0, 6)

    if mode == 0:
        # tiny alphabet, short -> dense palindromes
        alpha = "ab"
        n = rng.randint(1, 16)
        return "".join(rng.choice(alpha) for _ in range(n))
    if mode == 1:
        # alphabet of 3
        alpha = "abc"
        n = rng.randint(1, 22)
        return "".join(rng.choice(alpha) for _ in range(n))
    if mode == 2:
        # single distinct char repeated (whole string palindrome -> 0 cuts)
        c = rng.choice("abcde")
        n = rng.randint(1, 30)
        return c * n
    if mode == 3:
        # build by concatenating random palindromes (controls cut count)
        alpha = "abc"
        parts = []
        for _ in range(rng.randint(1, 5)):
            half = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 4)))
            mid = rng.choice(alpha) if rng.random() < 0.5 else ""
            parts.append(half + mid + half[::-1])
        s = "".join(parts)
        if not s:
            s = rng.choice(alpha)
        return s
    if mode == 4:
        # near-palindrome: a real palindrome with a few random perturbations
        alpha = "ab"
        n = rng.randint(2, 24)
        half = [rng.choice(alpha) for _ in range(n // 2)]
        s = list(half + ([rng.choice(alpha)] if n % 2 else []) + half[::-1])
        for _ in range(rng.randint(0, 3)):
            s[rng.randrange(len(s))] = rng.choice(alpha)
        return "".join(s)
    if mode == 5:
        # larger alphabet, fewer palindromes -> close to n-1 cuts
        alpha = "abcdefgh"
        n = rng.randint(1, 20)
        return "".join(rng.choice(alpha) for _ in range(n))
    # mode == 6: single character
    return rng.choice("ab")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(gen(seed))


if __name__ == "__main__":
    main()
