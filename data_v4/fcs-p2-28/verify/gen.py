#!/usr/bin/env python3
"""Random + edge-case generator for the Longest Palindromic Subsequence problem.

Usage: gen.py <seed> [mode]
Prints a single line: the string s (or an empty line for the empty-string case).

Modes are chosen by seed buckets so that a run of many seeds covers:
  - tiny strings over a 2-letter alphabet (high palindrome density)
  - small strings over varying alphabet sizes
  - already-palindromic strings
  - the empty string and length-1 edge
  - medium strings up to a few hundred chars
"""
import random
import sys


def rand_string(rng, length, alphabet):
    return ''.join(rng.choice(alphabet) for _ in range(length))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    bucket = seed % 10

    if bucket == 0:
        # empty / length-1 edges
        choice = rng.choice(["", "a", "z", rng.choice("abc")])
        print(choice)
        return
    if bucket in (1, 2, 3):
        # tiny, 2-letter alphabet, dense palindromes -- exercises bruteforce cross-check
        length = rng.randint(1, 16)
        print(rand_string(rng, length, "ab"))
        return
    if bucket in (4, 5):
        # small, variable alphabet size
        asize = rng.randint(1, 6)
        alphabet = "abcdefghijklmnopqrstuvwxyz"[:asize]
        length = rng.randint(1, 12)
        print(rand_string(rng, length, alphabet))
        return
    if bucket == 6:
        # already a palindrome (even or odd)
        half_len = rng.randint(0, 8)
        asize = rng.randint(1, 4)
        alphabet = "abcdefghijklmnopqrstuvwxyz"[:asize]
        half = rand_string(rng, half_len, alphabet)
        mid = rng.choice([""] + list(alphabet))
        s = half + mid + half[::-1]
        if s == "":
            s = rng.choice(alphabet)
        print(s)
        return
    if bucket == 7:
        # medium length, small alphabet
        length = rng.randint(20, 200)
        asize = rng.randint(2, 4)
        alphabet = "abcdefghijklmnopqrstuvwxyz"[:asize]
        print(rand_string(rng, length, alphabet))
        return
    if bucket == 8:
        # medium length, full lowercase alphabet
        length = rng.randint(20, 300)
        print(rand_string(rng, length, "abcdefghijklmnopqrstuvwxyz"))
        return
    # bucket == 9: a single repeated character (whole string is a palindrome)
    length = rng.randint(1, 50)
    print(rng.choice("abcdefghijklmnopqrstuvwxyz") * length)


if __name__ == "__main__":
    main()
