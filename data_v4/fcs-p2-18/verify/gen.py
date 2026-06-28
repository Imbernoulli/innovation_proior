#!/usr/bin/env python3
"""Random + edge-biased test generator for the palindrome-insertions problem.

Usage: gen.py <seed>
Prints a single string (the stdin for one test case).
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = rng.randint(0, 6)

    if mode == 0:
        # very short random over a 2-letter alphabet (stresses the small DP corners)
        n = rng.randint(1, 6)
        s = ''.join(rng.choice('ab') for _ in range(n))
    elif mode == 1:
        # short random over a slightly bigger alphabet
        n = rng.randint(1, 6)
        s = ''.join(rng.choice('abc') for _ in range(n))
    elif mode == 2:
        # small/medium random, alphabet size 2..4
        n = rng.randint(1, 40)
        k = rng.randint(2, 4)
        alpha = 'abcdefghij'[:k]
        s = ''.join(rng.choice(alpha) for _ in range(n))
    elif mode == 3:
        # already a palindrome (answer should be 0) or near-palindrome
        half_n = rng.randint(0, 10)
        k = rng.randint(2, 5)
        alpha = 'abcdefghij'[:k]
        half = ''.join(rng.choice(alpha) for _ in range(half_n))
        mid = rng.choice(['', rng.choice(alpha)])
        s = half + mid + half[::-1]
        if not s:
            s = rng.choice(alpha)
        # optionally perturb to make it near-palindrome
        if s and rng.random() < 0.5:
            i = rng.randrange(len(s))
            s = s[:i] + rng.choice(alpha) + s[i + 1:]
    elif mode == 4:
        # single repeated character (always already a palindrome)
        n = rng.randint(1, 30)
        s = rng.choice('abc') * n
    elif mode == 5:
        # medium random, full lowercase alphabet
        n = rng.randint(1, 60)
        s = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n))
    else:
        # length-1 / length-2 micro edge cases
        n = rng.randint(1, 2)
        s = ''.join(rng.choice('ab') for _ in range(n))

    print(s)


if __name__ == "__main__":
    main()
