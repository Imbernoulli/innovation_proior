#!/usr/bin/env python3
"""Random + edge-case generator for the LCS problem.

Usage: gen.py <seed>

Prints two whitespace-separated strings (s on line 1, t on line 2). Mixes
several regimes designed to stress the greedy 'match-as-you-scan' bug:
 - tiny alphabets (forces many crossing matches where greedy commits early),
 - reordered / shuffled content,
 - strings that share characters but in adversarial order,
 - varied lengths including length-1 and equal strings.
A handful of fixed edge cases are emitted for small seeds.
"""
import sys
import random


FIXED = [
    ("a", "a"),
    ("a", "b"),
    ("abc", "abc"),
    ("ab", "ba"),          # greedy 'match-as-you-scan' classic trap
    ("abcbdab", "bdcaba"), # textbook LCS=4 (bdab / bcba)
    ("aaaa", "aa"),
    ("xyz", "zyx"),
    ("aab", "azb"),        # greedy that grabs the first 'a' can miss
    ("banana", "atana"),
    ("abcde", "edcba"),
]


def rand_string(rng, length, alpha):
    return "".join(rng.choice(alpha) for _ in range(length))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if seed < len(FIXED):
        s, t = FIXED[seed]
        sys.stdout.write(s + "\n" + t + "\n")
        return

    rng = random.Random(seed)

    mode = rng.randint(0, 5)
    if mode == 0:
        # tiny alphabet, small lengths -> brute exhaustive is feasible & greedy-hostile
        alpha = "ab"
        n = rng.randint(1, 14)
        m = rng.randint(1, 14)
        s = rand_string(rng, n, alpha)
        t = rand_string(rng, m, alpha)
    elif mode == 1:
        alpha = "abc"
        n = rng.randint(1, 20)
        m = rng.randint(1, 20)
        s = rand_string(rng, n, alpha)
        t = rand_string(rng, m, alpha)
    elif mode == 2:
        # t is a shuffle of s (same multiset, adversarial order)
        alpha = "abcd"
        n = rng.randint(2, 18)
        s = rand_string(rng, n, alpha)
        lst = list(s)
        rng.shuffle(lst)
        t = "".join(lst)
    elif mode == 3:
        # one short, one longer
        alpha = "abcde"
        s = rand_string(rng, rng.randint(1, 6), alpha)
        t = rand_string(rng, rng.randint(10, 25), alpha)
    elif mode == 4:
        # larger general case (table path in brute; still small enough to be fast)
        alpha = "abcdefgh"
        n = rng.randint(20, 60)
        m = rng.randint(20, 60)
        s = rand_string(rng, n, alpha)
        t = rand_string(rng, m, alpha)
    else:
        # share a hidden common subsequence embedded with noise around it
        alpha = "abcdef"
        core_len = rng.randint(1, 6)
        core = rand_string(rng, core_len, alpha)
        def embed(core):
            out = []
            for ch in core:
                k = rng.randint(0, 3)
                out.append(rand_string(rng, k, alpha))
                out.append(ch)
            out.append(rand_string(rng, rng.randint(0, 3), alpha))
            return "".join(out)
        s = embed(core)
        t = embed(core)

    sys.stdout.write(s + "\n" + t + "\n")


if __name__ == "__main__":
    main()
