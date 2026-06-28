#!/usr/bin/env python3
"""Random + edge-case generator for the wildcard matching problem.

Usage: gen.py SEED [MODE]
Prints two whitespace-separated tokens: pattern and string.
Empty pattern/string are emitted as the literal token "-".

Modes:
  rand     : small random pattern/string over a tiny alphabet (default mix)
  stars    : pattern heavy in '*' (stress backtracking / DP empty runs)
  derived  : string derived from a base, pattern built to (often) match it
  edge     : hand-picked corner cases keyed by seed
"""
import sys
import random


def tok(x: str) -> str:
    return x if x != "" else "-"


def gen_rand(rng, pat_alpha, str_alpha):
    pn = rng.randint(0, 8)
    sn = rng.randint(0, 8)
    p = "".join(rng.choice(pat_alpha) for _ in range(pn))
    s = "".join(rng.choice(str_alpha) for _ in range(sn))
    return p, s


def gen_stars(rng):
    # Pattern dominated by '*' and '?' to stress backtracking.
    pat_alpha = ['a', 'b', '*', '*', '*', '?']
    str_alpha = ['a', 'b']
    pn = rng.randint(1, 10)
    sn = rng.randint(0, 10)
    p = "".join(rng.choice(pat_alpha) for _ in range(pn))
    s = "".join(rng.choice(str_alpha) for _ in range(sn))
    return p, s


def gen_derived(rng):
    # Build a string, then a pattern that should usually match it,
    # so we exercise plenty of YES cases (not just random NOs).
    str_alpha = ['a', 'b', 'c']
    sn = rng.randint(0, 8)
    s = "".join(rng.choice(str_alpha) for _ in range(sn))
    p_parts = []
    i = 0
    while i < len(s):
        r = rng.random()
        if r < 0.4:
            p_parts.append(s[i])         # literal
            i += 1
        elif r < 0.6:
            p_parts.append('?')          # any single
            i += 1
        elif r < 0.8:
            p_parts.append('*')          # consume a run
            i += rng.randint(1, max(1, len(s) - i))
        else:
            p_parts.append('*')          # consume nothing (zero-length)
    if rng.random() < 0.5:
        p_parts.append('*')              # trailing star
    p = "".join(p_parts)
    return p, s


def gen_edge(rng, seed):
    cases = [
        ("", ""),
        ("*", ""),
        ("", "a"),
        ("a", ""),
        ("?", ""),
        ("?", "a"),
        ("*", "abc"),
        ("a*", "a"),
        ("*a", "a"),
        ("*a*", "ba"),
        ("a*b", "ab"),
        ("a*b", "axxxb"),
        ("a*b", "axxxc"),
        ("**", "ab"),
        ("*?*", "a"),
        ("*?*", ""),
        ("?*", ""),
        ("?*", "x"),
        ("a?c", "abc"),
        ("a?c", "abbc"),
        ("*a*a*a*", "aaa"),
        ("*a*a*a*", "aa"),
        ("a", "a"),
        ("a", "b"),
        ("abc", "abc"),
        ("ab*cd", "abXYZcd"),
        ("ab*cd", "abcd"),
        ("ab*cd", "abc"),
    ]
    return cases[seed % len(cases)]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "rand":
        pat_alpha = ['a', 'b', 'c', '?', '*']
        str_alpha = ['a', 'b', 'c']
        p, s = gen_rand(rng, pat_alpha, str_alpha)
    elif mode == "stars":
        p, s = gen_stars(rng)
    elif mode == "derived":
        p, s = gen_derived(rng)
    elif mode == "edge":
        p, s = gen_edge(rng, seed)
    else:
        p, s = gen_rand(rng, ['a', 'b', '?', '*'], ['a', 'b'])

    print(tok(p), tok(s))


if __name__ == "__main__":
    main()
