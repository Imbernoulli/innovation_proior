#!/usr/bin/env python3
"""Random + edge-case generator for 'minimum falling path sum'.

Usage: gen.py <seed> [mode]

Modes (chosen by seed if not given):
  tiny      n in {0,1,2}
  small     n in {1..5}, small values incl. negatives
  negatives n in {1..6}, all-negative or mixed values
  wide      n in {1..7}, wide value range
  default   n in {1..6}, generic mixed

Prints a grid in the stdin format expected by sol.cpp / brute.py:
  n
  then n rows, each with n integers.

n is kept small so the exponential brute oracle stays cheap.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    mode = sys.argv[2] if len(sys.argv) > 2 else rng.choice(
        ["tiny", "small", "negatives", "wide", "default"]
    )

    if mode == "tiny":
        n = rng.choice([0, 1, 1, 2, 2])
        lo, hi = -9, 9
    elif mode == "small":
        n = rng.randint(1, 5)
        lo, hi = -20, 20
    elif mode == "negatives":
        n = rng.randint(1, 6)
        lo, hi = -50, 0
    elif mode == "wide":
        n = rng.randint(1, 7)
        lo, hi = -1000, 1000
    else:  # default
        n = rng.randint(1, 6)
        lo, hi = -30, 30

    out = [str(n)]
    for _ in range(n):
        row = [str(rng.randint(lo, hi)) for _ in range(n)]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
