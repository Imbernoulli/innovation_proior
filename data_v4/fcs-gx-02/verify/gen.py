#!/usr/bin/env python3
"""Random small-case generator for fcs-gx-02.

Usage: python3 gen.py <seed>
Emits a case on stdout in the solver's stdin format:
    s
    k
Small n (<= 14) so the exponential brute force stays tractable. A deliberately
small alphabet makes equal-character runs and zigzags common, which is where the
monotonic-stack pop logic is most easily wrong. k ranges over 0..n (including the
"delete everything" boundary).
"""
import random
import sys


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of tiny and small lengths. |s| >= 1 (the empty-string input case is
    # excluded by the contract; an empty *result* via k>=n is still exercised).
    r = rng.random()
    if r < 0.12:
        n = 1
    elif r < 0.32:
        n = rng.randint(2, 5)
    else:
        n = rng.randint(6, 14)

    # Alphabet size: small alphabets create lots of ties/zigzags.
    alpha_size = rng.choice([1, 2, 2, 3, 3, 4, 5, 10])
    alphabet = "abcdefghij"[:alpha_size]

    # Occasionally bias toward digit strings (classic "remove k digits").
    if rng.random() < 0.25:
        alphabet = "0123456789"[: rng.choice([2, 3, 10])]

    s = "".join(rng.choice(alphabet) for _ in range(n))

    # k can be 0..n (n meaning delete everything). Bias toward interesting middle.
    rk = rng.random()
    if rk < 0.15:
        k = 0
    elif rk < 0.30:
        k = n            # delete all -> empty result
    else:
        k = rng.randint(0, n)

    sys.stdout.write(s + "\n")
    sys.stdout.write(str(k) + "\n")


if __name__ == "__main__":
    main()
