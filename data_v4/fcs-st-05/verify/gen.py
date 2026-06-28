#!/usr/bin/env python3
"""Random small-case generator for the least-rotation problem.

Usage: gen.py <seed>

Emits one line: a non-empty string over a small alphabet. Small alphabets and
short lengths are deliberately chosen to maximize ties and repeated blocks,
which are the cases that stress Booth's "skip a whole block" step.
"""
import random
import sys


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to exercise ties, periodicity, and single chars.
    regime = rng.randint(0, 4)
    if regime == 0:
        alpha_size = 2          # heavy ties / borders
    elif regime == 1:
        alpha_size = 3
    elif regime == 2:
        alpha_size = 1          # all-equal string
    else:
        alpha_size = rng.randint(2, 5)

    alphabet = "abcde"[:alpha_size]

    # Length up to 30 keeps brute force fast while covering structure.
    n = rng.randint(1, 30)

    if regime == 4:
        # Build a periodic string: repeat a small block, occasionally perturbed.
        block_len = rng.randint(1, max(1, n // 2))
        block = "".join(rng.choice(alphabet) for _ in range(block_len))
        s = (block * (n // block_len + 1))[:n]
        s = list(s)
        if rng.random() < 0.4 and n > 0:
            s[rng.randrange(n)] = rng.choice(alphabet)
        s = "".join(s)
    else:
        s = "".join(rng.choice(alphabet) for _ in range(n))

    print(s)


if __name__ == "__main__":
    main()
