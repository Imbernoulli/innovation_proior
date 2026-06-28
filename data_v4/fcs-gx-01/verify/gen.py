#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Emits a valid instance: first line n, then n lines "t w".
# Small n (<= 8) and small values so brute.py (n!) stays fast, while still
# exercising ties, equal ratios, extreme t/w skew, and zero-free positives.
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 8)

    # Mix value regimes so equal-ratio ties and t-vs-w skew both appear often.
    mode = rng.randint(0, 3)
    lines = [str(n)]
    for _ in range(n):
        if mode == 0:
            t = rng.randint(1, 5)
            w = rng.randint(1, 5)
        elif mode == 1:
            # force frequent equal ratios via a small shared multiplier set
            base_t = rng.choice([1, 2, 3])
            base_w = rng.choice([1, 2, 3])
            k = rng.randint(1, 4)
            t = base_t * k
            w = base_w * k
        elif mode == 2:
            # skewed: large processing time vs tiny weight and vice versa
            t = rng.choice([1, 1, 100])
            w = rng.choice([1, 1, 100])
        else:
            t = rng.randint(1, 50)
            w = rng.randint(1, 50)
        lines.append(f"{t} {w}")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
