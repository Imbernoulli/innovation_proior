import sys
import random

# Random SMALL-case generator for the bounded-composition counting problem.
# Usage: python3 gen.py <seed>
# Emits one line: k c S M
# M is chosen to be a prime strictly larger than any binomial argument the
# intended solution touches (its top argument is S + k - 1 < S + k), so the
# factorial-inverse table is always valid. We pick a big prime for most cases
# and a small prime that still clears S+k occasionally to stress modular reduction.

PRIMES_SMALL = [2, 3, 5, 7, 11, 13, 1009, 100003]
BIG_PRIME = 1000000007

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    k = rng.randint(0, 8)
    c = rng.randint(0, 8)
    Smax = max(1, k * c)
    choice = rng.random()
    if choice < 0.22:
        S = 0
    elif choice < 0.40 and k > 0:
        S = k * c                       # all children maxed: exactly one solution
    elif choice < 0.55:
        S = rng.randint(0, Smax + 3)    # may exceed k*c -> answer 0
    elif choice < 0.78 and c >= 0:
        # land near a multiple of (c+1) to exercise the inclusion-exclusion boundary
        step = c + 1
        j = rng.randint(0, k) if k > 0 else 0
        S = j * step + rng.randint(0, 2)
    else:
        S = rng.randint(0, Smax)

    top = S + k                         # strict upper bound on any binomial "top"
    if rng.random() < 0.5:
        M = BIG_PRIME
    else:
        cand = [p for p in PRIMES_SMALL if p > top]
        M = cand[0] if cand else BIG_PRIME

    print(k, c, S, M)

if __name__ == "__main__":
    main()
