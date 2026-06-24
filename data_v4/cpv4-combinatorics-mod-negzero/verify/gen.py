import sys
import random

# Small-case generator parameterized by an integer seed.
# Keeps n tiny so the O(2^n) brute force stays fast, and deliberately
# oversamples zeros, negatives, all-negative arrays, and small moduli to
# exercise the base-case / sign / empty corners.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 12)

    # Pick a "flavor" so we hit the corner distributions often.
    flavor = random.randint(0, 6)
    a = []
    for _ in range(n):
        if flavor == 0:      # all negative
            v = -random.randint(1, 9)
        elif flavor == 1:    # all zero
            v = 0
        elif flavor == 2:    # negatives and zeros only (no positive product possible)
            v = random.choice([0, -random.randint(1, 9)])
        elif flavor == 3:    # all positive
            v = random.randint(1, 9)
        elif flavor == 4:    # heavy on zeros
            v = random.choice([0, 0, 0, random.randint(-9, 9)])
        else:                # fully mixed, includes big magnitudes
            v = random.choice(
                [0, random.randint(-9, 9), random.randint(-1000000000, 1000000000)]
            )
        a.append(v)

    # Modulus: oversample tiny moduli (1, 2, ...) plus a big prime and a big composite.
    m = random.choice([1, 1, 2, 3, 4, 7, 10, 999983, 1000000000, 1000000007, 998244353])

    print(n, m)
    print(" ".join(str(x) for x in a))

if __name__ == "__main__":
    main()
