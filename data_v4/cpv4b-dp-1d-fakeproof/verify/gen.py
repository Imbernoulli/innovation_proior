import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to exercise the parity bucketing and the popcount predicate.
    mode = seed % 5
    if mode == 0:
        n = rng.randint(1, 10)
        hi = 7                       # tiny values, many collisions
    elif mode == 1:
        n = rng.randint(1, 12)
        hi = (1 << 30) - 1           # full 30-bit range
    elif mode == 2:
        n = rng.randint(1, 9)
        hi = 3                       # only 0..3, lots of even/odd popcount mixing
    elif mode == 3:
        n = rng.randint(0, 6)        # include n = 0 and tiny n
        hi = 1                       # only 0 and 1 (popcount 0 or 1)
    else:
        n = rng.randint(1, 14)
        hi = rng.choice([1, 15, 255, (1 << 30) - 1])

    vals = [rng.randint(0, hi) for _ in range(n)]
    out = [str(n)] + [str(v) for v in vals]
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
