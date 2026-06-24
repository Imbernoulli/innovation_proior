import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases respecting the contract, biased toward the tricky corners:
    # frequent negatives and zeros, occasional all-negative / all-positive / empty.
    mode = rng.randint(0, 9)
    if mode == 0:
        n = 0
    elif mode == 1:
        n = 1
    else:
        n = rng.randint(0, 8)

    print(n)
    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.30:
            v = -rng.randint(1, 6)        # negative
        elif r < 0.45:
            v = 0                         # zero
        else:
            v = rng.randint(1, 6)         # positive
        # occasionally push a value to a large magnitude to exercise 64-bit sums
        if rng.random() < 0.10:
            v *= rng.randint(1, 10**9 // 6)
        vals.append(v)
    if vals:
        print(" ".join(str(v) for v in vals))

main()
