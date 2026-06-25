import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed * 1000 + 7)
    n = rng.randint(0, 16)
    mode = rng.randint(0, 2)
    if mode == 0:
        lo, hi = 0, 9          # non-negative
    elif mode == 1:
        lo, hi = -6, 9         # mixed
    else:
        lo, hi = -9, -1        # all negative
    vals = [rng.randint(lo, hi) for _ in range(n)]
    print(n)
    if n > 0:
        print(" ".join(map(str, vals)))

if __name__ == "__main__":
    main()
