import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Tiny cases so the brute-force 2^n enumeration is cheap.
    n = rng.randint(0, 12)

    # Pick a small capacity. Sometimes 0 (empty-subset corner), sometimes
    # larger than anything reachable (IMPOSSIBLE corner).
    mode = rng.randint(0, 4)
    if mode == 0:
        C = 0
    elif mode == 1:
        C = rng.randint(0, 30)        # often reachable
    else:
        C = rng.randint(0, 20)

    print(n, C)
    for _ in range(n):
        w = rng.randint(1, 8)         # weights are positive
        # Values include negatives, zeros, and positives on purpose.
        v = rng.randint(-9, 9)
        print(w, v)

if __name__ == "__main__":
    main()
