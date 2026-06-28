#!/usr/bin/env python3
# Adversarial generator: clusters intervals to stress Hall's condition near the
# feasibility boundary, mixes tiny and wide intervals. Usage: gen2.py <seed>
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    T = rng.randint(1, 12)
    # bias n to hover around T so YES/NO is a near thing
    n = max(0, min(T + 1, rng.randint(0, T) + rng.choice([-1, 0, 0, 1, 2])))
    print(n, T)
    out = []
    for _ in range(n):
        mode = rng.randint(0, 2)
        if mode == 0:  # point interval
            a = rng.randint(1, T); b = a
        elif mode == 1:  # narrow
            a = rng.randint(1, T); b = min(T, a + rng.randint(0, 1))
        else:  # wide
            a = rng.randint(1, T); b = rng.randint(a, T)
        out.append(f"{a} {b}")
    if out:
        print("\n".join(out))

if __name__ == "__main__":
    main()
