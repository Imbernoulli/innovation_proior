#!/usr/bin/env python3
# Larger small-case generator (n up to ~40) for deeper structural stress;
# still within reach of the O(n^2) reachability-removal brute.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(2, 40)
    # bias toward sparser graphs (more interesting dominator structure)
    mode = rng.randint(0, 2)
    if mode == 0:          # sparse
        m = rng.randint(0, 2 * n)
    elif mode == 1:        # medium
        m = rng.randint(n, 4 * n)
    else:                  # dense
        m = rng.randint(0, n * n)
    s = rng.randint(1, n)
    out = [f"{n} {m} {s}"]
    for _ in range(m):
        a = rng.randint(1, n)
        b = rng.randint(1, n)
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
