#!/usr/bin/env python3
# Random small-case generator: python3 gen.py <seed>
# Keeps n small enough for the permutation brute force, but pushes values high
# enough that the product overflows 64-bit (forcing the __int128 / exact path).
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mode selection to exercise different regimes.
    r = rng.random()
    if r < 0.15:
        n = rng.choice([0, 1, 2])
    elif r < 0.45:
        n = rng.randint(2, 5)          # tiny, dense small values
    else:
        n = rng.randint(5, 8)          # larger, adversarial big values & ties

    # Value ranges: sometimes small (exercise ties / 1s), sometimes near the cap
    # (force 64-bit overflow on full products).
    big = rng.random() < 0.6
    if big:
        vlo, vhi = 30, 60
    else:
        vlo, vhi = 1, 6

    out = [str(n)]
    if n == 0:
        # Still emit a (degenerate) well-formed body of zero values.
        print("\n".join(out))
        return

    b = [rng.randint(vlo, vhi) for _ in range(n)]
    out.append(" ".join(map(str, b)))
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(str(rng.randint(vlo, vhi)))  # diagonal unused but present
            else:
                # Occasionally inject equal values to create exact ties that a
                # lossy (double/log) comparison could mis-resolve.
                if rng.random() < 0.25:
                    row.append(str(vhi))
                else:
                    row.append(str(rng.randint(vlo, vhi)))
        out.append(" ".join(row))
    print("\n".join(out))

if __name__ == "__main__":
    main()
