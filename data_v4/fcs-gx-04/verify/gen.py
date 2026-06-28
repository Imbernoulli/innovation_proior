#!/usr/bin/env python3
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    # Keep coordinates small so the brute-force grid scan stays feasible.
    n = rnd.randint(0, 12)
    coord_lo, coord_hi = -8, 8
    w_hi = rnd.choice([1, 1, 3, 7])  # mostly small weights, sometimes larger

    lines = [str(n)]
    for _ in range(n):
        x = rnd.randint(coord_lo, coord_hi)
        y = rnd.randint(coord_lo, coord_hi)
        w = rnd.randint(1, w_hi)
        lines.append(f"{x} {y} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
