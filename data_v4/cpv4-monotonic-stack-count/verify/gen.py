#!/usr/bin/env python3
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 9)
    # Deliberately small value range so EQUAL elements are common -> stresses the
    # leftmost-min tie-break / double-count pitfall.
    vmax = rng.choice([1, 2, 2, 3, 3, 4])
    a = [rng.randint(1, vmax) for _ in range(n)]

    out = [str(n)]
    if a:
        out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
