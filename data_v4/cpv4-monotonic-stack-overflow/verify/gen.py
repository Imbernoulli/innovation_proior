import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes: tiny n, ties-heavy small-value arrays (stress tie-breaking),
    # and a few large-value arrays.
    mode = seed % 4
    if mode == 0:
        n = rng.randint(0, 6)
        vlo, vhi = -5, 5            # negatives + ties
    elif mode == 1:
        n = rng.randint(1, 10)
        vlo, vhi = 1, 3             # heavy ties
    elif mode == 2:
        n = rng.randint(1, 12)
        vlo, vhi = -10**9, 10**9   # large magnitudes
    else:
        n = rng.randint(1, 8)
        vlo, vhi = 0, 4

    a = [rng.randint(vlo, vhi) for _ in range(n)]
    out = [str(n)]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
