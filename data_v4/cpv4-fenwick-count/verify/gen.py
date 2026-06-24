import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 9)
    # Small value range so equal values and band overlaps are common -- exactly
    # where the L==0 double-count and the inclusive-range off-by-one bite.
    vmax = rng.choice([2, 3, 5, 8])
    a = [rng.randint(-vmax, vmax) for _ in range(n)]

    R = rng.randint(0, 2 * vmax)
    L = rng.randint(0, R)
    # Bias hard toward L==0 to stress the merged-band / overlap path.
    if rng.random() < 0.5:
        L = 0

    out = [f"{n} {L} {R}"]
    if n > 0:
        out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
