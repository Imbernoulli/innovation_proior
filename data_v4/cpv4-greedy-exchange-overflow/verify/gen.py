import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Tiny case so the O(n!) brute force is feasible (n! over many trials).
    n = rng.randint(0, 7)
    # Small values keep brute fast but still exercise ties and ratio orderings.
    # Occasionally use larger values to vary the ratios; correctness is value-agnostic.
    hi = rng.choice([3, 5, 10, 50])
    p = [rng.randint(1, hi) for _ in range(n)]
    w = [rng.randint(1, hi) for _ in range(n)]

    out = [str(n)]
    out.append(" ".join(map(str, p)))
    out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
