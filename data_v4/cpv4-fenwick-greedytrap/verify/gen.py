import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the exponential brute force stays fast.
    n = rng.randint(0, 12)

    # Mix of value regimes to stress the greedy trap (ties, duplicates,
    # negatives, big positives next to small positives).
    mode = rng.randint(0, 3)
    if mode == 0:
        lo, hi = -6, 6           # small range, many ties / duplicates
    elif mode == 1:
        lo, hi = 0, 20           # all non-negative
    elif mode == 2:
        lo, hi = -20, -1         # all negative
    else:
        lo, hi = -50, 50         # wide range

    vals = [rng.randint(lo, hi) for _ in range(n)]

    out = [str(n)]
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
