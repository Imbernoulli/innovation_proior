import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # tiny cases so the exponential brute force stays fast
    n = rng.randint(0, 10)
    # Mostly small weights so collisions / tight fits are common (stresses the
    # greedy trap: many ties near the limit). Occasionally a "large" regime to
    # exercise the 64-bit boundary (a pair-sum can exceed 32-bit INT_MAX).
    if rng.random() < 0.2:
        maxw = 2 * 10**9
        L = rng.randint(1, 4 * 10**9)
    else:
        maxw = rng.choice([3, 5, 8, 20])
        L = rng.randint(1, 2 * maxw)
    w = [rng.randint(1, maxw) for _ in range(n)]

    out = [f"{n} {L}"]
    out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
