import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the brute force is fast and corners are hit often.
    n = rng.randint(1, 12)
    # w sometimes exceeds n (no valid batch), sometimes equals n, often < n.
    w = rng.randint(1, n + 2)

    # Weights: allow negatives and zeros so band logic is exercised.
    a = [rng.randint(-5, 9) for _ in range(n)]

    # Choose a band that frequently brackets actual batch sums.
    lo = rng.randint(-20, 20)
    hi = lo + rng.randint(0, 25)
    # Occasionally make a degenerate band (L == R) to stress exact equality.
    if rng.random() < 0.25:
        hi = lo

    out = []
    out.append(f"{n} {w} {lo} {hi}")
    out.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(out) + "\n")

main()
