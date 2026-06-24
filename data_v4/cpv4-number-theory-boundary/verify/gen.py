import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # small bounds so brute force is feasible and the L-1 / inclusive-R
    # boundary gets exercised at the edges (L == 1, L == R, etc.)
    m = rng.randint(1, 60)
    q = rng.randint(1, 6)
    MAXV = 80

    lines = [f"{m} {q}"]
    for _ in range(q):
        a = rng.randint(1, MAXV)
        b = rng.randint(1, MAXV)
        L, R = min(a, b), max(a, b)
        # deliberately bias toward boundary-heavy cases sometimes
        r = rng.random()
        if r < 0.25:
            L = 1
        elif r < 0.5:
            R = L  # single-point window
        lines.append(f"{L} {R}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
