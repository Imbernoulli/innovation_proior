import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the O(m^3) brute force stays fast, but deliberately use a
    # tiny coordinate range so columns/rows collide a lot (the counting regime),
    # and allow EXACT-duplicate points so the dedup pitfall is exercised.
    n = rng.randint(0, 9)
    coord_hi = rng.choice([1, 2, 3, 4])  # small => many shared rows/cols
    allow_dups = rng.random() < 0.5

    pts = []
    for _ in range(n):
        x = rng.randint(-coord_hi, coord_hi)
        y = rng.randint(-coord_hi, coord_hi)
        pts.append((x, y))

    if not allow_dups:
        # still may collide naturally; nothing special
        pass

    lines = [str(n)]
    for (x, y) in pts:
        lines.append(f"{x} {y}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
