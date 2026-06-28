#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Emits: n, then n lines "x y" with small integer coordinates.
# Mixes regimes that stress the geometry:
#   - small coordinate range (forces collinearities, duplicates, ties)
#   - occasional all-collinear / clustered inputs
#   - tiny n (0,1,2) to exercise degenerate output paths
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    regime = rng.randint(0, 9)
    if regime == 0:
        n = rng.randint(0, 2)            # degenerate tiny
    elif regime == 1:
        n = rng.randint(3, 8)            # very small
    else:
        n = rng.randint(3, 40)           # general small

    # coordinate magnitude: keep small so collinearities & duplicates appear often
    if regime in (2, 3):
        C = rng.choice([2, 3, 4])        # very tight -> many ties/collinear
    else:
        C = rng.choice([5, 10, 30, 100])

    pts = []
    if regime == 4 and n >= 1:
        # all points collinear on a random line through origin
        dx = rng.randint(-3, 3); dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx = 1
        for _ in range(n):
            t = rng.randint(-C, C)
            pts.append((t * dx, t * dy))
    elif regime == 5 and n >= 1:
        # heavy duplicates: pick from a tiny pool
        pool = [(rng.randint(-C, C), rng.randint(-C, C)) for _ in range(max(1, n // 3))]
        for _ in range(n):
            pts.append(rng.choice(pool))
    else:
        for _ in range(n):
            pts.append((rng.randint(-C, C), rng.randint(-C, C)))

    out = [str(n)]
    for (x, y) in pts:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
