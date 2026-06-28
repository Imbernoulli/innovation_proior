import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Pick a "shape" regime so we cover degenerate cases the hull/calipers must
    # survive: tiny n, all-identical, collinear, on a small lattice (forces
    # duplicates + collinear triples), and generic.
    regime = rng.randint(0, 6)

    if regime == 0:
        n = rng.randint(2, 4)            # tiny
        coord = lambda: rng.randint(-3, 3)
        pts = [(coord(), coord()) for _ in range(n)]
    elif regime == 1:
        n = rng.randint(2, 30)           # all identical
        x = rng.randint(-5, 5); y = rng.randint(-5, 5)
        pts = [(x, y) for _ in range(n)]
    elif regime == 2:
        n = rng.randint(2, 40)           # collinear (random line through lattice)
        # line: points (t, a*t + b) clamped to ints via a small slope
        a = rng.randint(-2, 2); b = rng.randint(-4, 4)
        pts = []
        for _ in range(n):
            t = rng.randint(-6, 6)
            pts.append((t, a * t + b))
    elif regime == 3:
        n = rng.randint(2, 40)           # vertical line (x constant) edge case
        x = rng.randint(-5, 5)
        pts = [(x, rng.randint(-7, 7)) for _ in range(n)]
    elif regime == 4:
        n = rng.randint(2, 60)           # small lattice: many dups + collinear
        pts = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(n)]
    elif regime == 5:
        n = rng.randint(2, 60)           # generic, moderate range
        pts = [(rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(n)]
    else:
        n = rng.randint(2, 80)           # wider range, sparse
        pts = [(rng.randint(-1000, 1000), rng.randint(-1000, 1000)) for _ in range(n)]

    out = [str(len(pts))]
    for (x, y) in pts:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")

main()
