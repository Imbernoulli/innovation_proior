import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes so the negative/zero/empty corners are exercised often.
    r = rng.random()
    if r < 0.15:
        n = rng.randint(0, 1)          # empty / single point -> NONE corner
    elif r < 0.30:
        n = rng.randint(2, 4)          # tiny pair counts
    else:
        n = rng.randint(2, 8)

    # Coordinate ranges chosen to produce negatives, zeros, and collinear ties.
    mode = rng.randint(0, 3)
    if mode == 0:
        lo, hi = -3, 3                 # lots of zeros and sign flips
    elif mode == 1:
        lo, hi = -1, 1                 # heavy zero / collinear pressure
    elif mode == 2:
        lo, hi = -10, 0                # all-nonpositive coordinates
    else:
        lo, hi = -1000, 1000

    pts = []
    for _ in range(n):
        pts.append((rng.randint(lo, hi), rng.randint(lo, hi)))

    out = [str(n)]
    for (xx, yy) in pts:
        out.append(f"{xx} {yy}")
    sys.stdout.write("\n".join(out) + "\n")

main()
