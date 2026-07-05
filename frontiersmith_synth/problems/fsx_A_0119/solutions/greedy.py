# TIER: greedy
# Anchors + a mediocre rank-1 lattice (coarse generator) for the free points.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    dim = int(next(it)); M = int(next(it)); K = int(next(it))
    anchors = [(float(next(it)), float(next(it))) for _ in range(K)]
    pts = list(anchors)
    rem = M - K
    g = 3.0 / 8.0  # mediocre generator
    for j in range(rem):
        x = (j + 0.5) / rem if rem > 0 else 0.5
        y = ((j * g) % 1.0)
        pts.append((x, y))
    print("\n".join("%.6f %.6f" % (x, y) for (x, y) in pts))

if __name__ == "__main__":
    main()
