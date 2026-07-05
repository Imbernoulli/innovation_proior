# TIER: trivial
# Anchors + all free points clustered at the origin -- reproduces the
# checker's baseline plan (bake the same corner recipe repeatedly).
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    dim = int(next(it)); M = int(next(it)); K = int(next(it))
    anchors = [(float(next(it)), float(next(it))) for _ in range(K)]
    pts = list(anchors)
    rem = M - K
    for _ in range(rem):
        pts.append((0.0, 0.0))
    print("\n".join("%.6f %.6f" % (x, y) for (x, y) in pts))

if __name__ == "__main__":
    main()
