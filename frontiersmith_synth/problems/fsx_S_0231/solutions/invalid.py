# TIER: invalid
# Deliberately infeasible: deploy one type-0 cabinet with a cell placed far outside the
# grid, which the checker must reject (score 0.0).
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); H = int(next(it))
    R = int(next(it))
    for _ in range(R):
        next(it); next(it)
    for _ in range(W * H):
        next(it)
    P = int(next(it))
    c0 = int(next(it)); s0 = int(next(it))
    cells = [(int(next(it)), int(next(it))) for _ in range(s0)]

    # emit s0 cells but push them out of bounds (guaranteed infeasible)
    coords = " ".join("%d %d" % (10000 + dx, 10000 + dy) for (dx, dy) in cells)
    sys.stdout.write("1\n0 " + coords + "\n")

if __name__ == "__main__":
    main()
