# TIER: invalid
import sys

def main():
    d = iter(sys.stdin.read().split())
    W = int(next(d)); H = int(next(d)); R = int(next(d))
    for _ in range(R):
        next(d); next(d)
    P = int(next(d))
    c = int(next(d)); s = int(next(d))
    shape = []
    for _ in range(s):
        shape.append((int(next(d)), int(next(d))))
    # Install one copy of type 0 but deliberately push a cell out of the grid
    # (right of column W) -> infeasible, must score 0.
    cells = [(x + W + 5, y) for (x, y) in shape]
    parts = ["0"]
    for (x, y) in cells:
        parts.append(str(x)); parts.append(str(y))
    sys.stdout.write("1\n" + " ".join(parts) + "\n")

main()
