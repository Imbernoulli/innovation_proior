# TIER: trivial
# Punch every target hole individually with zero folds (the checker's own baseline).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it))
    cells = []
    for _ in range(H):
        i = int(next(it)); j = int(next(it))
        cells.append((i, j))

    out = []
    for (i, j) in cells:
        out.append("PUNCH %d %d" % (i, j))
    out.append("UNFOLD_ALL")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
