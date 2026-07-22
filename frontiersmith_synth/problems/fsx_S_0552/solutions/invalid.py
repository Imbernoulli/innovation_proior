# TIER: invalid
# Emits a degenerate walk that immediately self-intersects (up then down revisits
# the origin), so the checker's self-avoidance gate rejects it -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    L = int(toks[0])
    if L - 1 <= 0:
        sys.stdout.write("\n")
        return
    # length L-1 but revisits: 2 3 2 3 ... (up,down,...) collapses onto two cells.
    sys.stdout.write(" ".join((["2", "3"] * L)[: L - 1]) + "\n")


if __name__ == "__main__":
    main()
