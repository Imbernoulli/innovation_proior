# TIER: invalid
"""Emits a wall count that is silently wrong (W-1 lines when W are
required) plus a degenerate zero-length "line" -- must score 0 under
strict feasibility checking."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); K = int(next(it)); _tid = int(next(it))
    Sx = float(next(it)); Sy = float(next(it))

    out = [str(max(0, W - 1))]
    for _ in range(max(0, W - 1)):
        out.append("%.6f %.6f %.6f %.6f" % (Sx, Sy, Sx, Sy))  # degenerate zero-length line
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
