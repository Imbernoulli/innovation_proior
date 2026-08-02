# TIER: invalid
"""Deliberately infeasible: emits an out-of-range neighbour index (N, which
is never a valid 0-indexed point) so the checker must reject it."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); R = int(next(it))

    out = []
    out.append(" ".join(str(k) for k in range(R)))
    for i in range(N):
        # one out-of-range neighbour (index N is invalid for an N-point set)
        out.append(f"1 {N}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
