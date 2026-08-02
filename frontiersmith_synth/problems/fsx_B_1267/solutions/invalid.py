# TIER: invalid
"""Infeasible: claims to investigate every claim while pretending the
budget doesn't exist (also throws in an out-of-range and a duplicate
index), so the checker must reject it outright."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))

    idxs = list(range(N)) + [N + 5, 0]  # out-of-range + duplicate on top of overspending
    out = [str(len(idxs))] + [str(i) for i in idxs]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
