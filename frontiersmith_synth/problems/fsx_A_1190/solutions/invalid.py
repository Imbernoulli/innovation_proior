# TIER: invalid
"""Deliberately infeasible: negative endmember entries and abundances that don't sum to 1.
Must score Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    _t = int(next(it))
    R = int(next(it)); K = int(next(it)); N = int(next(it))
    for _ in range(N * R):
        next(it)  # consume Y, unused

    out = []
    for _k in range(K):
        out.append(" ".join("-1.0" for _ in range(R)))
    for _j in range(N):
        out.append(" ".join("1.0" for _ in range(K)))  # sums to K, not 1
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
