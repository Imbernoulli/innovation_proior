# TIER: invalid
"""Emits out-of-range garbage: coordinates outside [0,1]. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    d, M = int(data[0]), int(data[1])
    out = []
    for i in range(M):
        # deliberately infeasible: values well outside [0,1]
        out.append(" ".join(str(7.0 + i) for _ in range(d)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
