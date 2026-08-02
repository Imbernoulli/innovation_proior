# TIER: invalid
"""Deliberately infeasible artifact: illegal mode tokens and a non-finite
adaptive threshold. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[2])
    lines = []
    for i in range(L):
        if i % 2 == 0:
            lines.append("ADAPT 3 nan")
        else:
            lines.append("BROADCAST 1 0.0")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
