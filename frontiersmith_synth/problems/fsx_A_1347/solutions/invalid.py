# TIER: invalid
"""Emits a garbage / infeasible artifact: colors clamped out of [1,K]."""
import sys


def main():
    data = sys.stdin.read().split()
    ti = 0
    n = int(data[ti]); ti += 1
    m = int(data[ti]); ti += 1
    K = int(data[ti]); ti += 1

    out = []
    out.append(" ".join(str(K + 50) for _ in range(n)))  # every color out of range
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
