# TIER: invalid
"""Emits a malformed schedule (contracts a tensor with itself), which the
checker rejects -> score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    m = int(toks[0]) if toks else 2
    lines = []
    for _ in range(max(1, m - 1)):
        lines.append("0 0")  # a==b is never a valid contraction
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
