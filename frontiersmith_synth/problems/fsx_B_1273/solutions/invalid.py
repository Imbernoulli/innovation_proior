# TIER: invalid
"""Emits a garbage / infeasible policy grid: correct token count but with
non-finite and out-of-range values -- must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    lines = []
    lines.append("nan nan nan nan nan")
    for t in range(2, T + 1):
        lines.append("1.7 -0.3 2.5 inf -1.0")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
