# TIER: invalid
"""Emits an infeasible grid (entries out of {-1,+1}), must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = [" ".join("2" for _ in range(N)) for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
