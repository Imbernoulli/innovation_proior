# TIER: invalid
# Emits an infeasible placement: n copies of the same mile-post (not distinct).
# The checker must reject it -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    print(" ".join(["0"] * n))


if __name__ == "__main__":
    main()
