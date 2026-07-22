# TIER: invalid
# Emits an infeasible artifact: K copies of the same out-of-range node id (violates both
# the distinct-sites rule and the [0,N) range check). Must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    K = int(toks[2])
    print(" ".join(str(N + 5) for _ in range(K)))


if __name__ == "__main__":
    main()
