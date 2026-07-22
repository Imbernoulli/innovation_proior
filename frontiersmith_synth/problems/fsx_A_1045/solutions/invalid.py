# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    # deliberately infeasible: repeats job 1 N times (not a permutation)
    print(" ".join(["1"] * N))


main()
