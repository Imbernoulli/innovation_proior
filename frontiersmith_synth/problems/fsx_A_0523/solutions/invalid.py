# TIER: invalid
# Emits an all-zero schedule: demand is never met -> checker must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0]); K = int(toks[1])
    line = " ".join(["0.0"] * K)
    sys.stdout.write("\n".join([line] * T) + "\n")


main()
