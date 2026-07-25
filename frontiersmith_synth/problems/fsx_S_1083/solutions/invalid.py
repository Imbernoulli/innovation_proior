# TIER: invalid
# Emits an infeasible artifact: entries far above the pmax bound.
import sys


def main():
    data = sys.stdin.read().split()
    n, m = int(data[0]), int(data[1])
    row = " ".join(["999999"] * n)
    sys.stdout.write("\n".join([row] * m) + "\n")


main()
