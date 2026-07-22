# TIER: invalid
# Emit an infeasible artifact (cell index out of range) -> checker must score 0.
import sys


def main():
    d = sys.stdin.read().split()
    N = int(d[0]); K = int(d[1])
    print("\n".join("%d 1" % (N + 5) for _ in range(K)))


main()
