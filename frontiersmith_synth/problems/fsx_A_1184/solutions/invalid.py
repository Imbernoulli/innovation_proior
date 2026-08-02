# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    testId = int(next(it)); T = int(next(it)); M = int(next(it))
    # emit non-finite garbage for every slot -- must be rejected before scoring
    print(" ".join("nan" for _ in range(M)))


main()
