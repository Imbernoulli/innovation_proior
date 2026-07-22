# TIER: invalid
# Blatantly infeasible: push arm 0 to speed floor(sqrt(P))+1 on tick one.
# By definition of floor(sqrt(P)), (floor(sqrt(P))+1)^2 > P, so this single
# entry alone already blows the shared power cap on its own tick,
# regardless of K or how generous P is.
import math
import sys


def main():
    data = sys.stdin.read().split("\n")
    K, P, A = (int(x) for x in data[0].split())
    D = [int(x) for x in data[1].split()]

    vmax = math.isqrt(P)
    bad_speed = vmax + 1

    row = [0] * K
    row[0] = bad_speed
    print(" ".join(map(str, row)))


if __name__ == "__main__":
    main()
