# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); F = int(next(it))
    # (rest of the instance is unused by this construction)

    # Borrow the entire funding need overnight at the very first maturity. Always
    # feasible (prefunds everything immediately) but concentrates all rollover
    # exposure on a single date -- exactly the checker's own reference baseline.
    p = [0] * T
    p[0] = F
    print(" ".join(map(str, p)))


main()
