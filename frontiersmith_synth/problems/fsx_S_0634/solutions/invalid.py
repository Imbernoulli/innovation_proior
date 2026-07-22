# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    G = int(next(it)); Tmin = int(next(it)); Tmax = int(next(it)); K = int(next(it))
    out = []
    for _ in range(K):
        for _ in range(G):
            out.append("%d %d" % (Tmax + 1000, Tmin))  # out-of-range teeth -> infeasible
    print("\n".join(out))


main()
