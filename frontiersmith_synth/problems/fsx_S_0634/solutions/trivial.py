# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    G = int(next(it)); Tmin = int(next(it)); Tmax = int(next(it)); K = int(next(it))
    for _ in range(K):
        next(it); next(it); next(it)  # skip P Q lambda -- never even looked at

    mid = (Tmin + Tmax) // 2
    out = []
    for _ in range(K):
        for _ in range(G):
            out.append("%d %d" % (mid, mid))
    print("\n".join(out))


main()
