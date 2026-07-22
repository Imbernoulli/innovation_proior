# TIER: trivial
# Baseline "star": wire every organ straight to the arterial source. No shared
# trunks, no junctions -- exactly the construction the checker uses as its
# internal baseline B, so this scores about 0.1.
import sys


def main():
    t = sys.stdin.read().split()
    it = iter(t)
    K = int(next(it)); M = int(next(it))
    next(it); next(it)          # Wm, Wd
    next(it); next(it)          # source x,y
    # skip the rest; we only need K
    out = []
    out.append("0")             # no steiner points
    for _ in range(K):
        out.append("0")         # every organ's parent is the source
    sys.stdout.write("\n".join(out) + "\n")


main()
