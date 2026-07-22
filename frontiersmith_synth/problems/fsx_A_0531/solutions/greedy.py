# TIER: greedy
# The obvious approach: "fold it in half each time" -- always fold at the middle
# line so the footprint shrinks as fast as possible. This front-loads thickness
# (it folds the heavy centre early) and blindly bends any central reinforced
# crease. Reaches the target but pays for carrying big stacks.
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); T = int(next(it)); R = int(next(it)); W = int(next(it))
    folds = []
    w = N
    while w > T:
        k = w // 2
        if k < 1:
            k = 1
        folds.append(k)
        w = w - min(k, w - k)
    print(len(folds))
    if folds:
        print(" ".join(str(x) for x in folds))


if __name__ == "__main__":
    main()
