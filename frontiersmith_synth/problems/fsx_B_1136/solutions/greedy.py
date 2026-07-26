# TIER: greedy
# The "obvious" first attempt: build one basin PER CLASS, in the order the classes
# appear in the input (class id 1..7), each basin sized just enough to reach that
# class's own residence-time threshold, using one fixed "reasonable-looking" depth
# for every basin. This never checks whether input order already matches threshold
# order, and never tunes depth per class for scour safety -- it treats the cascade
# as 7 independent sizing problems rather than a single ordered cut sequence.
import sys

FIXED_DEPTH = 4


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    Q = int(next(it)); VolumeBudget = int(next(it)); M = int(next(it))
    Dmax = int(next(it)); Lmax = int(next(it))
    PenNum = int(next(it)); PenDen = int(next(it))
    thr, mass, val, scour = {}, {}, {}, {}
    for c in range(1, 8):
        thr[c] = int(next(it)); mass[c] = int(next(it))
        val[c] = int(next(it)); scour[c] = int(next(it))

    d = min(FIXED_DEPTH, Dmax)
    basins = []
    total_vol = 0
    for c in range(1, 8):
        if len(basins) >= M:
            break
        need = Q * thr[c]
        l = -(-need // d)
        if l < 1:
            l = 1
        if l > Lmax:
            l = Lmax
        vol = d * l
        if total_vol + vol > VolumeBudget:
            continue
        total_vol += vol
        basins.append((d, l, c))

    print(len(basins))
    for (dd, ll, bb) in basins:
        print(dd, ll, bb)


if __name__ == "__main__":
    main()
