# TIER: trivial
# Build exactly ONE basin: sized for the single class with the smallest residence-time
# threshold, with a scour-safe depth. This is the "solve the easiest sub-problem, ignore
# the other six" baseline -- it matches the checker's own internal reference construction.
import sys


def find_basin(Q, thr_lo, scour_c, Dmax, Lmax):
    d_min = max(1, -(-Q // scour_c))
    for d in range(d_min, Dmax + 1):
        need = Q * thr_lo
        l = -(-need // d)
        if l < 1:
            l = 1
        if l > Lmax:
            continue
        return d, l
    return Dmax, Lmax


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

    c0 = min(range(1, 8), key=lambda c: (thr[c], c))
    d, l = find_basin(Q, thr[c0], scour[c0], Dmax, Lmax)

    print(1)
    print(d, l, c0)


if __name__ == "__main__":
    main()
