# TIER: strong
# Insight: a cascade of thresholds is a radix sort in disguise. Sort the 7 classes by
# their residence-time threshold ascending; build ONE basin per rank, in that order,
# each just wide/deep enough to clear its own threshold but still BELOW the next
# threshold in line -- so each basin resolves exactly one class boundary and drains
# purely into that class's own bin. Depth is chosen to be scour-safe (no resuspension)
# at (essentially) zero extra volume cost, since depth*length is what the budget
# charges for, not the split between them. This is a reordering + an exchange
# argument, not "greedy with a bigger constant".
import sys


def find_basin(Q, thr_lo, thr_hi_excl, scour_c, Dmax, Lmax):
    d_min = max(1, -(-Q // scour_c))
    for d in range(d_min, Dmax + 1):
        need = Q * thr_lo
        l = -(-need // d)
        if l < 1:
            l = 1
        if l > Lmax:
            continue
        prod = d * l
        if thr_hi_excl is not None and prod >= Q * thr_hi_excl:
            continue
        return d, l
    return None, None


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

    order = sorted(range(1, 8), key=lambda c: (thr[c], c))  # ascending threshold

    basins = []
    total_vol = 0
    for k in range(len(order)):
        if len(basins) >= M:
            break
        c = order[k]
        thr_hi_excl = thr[order[k + 1]] if k + 1 < len(order) else None
        d, l = find_basin(Q, thr[c], thr_hi_excl, scour[c], Dmax, Lmax)
        if d is None:
            continue
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
