import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    bands = []
    for _ in range(n):
        s = int(data[idx]); d = int(data[idx+1]); p = int(data[idx+2]); idx += 3
        bands.append((s, d, p))

    # Build, for each band, the set of slots it occupies as a bitmask over [0, m).
    # A band is bookable only if it fits in [0, m): need d > 0, s >= 0, s + d <= m.
    # The occupied slots are the half-open interval [s, s+d): s, s+1, ..., s+d-1.
    usable = []  # (mask, profit)
    for (s, d, p) in bands:
        if d <= 0 or s < 0 or s + d > m:
            continue
        mask = 0
        for k in range(s, s + d):
            mask |= (1 << k)
        usable.append((mask, p))

    best = 0  # booking nothing yields 0
    K = len(usable)
    # Try every subset of usable bands; keep the ones that are pairwise slot-disjoint.
    for r in range(1, K + 1):
        for combo in combinations(range(K), r):
            union = 0
            ok = True
            total = 0
            for ci in combo:
                mk, pr = usable[ci]
                if union & mk:
                    ok = False
                    break
                union |= mk
                total += pr
            if ok and total > best:
                best = total
    print(best)

main()
