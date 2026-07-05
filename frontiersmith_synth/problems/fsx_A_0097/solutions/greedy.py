# TIER: greedy
# Mian-Chowla greedy: scan 0..V, install an offset iff it keeps the set Sidon
# (all pairwise sums distinct). Capped at k fittings.
import sys


def main():
    d = sys.stdin.read().split()
    k = int(d[0]); V = int(d[1])

    A = []
    sums = set()
    for x in range(0, V + 1):
        if len(A) >= k:
            break
        new = []
        ok = True
        for a in A:
            s = x + a
            if s in sums:
                ok = False
                break
            new.append(s)
        if not ok:
            continue
        s2 = 2 * x
        if s2 in sums:
            continue
        # all a<x so x+a < 2x and distinct among themselves; safe to commit
        A.append(x)
        for s in new:
            sums.add(s)
        sums.add(s2)

    print(len(A))
    for x in A:
        print(x)


main()
