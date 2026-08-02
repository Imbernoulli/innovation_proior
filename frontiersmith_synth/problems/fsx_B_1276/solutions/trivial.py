# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    f0 = int(next(it)); a0 = int(next(it)); r0 = int(next(it))
    REV = int(next(it)); BUDGET = int(next(it)); MIN_COMPS = int(next(it))
    comps = []
    for _ in range(N):
        margin = int(next(it)); f = int(next(it)); a = int(next(it))
        r = int(next(it)); doc_cost = int(next(it))
        comps.append((margin, f, a, r, doc_cost))

    # no functional-distance reasoning at all -- just the MIN_COMPS candidates whose
    # margin is closest to the sample median margin (a stable, unremarkable pick), no
    # documentation (reproduces the checker's own internal baseline)
    margins_all = sorted(c[0] for c in comps)
    med = margins_all[len(margins_all) // 2]
    order = sorted(range(N), key=lambda idx: (abs(comps[idx][0] - med), idx))
    chosen = [idx + 1 for idx in order[:MIN_COMPS]]

    margins = [comps[i1 - 1][0] for i1 in chosen]
    M = sum(margins) // len(margins)

    print(len(chosen))
    for i1 in chosen:
        print(i1, 0)
    print(M)


main()
