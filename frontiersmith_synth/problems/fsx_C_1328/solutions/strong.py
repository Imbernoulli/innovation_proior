# TIER: strong
"""Insight: additivity is NOT assumed. Screen every candidate SET against the
sparse epistasis table (both the stability correction and the activity
correction) and explicitly cap how many selected mutations fall inside the
active-site neighbourhood, instead of ranking mutations one-at-a-time by their
individual delta. Because the budget K and site count n are kept small by the
generator, this reformulation is searched exhaustively (with the true,
epistasis- and crowding-aware feasibility/objective), which is exactly what lets
it find synergistic pairs that individual-delta ranking would never try
together and avoid antagonistic pairs that individual-delta ranking walks
straight into."""
import sys
from itertools import combinations


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); K = int(nxt()); C = int(nxt()); R = int(nxt())
    A0 = float(nxt()); ActMin = float(nxt()); alpha = float(nxt())
    dstab = [0.0] * n
    dact = [0.0] * n
    dist = [0] * n
    for i in range(n):
        dstab[i] = float(nxt())
        dact[i] = float(nxt())
        dist[i] = int(nxt())
    m_epi = int(nxt())
    epi_s = {}
    epi_a = {}
    for _ in range(m_epi):
        i = int(nxt()); j = int(nxt()); es = float(nxt()); ea = float(nxt())
        a, b = (i, j) if i < j else (j, i)
        epi_s[(a, b)] = es
        epi_a[(a, b)] = ea

    idxs = list(range(n))
    best_score = -1.0
    best_set = ()

    for k in range(0, K + 1):
        for combo in combinations(idxs, k):
            stab = 0.0
            act = A0
            for i in combo:
                stab += dstab[i]
                act += dact[i]
            if k >= 2:
                for a in range(k):
                    for b in range(a + 1, k):
                        i, j = combo[a], combo[b]
                        key = (i, j) if i < j else (j, i)
                        es = epi_s.get(key)
                        if es is not None:
                            stab += es
                            act += epi_a[key]
            cc = 0
            for i in combo:
                if dist[i] <= R:
                    cc += 1
            over = cc - C
            if over > 0:
                act -= alpha * over * over
            if act >= ActMin - 1e-9:
                F = stab if stab > 0.0 else 0.0
                if F > best_score:
                    best_score = F
                    best_set = combo

    print(len(best_set))
    print(*best_set)


if __name__ == "__main__":
    main()
