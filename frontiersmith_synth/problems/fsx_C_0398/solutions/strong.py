# TIER: strong
"""Conditional-independence skeleton + frequency ordering (a PC-style discrete rule).

Skeleton: start from the marginal mutual-information graph (an edge wherever
MI(X,Y) exceeds a threshold), then PRUNE every candidate edge X-Y for which some
third gallery Z screens it off -- i.e. the conditional mutual information
MI(X,Y | Z) collapses near zero.  This removes the TRANSITIVE-path false edges
(A->B->C leaves MI(A,C|B)~0) that marginal thresholding keeps, while the noisy-OR
collider structure never manufactures a co-parent edge to begin with.

Orientation: under small private leak rates a SOURCE gallery fires rarely while
its descendants inherit extra activation, so descendants are entered MORE often.
Sorting galleries by ascending marginal entry frequency estimates a topological
order; each surviving skeleton edge is oriented from the rarer (earlier) gallery
to the more-frequent (later) one.

Finite tours make the conditional tests noisy and near-saturated descendants blur
the frequency order, so leftover / missed edges and the odd mis-orientation keep
it short of perfect, especially on the larger, sparser, tour-poorer museums."""
import sys, json
from math import log


def _mi_counts(counts, tot):
    px = [(counts[0][0] + counts[0][1]) / tot, (counts[1][0] + counts[1][1]) / tot]
    py = [(counts[0][0] + counts[1][0]) / tot, (counts[0][1] + counts[1][1]) / tot]
    mi = 0.0
    for x in (0, 1):
        for y in (0, 1):
            pxy = counts[x][y] / tot
            if pxy > 0:
                mi += pxy * log(pxy / (px[x] * py[y]))
    return mi


def mutual_information(col_a, col_b):
    n = len(col_a)
    a = 0.5; tot = n + 4 * a
    c = [[a, a], [a, a]]
    for k in range(n):
        c[col_a[k]][col_b[k]] += 1.0
    return _mi_counts(c, tot)


def cond_mi(col_a, col_b, col_z):
    """Conditional MI(A;B|Z) for binary Z, 0.5-Laplace within each stratum."""
    n = len(col_a)
    a = 0.5
    strata = [[[a, a], [a, a]], [[a, a], [a, a]]]
    nz = [0, 0]
    for k in range(n):
        z = col_z[k]
        nz[z] += 1
        strata[z][col_a[k]][col_b[k]] += 1.0
    cmi = 0.0
    for z in (0, 1):
        tot = nz[z] + 4 * a
        pz = nz[z] / n
        cmi += pz * _mi_counts(strata[z], tot)
    return cmi


def main():
    inst = json.load(sys.stdin)
    data = inst["data"]
    n = len(data)
    d = int(inst["n_galleries"])
    cols = [[int(data[r][j]) for r in range(n)] for j in range(d)]

    mi_thr = 0.012
    cond_thr = 0.004

    # 1) marginal-MI skeleton
    skel = {}
    for i in range(d):
        for j in range(i + 1, d):
            m = mutual_information(cols[i], cols[j])
            if m > mi_thr:
                skel[(i, j)] = m

    # 2) prune edges screened off by a single conditioning gallery
    pruned = set()
    for (i, j) in list(skel.keys()):
        for z in range(d):
            if z == i or z == j:
                continue
            if cond_mi(cols[i], cols[j], cols[z]) < cond_thr:
                pruned.add((i, j))
                break

    # 3) frequency ordering: rarer gallery is earlier (source)
    freq = [sum(cols[j]) for j in range(d)]
    order = sorted(range(d), key=lambda k: (freq[k], k))   # ascending entry frequency
    rank = [0] * d
    for pos, node in enumerate(order):
        rank[node] = pos

    edges = []
    for (i, j) in skel:
        if (i, j) in pruned:
            continue
        if rank[i] < rank[j]:
            edges.append([i, j])
        else:
            edges.append([j, i])
    print(json.dumps({"edges": edges}))


main()
