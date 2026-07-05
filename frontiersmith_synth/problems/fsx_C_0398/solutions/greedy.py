# TIER: greedy
"""Marginal mutual-information thresholding.  For every pair of galleries compute
the pairwise mutual information from the binary tour logs; add an (undirected) edge
whenever it exceeds a fixed threshold, and orient it by raw column order
(i -> j for i < j).  This catches real driver-galleries but ALSO every TRANSITIVE
dependence along a flow path A->B->C (a false A-C edge), and because the galleries
were randomly relabelled the index-order orientation is essentially a coin flip --
so it recovers some skeleton yet pays heavily in extra and reversed edges."""
import sys, json
from math import log


def mutual_information(col_a, col_b):
    n = len(col_a)
    a = 0.5; tot = n + 4 * a          # 0.5-Laplace smoothing over the 2x2 table
    c = [[a, a], [a, a]]
    for k in range(n):
        c[col_a[k]][col_b[k]] += 1.0
    px = [(c[0][0] + c[0][1]) / tot, (c[1][0] + c[1][1]) / tot]
    py = [(c[0][0] + c[1][0]) / tot, (c[0][1] + c[1][1]) / tot]
    mi = 0.0
    for x in (0, 1):
        for y in (0, 1):
            pxy = c[x][y] / tot
            mi += pxy * log(pxy / (px[x] * py[y]))
    return mi


def main():
    inst = json.load(sys.stdin)
    data = inst["data"]
    d = int(inst["n_galleries"])
    cols = [[int(data[r][j]) for r in range(len(data))] for j in range(d)]

    thr = 0.015
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if mutual_information(cols[i], cols[j]) > thr:
                edges.append([i, j])       # orient by index order (uninformed)
    print(json.dumps({"edges": edges}))


main()
