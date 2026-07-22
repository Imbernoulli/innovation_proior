# TIER: greedy
"""The obvious first approach: pool the K languages' digraph tables by summing
raw counts, rank symbols by total pooled demand, rank slots by centrality
(low total travel to all other slots), and match highest-demand symbol to
most-central slot. This is exactly the classic "put frequent letters on the
home row" keyboard heuristic applied to the POOLED corpus.

It is a trap: pooled raw counts are dominated by whichever language happens
to have the largest absolute corpus size, so this recipe silently starves the
smaller-corpus languages' own high-value (highly concentrated) digraphs, even
though the graded objective cares about each language's OWN relative cost.
"""
import sys
import math


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); k = int(next(it))
    coords = []
    for _ in range(n):
        x = float(next(it)); y = float(next(it))
        coords.append((x, y))
    freq = []
    for _lang in range(k):
        mat = []
        for _i in range(n):
            row = [int(next(it)) for _ in range(n)]
            mat.append(row)
        freq.append(mat)

    # travel matrix
    T = [[0.0] * n for _ in range(n)]
    for u in range(n):
        xu, yu = coords[u]
        for v in range(n):
            if u != v:
                xv, yv = coords[v]
                T[u][v] = math.hypot(xu - xv, yu - yv)

    # pooled combined (undirected) weight
    W = [[0] * n for _ in range(n)]
    for lang in range(k):
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i][j] += freq[lang][i][j]

    symbol_weight = [sum(W[i][j] + W[j][i] for j in range(n) if j != i) for i in range(n)]
    slot_centrality = [sum(T[u][v] for v in range(n) if v != u) for u in range(n)]

    sym_order = sorted(range(n), key=lambda i: (-symbol_weight[i], i))
    slot_order = sorted(range(n), key=lambda u: (slot_centrality[u], u))

    perm = [0] * n
    for sym, slot in zip(sym_order, slot_order):
        perm[sym] = slot

    print(" ".join(str(perm[i]) for i in range(n)))


if __name__ == "__main__":
    main()
