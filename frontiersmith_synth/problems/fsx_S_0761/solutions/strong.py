# TIER: strong
# Insight: the union of K patterns is poisoned by K independent draws of
# private noise edges, but the INTERSECTION of all K edge lists is not --
# an edge only survives intersection if it appears in EVERY pattern, which
# random per-pattern noise essentially never does. The intersection therefore
# recovers (an approximation of) the hidden shared separator backbone that
# all K patterns were built around. Running minimum-degree elimination on
# this cleaned-up skeleton (instead of the noisy union) yields a leaves-
# before-separators order that stays close to good for every pattern at
# once, because it is aimed at the structure they actually share rather
# than at whichever pattern's noise happens to dominate the union.
import sys


def min_degree_order(N, edges):
    adj = [0] * (N + 1)
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    alive_mask = 0
    for v in range(1, N + 1):
        alive_mask |= (1 << v)
    order = []
    for _ in range(N):
        best_v, best_d = -1, None
        m = alive_mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            d = bin(adj[v] & alive_mask).count("1")
            if best_d is None or d < best_d or (d == best_d and v < best_v):
                best_d, best_v = d, v
            m &= m - 1
        v = best_v
        nbrs_mask = adj[v] & alive_mask
        mm = nbrs_mask
        while mm:
            low = mm & (-mm)
            u = low.bit_length() - 1
            adj[u] |= nbrs_mask
            adj[u] &= ~low
            mm &= mm - 1
        alive_mask &= ~(1 << v)
        order.append(v)
    return order


def main():
    it = iter(sys.stdin.read().split())

    def nxt():
        return int(next(it))

    N = nxt()
    K = nxt()
    pattern_edge_sets = []
    for _ in range(K):
        M = nxt()
        s = set()
        for _ in range(M):
            u = nxt()
            v = nxt()
            a, b = (u, v) if u < v else (v, u)
            s.add((a, b))
        pattern_edge_sets.append(s)

    inter = set(pattern_edge_sets[0])
    for s in pattern_edge_sets[1:]:
        inter &= s

    order = min_degree_order(N, inter)
    print(" ".join(str(v) for v in order))


if __name__ == "__main__":
    main()
