# TIER: greedy
# The obvious first idea: since one shared order must serve every pattern,
# just union all K edge lists into a single graph and run textbook DYNAMIC
# minimum-degree elimination on that union. This ignores the fact that the
# union also accumulates K independent draws of private noise edges, which
# distorts the degree signal relative to any single pattern (or the shared
# backbone) -- vertices that are noise-heavy look artificially "busy" and
# get eliminated late, and true separators can look artificially "cheap".
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
    union_edges = set()
    for _ in range(K):
        M = nxt()
        for _ in range(M):
            u = nxt()
            v = nxt()
            a, b = (u, v) if u < v else (v, u)
            union_edges.add((a, b))

    order = min_degree_order(N, union_edges)
    print(" ".join(str(v) for v in order))


if __name__ == "__main__":
    main()
