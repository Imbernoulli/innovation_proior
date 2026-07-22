# TIER: greedy
"""Textbook approach: for EACH query independently, run the standard O(n^3) matrix-chain
DP to find that query's own cost-optimal parenthesization. This is the obvious "strong
coder" move -- solve every brew order to local optimality. It never looks at what other
queries need, so it builds a completely fresh set of nodes per query and captures none
of the cross-query reuse economy, even when queries share large overlapping sub-ranges."""
import sys


def solve_query(L, R, dims, nodes):
    n = R - L
    if n == 1:
        nodes.append("L %d" % L)
        return len(nodes) - 1

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    sp = [[-1] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            a = dims[L + i]
            b = dims[L + j]
            best = None
            bk = -1
            for k in range(i + 1, j):
                c = dims[L + k]
                cost = dp[i][k] + dp[k][j] + a * c * b
                if best is None or cost < best:
                    best = cost
                    bk = k
            dp[i][j] = best
            sp[i][j] = bk

    def build(i, j):
        if j - i == 1:
            nodes.append("L %d" % (L + i))
            return len(nodes) - 1
        k = sp[i][j]
        c1 = build(i, k)
        c2 = build(k, j)
        nodes.append("S %d %d" % (c1, c2))
        return len(nodes) - 1

    return build(0, n)


def main():
    data = sys.stdin.read().split()
    p = 0
    m = int(data[p]); p += 1
    dims = [int(data[p + i]) for i in range(m + 1)]; p += m + 1
    Q = int(data[p]); p += 1
    queries = []
    for _ in range(Q):
        L = int(data[p]); R = int(data[p + 1]); p += 2
        queries.append((L, R))

    nodes = []
    roots = []
    for (L, R) in queries:
        roots.append(solve_query(L, R, dims, nodes))

    out = [str(len(nodes))]
    out.extend(nodes)
    out.append(" ".join(map(str, roots)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
