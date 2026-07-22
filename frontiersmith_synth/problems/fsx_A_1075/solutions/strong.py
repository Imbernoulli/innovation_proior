# TIER: strong
"""Insight: build one SHARED stockpot economy instead of re-optimizing every brew order
in isolation.

Per-query-optimal parenthesization (the greedy tier) has no reason to land on any
particular sub-range boundary -- the unrestricted optimum is free to cut wherever is
locally cheapest for that one query, so identical sub-ranges almost never recur as
declared nodes across different queries even when queries heavily overlap.

This solution instead RESTRICTS every query's top-level decomposition to a small shared
candidate point set C (every query endpoint, plus the chain's two ends) and solves a
single GLOBAL memoized DP over pairs of candidate points. Because the candidate set is
shared, the optimal coarse decomposition of a given (candidate-point) span is a pure
function of that span -- independent of which query is asking -- so whenever two
queries' brew orders route through the same span, the memo returns the SAME already-built
node, and the checker charges it once. Below the coarse grid (inside one "cell" with no
candidate point strictly inside it) we fall back to an ordinary per-cell-optimal DP,
also memoized by cell so a repeated cell is only ever built once.

This deliberately gives up per-query global optimality (a single query, taken alone,
might have a cheaper unrestricted tree) in exchange for a shared-intermediate economy
that wins in aggregate once many queries reuse the same spans."""
import sys


def cost_only_mcm(a, b, dims):
    """Optimal scalar cost (no node construction) of contracting dims[a..b]."""
    n = b - a
    if n <= 1:
        return 0
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            aa = dims[a + i]
            bb = dims[a + j]
            best = None
            for k in range(i + 1, j):
                c = dims[a + k]
                cst = dp[i][k] + dp[k][j] + aa * c * bb
                if best is None or cst < best:
                    best = cst
            dp[i][j] = best
    return dp[0][n]


def build_fine_tree(a, b, dims, nodes):
    """Actually construct the optimal parenthesization of dims[a..b] as nodes."""
    n = b - a
    if n == 1:
        nodes.append("L %d" % a)
        return len(nodes) - 1
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    sp = [[-1] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            aa = dims[a + i]
            bb = dims[a + j]
            best = None
            bk = -1
            for k in range(i + 1, j):
                c = dims[a + k]
                cst = dp[i][k] + dp[k][j] + aa * c * bb
                if best is None or cst < best:
                    best = cst
                    bk = k
            dp[i][j] = best
            sp[i][j] = bk

    def build(i, j):
        if j - i == 1:
            nodes.append("L %d" % (a + i))
            return len(nodes) - 1
        k = sp[i][j]
        c1 = build(i, k)
        c2 = build(k, j)
        nodes.append("S %d %d" % (c1, c2))
        return len(nodes) - 1

    return build(0, n)


def main():
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    p = 0
    m = int(data[p]); p += 1
    dims = [int(data[p + i]) for i in range(m + 1)]; p += m + 1
    Q = int(data[p]); p += 1
    queries = []
    for _ in range(Q):
        L = int(data[p]); R = int(data[p + 1]); p += 2
        queries.append((L, R))

    # ---- shared candidate cut set: every query endpoint + the chain ends ----
    cset = {0, m}
    for (L, R) in queries:
        cset.add(L)
        cset.add(R)
    C = sorted(cset)
    K = len(C)
    cidx = {c: i for i, c in enumerate(C)}

    # ---- phase 1: per-cell optimal cost (adjacent candidate points, no split freedom) ----
    cell_cost = [cost_only_mcm(C[i], C[i + 1], dims) for i in range(K - 1)]

    # ---- phase 2: coarse DP over ALL candidate-point pairs, cost only ----
    coarse_cost = [[0] * K for _ in range(K)]
    coarse_split = [[-1] * K for _ in range(K)]
    for i in range(K - 1):
        coarse_cost[i][i + 1] = cell_cost[i]
    for span in range(2, K):
        for i in range(0, K - span):
            j = i + span
            ai = dims[C[i]]
            bj = dims[C[j]]
            best = None
            bk = -1
            for k in range(i + 1, j):
                cst = coarse_cost[i][k] + coarse_cost[k][j] + ai * dims[C[k]] * bj
                if best is None or cst < best:
                    best = cst
                    bk = k
            coarse_cost[i][j] = best
            coarse_split[i][j] = bk

    # ---- phase 3: build the actual shared DAG, memoized by (i,j) so a repeated span
    #      across different queries is built exactly once and simply referenced again ----
    nodes = []
    fine_cache = {}
    coarse_cache = {}

    def fine_solve(a, b):
        key = (a, b)
        nid = fine_cache.get(key)
        if nid is None:
            nid = build_fine_tree(a, b, dims, nodes)
            fine_cache[key] = nid
        return nid

    def coarse_solve(i, j):
        key = (i, j)
        nid = coarse_cache.get(key)
        if nid is not None:
            return nid
        if j - i == 1:
            nid = fine_solve(C[i], C[j])
        else:
            k = coarse_split[i][j]
            c1 = coarse_solve(i, k)
            c2 = coarse_solve(k, j)
            nodes.append("S %d %d" % (c1, c2))
            nid = len(nodes) - 1
        coarse_cache[key] = nid
        return nid

    roots = []
    for (L, R) in queries:
        roots.append(coarse_solve(cidx[L], cidx[R]))

    out = [str(len(nodes))]
    out.extend(nodes)
    out.append(" ".join(map(str, roots)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
