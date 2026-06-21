The assignment problem asks us to pair n rows with n columns in an n × n cost or rating matrix so that each row and each column is used exactly once and the total selected value is optimal. A legal solution is a permutation of the columns, and the naive way to find the best one is to enumerate all n! permutations. That is already hopeless at n = 12, where 12! is nearly half a billion, so we need structural insight rather than brute force.

The problem is a linear program, but throwing a general simplex solver at it is the wrong move. The assignment polytope is the most degenerate special case of the transportation problem: many bases describe the same vertex, and simplex grinds through pivots that do not improve the objective. The 0/1 matching result of König solves only the qualification special case, not matrices with general integer weights. What we want is an exact method that respects the bipartite combinatorics, runs in polynomial time, and hands back a small certificate of optimality.

The right tool is the Hungarian Method, also known as the Kuhn–Munkres algorithm. It is a primal-dual method built on the linear-programming dual of the assignment problem. We keep a row potential u_i for each person and a column potential v_j for each job so that every reduced cost c_ij − u_i − v_j is nonnegative. Because any feasible (u, v) gives a lower bound Σ_i u_i + Σ_j v_j on every assignment, an assignment that uses only tight edges — those with reduced cost exactly zero — must be optimal. This is complementary slackness: when the lower bound meets the assignment value, both are certified best possible.

The algorithm therefore alternates between two combinatorial steps. First it builds a maximum matching on the tight edges using augmenting paths; if the matching is perfect, we are done. If the matching is short, König's theorem gives a minimum vertex cover of the tight graph, which means the uncovered entries all have strictly positive reduced cost. Let δ be the smallest such uncovered slack. We update the potentials by subtracting δ from the uncovered rows and adding δ to the covered columns. This keeps every reduced cost nonnegative, creates at least one new tight edge, and strictly improves the dual objective. Repeating the loop must terminate, because each dual update lowers the gap by a positive amount and the optimum is bounded.

A clean way to implement the same logic in O(n^3) is to add one row at a time and grow a single shortest augmenting path while raising potentials by the minimum slack encountered along the way. The version below accepts a rectangular n × m cost matrix with m ≥ n and returns the optimal assignment together with its total cost. For a maximization problem, negate the matrix before calling the function and negate the returned total.

```python
INF = float("inf")

def hungarian(cost):
    """
    Min-cost assignment by the Hungarian (Kuhn-Munkres) method.
    cost is n x m with m >= n.  Returns (assignment, total_cost),
    where assignment[i] is the column chosen for row i.
    For maximization, pass -cost and negate the total.
    """
    n, m = len(cost), len(cost[0])
    u = [0] * (n + 1)       # row potentials
    v = [0] * (m + 1)       # column potentials
    p = [0] * (m + 1)       # matching: p[j] = row matched to column j (0 = free)
    way = [0] * (m + 1)     # predecessor column for path tracing

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # dual update: keep tight edges tight, reduce slack elsewhere
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:        # reached a free column -> augment
                break

        # flip the alternating path to enlarge the matching
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [0] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    total = sum(cost[i][assignment[i]] for i in range(n))
    return assignment, total


if __name__ == "__main__":
    # Example: maximize ratings by minimizing their negatives.
    ratings = [
        [8, 7, 9, 9],
        [5, 2, 7, 8],
        [5, 1, 4, 8],
        [2, 2, 2, 6],
    ]
    neg = [[-x for x in row] for row in ratings]
    assign, cost = hungarian(neg)
    print("assignment:", assign)
    print("maximum rating sum:", -cost)
```

The Hungarian Method is the canonical polynomial-time algorithm for the assignment problem. It is strongly polynomial, independent of the size of the matrix entries, and the potentials it produces serve as a compact, hand-checkable proof that the returned permutation is optimal.