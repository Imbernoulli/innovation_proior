# TIER: greedy
import sys

# The obvious resolver: walk packages in declaration order 0..n-1, at each
# package try versions NEWEST-first, keep the first version consistent with
# every already-decided package that requires something of it, and on a
# dead end backtrack to the MOST RECENTLY decided package (chronological
# backtracking, no memory of *why* a branch failed). Accepts the first
# fully feasible assignment it finds -- it does not keep searching for a
# higher-preference one. Bounded by a fixed node-visit budget (never wall
# clock) so it terminates deterministically; if the budget runs out before
# a full assignment is found, it falls back to the universal safe
# construction (install everything at its oldest version).

NODE_BUDGET = 6000


class BudgetExceeded(Exception):
    pass


def build_incoming(n, m, reqs):
    incoming = [set() for _ in range(n)]
    for s in range(n):
        for v in range(1, m[s] + 1):
            for (j, lo, hi) in reqs[s][v]:
                incoming[j].add(s)
    return [sorted(x) for x in incoming]


def solve(n, m, reqs, incoming):
    assign = [0] * n
    steps = [0]

    def consistent(t, v):
        for s in incoming[t]:
            vs = assign[s]
            for (j, lo, hi) in reqs[s][vs]:
                if j == t and (v < lo or v > hi):
                    return False
        return True

    def dfs(idx):
        if idx == n:
            return True
        for v in range(m[idx], 0, -1):
            steps[0] += 1
            if steps[0] > NODE_BUDGET:
                raise BudgetExceeded()
            if consistent(idx, v):
                assign[idx] = v
                if dfs(idx + 1):
                    return True
        assign[idx] = 0
        return False

    try:
        ok = dfs(0)
    except BudgetExceeded:
        return None
    return assign if ok else None


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = [0] * n
    reqs = [None] * n
    for i in range(n):
        mi = int(next(it))
        m[i] = mi
        reqs[i] = [None] * (mi + 1)
        for v in range(1, mi + 1):
            next(it)  # pref (greedy does not need it to search feasibility)
            r = int(next(it))
            edges = []
            for _ in range(r):
                j = int(next(it)); lo = int(next(it)); hi = int(next(it))
                edges.append((j, lo, hi))
            reqs[i][v] = edges

    incoming = build_incoming(n, m, reqs)
    assign = solve(n, m, reqs, incoming)
    if assign is None:
        assign = [1] * n  # node budget exhausted -> universal safe fallback
    print(" ".join(str(x) for x in assign))


if __name__ == "__main__":
    main()
