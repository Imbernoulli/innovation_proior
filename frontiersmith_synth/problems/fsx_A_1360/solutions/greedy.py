# TIER: greedy
# The "obvious" approach: start from a uniform-random 2-coloring (seeded from the instance,
# so it is deterministic), then run a bounded local-repair pass -- for a fixed budget of
# candidate edges (shuffled order), flip an edge's channel if doing so strictly reduces the
# number of monochromatic k-cliques through that edge. This is exactly the textbook
# "random + local repair" recipe. The budget is a FIXED constant, not scaled with n, so on
# the larger instances it only touches a small fraction of the edge set and plateaus well
# short of what a good starting point (see strong.py) reaches with the same repair budget.
import sys, random
from itertools import combinations

REPAIR_BUDGET = 40


def build_adj(n, color, c):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if color[i][j] == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def count_mono_containing(adj, n, k, must):
    pool = (1 << n) - 1
    for v in must:
        pool &= adj[v]
    for v in must:
        pool &= ~(1 << v)
    remaining = k - len(must)
    if remaining <= 0:
        return 1
    poollist = [v for v in range(n) if (pool >> v) & 1]
    if len(poollist) < remaining:
        return 0
    cnt = 0
    for T in combinations(poollist, remaining):
        ok = True
        for a in range(len(T)):
            row = adj[T[a]]
            for b in range(a + 1, len(T)):
                if not (row >> T[b]) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            cnt += 1
    return cnt


def main():
    t = sys.stdin.read().split()
    n = int(t[0]); k = int(t[1])

    rnd = random.Random(55555 * n + 31 * k + 9)
    color = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = rnd.randint(0, 1)
            color[i][j] = c
            color[j][i] = c

    adj0 = build_adj(n, color, 0)
    adj1 = build_adj(n, color, 1)

    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rnd2 = random.Random(77 * n + k)
    rnd2.shuffle(edges)

    for (a, b) in edges[:REPAIR_BUDGET]:
        cur = color[a][b]
        adj_cur = adj0 if cur == 0 else adj1
        adj_other = adj1 if cur == 0 else adj0
        before = count_mono_containing(adj_cur, n, k, (a, b))
        after = count_mono_containing(adj_other, n, k, (a, b))
        if after < before:
            newc = 1 - cur
            color[a][b] = newc; color[b][a] = newc
            adj_cur[a] &= ~(1 << b); adj_cur[b] &= ~(1 << a)
            adj_other[a] |= (1 << b); adj_other[b] |= (1 << a)

    out = []
    for i in range(n):
        out.append(" ".join(str(x) for x in color[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
