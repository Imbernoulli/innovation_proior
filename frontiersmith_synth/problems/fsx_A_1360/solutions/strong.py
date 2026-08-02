# TIER: strong
# The insight: a uniform-random coloring cannot be certified and plateaus under bounded
# repair (see greedy.py). An EXPLICIT algebraic construction does much better as a starting
# point and is fully verifiable: pick the smallest prime p >= n with p = 1 (mod 4) (so the
# quadratic-residue set QR(p) is symmetric under negation), then color edge (i,j) by
# channel 1 iff (i-j) mod p is a quadratic residue mod p, else channel 0. Restricted to our
# n <= p vertices this is an induced sub-mesh of the Paley tournament / Paley graph on
# GF(p) -- the same family that gives the extremal (4,4;17) Ramsey graph -- and it already
# has far fewer monochromatic K4's than a random coloring before any repair at all. We then
# spend the SAME fixed local-repair budget as greedy.py (identical procedure) to polish the
# few residual violations; because the starting point is already close to a good local
# optimum, the same repair budget goes much further here.
import sys
from itertools import combinations

REPAIR_BUDGET = 40


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True


def qr_prime_at_least(n):
    p = max(n, 2)
    while not (is_prime(p) and p % 4 == 1):
        p += 1
    return p


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

    p = qr_prime_at_least(n)
    QR = set((x * x) % p for x in range(1, p))

    color = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % p
            c = 1 if d in QR else 0
            color[i][j] = c
            color[j][i] = c

    adj0 = build_adj(n, color, 0)
    adj1 = build_adj(n, color, 1)

    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    import random
    rnd2 = random.Random(99 * n + k)
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
