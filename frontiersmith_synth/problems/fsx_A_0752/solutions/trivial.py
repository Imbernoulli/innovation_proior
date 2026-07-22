# TIER: trivial
# Flat baseline: ignore the frequencies entirely. Find the smallest length L at
# which at least n distinct run-legal codewords exist, then hand out the first n
# of them (lexicographic DFS) in input order. Reproduces the checker's own
# internal baseline exactly (F == B), so this always scores ~0.1.
import sys


def count_legal(L, d):
    if L <= 0:
        return 1
    g = [0] * (d + 1)
    g[1] = 2
    for _ in range(2, L + 1):
        s = sum(g[1:d + 1])
        newg = [0] * (d + 1)
        for r in range(1, d):
            newg[r + 1] += g[r]
        newg[1] += s
        g = newg
    return sum(g[1:d + 1])


def enumerate_legal(L, d, need):
    results = []

    def dfs(prefix, last_bit, run):
        if len(results) >= need:
            return
        if len(prefix) == L:
            results.append(prefix)
            return
        for b in (0, 1):
            if last_bit is None or b != last_bit:
                dfs(prefix + str(b), b, 1)
            elif run < d:
                dfs(prefix + str(b), b, run + 1)
            if len(results) >= need:
                return

    dfs("", None, 0)
    return results[:need]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); d = int(toks[1])
    p = [int(x) for x in toks[2:2 + n]]

    L = 0
    while count_legal(L, d) < n:
        L += 1

    words = enumerate_legal(L, d, n)
    print(" ".join(words))


if __name__ == "__main__":
    main()
