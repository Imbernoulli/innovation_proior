# TIER: greedy
# The obvious approach: pairwise-merge heuristic (Eppstein).  Precompute the shortest
# merging word for every pair via one backward BFS from the diagonal, then repeatedly merge
# the currently-active pair whose merging word is SHORTEST, applying that word to the whole
# active set.  Polynomial and reasonable, but blind to the automaton's global cyclic algebra,
# so it lands well above the (n-1)^2 optimum on these Cerny-family instances.
from collections import deque
import sys

def apply_word(delta, word, S):
    cur = set(S)
    for s in word:
        cur = set(delta[s][i] for i in cur)
    return cur

def all_pairs(delta, m, n):
    dist = {}; nsym = {}; npair = {}
    dq = deque()
    for c in range(n):
        dist[(c, c)] = 0; dq.append((c, c))
    pre = [[[] for _ in range(n)] for _ in range(m)]
    for s in range(m):
        for x in range(n):
            pre[s][delta[s][x]].append(x)
    while dq:
        u = dq.popleft(); ua, ub = u
        for s in range(m):
            for x in pre[s][ua]:
                for y in pre[s][ub]:
                    key = (x, y) if x < y else (y, x)
                    if key not in dist:
                        dist[key] = dist[u] + 1; nsym[key] = s; npair[key] = u
                        dq.append(key)
    return dist, nsym, npair

def merge_word(nsym, npair, p, q):
    key = (p, q) if p < q else (q, p)
    w = []
    while key[0] != key[1]:
        s = nsym[key]; w.append(s); key = npair[key]
    return w

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    delta = [[int(next(it)) for _ in range(n)] for _ in range(m)]
    dist, nsym, npair = all_pairs(delta, m, n)
    S = set(range(n)); word = []
    while len(S) > 1:
        Sl = sorted(S); sel = None; selv = None
        for i in range(len(Sl)):
            for j in range(i + 1, len(Sl)):
                k = (Sl[i], Sl[j]); d = dist.get(k)
                if d is not None and (selv is None or d < selv):
                    selv = d; sel = k
        w = merge_word(nsym, npair, sel[0], sel[1])
        word += w; S = apply_word(delta, w, S)
    sys.stdout.write(" ".join(map(str, word)) + "\n")

if __name__ == "__main__":
    main()
