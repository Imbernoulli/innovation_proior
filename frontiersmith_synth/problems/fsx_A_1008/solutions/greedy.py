# TIER: greedy
import sys
from collections import deque


def parse_input():
    toks = sys.stdin.read().split()
    it = iter(toks)
    k = int(next(it))
    sigma = int(next(it))
    Lmax = int(next(it))
    automata = []
    for _ in range(k):
        n = int(next(it))
        table = [[int(next(it)) for _ in range(n)] for _l in range(sigma)]
        automata.append((n, table))
    return k, sigma, Lmax, automata


def find_reset_word(n, table, sigma):
    """Standard pairwise-state-merging heuristic (Eppstein-style): repeatedly find
    the shortest word merging some current pair of distinct reachable states (via
    BFS on the pair space), apply it to the whole reachable set, repeat."""

    def canon(x, y):
        return (x, y) if x <= y else (y, x)

    def bfs_merge(a, b):
        start = canon(a, b)
        if start[0] == start[1]:
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            x, y = cur
            for l in range(sigma):
                nxt = canon(table[l][x], table[l][y])
                if nxt not in prev:
                    prev[nxt] = (cur, l)
                    if nxt[0] == nxt[1]:
                        path = []
                        node = nxt
                        while prev[node] is not None:
                            par, lab = prev[node]
                            path.append(lab)
                            node = par
                        path.reverse()
                        return path
                    q.append(nxt)
        return []  # defensive: automaton not synchronizable from this pair

    S = list(range(n))
    word = []
    while len(S) > 1:
        a, b = S[0], S[1]
        seg = bfs_merge(a, b)
        if not seg:
            break
        word.extend(seg)
        newS = set()
        for s in S:
            cur = s
            for l in seg:
                cur = table[l][cur]
            newS.add(cur)
        S = list(newS)
    return word


def main():
    k, sigma, Lmax, automata = parse_input()
    items = []  # (cost, word_letters)
    for n, table in automata:
        w = find_reset_word(n, table, sigma)
        items.append((len(w), w))
    items.sort(key=lambda x: x[0])
    total = 0
    chosen = []
    for cost, w in items:
        if total + cost <= Lmax:
            chosen.append(w)
            total += cost
    word = "".join(str(l) for w in chosen for l in w)
    print(word)


if __name__ == "__main__":
    main()
