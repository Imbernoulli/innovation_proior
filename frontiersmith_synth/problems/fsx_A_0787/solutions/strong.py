# TIER: strong
# Insight: reaching position s only depends on the pitch classes written in a
# bounded window behind s (the maximum voice delay), i.e. the melody is a
# WALK IN A PRODUCT AUTOMATON whose states are windows of recent pitch
# classes -- exactly the structure the problem's "impossible lookahead"
# hides. Instead of committing to the first locally-safe pitch (greedy's
# trap), we search that automaton with backtracking: order candidate pitches
# by which pitch class is currently under-represented (drives entropy up),
# and before committing verify the state still has an outgoing edge (a
# 1-step lookahead into the automaton) so we don't walk into a dead state
# that greedy would only discover one note later. If we do hit a dead end we
# backtrack -- true graph search, not a single doomed pass.
import sys


def read_instance():
    toks = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    K = int(nxt()); L = int(nxt())
    C = [[0] * 12 for _ in range(12)]
    for i in range(12):
        for j in range(12):
            C[i][j] = int(nxt())
    voices = []
    for _ in range(K):
        d = int(nxt()); t = int(nxt())
        voices.append((d, t))
    return K, L, C, voices


def derive_rules(voices):
    rules = []
    for (d, t) in voices:
        rules.append((d, t % 12))
    n = len(voices)
    for i in range(n):
        for j in range(i + 1, n):
            d1, t1 = voices[i]
            d2, t2 = voices[j]
            if d1 == d2:
                continue
            if d1 < d2:
                Delta = d2 - d1
                delta_t = (t1 - t2) % 12
            else:
                Delta = d1 - d2
                delta_t = (t2 - t1) % 12
            rules.append((Delta, delta_t))
    return rules


def valid_next(c, s, cand, C, rules):
    for (Delta, delta_t) in rules:
        if Delta > s:
            continue
        y = (c[s - Delta] + delta_t) % 12
        if C[cand][y] != 1:
            return False
    return True


def has_outgoing_edge(c, s, C, rules):
    for cand in range(12):
        if valid_next(c, s, cand, C, rules):
            return True
    return False


def search(L, C, rules, node_budget):
    c = []
    usage = [0] * 12
    best = []
    budget = [node_budget]

    def dfs(s):
        nonlocal best
        if len(c) > len(best):
            best = list(c)
        if s == L:
            return True
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        cands = sorted(range(12), key=lambda p: (usage[p], p))
        for cand in cands:
            if not valid_next(c, s, cand, C, rules):
                continue
            c.append(cand)
            usage[cand] += 1
            ok_lookahead = True
            if s + 1 < L:
                ok_lookahead = has_outgoing_edge(c, s + 1, C, rules)
            if ok_lookahead and dfs(s + 1):
                return True
            c.pop()
            usage[cand] -= 1
            if budget[0] <= 0:
                return False
        return False

    dfs(0)
    return best if best else [0]


def main():
    K, L, C, voices = read_instance()
    rules = derive_rules(voices)
    sys.setrecursionlimit(10000)
    c = search(L, C, rules, node_budget=60000)
    if not c:
        c = [0]
    print(len(c))
    print(" ".join(map(str, c)))


if __name__ == "__main__":
    main()
