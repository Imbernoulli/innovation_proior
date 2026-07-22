# TIER: strong
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
        return []

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


def final_set_size(n, table, word_letters):
    S = set(range(n))
    for l in word_letters:
        if len(S) == 1:
            break
        S = {table[l][s] for s in S}
    return len(S)


def main():
    k, sigma, Lmax, automata = parse_input()

    # Stage 1: for each automaton, find A synchronizing word for it in isolation
    # (same generic recipe as the naive/greedy tier).
    candidates = []  # (cost, word_letters)
    for n, table in automata:
        w = find_reset_word(n, table, sigma)
        candidates.append((len(w), w))

    # Stage 2 -- the insight: a word built for automaton i may, by shared/product
    # transition structure, ALSO reset OTHER automata "for free". Detect this
    # empirically instead of assuming per-automaton words are independent needs:
    # test every candidate word against every automaton.
    kk = len(candidates)
    coverage = []  # coverage[i] = bitmask of automata fully reset by candidate i's word
    for _cost, w in candidates:
        mask = 0
        for j, (nj, tj) in enumerate(automata):
            if final_set_size(nj, tj, w) == 1:
                mask |= (1 << j)
        coverage.append(mask)

    # Stage 3: choose a subset of candidate words (their concatenation, in any
    # order, preserves every individual reset -- a synced automaton stays synced
    # under further letters) to maximize the number of DISTINCT automata reset,
    # subject to the total length budget. kk is small -> brute force the subset.
    best_count = -1
    best_cost = 0
    best_bits = 0
    for bits in range(1 << kk):
        total_cost = 0
        union_mask = 0
        feasible = True
        for i in range(kk):
            if bits & (1 << i):
                total_cost += candidates[i][0]
                if total_cost > Lmax:
                    feasible = False
                    break
                union_mask |= coverage[i]
        if not feasible:
            continue
        cnt = bin(union_mask).count("1")
        if cnt > best_count or (cnt == best_count and total_cost < best_cost):
            best_count = cnt
            best_cost = total_cost
            best_bits = bits

    chosen_words = [candidates[i][1] for i in range(kk) if best_bits & (1 << i)]
    word = "".join(str(l) for w in chosen_words for l in w)
    print(word)


if __name__ == "__main__":
    main()
