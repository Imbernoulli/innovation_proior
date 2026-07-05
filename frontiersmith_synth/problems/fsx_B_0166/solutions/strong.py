# TIER: strong
# Structure-aware ordering that generalizes from the training stream:
#   1. Two principled initial orders computed on the training queries:
#        - frequency-descending (globally hot channels lead)
#        - co-occurrence greedy path (grow the order by repeatedly appending the unplaced
#          channel most co-queried with the channels already placed -> keeps sensor 'stations'
#          that appear together contiguous, which is what actually merges prefixes)
#   2. A light swap-based hill-climb over the TRAINING trie-node count from each init
#      (few passes -> refine without overfitting the finite training log).
#   3. Keep whichever refined order materializes the fewest prefixes on TRAINING.
# The chosen order is then applied to the (unseen) held-out stream; because it targets the
# stable motif structure rather than the exact training paths, it transfers well.
import sys, json


def nodes(queries, rank):
    trie = {}
    cnt = 0
    for q in queries:
        seq = sorted(q, key=lambda c: rank[c])
        node = trie
        for c in seq:
            nxt = node.get(c)
            if nxt is None:
                nxt = {}
                node[c] = nxt
                cnt += 1
            node = nxt
    return cnt


def rank_of(order):
    return {c: i for i, c in enumerate(order)}


def freq_order(N, queries):
    freq = [0] * N
    for q in queries:
        for c in q:
            freq[c] += 1
    return sorted(range(N), key=lambda c: (-freq[c], c))


def cooc_order(N, queries):
    C = [[0] * N for _ in range(N)]
    freq = [0] * N
    for q in queries:
        for c in q:
            freq[c] += 1
        L = len(q)
        for a in range(L):
            qa = q[a]
            for b in range(a + 1, L):
                qb = q[b]
                C[qa][qb] += 1
                C[qb][qa] += 1
    start = max(range(N), key=lambda c: (freq[c], c))
    order = [start]
    placed = {start}
    while len(order) < N:
        best = None
        best_key = None
        for c in range(N):
            if c in placed:
                continue
            sc = sum(C[c][p] for p in order)
            key = (sc, freq[c], -c)
            if best_key is None or key > best_key:
                best_key = key
                best = c
        order.append(best)
        placed.add(best)
    return order


def hill_climb(N, queries, start, passes=3):
    order = list(start)
    best = nodes(queries, rank_of(order))
    improved = True
    p = 0
    while improved and p < passes:
        improved = False
        p += 1
        for i in range(N):
            for j in range(i + 1, N):
                order[i], order[j] = order[j], order[i]
                c = nodes(queries, rank_of(order))
                if c < best:
                    best = c
                    improved = True
                else:
                    order[i], order[j] = order[j], order[i]
    return order, best


def main():
    inst = json.load(sys.stdin)
    N = inst["N"]
    train = inst["train_queries"]

    best_order = None
    best_cost = None
    for init in (freq_order(N, train), cooc_order(N, train)):
        o, c = hill_climb(N, train, init, passes=3)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_order = o

    print(json.dumps({"order": best_order}))


main()
