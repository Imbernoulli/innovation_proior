# TIER: strong
# Insight: this is a graph-embedding problem, not a coding problem. Huffman only ever
# sees the diagonal (per-chime strike counts); it never looks at which chimes strike
# back-to-back. We instead build the undirected chime-adjacency graph straight from the
# peal (edge weight = how often two chimes strike consecutively), order chimes with a
# max-weight nearest-neighbor chain over that graph (a 1-D linear-arrangement heuristic
# that pulls frequently-co-occurring chimes next to each other), and then recursively
# bisect that chain into a balanced binary rack. This embeds the transition graph into
# the tree metric directly, instead of only balancing depth against raw frequency.
import sys


def build_weight_matrix(N, trace):
    W = [[0] * (N + 1) for _ in range(N + 1)]
    for k in range(len(trace) - 1):
        a, b = trace[k], trace[k + 1]
        if a != b:
            W[a][b] += 1
            W[b][a] += 1
    return W


def greedy_chain_order(symbols, W):
    if len(symbols) <= 1:
        return symbols[:]
    remaining = set(symbols)

    def total_w(s):
        row = W[s]
        return sum(row[t] for t in symbols if t != s)

    start = max(symbols, key=lambda s: (total_w(s), -s))
    order = [start]
    remaining.discard(start)
    cur = start
    while remaining:
        row = W[cur]
        nxt = max(remaining, key=lambda t: (row[t], -t))
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return order


def build_tree(symbols, W, prefix, addr):
    if len(symbols) == 1:
        addr[symbols[0]] = prefix if prefix else "0"
        return
    order = greedy_chain_order(symbols, W)
    mid = len(order) // 2
    build_tree(order[:mid], W, prefix + "0", addr)
    build_tree(order[mid:], W, prefix + "1", addr)


def main():
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    next(it)  # Dmax
    L = int(next(it))
    trace = [int(next(it)) for _ in range(L)]

    W = build_weight_matrix(N, trace)
    addr = {}
    build_tree(list(range(1, N + 1)), W, "", addr)

    out = [addr[i] for i in range(1, N + 1)]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
