# TIER: greedy
# The textbook approach: build the ordinary (unconstrained) Huffman code from the
# frequencies -- optimal for expected length WITHOUT the run constraint -- and
# use it as-is. Then simply VALIDATE it against the run ceiling: if it happens
# to already be run-legal, ship it; if not (which is what happens as soon as the
# distribution is skewed enough for Huffman to build a deep, comb-shaped branch,
# since a comb branch is a long run of the same bit), fall back to the safe flat
# code (same length for every symbol) rather than reasoning about how to reshape
# the tree around the constraint. This is a common, defensible engineering
# instinct ("try the optimum, validate, and fall back to something safe if it's
# infeasible") but it throws away all of the probability information exactly on
# the instances where the constraint actually bites.
import sys, heapq


def huffman_codes(items):
    heap = []
    counter = 0
    for sym, w in items:
        heap.append((w, counter, ('leaf', sym)))
        counter += 1
    heapq.heapify(heap)
    if len(heap) == 1:
        _, _, node = heap[0]
        return {node[1]: "0"}
    while len(heap) > 1:
        w1, t1, n1 = heapq.heappop(heap)
        w2, t2, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, counter, ('node', n1, n2)))
        counter += 1
    root = heap[0][2]
    codes = {}

    def dfs(node, path):
        if node[0] == 'leaf':
            codes[node[1]] = path if path else "0"
        else:
            dfs(node[1], path + "0")
            dfs(node[2], path + "1")

    dfs(root, "")
    return codes


def max_run(s):
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


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


def flat_code(n, d):
    L = 0
    while count_legal(L, d) < n:
        L += 1
    return enumerate_legal(L, d, n)


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); d = int(toks[1])
    p = [int(x) for x in toks[2:2 + n]]

    items = [(i, p[i]) for i in range(n)]
    codes = huffman_codes(items)
    words = [codes[i] for i in range(n)]

    if any(max_run(w) > d for w in words):
        words = flat_code(n, d)

    print(" ".join(words))


if __name__ == "__main__":
    main()
