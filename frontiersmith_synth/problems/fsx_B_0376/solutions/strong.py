# TIER: strong
# Correlation-aware ordering + local search. The prefix cache rewards keeping
# co-occurring stage clusters CONTIGUOUS (so a whole latent mining profile
# shares a long cached prefix), which marginal frequency alone misses. We seed
# from several constructions (frequency order, identity, random restarts) and
# run insertion-move local search on the real objective -- the exact distinct
# prefix-trie node count -- keeping the best order found. Deterministic (seeded).
import sys, json, random


def trie_nodes(order, probes):
    pos = {c: i for i, c in enumerate(order)}
    root = {}
    count = 0
    for S in probes:
        node = root
        for c in sorted(S, key=lambda x: pos[x]):
            nxt = node.get(c)
            if nxt is None:
                nxt = {}
                node[c] = nxt
                count += 1
            node = nxt
    return count


def freq_order(probes, C):
    f = [0] * C
    for S in probes:
        for c in S:
            f[c] += 1
    return sorted(range(C), key=lambda c: (-f[c], c))


def insertion_ls(order, probes, C, iters, rng):
    best = order[:]
    bn = trie_nodes(best, probes)
    for _ in range(iters):
        i = rng.randrange(C)
        j = rng.randrange(C)
        if i == j:
            continue
        o = best[:]
        x = o.pop(i)
        o.insert(j, x)
        n = trie_nodes(o, probes)
        if n < bn:
            bn = n
            best = o
    return bn, best


inst = json.load(sys.stdin)
C = inst["n_stages"]
probes = inst["probes"]
rng = random.Random(20250376)

starts = [freq_order(probes, C), list(range(C))]
for _ in range(4):
    p = list(range(C))
    rng.shuffle(p)
    starts.append(p)

best_order = starts[0]
best_n = trie_nodes(best_order, probes)
for s in starts:
    n, o = insertion_ls(s, probes, C, 2000, rng)
    if n < best_n:
        best_n = n
        best_order = o

print(json.dumps({"order": best_order}))
