# TIER: strong
# All the public data (directives + weights) is available, so the true weighted
# prefix-cache hit rate can be simulated exactly for ANY candidate order. This
# solver therefore optimizes the real objective directly:
#   1. seed with the marginal-frequency order (a good starting point),
#   2. deterministic hill-climb over the permutation using three move families --
#      adjacent swaps, move-to-front, and single-element re-insertion -- accepting
#      any move that strictly increases the simulated hit rate, until no move helps.
# The re-insertion / move-to-front moves capture co-occurrence structure that pure
# marginal frequency misses. The search stops at a local optimum (the permutation
# space is far too large to exhaust), so it beats greedy while leaving headroom.
import sys, json

inst = json.load(sys.stdin)
C = inst["n_clauses"]
weights = inst["weights"]
directives = inst["directives"]


def hit_rate(order):
    pos = [0] * len(order)
    for i, c in enumerate(order):
        pos[c] = i
    cache = set()
    total = 0
    hit = 0
    for d in directives:
        seq = sorted(d, key=lambda c: pos[c])
        prefixes = []
        t = ()
        for c in seq:
            t = t + (c,)
            prefixes.append(t)
        p = 0
        for k in range(len(prefixes)):
            if prefixes[k] in cache:
                p = k + 1
            else:
                break
        for i, c in enumerate(seq):
            w = weights[c]
            total += w
            if i < p:
                hit += w
        for t in prefixes:
            cache.add(t)
    return hit / total if total > 0 else 0.0


freq = [0] * C
for d in directives:
    for c in d:
        freq[c] += 1
order = sorted(range(C), key=lambda c: (-freq[c], -weights[c], c))
best = hit_rate(order)

MAX_PASSES = 60
for _ in range(MAX_PASSES):
    improved = False
    # adjacent swaps
    for i in range(C - 1):
        cand = order[:]
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        h = hit_rate(cand)
        if h > best + 1e-12:
            order, best, improved = cand, h, True
    # single-element re-insertion (includes move-to-front as a special case)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            cand = order[:]
            x = cand.pop(i)
            cand.insert(j, x)
            h = hit_rate(cand)
            if h > best + 1e-12:
                order, best, improved = cand, h, True
    if not improved:
        break

print(json.dumps({"order": order}))
