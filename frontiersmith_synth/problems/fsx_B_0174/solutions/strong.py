# TIER: strong
# Frequency-seeded local search on the actual reuse objective.
# Maximising reused calibration steps == minimising the number of distinct
# prefixes in the trie of tool-string sequences. We start from the
# frequency-descending order and hill-climb with pairwise swaps, evaluating the
# true reuse each time, so common modules stay near the front AND tightly
# co-occurring module groups become contiguous (which plain frequency-sort
# fails to do). Deterministic: no randomness.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
wells = inst["wells"]


def reuse(order):
    rank = {m: i for i, m in enumerate(order)}
    trie = {}
    reused = 0
    for w in wells:
        seq = sorted(w, key=lambda m: rank[m])
        node = trie
        i = 0
        while i < len(seq) and seq[i] in node:
            node = node[seq[i]]
            i += 1
        reused += i
        for s in seq[i:]:
            node[s] = {}
            node = node[s]
    return reused


cnt = [0] * K
for w in wells:
    for m in w:
        cnt[m] += 1
order = sorted(range(K), key=lambda m: (-cnt[m], m))

best = reuse(order)
improved = True
passes = 0
while improved and passes < 60:
    improved = False
    passes += 1
    for i in range(K):
        for j in range(i + 1, K):
            order[i], order[j] = order[j], order[i]
            r = reuse(order)
            if r > best:
                best = r
                improved = True
            else:
                order[i], order[j] = order[j], order[i]

print(json.dumps({"order": order}))
