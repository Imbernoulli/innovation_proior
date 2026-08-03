# TIER: strong
# Greedy CONSTRUCTION that scores against the real prefix-cache objective. Build the
# global order left to right; at each slot, try every remaining predicate in that slot
# (with the rest trailing in a fixed order) and keep the one that maximizes total hit
# tokens. This captures ordering interactions between overlapping check families that
# pure frequency ranking misses, so it matches or beats the frequency layout -- but it
# is still a local greedy, not the true (hard) optimum, leaving headroom.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
w = inst["weights"]
Q = inst["queries"]


def objective(order):
    rank = [0] * M
    for pos, a in enumerate(order):
        rank[a] = pos
    trie = {}
    hit = 0
    for q in Q:
        seq = sorted(q, key=lambda a: rank[a])
        node = trie
        matched = 0
        broke = False
        for a in seq:
            if (not broke) and (a in node):
                matched += w[a]
                node = node[a]
            else:
                broke = True
                nxt = node.get(a)
                if nxt is None:
                    nxt = {}
                    node[a] = nxt
                node = nxt
        hit += matched
    return hit


# seed the tail with a frequency ranking so the "rest trailing" term is sensible
freq = [0] * M
for q in Q:
    for a in q:
        freq[a] += 1
tail = sorted(range(M), key=lambda a: (-freq[a], -w[a], a))

order = []
remaining = list(tail)
while remaining:
    best = None
    best_v = None
    for a in remaining:
        rest = [x for x in remaining if x != a]
        v = objective(order + [a] + rest)
        if best_v is None or v > best_v:
            best_v = v
            best = a
    order.append(best)
    remaining.remove(best)

print(json.dumps({"order": order}))
