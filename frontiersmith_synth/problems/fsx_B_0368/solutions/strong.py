# TIER: strong
# Season-weighted greedy CONSTRUCTION scored against the real weighted prefix-cache
# objective. Build the global order left to right; at each slot try every remaining
# predicate in that slot (rest trailing in a fixed season-weighted-frequency order)
# and keep whichever maximizes total SEASON-WEIGHTED hit tokens. This captures both
# the season frequency of each check family and the ordering interactions between
# overlapping families that a blind frequency ranking misses -- yet it is still a
# local greedy, not the true (hard) optimum, so it leaves headroom.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
w = inst["weights"]
Q = inst["queries"]
QW = inst["qweights"]


def objective(order):
    rank = [0] * M
    for pos, a in enumerate(order):
        rank[a] = pos
    trie = {}
    hit = 0
    for q, qw in zip(Q, QW):
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
        hit += qw * matched
    return hit


# seed the tail with a season-WEIGHTED frequency ranking so the trailing term is sane
wfreq = [0] * M
for q, qw in zip(Q, QW):
    for a in q:
        wfreq[a] += qw
tail = sorted(range(M), key=lambda a: (-wfreq[a], -w[a], a))

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
