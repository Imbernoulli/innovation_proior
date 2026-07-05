# TIER: greedy
# First-appearance locality heuristic: order channels by the order in which they are first
# seen while scanning the training query stream. Captures some coarse locality (channels that
# appear early in the log go first) but ignores frequency and co-occurrence structure, so it
# clusters related channels only by accident.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
train = inst["train_queries"]

seen = {}
nxt = 0
for q in train:
    for c in q:
        if c not in seen:
            seen[c] = nxt
            nxt += 1
for c in range(N):
    if c not in seen:
        seen[c] = nxt
        nxt += 1

order = sorted(range(N), key=lambda c: seen[c])
print(json.dumps({"order": order}))
