# TIER: trivial
# Uninformed query order: ignore the feature geometry entirely and send pool
# examples in a fixed (seeded) RANDOM order.  This is exactly the weak anchor the
# evaluator uses to define the 0.1 baseline, so it scores ~0.1 on every instance:
# it wastes budget re-sampling the common clusters and covers the rare classes no
# faster than chance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

# deterministic 64-bit LCG shuffle (no reliance on labels or geometry)
state = (2654435761 * (n + 12345) + 1013904223) & ((1 << 64) - 1)


def u():
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)


order = list(range(n))
for i in range(n - 1, 0, -1):
    j = int(u() * (i + 1))
    if j > i:
        j = i
    order[i], order[j] = order[j], order[i]

print(json.dumps({"order": order}))
