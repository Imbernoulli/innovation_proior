# TIER: invalid
# Over-concentrate the whole stockpile on one "hotspot" leaf district. A leaf's
# stock can only cover its OWN demand (no descendants to pool for), so every
# other district is left completely unserved -> the network fill rate collapses
# far below the service target -> the feasibility gate rejects it -> score 0.
import sys, json
inst = json.load(sys.stdin)
N = inst["N"]; B = inst["B"]; parent = inst["parent"]
has_child = [False] * (N + 1)
for i in range(1, N + 1):
    has_child[parent[i]] = True
leaf = next(i for i in range(1, N + 1) if not has_child[i])
x = [0] * (N + 1)
x[leaf] = B
print(json.dumps({"stock": x}))
