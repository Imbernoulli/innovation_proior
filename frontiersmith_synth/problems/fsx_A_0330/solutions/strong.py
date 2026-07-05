# TIER: strong
# Nearest-station chain + 2-opt edge-uncrossing local search.
#   1) Build an initial circuit by greedy nearest-neighbour from the depot.
#   2) 2-opt: repeatedly look for a pair of circuit edges (i,i+1) and (j,j+1) whose
#      total length drops if the segment between them is reversed; apply the best
#      improving reversal.  Iterate in a deterministic scan order until no improving
#      move remains or a fixed pass budget is exhausted (the op budget).
# 2-opt removes the long crossing "return" hops greedy leaves behind, so it beats the
# nearest-station chain, yet the loose lower bound keeps the normalized score < 1.0.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
coords = inst["coords"]


def dist(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return int(round(math.sqrt(dx * dx + dy * dy + dz * dz)))


# precompute depot-relative and full distance via a helper closure
def D(i, j):
    return dist(coords[i], coords[j])


# 1) greedy nearest-neighbour from depot (index 0)
unvisited = set(range(1, n + 1))
order = []
cur = 0
while unvisited:
    nxt = min(unvisited, key=lambda j: (D(cur, j), j))
    order.append(nxt)
    unvisited.discard(nxt)
    cur = nxt

# Represent the closed circuit as node sequence with depot at both ends.
# route = [0, order..., 0]
route = [0] + order + [0]

# 2) 2-opt on the closed circuit.  Edges are (route[k], route[k+1]).
m = len(route)
MAX_PASSES = 60
for _ in range(MAX_PASSES):
    improved = False
    for i in range(0, m - 2):
        a, b = route[i], route[i + 1]
        dab = D(a, b)
        for k in range(i + 2, m - 1):
            c, d = route[k], route[k + 1]
            # reversing route[i+1 .. k] replaces edges (a,b)+(c,d) with (a,c)+(b,d)
            delta = (D(a, c) + D(b, d)) - (dab + D(c, d))
            if delta < 0:
                route[i + 1:k + 1] = route[i + 1:k + 1][::-1]
                improved = True
                b = route[i + 1]
                dab = D(a, b)
    if not improved:
        break

final = [v for v in route if v != 0]
print(json.dumps({"order": final}))
