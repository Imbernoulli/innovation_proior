# TIER: greedy
# Nearest-station chain (greedy nearest-neighbour).  Start at the base depot and
# repeatedly drive to the closest not-yet-inspected lift station.  This slashes the
# roster circuit dramatically, but it is myopic: the last few hops can be long
# "returns" across the mountain because greedy paints itself into a corner, and it
# never uncrosses edges -- so it leaves length on the table that 2-opt recovers.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
coords = inst["coords"]


def dist(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return int(round(math.sqrt(dx * dx + dy * dy + dz * dz)))


unvisited = set(range(1, n + 1))
order = []
cur = 0  # depot
while unvisited:
    nxt = min(unvisited, key=lambda j: (dist(coords[cur], coords[j]), j))
    order.append(nxt)
    unvisited.discard(nxt)
    cur = nxt

print(json.dumps({"order": order}))
