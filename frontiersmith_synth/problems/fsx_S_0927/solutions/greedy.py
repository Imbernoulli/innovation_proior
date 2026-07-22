# TIER: greedy
# Plain nearest-neighbor construction: start at stop 0, always hop to the
# closest unvisited stop. This is the single textbook recipe most solvers reach
# for first. It beats the x-sort baseline on most layouts, but it is a single,
# un-refined recipe with no notion of "what kind of territory is this" -- it
# gets systematically stranded on star/spoke networks (commits to one spoke all
# the way to the tip, then pays a huge jump back) and on winding corridors
# (nearest unvisited stop can be across the fold rather than along the path),
# and it never polishes the resulting loop.
import sys, json, math

inst = json.load(sys.stdin)
pts = [tuple(p) for p in inst["points"]]
n = inst["n"]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


visited = [False] * n
order = [0]
visited[0] = True
cur = 0
for _ in range(n - 1):
    best, bd = -1, float("inf")
    for j in range(n):
        if not visited[j]:
            d = dist(pts[cur], pts[j])
            if d < bd:
                bd, best = d, j
    order.append(best)
    visited[best] = True
    cur = best

print(json.dumps({"tour": order}))
