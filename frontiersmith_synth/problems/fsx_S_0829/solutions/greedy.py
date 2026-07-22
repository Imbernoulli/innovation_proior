# TIER: greedy
# The obvious first algorithm: nearest-neighbor construction from point 0,
# always hop to the closest unvisited point. This is a fine general-purpose
# recipe and needs no knowledge of the instance's layout -- which is exactly
# its weakness. On a strongly clustered point set, NN greedily empties out
# each cluster it visits, then must occasionally take a long "return" jump
# across the map to an early-skipped, far-away cluster; with only a small
# shared refine budget downstream, those few bad jumps are often too many
# to fully iron out. This solution never looks at the shape of the point
# set at all, so it cannot see that coming.
import sys, json, math


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def nearest_neighbor(points, start=0):
    n = len(points)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        best_j, best_d = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                d = dist(points[cur], points[j])
                if d < best_d:
                    best_d, best_j = d, j
        visited[best_j] = True
        tour.append(best_j)
        cur = best_j
    return tour


inst = json.load(sys.stdin)
points = inst["points"]
tour = nearest_neighbor(points, start=0)
print(json.dumps({"tour": tour}))
