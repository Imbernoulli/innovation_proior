# TIER: trivial
# Reproduce the evaluator's weak FUEL-BLIND NEAREST-HOP operator: from the starbase,
# always jump to the nearest unvisited system that still fits the remaining fuel,
# ignoring science value entirely.  This exactly matches q_base, so it scores ~0.1
# on every instance.
import sys, json, math

inst = json.load(sys.stdin)
N, L = inst["N"], inst["L"]
x, y = inst["x"], inst["y"]
cx, cy = inst["bx"], inst["by"]


def dist(ax, ay, bxx, byy):
    return math.hypot(ax - bxx, ay - byy)


used = 0.0
visited = [False] * N
route = []
while True:
    best_j = -1
    best_d = None
    for j in range(N):
        if visited[j]:
            continue
        d = dist(cx, cy, x[j], y[j])
        if used + d <= L + 1e-9 and (best_d is None or d < best_d):
            best_d = d
            best_j = j
    if best_j < 0:
        break
    used += best_d
    cx = x[best_j]
    cy = y[best_j]
    visited[best_j] = True
    route.append(best_j)

print(json.dumps({"route": route}))
