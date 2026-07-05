# TIER: greedy
# Value-density insertion greedy.  Repeatedly take the unvisited system with the
# best score = prize / (extra fuel it would cost to insert it into the current
# route at its cheapest position), as long as the insertion still fits the fuel
# budget.  Unlike the fuel-blind nearest hop this actually chases the rich anomalies
# and packs cheap detours -- but it never revises a placement, so early insertions
# that later block a better ordering leave value stranded.
import sys, json, math

inst = json.load(sys.stdin)
N, L = inst["N"], inst["L"]
x, y, p = inst["x"], inst["y"], inst["p"]
bx, by = inst["bx"], inst["by"]


def dist(ax, ay, bxx, byy):
    return math.hypot(ax - bxx, ay - byy)


# path is the ordered list of system indices; length is the flown distance
# starbase -> path[0] -> ... -> path[-1] (open route, no return).
def insertion_cost(path, length, j):
    """Cheapest extra fuel to insert system j, and the position; open path."""
    if not path:
        return dist(bx, by, x[j], y[j]), 0
    best_extra = None
    best_pos = 0
    # position 0: before current first (new start from starbase)
    d_new = dist(bx, by, x[j], y[j]) + dist(x[j], y[j], x[path[0]], y[path[0]])
    d_old = dist(bx, by, x[path[0]], y[path[0]])
    best_extra = d_new - d_old
    best_pos = 0
    # internal positions between consecutive systems
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        extra = (dist(x[a], y[a], x[j], y[j]) + dist(x[j], y[j], x[b], y[b])
                 - dist(x[a], y[a], x[b], y[b]))
        if extra < best_extra:
            best_extra = extra
            best_pos = k + 1
    # append at the end
    last = path[-1]
    extra_end = dist(x[last], y[last], x[j], y[j])
    if extra_end < best_extra:
        best_extra = extra_end
        best_pos = len(path)
    return best_extra, best_pos


path = []
length = 0.0
used = set()
while True:
    best_j = -1
    best_score = 0.0
    best_extra = 0.0
    best_pos = 0
    for j in range(N):
        if j in used:
            continue
        extra, pos = insertion_cost(path, length, j)
        if length + extra > L + 1e-9:
            continue
        denom = extra if extra > 1e-9 else 1e-9
        score = p[j] / denom
        if score > best_score + 1e-15:
            best_score = score
            best_j = j
            best_extra = extra
            best_pos = pos
    if best_j < 0:
        break
    path.insert(best_pos, best_j)
    length += best_extra
    used.add(best_j)

print(json.dumps({"route": path}))
