# TIER: greedy
# Single-pass area growth in input order.  Each pool starts as a 1x1 cell on its
# organism and greedily expands one side at a time toward its target area, taking
# the largest feasible extension that does not overshoot the target, cross the
# shore boundary, or collide with an already-placed pool.  Earlier organisms grab
# space first and can starve later ones -- no reordering, no backtracking.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
W, H = inst["W"], inst["H"]
xs, ys, areas = inst["x"], inst["y"], inst["a"]

# start every pool as a unit cell on its survey point
rects = [[xs[i], ys[i], xs[i] + 1, ys[i] + 1] for i in range(n)]


def free_extent(idx, side, others):
    """Max units rect idx can extend on `side` without overlap/out-of-bounds."""
    x1, y1, x2, y2 = rects[idx]
    if side == 0:      # left  (decrease x1)
        limit = 0
        for j in others:
            ox1, oy1, ox2, oy2 = rects[j]
            if oy1 < y2 and y1 < oy2 and ox2 <= x1:
                if ox2 > limit:
                    limit = ox2
        return x1 - limit
    if side == 1:      # right (increase x2)
        limit = W
        for j in others:
            ox1, oy1, ox2, oy2 = rects[j]
            if oy1 < y2 and y1 < oy2 and ox1 >= x2:
                if ox1 < limit:
                    limit = ox1
        return limit - x2
    if side == 2:      # down  (decrease y1)
        limit = 0
        for j in others:
            ox1, oy1, ox2, oy2 = rects[j]
            if ox1 < x2 and x1 < ox2 and oy2 <= y1:
                if oy2 > limit:
                    limit = oy2
        return y1 - limit
    # side == 3       # up    (increase y2)
    limit = H
    for j in others:
        ox1, oy1, ox2, oy2 = rects[j]
        if ox1 < x2 and x1 < ox2 and oy1 >= y2:
            if oy1 < limit:
                limit = oy1
    return limit - y2


for i in range(n):
    others = [j for j in range(n) if j != i]
    target = areas[i]
    # cycle through sides until no side can help or target reached
    for _ in range(400):
        x1, y1, x2, y2 = rects[i]
        area = (x2 - x1) * (y2 - y1)
        if area >= target:
            break
        best_side = -1
        best_ext = 0
        for side in range(4):
            avail = free_extent(i, side, others)
            if avail <= 0:
                continue
            span = (y2 - y1) if side in (0, 1) else (x2 - x1)
            need = target - area
            # units on this side to (roughly) reach target, capped by availability
            step = need // span if span > 0 else 0
            if step < 1:
                step = 1
            step = min(step, avail)
            if step > best_ext:
                best_ext = step
                best_side = side
        if best_side < 0:
            break
        if best_side == 0:
            rects[i][0] -= best_ext
        elif best_side == 1:
            rects[i][2] += best_ext
        elif best_side == 2:
            rects[i][1] -= best_ext
        else:
            rects[i][3] += best_ext

print(json.dumps({"rects": rects}))
