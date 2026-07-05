# TIER: strong
# Best-of-several construct-then-refine.  For each of several growth ORDERS
# (input order, small-target-first, large-target-first) we grow every pool toward
# its target one side at a time, then run a seeded local search that grows/shrinks
# random sides accepting any non-worsening move (an overshooting pool can shrink,
# freeing space a neighbour later reclaims).  We keep whichever finished layout has
# the highest self-computed total quality.  Because the input-order construction is
# one of the candidates, this never scores below the single-pass greedy tier, and
# the alternate orders + refinement usually beat it -- yet the ~92-99% packing
# pressure keeps the normalized score well below 1.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
W, H = inst["W"], inst["H"]
xs, ys, areas = inst["x"], inst["y"], inst["a"]


def rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


def free_extent(rc, idx, side):
    x1, y1, x2, y2 = rc[idx]
    if side == 0:
        limit = 0
        for j in range(n):
            if j == idx:
                continue
            ox1, oy1, ox2, oy2 = rc[j]
            if oy1 < y2 and y1 < oy2 and ox2 <= x1 and ox2 > limit:
                limit = ox2
        return x1 - limit
    if side == 1:
        limit = W
        for j in range(n):
            if j == idx:
                continue
            ox1, oy1, ox2, oy2 = rc[j]
            if oy1 < y2 and y1 < oy2 and ox1 >= x2 and ox1 < limit:
                limit = ox1
        return limit - x2
    if side == 2:
        limit = 0
        for j in range(n):
            if j == idx:
                continue
            ox1, oy1, ox2, oy2 = rc[j]
            if ox1 < x2 and x1 < ox2 and oy2 <= y1 and oy2 > limit:
                limit = oy2
        return y1 - limit
    limit = H
    for j in range(n):
        if j == idx:
            continue
        ox1, oy1, ox2, oy2 = rc[j]
        if ox1 < x2 and x1 < ox2 and oy1 >= y2 and oy1 < limit:
            limit = oy1
    return limit - y2


def pool_quality(rc, idx):
    x1, y1, x2, y2 = rc[idx]
    if not (x1 <= xs[idx] < x2 and y1 <= ys[idx] < y2):
        return 0.0
    s = (x2 - x1) * (y2 - y1)
    a = areas[idx]
    p = 1.0 - min(a, s) / max(a, s)
    return 1.0 - p * p


def total_quality(rc):
    return sum(pool_quality(rc, i) for i in range(n))


def construct(order):
    rc = [[xs[i], ys[i], xs[i] + 1, ys[i] + 1] for i in range(n)]
    for i in order:
        target = areas[i]
        for _ in range(400):
            x1, y1, x2, y2 = rc[i]
            area = (x2 - x1) * (y2 - y1)
            if area >= target:
                break
            best_side, best_ext = -1, 0
            for side in range(4):
                avail = free_extent(rc, i, side)
                if avail <= 0:
                    continue
                span = (y2 - y1) if side in (0, 1) else (x2 - x1)
                need = target - area
                step = need // span if span > 0 else 0
                if step < 1:
                    step = 1
                step = min(step, avail)
                if step > best_ext:
                    best_ext, best_side = step, side
            if best_side < 0:
                break
            if best_side == 0:
                rc[i][0] -= best_ext
            elif best_side == 1:
                rc[i][2] += best_ext
            elif best_side == 2:
                rc[i][1] -= best_ext
            else:
                rc[i][3] += best_ext
    return rc


def refine(rc, iters, rnd):
    for _ in range(iters):
        i = rnd(0, n - 1)
        side = rnd(0, 3)
        grow = rnd(0, 1)
        before = pool_quality(rc, i)
        x1, y1, x2, y2 = rc[i]
        if grow == 1:
            avail = free_extent(rc, i, side)
            if avail <= 0:
                continue
            d = rnd(1, avail)
            if side == 0:
                rc[i][0] -= d
            elif side == 1:
                rc[i][2] += d
            elif side == 2:
                rc[i][1] -= d
            else:
                rc[i][3] += d
        else:
            if side == 0:
                cap = min(x2 - 1, xs[i]) - x1
                if cap <= 0:
                    continue
                rc[i][0] += rnd(1, cap)
            elif side == 1:
                cap = x2 - max(x1 + 1, xs[i] + 1)
                if cap <= 0:
                    continue
                rc[i][2] -= rnd(1, cap)
            elif side == 2:
                cap = min(y2 - 1, ys[i]) - y1
                if cap <= 0:
                    continue
                rc[i][1] += rnd(1, cap)
            else:
                cap = y2 - max(y1 + 1, ys[i] + 1)
                if cap <= 0:
                    continue
                rc[i][3] -= rnd(1, cap)
        if pool_quality(rc, i) < before - 1e-12:
            rc[i] = [x1, y1, x2, y2]
    return rc


base_seed = 1000003 + n * 97 + (sum(areas) % 1000000)
iters = 300 * n + 4000

orders = [
    list(range(n)),                                   # input order (>= greedy floor)
    sorted(range(n), key=lambda i: areas[i]),         # small targets first
    sorted(range(n), key=lambda i: -areas[i]),        # large targets first
]

best_rc, best_q = None, -1.0
for k, order in enumerate(orders):
    rc = construct(order)
    rc = refine(rc, iters, rng(base_seed + 7919 * k))
    q = total_quality(rc)
    if q > best_q:
        best_q, best_rc = q, rc

print(json.dumps({"rects": best_rc}))
