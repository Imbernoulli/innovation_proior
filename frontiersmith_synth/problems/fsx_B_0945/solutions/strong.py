# TIER: strong
# Insight: turn "when should I come back?" into a number. We estimate the
# regrowth-diffusion recovery time constant tau -- how many undisturbed days a
# freshly-grazed paddock needs, while sitting next to paddocks at the grid's
# typical resting level, to climb back to the daily requirement -- by
# replaying the exact public dynamics on a tiny local model. We then build a
# short, compact tour (nearest-neighbour order over the k paddocks closest to
# the start) for a handful of candidate tour sizes k centred on tau, REPLAY
# the whole season internally for each candidate (own re-implementation of the
# public formula) and keep whichever schedule the internal replay scores
# highest. This directly matches the revisit period to the recovery time
# constant instead of reacting to today's greenest cell.
import sys, json


def neighbors(i, R, C):
    row, col = divmod(i, C)
    out = []
    if row > 0:
        out.append(i - C)
    if row < R - 1:
        out.append(i + C)
    if col > 0:
        out.append(i - 1)
    if col < C - 1:
        out.append(i + 1)
    return out


def manhattan(a, b, C):
    ra, ca = divmod(a, C)
    rb, cb = divmod(b, C)
    return abs(ra - rb) + abs(ca - cb)


def evolve(g, nbrs, r, D):
    P = len(g)
    out = [0.0] * P
    for i in range(P):
        gi = g[i]
        s = 0.0
        for j in nbrs[i]:
            s += g[j] - gi
        val = gi + r * gi * (1.0 - gi) + D * s
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        out[i] = val
    return out


def simulate(R, C, T, r, D, D_req, mf, mpd, sm, start, g0, nbrs, visits):
    g = list(g0)
    prev = start
    total_intake = total_supp = total_move = 0.0
    for t in range(T):
        c = visits[t]
        if c != prev:
            total_move += mf + mpd * manhattan(c, prev, C)
        eaten = g[c] if g[c] < D_req else D_req
        total_intake += eaten
        total_supp += sm * (D_req - eaten)
        g[c] -= eaten
        g = evolve(g, nbrs, r, D)
        prev = c
    return total_intake - total_supp - total_move


def estimate_tau(r, D, D_req, avg_deg, resting_level):
    """Days for a freshly-grazed cell (starts at 0) to reach D_req, assuming
    its neighbours hover near `resting_level` (a small local scalar model of
    the diffusion pull). Capped so the search stays cheap."""
    g = 0.0
    for day in range(1, 400):
        g = g + r * g * (1.0 - g) + D * avg_deg * (resting_level - g)
        if g < 0.0:
            g = 0.0
        elif g > 1.0:
            g = 1.0
        if g >= D_req:
            return day
    return 400


def nn_tour(cells, start_pos, C):
    remaining = list(cells)
    tour = []
    cur = start_pos
    while remaining:
        remaining.sort(key=lambda c: (manhattan(c, cur, C), c))
        nxt = remaining.pop(0)
        tour.append(nxt)
        cur = nxt
    return tour


inst = json.load(sys.stdin)
R, C, T = inst["R"], inst["C"], inst["T"]
r, D, D_req = inst["r"], inst["D"], inst["D_req"]
mf, mpd, sm = inst["move_fixed"], inst["move_per_dist"], inst["supp_mult"]
start = inst["start"]
g0 = list(inst["g0"])
P = R * C
nbrs = [neighbors(i, R, C) for i in range(P)]

# --- resting level: let the whole grid coast untouched for a bit to see
# what "typical" neighbour grass looks like (informs the local tau model). ---
g_rest = list(g0)
for _ in range(min(30, T)):
    g_rest = evolve(g_rest, nbrs, r, D)
resting_level = sum(g_rest) / P
avg_deg = sum(len(n) for n in nbrs) / P

tau = estimate_tau(r, D, D_req, avg_deg, resting_level)
tau = max(1, min(P, tau))

# --- candidate tour sizes: centred on tau, plus a few robustness anchors ---
candidates = set()
for delta in (-3, -2, -1, 0, 1, 2, 3, 5, 8):
    k = tau + delta
    if 1 <= k <= P:
        candidates.add(k)
candidates.update({1, max(1, P // 4), max(1, P // 2), P})

best_visits = None
best_obj = None
for k in sorted(candidates):
    cells = sorted(range(P), key=lambda c: (manhattan(c, start, C), c))[:k]
    tour = nn_tour(cells, start, C)
    visits = [tour[t % len(tour)] for t in range(T)]
    obj = simulate(R, C, T, r, D, D_req, mf, mpd, sm, start, g0, nbrs, visits)
    if best_obj is None or obj > best_obj:
        best_obj = obj
        best_visits = visits

print(json.dumps({"visits": best_visits}))
