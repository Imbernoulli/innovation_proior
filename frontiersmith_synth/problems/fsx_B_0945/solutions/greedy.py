# TIER: greedy
# The obvious first idea: every day, move the herd to whichever paddock has
# the MOST standing grass right now (ties broken by nearest, then lowest
# index), then graze it. This never looks at the move-cost budget or at how
# grazing today reshapes tomorrow's diffusion neighbourhood -- it just chases
# instantaneous greenness. To know "right now" the candidate must replay the
# public dynamics itself (everything needed is in the instance).
#
# This is exactly the trap the family is named for: on fields with far-apart
# lush patches it ping-pongs between them paying huge relocation fees every
# single day, and on fields with one diffusion-fed cluster it keeps re-draining
# the cluster (and the neighbours that keep "lending" it grass) until the
# whole neighbourhood collapses.
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


inst = json.load(sys.stdin)
R, C, T = inst["R"], inst["C"], inst["T"]
r, D, D_req = inst["r"], inst["D"], inst["D_req"]
P = R * C
nbrs = [neighbors(i, R, C) for i in range(P)]

g = list(inst["g0"])
prev = inst["start"]
visits = []
for _ in range(T):
    best_key = None
    best_c = 0
    for i in range(P):
        d = manhattan(i, prev, C)
        key = (-g[i], d, i)
        if best_key is None or key < best_key:
            best_key = key
            best_c = i
    visits.append(best_c)
    eaten = g[best_c] if g[best_c] < D_req else D_req
    g[best_c] -= eaten
    g = evolve(g, nbrs, r, D)
    prev = best_c

print(json.dumps({"visits": visits}))
