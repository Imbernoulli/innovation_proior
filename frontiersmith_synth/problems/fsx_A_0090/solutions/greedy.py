# TIER: greedy
# Submodular marginal-gain greedy (no refinement).  Build the well field one well at
# a time: at each step drill at the cell whose (2R+1)x(2R+1) window adds the most
# heat NOT YET tapped by the wells chosen so far.  This discounts overlap, so wells
# spread across separate plumes and it clears the overlap-blind baseline easily.
# It is myopic, though: an early well can block a better later configuration, and it
# never revisits a placement -- so it leaves yield on the table that local search
# (the strong tier) can still recover.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; R = inst["radius"]; heat = inst["heat"]


def window(r, c):
    r0 = max(0, r - R); r1 = min(n - 1, r + R)
    c0 = max(0, c - R); c1 = min(n - 1, c + R)
    out = []
    for rr in range(r0, r1 + 1):
        base = rr * n
        for cc in range(c0, c1 + 1):
            out.append(base + cc)
    return out


win = {}
for r in range(n):
    for c in range(n):
        win[(r, c)] = window(r, c)

covered = set()
chosen = []
chosen_set = set()

for _ in range(k):
    best = None; best_g = -1
    for r in range(n):
        for c in range(n):
            if (r, c) in chosen_set:
                continue
            g = 0
            for idx in win[(r, c)]:
                if idx not in covered:
                    g += heat[idx // n][idx % n]
            if g > best_g:
                best_g = g; best = (r, c)
    chosen.append(best); chosen_set.add(best)
    covered.update(win[best])

wells = [[r, c] for (r, c) in chosen]
print(json.dumps({"wells": wells}))
