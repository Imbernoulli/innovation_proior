# TIER: strong
# Marginal-gain greedy (submodular max-coverage) + local search.
#   1) Build wells one at a time: each step drill at the cell whose window adds the
#      most heat NOT YET tapped by chosen wells.  This explicitly discounts overlap,
#      so wells spread across separate plumes instead of piling on the tallest one.
#   2) Local search: repeatedly try moving any single well to any free cell; keep a
#      move if it strictly increases the union yield.  Deterministic scan order and a
#      fixed pass cap keep it reproducible within the op budget.
# The (1-1/e) coverage guarantee plus swap refinement beats both overlap-blind rules,
# but the loose full-tract upper bound keeps the normalized score below 1.0.
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

covered_amt = {}   # cell idx -> heat, only for currently covered cells
chosen = []
chosen_set = set()


def marginal(cell):
    g = 0
    for idx in win[cell]:
        if idx not in covered_amt:
            g += heat[idx // n][idx % n]
    return g


# 1) submodular greedy build
for _ in range(k):
    best = None; best_g = -1
    for r in range(n):
        for c in range(n):
            if (r, c) in chosen_set:
                continue
            g = marginal((r, c))
            if g > best_g:
                best_g = g; best = (r, c)
    chosen.append(best); chosen_set.add(best)
    for idx in win[best]:
        if idx not in covered_amt:
            covered_amt[idx] = heat[idx // n][idx % n]


def total_union(wells):
    cov = set()
    for cell in wells:
        cov.update(win[cell])
    s = 0
    for idx in cov:
        s += heat[idx // n][idx % n]
    return s


# 2) local search: relocate one well at a time
cur = total_union(chosen)
for _ in range(8):
    improved = False
    for i in range(len(chosen)):
        base_others = chosen[:i] + chosen[i + 1:]
        best_cell = chosen[i]; best_val = cur
        for r in range(n):
            for c in range(n):
                if (r, c) in chosen_set and (r, c) != chosen[i]:
                    continue
                val = total_union(base_others + [(r, c)])
                if val > best_val:
                    best_val = val; best_cell = (r, c)
        if best_cell != chosen[i]:
            chosen_set.discard(chosen[i])
            chosen[i] = best_cell
            chosen_set.add(best_cell)
            cur = best_val
            improved = True
    if not improved:
        break

wells = [[r, c] for (r, c) in chosen]
print(json.dumps({"wells": wells}))
