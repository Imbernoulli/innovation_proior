# TIER: strong
"""
The insight: since the reaction network is a 2-level DAG (R,Z -> X,Y -> T),
firing order never changes the FINAL target count for a fixed multiset of
firings -- any order that fires every level-1 (precursor) reaction before
the level-2 (yield) reaction that needs it is feasible. So the true
objective reduces to a pure RESOURCE-ALLOCATION problem: choose how many
times to fire each "compound unit" --

    Direct_g : costs (dx_g*px_g) R, 0 Z -> dv_g T
    Combo_g  : costs (cx_g*px_g + cy_g*py_g) R, 1 Z -> cw_g T

subject to a total R budget and a total Z budget -- an UNBOUNDED 2D
knapsack. This is exactly the "precursor reservation" insight: Z (and the
Y_g stock it unlocks) must be deliberately RESERVED for the globally best
combo unit(s) instead of being ignored in favor of whichever single
reaction shows the best immediate per-firing number.

We solve the 2D unbounded knapsack by DP over (z, r) in O(N_Z*N_R*m), then
reconstruct the optimal firing multiset and finally emit a fully valid,
DAG-respecting sequence: all precursor (X/Y) firings first, then all
yield (D/C) firings.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    N_R = int(data[idx]); idx += 1
    N_Z = int(data[idx]); idx += 1
    groups = []
    for g in range(m):
        vals = [int(data[idx + k]) for k in range(7)]
        idx += 7
        groups.append(vals)

    # items: (costR, costZ, value, kind, group)
    items = []
    for g, (px, py, dx, dv, cx, cy, cw) in enumerate(groups):
        items.append((dx * px, 0, dv, "D", g))
        items.append((cx * px + cy * py, 1, cw, "C", g))

    NR, NZ = N_R, N_Z
    NEG = -1
    # dp[z][r] = best value; choice[z][r] = index into items used for the
    # LAST unit added on the optimal path reaching (z,r), or -1 if this
    # cell is just carried from (z-1,r) / (z,r-1) / is the base case.
    dp = [[0] * (NR + 1) for _ in range(NZ + 1)]
    choice = [[-1] * (NR + 1) for _ in range(NZ + 1)]  # -2 = carry-z, -3 = carry-r, >=0 item idx

    for z in range(NZ + 1):
        row = dp[z]
        for r in range(NR + 1):
            best = 0
            best_choice = -1
            if z > 0 and dp[z - 1][r] > best:
                best = dp[z - 1][r]
                best_choice = -2
            if r > 0 and row[r - 1] > best:
                best = row[r - 1]
                best_choice = -3
            for ii, (costR, costZ, value, kind, g) in enumerate(items):
                if costZ == 0:
                    if costR <= r:
                        cand = row[r - costR] + value
                        if cand > best:
                            best = cand
                            best_choice = ii
                else:
                    if z >= 1 and costR <= r:
                        cand = dp[z - 1][r - costR] + value
                        if cand > best:
                            best = cand
                            best_choice = ii
            row[r] = best
            choice[z][r] = best_choice

    # ---- backtrack to recover the multiset of item firings ----
    counts = [0] * len(items)
    z, r = NZ, NR
    while True:
        c = choice[z][r]
        if c == -1:
            break
        elif c == -2:
            z -= 1
        elif c == -3:
            r -= 1
        else:
            costR, costZ, value, kind, g = items[c]
            counts[c] += 1
            r -= costR
            z -= costZ

    # ---- emit a DAG-valid firing sequence: precursors first, then yields ----
    need_X = [0] * m
    need_Y = [0] * m
    yield_tokens = []
    for ii, (costR, costZ, value, kind, g) in enumerate(items):
        cnt = counts[ii]
        if cnt == 0:
            continue
        px, py, dx, dv, cx, cy, cw = groups[g]
        if kind == "D":
            need_X[g] += cnt * dx
            yield_tokens.extend(["%dD" % g] * cnt)
        else:
            need_X[g] += cnt * cx
            need_Y[g] += cnt * cy
            yield_tokens.extend(["%dC" % g] * cnt)

    precursor_tokens = []
    for g in range(m):
        precursor_tokens.extend(["%dX" % g] * need_X[g])
        precursor_tokens.extend(["%dY" % g] * need_Y[g])

    tokens = precursor_tokens + yield_tokens
    out = []
    out.append(str(len(tokens)))
    out.append(" ".join(tokens))
    print("\n".join(out))


if __name__ == "__main__":
    main()
