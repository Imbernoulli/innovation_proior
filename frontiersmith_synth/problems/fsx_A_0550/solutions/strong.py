# TIER: strong
# INSIGHT: spend commitment on the FUTURE frontier, not the present flame edge.
#   1) Simulate the fire with no firebreaks to get the arrival-time field AND the
#      shortest-path tree (which cell each burning cell caught fire FROM).
#   2) Score every cell by "gated value" = the total asset value of the sub-tree that
#      the fire can only reach by passing THROUGH that cell.  A narrow chokepoint that
#      the wind funnels the whole front through -- the gap in the ridge that guards the
#      town -- gates a huge amount of value, even though nothing is burning there yet.
#   3) Consider only cells that are still BUILDABLE (arrival >= build_time, so a break
#      dispatched now finishes before the front arrives) and rank them by gated value.
#   4) Greedily commit crews to the top gated cells, re-simulating after each pick so
#      the choice accounts for how earlier breaks reroute the fire, up to the crew cap.
# Pre-positioning the wall on the anticipated frontier seals off the ground behind the
# chokepoint; the ground near the seed still burns and crews are scarce, so the score
# stays below 1 -- there is headroom above this reference (dispatch timing, min-cuts,
# multi-gap trade-offs are left on the table).
import sys, json, heapq

inst = json.load(sys.stdin)
n = inst["n"]; res = inst["res"]; val = inst["value"]
wr, wc = inst["wind"]; ws = inst["wind_strength"]
b = inst["build_time"]; T = inst["horizon"]; C = inst["crews"]
seeds = inst["seeds"]
dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = float("inf")


def simulate(breaks):
    """Return (saved_value, arrival_list, pred_list) under the given breaks."""
    comp = {}
    for (r, c, d) in breaks:
        comp[r * n + c] = d + b
    arrival = [INF] * (n * n)
    pred = [-1] * (n * n)
    done = [False] * (n * n)
    pq = []
    for (sr, sc) in seeds:
        idx = sr * n + sc
        if arrival[idx] > 0:
            arrival[idx] = 0
            heapq.heappush(pq, (0, idx))
    deficit = 0.0
    Tf = float(T)
    while pq:
        a, idx = heapq.heappop(pq)
        if done[idx]:
            continue
        if a > T:
            break
        done[idx] = True
        ct = comp.get(idx)
        if ct is not None and ct <= a:
            continue
        r, c = divmod(idx, n)
        deficit += val[r][c] * (1.0 - a / Tf)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                nidx = nr * n + nc
                if done[nidx]:
                    continue
                cost = res[nr][nc] + ws * (-(dr * wr + dc * wc))
                if cost < 1:
                    cost = 1
                na = a + cost
                if na < arrival[nidx]:
                    arrival[nidx] = na
                    pred[nidx] = idx
                    heapq.heappush(pq, (na, nidx))
    total = sum(sum(row) for row in val)
    return total - deficit, arrival, pred


base_saved, arrival, pred = simulate([])

# gated value: sub-tree value that can only be reached through each cell
order = sorted((a, i) for i, a in enumerate(arrival) if a != INF and a <= T)
gated = [0] * (n * n)
for _a, idx in order:
    r, c = divmod(idx, n)
    gated[idx] += val[r][c]
for _a, idx in reversed(order):      # leaves (high arrival) first
    p = pred[idx]
    if p != -1:
        gated[p] += gated[idx]

# buildable candidates on the anticipated frontier, richest gate first
cand = []
for idx in range(n * n):
    a = arrival[idx]
    if a != INF and b <= a <= T:
        cand.append((-gated[idx], idx))
cand.sort()
cand_idx = [idx for (_, idx) in cand[:20]]

chosen = []
chosen_set = set()
cur_saved = base_saved
for _ in range(C):
    best_gain = 0
    best_idx = -1
    for idx in cand_idx:
        if idx in chosen_set:
            continue
        r, c = divmod(idx, n)
        trial = chosen + [[r, c, 0]]
        s, _arr, _pr = simulate(trial)
        if s - cur_saved > best_gain:
            best_gain = s - cur_saved
            best_idx = idx
    if best_idx < 0:
        break
    r, c = divmod(best_idx, n)
    chosen.append([r, c, 0])
    chosen_set.add(best_idx)
    cur_saved += best_gain

print(json.dumps({"breaks": chosen}))
