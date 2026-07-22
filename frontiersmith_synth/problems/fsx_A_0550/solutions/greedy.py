# TIER: greedy
# The obvious approach: build a wall right in front of the advancing fire.  Compute the
# fire's arrival-time field, take the cells the front reaches SOONEST that are still
# buildable in time (arrival >= build_time), and dispatch every crew there at t=0.
# This chases the CURRENT flame edge near the seed.  In open terrain the front simply
# streams around the short wall and re-converges, so the fire still funnels through the
# far chokepoint and burns the town behind it -- the locked crews were spent walling
# ground the fire was going to sweep past anyway.
import sys, json, heapq

inst = json.load(sys.stdin)
n = inst["n"]; res = inst["res"]; val = inst["value"]
wr, wc = inst["wind"]; ws = inst["wind_strength"]
b = inst["build_time"]; T = inst["horizon"]; C = inst["crews"]

INF = float("inf")
arrival = [INF] * (n * n)
done = [False] * (n * n)
pq = []
for (sr, sc) in inst["seeds"]:
    idx = sr * n + sc
    arrival[idx] = 0
    heapq.heappush(pq, (0, idx))
dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
while pq:
    a, idx = heapq.heappop(pq)
    if done[idx]:
        continue
    done[idx] = True
    r, c = divmod(idx, n)
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
                heapq.heappush(pq, (na, nidx))

# buildable cells the front reaches soonest (the current-edge wall)
cand = []
for idx in range(n * n):
    a = arrival[idx]
    if a != INF and b <= a <= T:
        r, c = divmod(idx, n)
        cand.append((a, -val[r][c], r, c))
cand.sort()
breaks = [[r, c, 0] for (_, _, r, c) in cand[:C]]
print(json.dumps({"breaks": breaks}))
