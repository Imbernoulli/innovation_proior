# TIER: greedy
# First-fit in ARRIVAL order (no reordering): load each container onto the
# lowest-index truck that still has room on BOTH axes, reusing earlier gaps but
# never sorting the queue.  Better than next-fit because it looks back at open
# trucks, but leaving arrival order intact wastes capacity vs decreasing-order.
import sys, json

inst = json.load(sys.stdin)
W, V = inst["W"], inst["V"]
mass, bulk = inst["mass"], inst["bulk"]
n = len(mass)

rem_m, rem_b = [], []          # remaining mass/bulk per open truck
assign = [0] * n
for i in range(n):
    m, b = mass[i], bulk[i]
    placed = -1
    for t in range(len(rem_m)):
        if rem_m[t] >= m and rem_b[t] >= b:
            placed = t
            break
    if placed < 0:
        rem_m.append(W - m); rem_b.append(V - b)
        assign[i] = len(rem_m) - 1
    else:
        rem_m[placed] -= m; rem_b[placed] -= b
        assign[i] = placed

print(json.dumps({"assign": assign}))
