# TIER: trivial
# Minimal-effort baseline: use ONLY bay 0, ignore the other N-1 bays
# entirely, and on that one bay run plain FCFS with no preemption --
# whoever has been waiting longest goes next, acuity/escalation is
# completely ignored for PRIORITY. This is the "didn't even try to use the
# department's real capacity" reference: no urgency awareness AND no
# parallelism. (We still have to locally track each patient's acuity and
# the public need() formula just to know how many steps a treatment takes
# on the one bay we use -- that bookkeeping is not a strategy, it's
# mechanical replay of the rules.)
import sys, json, math

inst = json.load(sys.stdin)
T, N, P = inst["T"], inst["N"], inst["P"]
arrival, L0, rate = inst["arrival"], inst["L0"], inst["rate"]
C = inst["C"]
base_need, slope = inst["base_need"], inst["slope"]


def need_of(a):
    if a <= C + 1e-9:
        return base_need
    return max(base_need, int(math.ceil(base_need + slope * (a - C) - 1e-9)))


status = ["not_arrived"] * P
acuity = [0.0] * P
queue = []

bay_patient = [None] * N
bay_progress = [0] * N
bay_need = [0] * N

assign_rows = []

for t in range(T):
    for p in range(P):
        if status[p] == "not_arrived" and arrival[p] == t:
            status[p] = "waiting"
            acuity[p] = L0[p]
            queue.append(p)
    for p in range(P):
        if status[p] == "waiting" and arrival[p] < t:
            acuity[p] += rate[p]

    row = [0] * N
    for b in range(min(1, N)):   # ONLY bay 0 -- the rest of the department sits idle
        if bay_patient[b] is not None:
            p = bay_patient[b]
            row[b] = p + 1
            bay_progress[b] += 1
            if bay_progress[b] >= bay_need[b]:
                status[p] = "done"
                bay_patient[b] = None
                bay_progress[b] = 0
            continue
        while queue and status[queue[0]] != "waiting":
            queue.pop(0)
        if queue:
            p = queue.pop(0)
            bay_patient[b] = p
            bay_progress[b] = 1
            bay_need[b] = need_of(acuity[p])
            status[p] = "treating"
            row[b] = p + 1
            if bay_progress[b] >= bay_need[b]:
                status[p] = "done"
                bay_patient[b] = None
                bay_progress[b] = 0
    assign_rows.append(row)

print(json.dumps({"assign": assign_rows}))
