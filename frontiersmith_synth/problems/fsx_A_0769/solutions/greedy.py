# TIER: greedy
# Chase-the-sickest: every step, point every bay at whichever unclaimed
# waiting patient currently has the highest acuity. If a bay is already
# treating someone and a waiting patient looks worse right now, PREEMPT --
# this is the "obvious" reactive triage rule (always help whoever looks
# worst), and it is exactly what a naive coder writes first for an ER
# scheduling problem. It reacts to crises instead of anticipating them, so
# it repeatedly preempts bays as flashier newcomers arrive (paying the
# switch penalty + setup downtime over and over) while quieter,
# fast-climbing patients get ignored until THEY also look "worst" -- often
# too late.
import sys, json, math

inst = json.load(sys.stdin)
T, N, P = inst["T"], inst["N"], inst["P"]
arrival, L0, rate = inst["arrival"], inst["L0"], inst["rate"]
C, D = inst["C"], inst["D"]
base_need, slope = inst["base_need"], inst["slope"]


def need_of(a):
    if a <= C + 1e-9:
        return base_need
    return max(base_need, int(math.ceil(base_need + slope * (a - C) - 1e-9)))


status = ["not_arrived"] * P
acuity = [0.0] * P

bay_status = ["idle"] * N
bay_patient = [None] * N
bay_progress = [0] * N
bay_need = [0] * N
bay_setup_remaining = [0] * N

assign_rows = []
S = inst["S"]

for t in range(T):
    for p in range(P):
        if status[p] == "not_arrived" and arrival[p] == t:
            status[p] = "waiting"
            acuity[p] = L0[p]
    for p in range(P):
        if status[p] == "waiting" and arrival[p] < t:
            acuity[p] += rate[p]
            if acuity[p] > D + 1e-9:
                status[p] = "dead"

    waiting = sorted((p for p in range(P) if status[p] == "waiting"),
                      key=lambda p: -acuity[p])
    claimed = set()
    row = [0] * N

    for b in range(N):
        if bay_status[b] == "setup":
            bay_setup_remaining[b] -= 1
            if bay_setup_remaining[b] <= 0:
                bay_status[b] = "idle"
            continue

        best = None
        for p in waiting:
            if p not in claimed:
                best = p
                break

        if bay_status[b] == "treating":
            cur = bay_patient[b]
            cur_level = bay_start_level = acuity[cur]  # frozen value tracked locally
            if best is not None and acuity[best] > cur_level:
                # preempt: chase the worse-looking newcomer
                claimed.add(best)
                row[b] = best + 1
                status[cur] = "waiting"
                bay_status[b] = "setup"
                bay_patient[b] = None
                bay_progress[b] = 0
                bay_setup_remaining[b] = S - 1
                if bay_setup_remaining[b] <= 0:
                    bay_status[b] = "idle"
            else:
                row[b] = cur + 1
                bay_progress[b] += 1
                if bay_progress[b] >= bay_need[b]:
                    status[cur] = "done"
                    bay_status[b] = "idle"
                    bay_patient[b] = None
                    bay_progress[b] = 0
            continue

        # idle
        if best is not None:
            claimed.add(best)
            row[b] = best + 1
            bay_status[b] = "treating"
            bay_patient[b] = best
            bay_progress[b] = 1
            bay_need[b] = need_of(acuity[best])
            status[best] = "treating"
            if bay_progress[b] >= bay_need[b]:
                status[best] = "done"
                bay_status[b] = "idle"
                bay_patient[b] = None
                bay_progress[b] = 0

    assign_rows.append(row)

print(json.dumps({"assign": assign_rows}))
