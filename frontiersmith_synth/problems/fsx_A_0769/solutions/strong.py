# TIER: strong
# Insight: every waiting patient's own public rate/acuity gives a
# DETERMINISTIC deadline -- time_to_tip = (C - acuity) / rate, the number of
# steps until they cross the mid threshold and become a reactive (0.35x,
# longer) rescue instead of a cheap full-value treatment. Rather than
# reacting to whoever currently LOOKS worst, fill idle bays with whoever
# is CLOSEST to tipping (already-tipped-but-alive patients get first call,
# ranked by their own time-to-death, since letting them die is catastrophic;
# among not-yet-tipped patients, smallest time-to-tip goes first). Crucially,
# we almost NEVER preempt a bay that is already mid-treatment -- every
# preemption pays switch_penalty + S steps of downtime AND throws away the
# abandoned patient's progress, which is worse than just finishing them and
# grabbing the next bay that frees up. The ONE exception: if a waiting
# patient will die on the very next step and no bay is idle, sacrifice the
# single cheapest in-progress bay (the one whose patient started at or below
# C, i.e. was never a crisis case, and has made the least progress) to save
# a life -- a bounded, rare trade, not a habit.
import sys, json, math

inst = json.load(sys.stdin)
T, N, P = inst["T"], inst["N"], inst["P"]
arrival, L0, rate = inst["arrival"], inst["L0"], inst["rate"]
C, D = inst["C"], inst["D"]
base_need, slope = inst["base_need"], inst["slope"]
S = inst["S"]


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
bay_start_acuity = [0.0] * N
bay_setup_remaining = [0] * N

assign_rows = []

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

    def priority(p):
        a = acuity[p]
        r = max(rate[p], 1e-6)
        if a > C + 1e-9:
            time_to_death = (D - a) / r
            return (0, time_to_death)
        time_to_tip = (C - a) / r
        return (1, time_to_tip)

    waiting = sorted((p for p in range(P) if status[p] == "waiting"), key=priority)
    claimed = set()

    idle_bays = [b for b in range(N)
                 if bay_status[b] == "idle"]
    # emergency check: a waiting patient about to die next step, with no
    # idle bay -- sacrifice the least-committed non-critical in-progress bay
    for p in waiting:
        if p in claimed:
            continue
        a = acuity[p]
        r = max(rate[p], 1e-6)
        will_die_next_step = (a <= C + 1e-9 or a <= D + 1e-9) and (D - a) <= r + 1e-9
        if not will_die_next_step or idle_bays:
            continue
        candidates = [b for b in range(N)
                      if bay_status[b] == "treating" and bay_start_acuity[b] <= C + 1e-9]
        if not candidates:
            continue
        b = min(candidates, key=lambda b: bay_progress[b])
        cur = bay_patient[b]
        status[cur] = "waiting"
        bay_status[b] = "setup"
        bay_patient[b] = None
        bay_progress[b] = 0
        bay_setup_remaining[b] = S - 1
        # (bay becomes free next step or later; the emergency patient still
        #  needs an idle bay to actually be picked up)
        break  # at most one emergency sacrifice per step, keep it rare

    row = [0] * N
    for b in range(N):
        if bay_status[b] == "setup":
            bay_setup_remaining[b] -= 1
            if bay_setup_remaining[b] <= 0:
                bay_status[b] = "idle"
            continue
        if bay_status[b] == "treating":
            cur = bay_patient[b]
            row[b] = cur + 1
            bay_progress[b] += 1
            if bay_progress[b] >= bay_need[b]:
                status[cur] = "done"
                bay_status[b] = "idle"
                bay_patient[b] = None
                bay_progress[b] = 0
            continue
        # idle: fill with the highest-priority unclaimed waiting patient
        best = None
        for p in waiting:
            if p not in claimed:
                best = p
                break
        if best is not None:
            claimed.add(best)
            row[b] = best + 1
            bay_status[b] = "treating"
            bay_patient[b] = best
            bay_progress[b] = 1
            bay_start_acuity[b] = acuity[best]
            bay_need[b] = need_of(acuity[best])
            status[best] = "treating"
            if bay_progress[b] >= bay_need[b]:
                status[best] = "done"
                bay_status[b] = "idle"
                bay_patient[b] = None
                bay_progress[b] = 0

    assign_rows.append(row)

print(json.dumps({"assign": assign_rows}))
