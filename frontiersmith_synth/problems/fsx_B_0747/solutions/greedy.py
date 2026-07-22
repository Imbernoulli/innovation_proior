# TIER: greedy
# The obvious first-pass dispatcher: ALWAYS PREFER THE FASTEST AVAILABLE UNIT.
# Every tick, jobs are packed into units in decreasing-capacity order -- fill the
# fastest available (not repairing, not yet at this tick's capacity) unit first,
# then the next fastest, etc. This maximizes THIS TICK's throughput and never
# looks at fatigue or the hazard cliff at all. Because capacity and fatigue_rate
# are correlated in this fleet (the fast unit also wears the fastest), the fast
# unit gets driven hard through the whole quiet part of the horizon; nothing
# ever tells it to bank headroom before a surge, so it is already worn down --
# or mid-repair -- right when the seeded demand peak needs it most.
import sys, json

def main():
    inst = json.load(sys.stdin)
    n = inst["n_units"]; units = inst["units"]; T = inst["horizon"]
    fatigue = [0.0] * n
    repair_until = [0] * n
    order = sorted(range(n), key=lambda i: (-units[i]["capacity"], i))

    by_tick = {}
    for j in inst["jobs"]:
        by_tick.setdefault(j["t"], []).append(j)

    assign_of = {}
    for t in range(T):
        this_tick = by_tick.get(t, [])
        processed_count = [0] * n
        for j in this_tick:
            u = None
            for cand in order:
                if t >= repair_until[cand] and processed_count[cand] < units[cand]["capacity"]:
                    u = cand
                    break
            if u is None:
                u = order[0]
            assign_of[j["id"]] = u
            if t >= repair_until[u] and processed_count[u] < units[u]["capacity"]:
                processed_count[u] += 1
                fatigue[u] += units[u]["fatigue_rate"]
                if fatigue[u] >= units[u]["hazard_cliff"]:
                    repair_until[u] = t + 1 + units[u]["repair_time"]
                    fatigue[u] = 0.0
        for u in range(n):
            if t >= repair_until[u] and processed_count[u] == 0:
                fatigue[u] = max(0.0, fatigue[u] - units[u]["recover_rate"])

    assignment = [assign_of[j["id"]] for j in inst["jobs"]]
    print(json.dumps({"assignment": assignment}))

if __name__ == "__main__":
    main()
