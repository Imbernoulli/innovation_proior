# TIER: strong
# Insight: don't spend a fast unit's fatigue budget on routine load just because
# it is fast. Order units by CAPACITY ASCENDING (least-capable unit first) for
# every routing decision, spilling over to the next-more-capable unit only when
# the current candidate is unavailable (repairing) or already saturated for this
# tick. In this fleet, the low-capacity units alone can absorb almost all of the
# routine background traffic, so the high-capacity, high-fatigue-rate unit(s)
# stay essentially idle -- and idle ticks bank fatigue headroom via recovery --
# through the quiet part of the horizon. That banked cumulative-fatigue budget
# is exactly what is needed the moment the seeded demand peak arrives: the fast
# unit walks INTO the surge close to fully rested instead of already worn down
# or mid-repair, so it can absorb several ticks of surge-level throughput before
# it ever approaches its hazard cliff. This is a genuine reformulation, not
# "greedy plus lookahead": the priority order is the exact REVERSE of the
# throughput-greedy dispatcher's, trading a little routine-time throughput
# (a low-capacity unit is slower even when it is not overloaded) to buy a much
# larger amount of surge-time throughput.
import sys, json

def main():
    inst = json.load(sys.stdin)
    n = inst["n_units"]; units = inst["units"]; T = inst["horizon"]
    fatigue = [0.0] * n
    repair_until = [0] * n
    order = sorted(range(n), key=lambda i: (units[i]["capacity"], i))

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
