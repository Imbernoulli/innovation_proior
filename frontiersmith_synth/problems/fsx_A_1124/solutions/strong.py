# TIER: strong
# Two-part insight over the textbook list-scheduling desk:
#
# 1. COMMITMENT, NOT FORECAST. Once a job is worth admitting at all, quote it at
#    EXACTLY the patience ceiling -- the loosest delay the customer will still
#    accept -- instead of your best-guess forecast. A tighter promise buys you
#    nothing (the customer already joins at any delay <= patience) and only adds
#    violation risk. This alone is "option-writing" done correctly: sell the
#    loosest option the counterparty will still buy.
#
# 2. GLOBAL, VALUE-FIRST ADMISSION. The whole trace is visible up front, so don't
#    decide job-by-job in arrival order. Offer jobs in DESCENDING VALUE order
#    instead, each with a tentative patience-ceiling deadline, and re-run the
#    real earliest-deadline-first dispatch (the same engine the evaluator uses)
#    on the admitted-so-far set each time. Keep the job only if EVERY currently
#    admitted job -- the new one and every higher-value job admitted earlier --
#    still finishes on time under that dispatch. This is an exchange argument:
#    a low-value job is only ever allowed to buy bench priority if doing so does
#    not cost any higher-value job (already locked in) its own promise. Bursts of
#    low-value, tightly-patient jobs are exactly what gets triaged away, freeing
#    capacity for the high-value jobs the burst also contains.
#
# This duplicates the evaluator's frozen EDF dispatcher locally (candidates run
# isolated and never see evaluator internals) so it can check feasibility before
# committing to each promise -- O(n^2 log n), comfortably inside the time limit.
import sys, json, heapq

inst = json.load(sys.stdin)
C = inst["capacity"]
jobs = inst["jobs"]
n = len(jobs)


def simulate(admitted):
    """admitted: list of dicts with idx,t,service,deadline. -> {idx: completion}."""
    order = sorted(admitted, key=lambda j: (j["t"], j["idx"]))
    by_idx = {j["idx"]: j for j in admitted}
    m = len(order)
    bench_free = [0] * C
    heapq.heapify(bench_free)
    waiting = []
    i = 0
    completion = {}
    while i < m or waiting:
        next_arrival_t = order[i]["t"] if i < m else None
        if waiting:
            next_free_t = bench_free[0]
            T = next_free_t if (next_arrival_t is None or next_free_t <= next_arrival_t) else next_arrival_t
        else:
            T = next_arrival_t
        while i < m and order[i]["t"] <= T:
            j = order[i]
            heapq.heappush(waiting, (j["deadline"], j["t"], j["idx"]))
            i += 1
        while waiting and bench_free[0] <= T:
            heapq.heappop(bench_free)
            _, _, idx = heapq.heappop(waiting)
            j = by_idx[idx]
            fin = T + j["service"]
            completion[idx] = fin
            heapq.heappush(bench_free, fin)
    return completion


def feasible(admitted):
    completion = simulate(admitted)
    for j in admitted:
        fin = completion.get(j["idx"])
        if fin is None or fin > j["deadline"]:
            return False
    return True


order = sorted(range(n), key=lambda i: (-jobs[i]["value"], jobs[i]["service"], jobs[i]["t"], i))

admitted = []
admitted_set = set()
for idx in order:
    job = jobs[idx]
    deadline = job["t"] + job["patience"]
    trial = admitted + [{"idx": idx, "t": job["t"], "service": job["service"], "deadline": deadline}]
    if feasible(trial):
        admitted = trial
        admitted_set.add(idx)

decisions = [None] * n
for idx, job in enumerate(jobs):
    if idx in admitted_set:
        decisions[idx] = {"action": "quote", "delay": int(job["patience"])}
    else:
        decisions[idx] = {"action": "reject"}

print(json.dumps({"decisions": decisions}))
