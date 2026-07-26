# TIER: greedy
# Textbook online list-scheduling desk: maintain the C benches' actual free-times
# and, for each job in arrival order, compute the TRUE earliest-available-bench
# delay right now (first-fit across benches -- the standard parallel-machine
# scheduling heuristic). Admit iff that honest forecast fits within patience, and
# QUOTE THAT FORECAST -- i.e. treat the quote as an honest prediction of when the
# desk expects to finish, not as a commitment with deliberate slack.
#
# This is strictly more accurate bookkeeping than the pooled mean-field baseline
# (it tracks per-bench state, not one averaged counter), so it out-admits and
# out-quotes the baseline in general. But it is still a single, arrival-order
# pass with a self-consistent FIFO/first-fit mental model of dispatch -- it has
# no idea the real desk dispatches by EARLIEST-PROMISED-DEADLINE, not by arrival
# order. When a burst of tightly-quoted jobs lands, later jobs with even tighter
# promises legally jump the real EDF queue ahead of jobs this policy already
# promised (and quoted with zero slack), so those earlier promises blow their own
# deadline even though the forecast that produced them was "honest" at the time.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
jobs = inst["jobs"]

bench_free = [0] * C
decisions = []
for job in jobs:
    t = job["t"]
    best_i = min(range(C), key=lambda i: bench_free[i])
    avail = bench_free[best_i]
    start = max(t, avail)
    delay = start + job["service"] - t
    if delay <= job["patience"]:
        decisions.append({"action": "quote", "delay": int(delay)})
        bench_free[best_i] = start + job["service"]
    else:
        decisions.append({"action": "reject"})

print(json.dumps({"decisions": decisions}))
