# TIER: greedy
# THE OBVIOUS RECIPE (the trap): Giffler-Thompson active-schedule generation with a
# shortest-processing-time (SPT) conflict tie-break -- the standard first-attempt JSP
# heuristic taught in every scheduling course.  At each step it finds the operation with
# the earliest possible completion time across all job "frontiers"; among operations that
# genuinely compete for that same machine before it locks in, it always runs the SHORTEST
# one next.  This has no notion of how much work is still queued *after* an operation (its
# tail) -- it only looks at the operation's own duration.  On trap instances a machine
# carries a handful of operations squeezed between long job chains: one competing operation
# is short but nearly done afterward (small tail), another is longer but has a long chain
# still to run (large tail).  SPT always runs the short one first, delaying the long-tail
# one and stretching the makespan -- exactly the case a tail-aware rule would get right.
import sys, json

inst = json.load(sys.stdin)
n_jobs = inst["n_jobs"]
n_machines = inst["n_machines"]
n_ops = inst["n_ops"]
job_ops = inst["job_ops"]
dur = {o["id"]: o["dur"] for o in inst["ops"]}
mach = {o["id"]: o["machine"] for o in inst["ops"]}
oid2job = {o["id"]: o["job"] for o in inst["ops"]}

ptr = [0] * n_jobs
job_ready = [0.0] * n_jobs
mach_free = [0.0] * n_machines
machine_order = [[] for _ in range(n_machines)]

scheduled = 0
while scheduled < n_ops:
    frontier = [job_ops[j][ptr[j]] for j in range(n_jobs) if ptr[j] < len(job_ops[j])]
    est, efin = {}, {}
    for oid in frontier:
        j = oid2job[oid]
        m = mach[oid]
        s = max(job_ready[j], mach_free[m])
        est[oid] = s
        efin[oid] = s + dur[oid]
    star = min(frontier, key=lambda o: efin[o])
    mstar = mach[star]
    tstar = efin[star]
    conflict = [o for o in frontier if mach[o] == mstar and est[o] < tstar - 1e-9]
    if not conflict:
        conflict = [star]
    # SPT tie-break; break further ties by job index for determinism
    chosen = min(conflict, key=lambda o: (dur[o], oid2job[o]))
    j = oid2job[chosen]
    m = mach[chosen]
    s = max(job_ready[j], mach_free[m])
    f = s + dur[chosen]
    machine_order[m].append(chosen)
    mach_free[m] = f
    job_ready[j] = f
    ptr[j] += 1
    scheduled += 1

print(json.dumps({"machine_order": machine_order}))
