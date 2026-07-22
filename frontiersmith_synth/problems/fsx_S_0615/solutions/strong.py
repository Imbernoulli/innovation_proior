# TIER: strong
# INSIGHT (reserve latency headroom, not "admit whatever fits right now"):
#  (1) The whole arrival stream is PUBLIC, so premium jobs that arrive soon are known.
#      Reformulate admission as SUBSET SELECTION on the single FIFO-by-arrival server:
#      choose a set of jobs that all finish on time and maximizes banked value.
#  (2) Process candidates in VALUE ORDER (highest first, ties: tighter deadline).  A job
#      is admitted only if, together with everything already selected, EVERY selected job
#      still finishes by its deadline under the FIFO simulation.  This is a weighted,
#      congestion-aware admission: the premium jobs claim the server's headroom first,
#      and a marginal low-value job is REFUSED exactly when its service size would push
#      the FIFO tail past a premium job's deadline -- i.e. it reserves headroom for the
#      predictably-arriving high-value work instead of drowning it in low-value backlog.
#  (3) Everything selected is on time, so no SLA penalty is ever paid; the objective is
#      the pure value of the protected set.  This is a genuine reformulation (value-first
#      global feasibility) rather than the myopic arrival-order / admit-all recipe -- and
#      because value-greedy selection is not optimal for weighted on-time scheduling, it
#      deliberately leaves score headroom above itself.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
a = inst["a"]; s = inst["s"]; v = inst["v"]; d = inst["d"]


def feasible(sel):
    """True iff every job in sel finishes by its deadline under FIFO-by-arrival."""
    order = sorted(sel, key=lambda i: (a[i], i))
    free = 0.0
    for i in order:
        start = a[i] if a[i] > free else free
        finish = start + s[i]
        if finish > d[i] + 1e-9:
            return False
        free = finish
    return True


cand_order = sorted(range(N), key=lambda i: (-v[i], d[i], i))
selected = []
for j in cand_order:
    selected.append(j)
    if not feasible(selected):
        selected.pop()

admit = [0] * N
for i in selected:
    admit[i] = 1
print(json.dumps({"admit": admit}))
