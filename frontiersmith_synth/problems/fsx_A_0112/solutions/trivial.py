# TIER: trivial
# Color-aware NEXT-FIT: keep one pod open; admit each arriving contact iff its
# load still fits AND it would not push the pod above K distinct lineages.
# Otherwise close the pod and open a fresh one -- never look back.  This exactly
# reproduces the evaluator's weak baseline, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_strains"]
loads = inst["loads"]
strains = inst["strains"]

assign = []
p = 0
rem = C
cur = set()
for w, s in zip(loads, strains):
    ok_load = w <= rem
    ok_color = (s in cur) or (len(cur) < K)
    if ok_load and ok_color:
        rem -= w
        cur.add(s)
        assign.append(p)
    else:
        p += 1
        rem = C - w
        cur = {s}
        assign.append(p)

print(json.dumps({"assign": assign}))
