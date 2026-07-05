# TIER: trivial
# 2-D next-fit: keep loading the current tour until the next group would break
# EITHER the crowd cap or the docent-minute budget, then open a fresh tour and
# never look back.  Reproduces the evaluator's weak baseline -> scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
C = inst["C"]; T = inst["T"]
people = inst["people"]; minutes = inst["minutes"]

assign = []
t = 0
rp, rf = C, T
for p, f in zip(people, minutes):
    if p <= rp and f <= rf:
        assign.append(t)
        rp -= p; rf -= f
    else:
        t += 1
        rp = C - p; rf = T - f
        assign.append(t)

print(json.dumps({"assign": assign}))
