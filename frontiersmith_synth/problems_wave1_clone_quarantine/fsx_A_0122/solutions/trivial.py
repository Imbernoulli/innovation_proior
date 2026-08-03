# TIER: trivial
# Next-fit: keep filling ONE open gantry; the moment a platoon doesn't fit, close
# it and open a fresh one.  Uses a single open gantry at a time (K=1), so bounded
# space is trivially satisfied.  This reproduces the evaluator's weak baseline -> ~0.1.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
platoons = inst["platoons"]

assign = []
cur = 0
rem = C
for s in platoons:
    if s <= rem:
        rem -= s
    else:
        cur += 1
        rem = C - s
    assign.append(cur)

print(json.dumps({"assign": assign}))
