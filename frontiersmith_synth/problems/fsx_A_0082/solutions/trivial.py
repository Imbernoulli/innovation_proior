# TIER: trivial
# Next-fit dispatch: fill the current gondola until a party doesn't fit, then
# open the next one and never look back.  Reproduces the evaluator's weak
# baseline, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
parties = inst["parties"]

assign = []
g = 0
rem = C
for s in parties:
    if s <= rem:
        assign.append(g)
        rem -= s
    else:
        g += 1
        rem = C - s
        assign.append(g)

print(json.dumps({"assign": assign}))
