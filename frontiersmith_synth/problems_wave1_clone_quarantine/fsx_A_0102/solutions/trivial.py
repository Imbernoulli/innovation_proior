# TIER: trivial
# Next-fit slotting: load the current stage until the next act does not fit --
# either it would exceed the resource capacity C, or the stage already holds K
# acts -- then open a new stage and never look back.  This reproduces the
# evaluator's weak baseline operator, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_acts"]
acts = inst["acts"]

assign = []
g = 0
rem = C
cnt = 0
for s in acts:
    if s <= rem and cnt < K:
        assign.append(g)
        rem -= s
        cnt += 1
    else:
        g += 1
        rem = C - s
        cnt = 1
        assign.append(g)

print(json.dumps({"assign": assign}))
