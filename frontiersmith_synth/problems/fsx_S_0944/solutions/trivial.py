# TIER: trivial
# Next-fit-commit: fill the current truck until a parcel doesn't fit, then open
# the next one and never look back. Reproduces the evaluator's weak reference
# exactly, so it scores ~0.1 on every instance. No repack moves used.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
sizes = inst["sizes"]

assign = []
g = 0
rem = C
for s in sizes:
    if s <= rem:
        assign.append(g)
        rem -= s
    else:
        g += 1
        rem = C - s
        assign.append(g)

print(json.dumps({"placements": assign, "moves": []}))
