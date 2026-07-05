# TIER: trivial
# Reproduce the on-board autoloader (Next-Fit): keep ONE hauler open, drop each
# arriving fragment into it, and the instant a fragment does not fit, seal that
# hauler and open a fresh one.  This is exactly the evaluator's weak baseline, so it
# never back-fills a sealed hauler and scores ~0.1 on every shift.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
masses = inst["masses"]

assign = []
cur = -1
load = None
for m in masses:
    if load is None or load + m > C:
        cur += 1
        load = m
    else:
        load += m
    assign.append(cur)

print(json.dumps({"assign": assign}))
