# TIER: trivial
# Next-Fit: keep a single open block, flush and open a fresh one whenever the next
# request will not fit. This reproduces the evaluator's weak baseline (~0.1).
import sys, json

inst = json.load(sys.stdin)
cap = inst["capacity"]
items = inst["items"]

assign = []
blk = 0
cur = 0
for x in items:
    if cur + x <= cap:
        cur += x
    else:
        blk += 1
        cur = x
    assign.append(blk)

print(json.dumps({"assign": assign}))
