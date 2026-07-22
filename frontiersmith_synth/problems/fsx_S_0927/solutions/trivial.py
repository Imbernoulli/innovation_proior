# TIER: trivial
# Sort stops by x-coordinate and visit them left to right. This is exactly the
# evaluator's internal weak reference, so it scores ~0.1 on every instance --
# it ignores the y-axis entirely and does no dispatch, no probing, no polish.
import sys, json

inst = json.load(sys.stdin)
pts = inst["points"]
n = inst["n"]

order = sorted(range(n), key=lambda i: pts[i][0])
print(json.dumps({"tour": order}))
