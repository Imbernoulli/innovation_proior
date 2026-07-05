# TIER: trivial
# All-1x1 delineation: draw a single unit cell around each organism's survey
# point.  This is exactly the manager's weak reference layout, so it reproduces
# the evaluator's baseline quality and scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
x = inst["x"]
y = inst["y"]

rects = [[x[i], y[i], x[i] + 1, y[i] + 1] for i in range(n)]
print(json.dumps({"rects": rects}))
