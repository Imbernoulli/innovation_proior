# TIER: invalid
# Delineate every pool as the ENTIRE shore.  With more than one organism (true
# for every instance in this family) the rectangles overlap, so the layout is
# infeasible and the evaluator scores it 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
W, H = inst["W"], inst["H"]

rects = [[0, 0, W, H] for _ in range(n)]
print(json.dumps({"rects": rects}))
