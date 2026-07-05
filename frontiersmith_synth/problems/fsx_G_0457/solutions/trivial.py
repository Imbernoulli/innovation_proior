# TIER: trivial
# Equal-weight committee: the fused forecast for each test event is just the plain
# average of the k member forecasts.  This is exactly the evaluator's weak reference,
# so it scores ~0.1 -- it lets the weak / miscalibrated members drag the committee down.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
test_pred = inst["test_pred"]
invk = 1.0 / k
q = []
for row in test_pred:
    s = 0.0
    for v in row:
        s += v
    q.append(s * invk)
print(json.dumps({"forecast": q}))
