# TIER: trivial
# Seasonal-naive: forecast every future hour by repeating the value observed
# exactly one day (period) earlier. This is the evaluator's baseline, so it
# scores ~0.1 by construction.
import sys, json
inst = json.load(sys.stdin)
y = inst["y"]; m = inst["period"]; H = inst["horizon"]
n = len(y)
fc = [y[n - m + (i % m)] for i in range(H)]
print(json.dumps({"forecast": fc}))
