# TIER: trivial
# Spread the step's carbon budget evenly across every currently active tip,
# every step, ignoring the sensed readings entirely. Reproduces the
# evaluator's weak reference recipe almost exactly (score ~0.1).
import sys, json

inst = json.load(sys.stdin)
tips = inst["tips"]
B = inst["budget_step"]
n = len(tips)
share = (B / n) if n else 0.0
alloc = {str(t["id"]): share for t in tips}
print(json.dumps({"alloc": alloc}))
