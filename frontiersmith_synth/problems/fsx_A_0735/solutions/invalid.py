# TIER: invalid
# Always requests far more carbon than the step budget allows -> the
# evaluator must reject the whole instance (score 0.0).
import sys, json

inst = json.load(sys.stdin)
tips = inst["tips"]
B = inst["budget_step"]
alloc = {str(t["id"]): B * 50.0 + 1000.0 for t in tips}
print(json.dumps({"alloc": alloc}))
