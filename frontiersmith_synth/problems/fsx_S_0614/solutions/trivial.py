# TIER: trivial
# Do nothing: leave every item out.  obj = 0, which is exactly the evaluator's reference,
# so this scores 0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"assign": [-1] * N}))
