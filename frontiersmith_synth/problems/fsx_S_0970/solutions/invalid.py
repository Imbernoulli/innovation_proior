# TIER: invalid
# Over-provisioned guess: proposes t+3 "terms" (more than the stated sparsity t
# allows). The evaluator strictly caps the answer at <= t terms, so this is
# REJECTED outright on every instance -> scores 0 everywhere.
import sys, json

inst = json.load(sys.stdin)
t = inst["t"]
print(json.dumps({"terms": [[i, 1] for i in range(t + 3)]}))
