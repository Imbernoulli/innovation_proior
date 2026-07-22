# TIER: trivial
# No construction at all: visit the points in the order they were given
# (0, 1, 2, ..., n-1). This is exactly the "identity" tour the evaluator
# itself uses to compute its weak reference q_base, so after the shared
# budgeted refine this scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"tour": list(range(n))}))
