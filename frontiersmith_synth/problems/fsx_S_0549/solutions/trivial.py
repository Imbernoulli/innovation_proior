# TIER: trivial
# Index round-robin: send job i to machine i % k.  Ignores sizes AND covariance,
# and it is exactly the evaluator's weak reference, so it scores ~0.1 everywhere.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
k = inst["k"]
print(json.dumps({"assign": [i % k for i in range(n)]}))
