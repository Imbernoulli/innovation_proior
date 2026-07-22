# TIER: trivial
# Do nothing: never spend the budget. This is the evaluator's own weak reference
# (maps to ~0.1 by construction, since "no intervention" is exactly its `noint` baseline).
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"schedule": [[] for _ in range(T)]}))
