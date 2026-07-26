# TIER: trivial
# Do nothing: keep the jobs in the order they already arrive at the head of the line.
# This is exactly the evaluator's baseline construction, so it maps to ratio ~0.1 by
# construction on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": list(range(n))}))
