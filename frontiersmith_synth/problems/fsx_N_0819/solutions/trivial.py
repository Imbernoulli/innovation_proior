# TIER: trivial
# Do nothing: no shipments at all, on every epoch call. This is exactly the
# evaluator's internal do-nothing baseline, so it scores ~0.1 on every
# instance by construction.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"shipments": []}))
