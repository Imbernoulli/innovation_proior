# TIER: invalid
# Emits "schedule" instead of the required "rounds" key.  The evaluator's
# top-level structural check rejects the whole answer -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"schedule": []}))
