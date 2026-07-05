# TIER: invalid
# Emits an activation with an out-of-range gain (a = 999, past the |a|<=10 cap) and
# an unknown basis name.  The evaluator's validator rejects it, so every instance
# scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"components": [{"base": "megarelu", "a": 999.0, "b": 0.0, "w": 5.0}]}))
