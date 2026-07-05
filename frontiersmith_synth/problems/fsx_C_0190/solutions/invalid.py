# TIER: invalid
# Emits a labels list of the wrong length (and with an out-of-set entry), so the
# evaluator's strict validation rejects it -> 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"labels": [0, 1, 7]}))
