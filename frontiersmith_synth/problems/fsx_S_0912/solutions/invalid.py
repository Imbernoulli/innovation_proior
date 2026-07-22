# TIER: invalid
# Emits a tape entry far outside the legal byte range [0,255] (and no attempt
# to respect L_max), so the evaluator's strict answer validation rejects it
# and the instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"tape": [999999]}))
