# TIER: invalid
# Reports an empty step schedule: covers nothing, hits no checkpoint, never
# reaches T. Strictly rejected by the evaluator's feasibility check on every
# instance -> scores 0 everywhere.
import sys, json

json.load(sys.stdin)
print(json.dumps({"steps": []}))
