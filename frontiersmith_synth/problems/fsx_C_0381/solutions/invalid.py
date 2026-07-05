# TIER: invalid
# A mis-specified schedule: it emits one coefficient too many per array (length
# K+1 instead of K), so the evaluator's strict shape check rejects it on every
# instance -> 0.0.  (Represents an off-by-one round-budget bug.)
import sys, json

inst = json.load(sys.stdin)
K = inst["budget"]
eta = inst["ref_step"]

print(json.dumps({"alpha": [eta] * (K + 1),
                  "beta": [0.0] * (K + 1),
                  "gamma": [0.0] * (K + 1)}))
