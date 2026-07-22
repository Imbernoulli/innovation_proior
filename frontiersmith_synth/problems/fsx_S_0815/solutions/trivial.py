# TIER: trivial
# Never hold. Reproduces the evaluator's weak never-hold baseline exactly, so it
# scores ~0.1 on every instance. Dwell-feedback runs completely free.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"gain_back": 0.0, "gain_fwd": 0.0, "target_frac": 1.0, "cap_frac": 0.0}))
