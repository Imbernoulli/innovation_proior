# TIER: trivial
# Do nothing clever: keep iterating the plain fixed-point map (omega = 1, no
# Aitken). This is exactly the evaluator's baseline, so it scores ~0.10.
import sys, json
json.load(sys.stdin)
print(json.dumps({"omega": 1.0, "aitken_period": 0}))
