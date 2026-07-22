# TIER: trivial
# Always-LRU: ignore the runner's forecast entirely. This exactly reproduces
# the evaluator's weak baseline reference, so it scores ~0.1 on every instance
# by construction.
import sys, json

json.load(sys.stdin)  # calibration data is available but unused
print(json.dumps({"mode": "lru"}))
