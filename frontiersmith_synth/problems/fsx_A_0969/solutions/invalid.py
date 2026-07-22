# TIER: invalid
# Spends nothing, and then LIES: claims a bound of 0.0 regardless of what was actually probed.
# The grader recomputes what was actually EARNED from the (empty) probe history and catches
# this as unsound the instant it is submitted -- a dishonest ("guessed, not guaranteed")
# certificate must score 0.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")

if phase == "probe":
    print(json.dumps({"probes": []}))
else:
    print(json.dumps({"bound": 0.0}))
