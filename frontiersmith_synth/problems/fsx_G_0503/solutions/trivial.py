# TIER: trivial
# Conservative hospital default: alert only when the early-warning score is at least
# 0.50 in every unit/acuty stratum.  This reproduces the evaluator's weak baseline.
import json
import sys

inst = json.load(sys.stdin)
print(json.dumps({"default_threshold": inst.get("baseline_threshold", 0.50), "thresholds": {}}))
