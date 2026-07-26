# TIER: invalid
# Malformed: claims a merge using rule index 99999, which is out of range for every
# instance's (much shorter) merge ruleset -> the evaluator's strict feasibility check
# rejects it on every instance -> score 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"moves": [{"op": "merge", "pos": 0, "rule": 99999}]}))
