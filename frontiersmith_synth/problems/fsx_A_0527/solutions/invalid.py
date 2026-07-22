# TIER: invalid
# Wrong-shaped policy table (a flat per-class list, not the required
# [class][rem_bucket][sig_bucket] cube).  The evaluator's strict shape check
# rejects it -> every tide scores 0.0.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"bars": [0.0 for _ in range(K)]}))
