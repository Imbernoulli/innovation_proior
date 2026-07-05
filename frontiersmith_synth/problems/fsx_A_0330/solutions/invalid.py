# TIER: invalid
# Broken route: visit station 1 N times.  For N >= 2 this is not a permutation of
# {1..N} (station 1 repeated, others missing), so it fails validation -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": [1] * n}))
