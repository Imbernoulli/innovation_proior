# TIER: trivial
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]
# one sensor at the domain centre -> a single on-front point
print(json.dumps({"points": [[0.5] * n]}))
