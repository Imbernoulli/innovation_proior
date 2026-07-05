# TIER: trivial
# Null clustering: put every intersection in a single group.
# ARI(single cluster, anything) == 0  ->  normalized score ~ 0.1 on every city.
import sys, json

inst = json.load(sys.stdin)
n = int(inst["n"])
print(json.dumps([0] * n))
