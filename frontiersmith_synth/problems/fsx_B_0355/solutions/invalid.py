# TIER: invalid
import sys, json
inst = json.load(sys.stdin)
N = inst["N"]
# wrong shape: a single row instead of an N x N phase array -> rejected
print(json.dumps({"phase": [0.0] * N}))
