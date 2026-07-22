# TIER: trivial
# All weights zero: score(y)=0 for every resident line, so eviction falls through
# to the tie-break (lowest line id) -- a static, signal-free rule.
import sys, json

json.load(sys.stdin)
print(json.dumps({"w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0, "w5": 0.0}))
