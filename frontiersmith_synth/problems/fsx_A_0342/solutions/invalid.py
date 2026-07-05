# TIER: invalid
# Energize every load block on a single transformer.  Because every instance in
# this family has total demand well above one transformer's thermal capacity (and
# more than K blocks), transformer 0 is both thermally overloaded and over its
# breaker-channel count -> the dispatch is infeasible -> the evaluator scores 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"assign": [0] * N}))
