# TIER: invalid
# Claim to "reinvest 500%" every turn -- an out-of-[0,1] fraction. The
# evaluator strictly validates each entry of `invest`; any value outside
# [0,1] makes the whole instance infeasible -> scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_turns"]

print(json.dumps({"invest": [5.0] * N}))
