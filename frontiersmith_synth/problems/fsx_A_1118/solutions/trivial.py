# TIER: trivial
# Admit every lot that still fits, no filtering at all (base=0 <= any positive
# density, and the other three coefficients are inert). This reproduces the
# evaluator's own weak "accept everything" baseline exactly.
import sys, json

inst = json.load(sys.stdin)

policy = {"base": 0.0, "cap_gain": 0.0, "drift_gain": 0.0, "time_gain": 0.0}
print(json.dumps({"policy": policy}))
