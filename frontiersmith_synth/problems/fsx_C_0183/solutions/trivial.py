# TIER: trivial
import sys, json
inst = json.load(sys.stdin)
# Do nothing: zero step sizes -> z_T = z0 -> residual = baseline -> ratio 0.1.
print(json.dumps({"steps": [0.0] * inst["T"]}))
