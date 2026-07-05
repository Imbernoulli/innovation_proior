# TIER: invalid
# Uniform MINIMUM cross-section on every bar. The members are far too thin: axial stresses
# blow past the yield limit, slender compression members buckle, and the gantry sags well
# beyond disp_limit -- so every instance is infeasible and scores 0.
import sys, json

inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_min"]] * M}))
