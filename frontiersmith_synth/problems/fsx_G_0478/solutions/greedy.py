# TIER: greedy
# Uniform ridge (classic weight decay) with a single hand-picked strength.
# Helps on some instances, hurts on others -- it penalizes the useful low-frequency
# features exactly as hard as the noise-fitting high-frequency ones.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
print(json.dumps({"ridge": [0.03] * M}))
