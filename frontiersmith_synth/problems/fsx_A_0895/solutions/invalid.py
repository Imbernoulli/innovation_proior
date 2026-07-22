# TIER: invalid
# Ignores the declared dose_max cap and blasts every patient with a wildly
# oversized flat dose at every step. dose_max is a hard feasibility bound
# (announced in the public instance); any dose above it makes the whole
# instance's answer infeasible -> evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["treatment_steps"]
dose_max = inst["dose_max"]
n = len(inst["patients"])

huge = dose_max * 5.0
doses = [[huge] * T for _ in range(n)]
print(json.dumps({"doses": doses}))
