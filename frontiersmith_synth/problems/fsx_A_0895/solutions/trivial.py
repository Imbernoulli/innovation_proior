# TIER: trivial
# One-size-fits-all: ignore the probe readings entirely and dose every patient,
# every step, at a constant amount equal to the published target concentration
# (a common back-of-envelope guess: "give roughly what you want to see").
# Does not use rho, does not use the probe curve, does not individualize.
import sys, json

inst = json.load(sys.stdin)
T = inst["treatment_steps"]
target = inst["target"]
n = len(inst["patients"])

doses = [[target] * T for _ in range(n)]
print(json.dumps({"doses": doses}))
