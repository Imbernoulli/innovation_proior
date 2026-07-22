# TIER: trivial
# Naive "discount to guarantee a sellout" instinct: read only the very first
# period's demand parameters, undercut the day-one monopolist price by half
# (a common unsophisticated urge -- "price it cheap so it definitely moves"),
# and freeze that ONE price for the entire event. Ignores that a/b drift over
# the horizon, ignores the reference-price dynamics, ignores inventory.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
p_max = inst["p_max"]
a0 = inst["a"][0]
b0 = inst["b"][0]

p = (a0 / (2.0 * b0)) * 0.5 if b0 > 0 else 0.0
p = max(0.0, min(p_max, p))

print(json.dumps({"prices": [p] * T}))
