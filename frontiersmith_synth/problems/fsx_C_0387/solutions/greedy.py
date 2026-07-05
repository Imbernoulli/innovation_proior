# TIER: greedy
# Push the constant extragradient step to the edge of stability: use twice the
# reference step (0.8/||H|| instead of 0.4/||H||).  A larger constant step
# contracts the low-frequency error faster than the timid reference, so it
# beats trivial on most reefs -- but a single fixed step can neither adapt to
# the spectrum nor accelerate, so it stays well short of a designed schedule.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
step = 2.0 * inst["ref_alpha"]          # ref_alpha = 0.4/Lspec  ->  0.8/Lspec
print(json.dumps({"alpha": [step] * K, "beta": [step] * K}))
