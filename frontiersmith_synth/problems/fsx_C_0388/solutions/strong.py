# TIER: strong
# The right inductive bias: use ONLY the length/order-invariant structural features.
#   VALID iff abs_balance == 0  AND  max_depth <= Dmax.
# Both conditions are one-sided ("small is good"), so the AND is expressible as a single linear
# rule:  w.f + b > 0  with a huge penalty on abs_balance and a unit penalty on max_depth, bias at
# Dmax+0.5.  These cues are invariant to sequence LENGTH, so ID performance transfers to the
# longer OOD logs unchanged. Ceiling < 1: the type-swap corruptions are invisible to bag features.
import sys, json
inst = json.load(sys.stdin)
m = inst["m"]
Dmax = inst["Dmax"]
ABS_BAL, MAX_DEPTH = 0, 1
w = [0.0] * m
w[ABS_BAL] = -1000.0
w[MAX_DEPTH] = -1.0
b = float(Dmax) + 0.5
print(json.dumps({"w": w, "b": b}))
