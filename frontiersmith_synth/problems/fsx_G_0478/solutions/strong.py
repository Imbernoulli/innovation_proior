# TIER: strong
# Frequency-weighted per-feature ridge: a smoothness prior.
# The teacher is smooth (low frequency); the overfitting comes from high-|omega|
# random features memorizing label noise. So penalize each feature in proportion to
# its squared frequency omega_j^2 (plus a tiny uniform floor to tame the mid band).
# This suppresses the noise-fitting features while sparing the signal-carrying ones,
# closing the generalization gap far more than uniform weight decay.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]
omega = inst["omega"]
ow = [w * w for w in omega]
mean_ow = sum(ow) / len(ow)
c = 0.10
floor = 0.005
ridge = [min(20.0, c * (v / mean_ow) + floor) for v in ow]
print(json.dumps({"ridge": ridge}))
