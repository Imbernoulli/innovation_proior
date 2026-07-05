# TIER: strong
# A deliberately SHAPED activation: an even, localized pair -- absolute-value plus a
# Gaussian bump -- g(x) = |x| + exp(-(1.2 x)^2).
#   * |x| is even and unbounded: it captures the rectified / V-shaped responses
#     (clip magnitude, derating kink) AND, unlike a single ReLU, treats both signs,
#     which matters for the even inverter laws (efficiency bell, MPP parabola).
#   * the Gaussian bump adds a compact, tunable feature that lets the ridge head
#     localize the peak of the efficiency curve and the trough of the MPP loss.
# Together they span more of the shared response geometry than any single rectifier,
# so the surrogate generalizes to the HELD-OUT setting too -- beating the ReLU
# default across the cross-setting geometric mean.  Crucially the designer must know
# to AVOID naive additions (e.g. a periodic sin term) that inject high-frequency
# features and hurt the non-periodic settings; the irreducible noise floor keeps even
# this design short of 1.0.
import sys, json

json.load(sys.stdin)
components = [
    {"base": "abs",   "a": 1.0, "b": 0.0, "w": 1.0},
    {"base": "gauss", "a": 1.2, "b": 0.0, "w": 1.0},
]
print(json.dumps({"components": components}))
