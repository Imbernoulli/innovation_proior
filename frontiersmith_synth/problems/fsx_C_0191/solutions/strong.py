# TIER: strong
# Strong curve: a gain-tuned saturating unit, phi(x) = 2*tanh(1.5*x). The steeper inner
# slope gives a strong live gradient near 0 (fast, expressive early training) while the
# wider +/-2 saturation plateau supplies bounded, decorrelating nonlinearity. This blend
# separates the nonlinear halls (XOR / rings / spirals / moons / checkerboard) AND stays
# stable on the near-linear blobs hall, giving a high geometric mean across the family --
# a genuinely designed curve, not merely "any nonlinearity".
import sys, json, math

inst = json.load(sys.stdin)
grid = inst["grid"]
ys = [2.0 * math.tanh(1.5 * x) for x in grid]
print(json.dumps({"activation": ys}))
