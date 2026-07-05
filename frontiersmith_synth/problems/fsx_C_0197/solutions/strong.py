# TIER: strong
# Smooth self-gated foraging curve: swish-like  f(s) = s * sigmoid(1.7 s), which is smooth,
# non-monotone near the origin, and bounded-below -- a strong general-purpose activation shape.
import sys, json, math
inst = json.load(sys.stdin)
grid = inst["grid"]
def sig(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
ys = [float(x) * sig(1.7 * float(x)) for x in grid]
print(json.dumps({"ys": ys}))
