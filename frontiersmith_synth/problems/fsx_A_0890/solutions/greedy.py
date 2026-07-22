# TIER: greedy
# The textbook safety-stock recipe: order-up-to (historical mean + 1 standard
# deviation), applied IDENTICALLY to every one of the six lines regardless of what
# their history actually looks like.  No classification, no seasonal shape, no
# trend extrapolation, no reactive hedge, no use of the cost parameters to size the
# buffer.  This is the "one size fits all" base-stock rule an average strong coder
# writes first -- fine for calm, roughly-symmetric demand, but it does not detect
# (and cannot survive) a demand family whose shape or timing violates the
# flat-Gaussian assumption baked into "mean + z*std".
import sys, json, math

inst = json.load(sys.stdin)
period = inst["period"]
Z = 1.0

policies = []
for tr in inst["traces"]:
    hist = tr["history"]
    n = len(hist)
    mean = sum(hist) / float(n)
    var = sum((x - mean) ** 2 for x in hist) / float(n)
    std = math.sqrt(var)
    level0 = mean + Z * std
    policies.append({
        "trace_id": tr["trace_id"],
        "level": [level0] * period,
        "trend": 0.0,
        "react": 0.0,
    })

print(json.dumps({"policies": policies}))
