# TIER: strong
# A rectifier: phi(x) = max(0, x).  Non-saturating on the positive side means the
# gradient does not vanish there, so under the shared full-batch GD schedule the
# battery of tiny MLPs trains faster and reaches higher geometric-mean accuracy than
# the smooth saturating activations -- while the label noise keeps every battery
# strictly below a perfect score, leaving headroom.
import sys, json

inst = json.load(sys.stdin)
grid = inst["grid"]
y = [x if x > 0.0 else 0.0 for x in grid]
print(json.dumps({"y": y}))
