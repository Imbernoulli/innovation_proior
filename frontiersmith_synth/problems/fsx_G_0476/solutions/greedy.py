# TIER: greedy
# A classic smooth saturating nonlinearity: phi(x) = tanh(x).  Bounded, smooth,
# introduces genuine curvature so the nets can bend decision boundaries -- a solid
# step above the linear baseline, but its vanishing gradients in the tails leave
# accuracy on the table for the harder batteries.
import sys, json, math

inst = json.load(sys.stdin)
grid = inst["grid"]
y = [math.tanh(x) for x in grid]
print(json.dumps({"y": y}))
