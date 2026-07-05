# TIER: trivial
# Plain gradient descent-ascent at the DEFAULT step size (theta=0, alpha=0).
# This is exactly the weak reference the evaluator uses to anchor r=0.1, so it
# reproduces the baseline (and on strongly-coupled instances it diverges just
# like the reference).  Scores ~0.1 everywhere.
import sys, json

json.load(sys.stdin)  # read the instance; we ignore its structure
print(json.dumps({"eta_x": 0.05, "eta_y": 0.05, "theta": 0.0, "alpha": 0.0}))
