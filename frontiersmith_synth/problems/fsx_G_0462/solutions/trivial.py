# TIER: trivial
# Conservative constant learning rate eta_t = 1/L, where L is the largest curvature.
# This is the "safe" fixed step every practitioner reaches for -- it is exactly the
# evaluator's weak baseline schedule, so it reproduces G_base and scores ~0.1 on
# every instance.  It barely moves the low-curvature modes, so the gradient stays big.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_steps"]
L = max(inst["curv"])
print(json.dumps({"lr": [1.0 / L] * N}))
