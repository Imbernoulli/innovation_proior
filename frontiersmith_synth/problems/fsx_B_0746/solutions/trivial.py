# TIER: trivial
# Uniform, non-adaptive grid of explicit RK4 steps spending (almost) the full
# evaluation budget, snapped to land exactly on every published checkpoint.
# No stiffness awareness at all -- this is exactly the evaluator's own
# baseline construction, so it always maps to ratio ~0.1.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
max_evals = inst["max_evals"]
ce = inst["cost_explicit"]
checkpoints = inst["checkpoints"]
n_check = len(checkpoints)

n_uniform = max(1, max_evals // ce - n_check - 2)
h = T / n_uniform

boundaries = []
t = 0.0
ci = 0
while ci < len(checkpoints):
    nxt = t + h
    if nxt >= checkpoints[ci] - 1e-9:
        nxt = checkpoints[ci]
        ci += 1
    boundaries.append({"t1": nxt, "method": "explicit"})
    t = nxt

print(json.dumps({"steps": boundaries}))
