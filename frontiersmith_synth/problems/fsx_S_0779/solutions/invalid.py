# TIER: invalid
# Malformed answer: seed_density has the wrong length (half the domain), so
# the evaluator's strict shape check must reject it -> scores 0 everywhere.
import sys, json

inst = json.load(sys.stdin)
nx, ny = inst["nx"], inst["ny"]
n_half = max(1, (nx * ny) // 2)
answer = {
    "seed_density": [0.5] * n_half,
    "filter_radius": 1.0,
    "move_limit": 0.1,
}
print(json.dumps(answer))
