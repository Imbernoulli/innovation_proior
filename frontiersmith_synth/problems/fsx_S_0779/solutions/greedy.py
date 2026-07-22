# TIER: greedy
# The obvious first idea: "put more material exactly where the sensed strain
# energy is high" -- a purely local stress-following remodeling law with no
# spatial regularization (filter_radius=0). This chases high-frequency,
# per-cell sensitivity noise straight into spatially incoherent (checkerboard-
# like) density patterns: numerically "stiff" in the coarse per-cell energy
# sense, but not backed by comparable-density neighbours.
import sys, json

inst = json.load(sys.stdin)
nx, ny, volfrac = inst["nx"], inst["ny"], inst["volfrac"]
answer = {
    "seed_density": [volfrac] * (nx * ny),
    "filter_radius": 0.0,
    "move_limit": 0.15,
}
print(json.dumps(answer))
