# TIER: strong
# The insight: spatially regularize the sensed signal before letting it drive
# density change. Filtering the strain-energy sensitivity over a radius of a
# couple of cells (mesh-independent smoothing) keeps every remodeling step
# spatially coherent -- no isolated corner-touching density spikes -- so the
# structure that emerges is genuinely load-bearing rather than an artifact of
# purely local greedy stress-chasing. A smaller move_limit lets the filtered
# signal accumulate into a real, converged shape over the iteration budget
# instead of overshooting.
import sys, json

inst = json.load(sys.stdin)
nx, ny, volfrac = inst["nx"], inst["ny"], inst["volfrac"]
answer = {
    "seed_density": [volfrac] * (nx * ny),
    "filter_radius": 2.0,
    "move_limit": 0.05,
}
print(json.dumps(answer))
