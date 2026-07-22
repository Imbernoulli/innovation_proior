# TIER: trivial
# Do nothing: seed a uniform density field at the target volume fraction and
# never remodel (move_limit=0 freezes the field). This is exactly the
# baseline() construction the evaluator computes itself.
import sys, json

inst = json.load(sys.stdin)
nx, ny, volfrac = inst["nx"], inst["ny"], inst["volfrac"]
answer = {
    "seed_density": [volfrac] * (nx * ny),
    "filter_radius": 0.0,
    "move_limit": 0.0,
}
print(json.dumps(answer))
