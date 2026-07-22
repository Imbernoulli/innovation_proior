# TIER: trivial
# Blind uniform sweep: ignore every pilot reading, spend the whole budget as
# fixed-seed uniform-random points across the box. Reproduces (in spirit) the
# evaluator's own weak blind-sweep reference, so it scores ~0.1.
import sys, json, random

inst = json.load(sys.stdin)
D = inst["dim"]
box = inst["box"]
Q = inst["budget"]

rnd = random.Random(1234567)
queries = [[rnd.uniform(lo, hi) for (lo, hi) in box] for _ in range(Q)]

print(json.dumps({"queries": queries}))
