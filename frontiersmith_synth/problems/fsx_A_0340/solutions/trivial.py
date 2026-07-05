# TIER: trivial
# Overlap-BLIND, cooling-BLIND row-major placement: put job j at slot (j//N, j%N).
# This is exactly the evaluator's weak reference -- jobs pack into a corner, their 3x3
# footprints stack, and the strong-vent slots go unused -> the instance scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; j = inst["j"]

place = [[idx // n, idx % n] for idx in range(j)]
print(json.dumps({"place": place}))
