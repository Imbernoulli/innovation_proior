# TIER: trivial
# Submit the single "centre" wiring layout: every decision variable = 0.5.
# The distance vars at 0.5 put it on the DTLZ2 sphere, but it is one lone
# middle trade-off point -> a small hypervolume (~ the calibrated baseline).
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"points": [[0.5] * n]}))
