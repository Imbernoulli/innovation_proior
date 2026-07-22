# TIER: invalid
# Deliberately broken: emits a "tour" that repeats stop 0 and omits the last
# stop, so validation must reject it on every instance (score 0.0 throughout).
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

tour = [0] * n  # not a permutation -> invalid everywhere
print(json.dumps({"tour": tour}))
