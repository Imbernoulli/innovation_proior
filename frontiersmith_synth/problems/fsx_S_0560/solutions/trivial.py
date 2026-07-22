# TIER: trivial
# Do-nothing / majority readout: paint EVERY cell the central stripe (label 1).
# Encoded as a relative-feature network whose cut points bracket the whole [0,1]
# range, so every cell falls in the middle band.  This reproduces the evaluator's
# majority anchor -> normalises to ~0.1.
import sys, json

json.load(sys.stdin)  # ignore the field entirely
print(json.dumps({"feature": "relative", "smooth": 0, "cuts": [-1.0, 2.0]}))
