# TIER: trivial
# Do nothing: dispatch no crews.  This reproduces the evaluator's do-nothing baseline
# (the fire burns whatever it reaches within the horizon), so every instance normalizes
# to ~0.1.  It is the floor the ladder is measured against.
import sys, json

json.load(sys.stdin)
print(json.dumps({"breaks": []}))
