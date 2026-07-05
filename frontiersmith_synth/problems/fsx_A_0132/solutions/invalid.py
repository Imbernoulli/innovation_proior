# TIER: invalid
# Emits a malformed answer: "weights" is a string, not a list of 4 finite numbers.
# The evaluator's strict validation rejects it, so every instance scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"weights": "best-fit-please"}))
