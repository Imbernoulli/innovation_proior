# TIER: trivial
# Do the absolute minimum: process jobs in the exact order they arrive in the input,
# no sorting, no family-awareness. The evaluator's own admission control auto-skips
# whatever doesn't fit, so this is a valid (if weak) answer.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": list(range(n))}))
