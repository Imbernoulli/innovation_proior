# TIER: invalid
# Returns a score list of the WRONG length (and the wrong key shape): the evaluator's
# strict validation rejects it -> 0 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"scores": [0.0] * (n // 2)}))
