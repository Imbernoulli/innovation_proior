# TIER: invalid
# Emit a malformed schedule: the beta list has the wrong length (K-1 instead of K).
# The evaluator's shape check rejects it, so every reef scores 0.0.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"alpha": [0.1] * K, "beta": [0.1] * (K - 1)}))
