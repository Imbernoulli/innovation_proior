# TIER: invalid
# Emit a malformed table: wrong length AND a non-finite value.  The evaluator's
# strict validator rejects it (length != M, and NaN fails the finiteness check),
# so every battery scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"y": [float("nan"), 0.0, 1.0]}))
