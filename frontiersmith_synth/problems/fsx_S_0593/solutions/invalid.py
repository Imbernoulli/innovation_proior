# TIER: invalid
# Emit a malformed quote: negative half-spreads and a wrong-length ask vector.  The
# evaluator rejects any negative / non-finite / mis-shaped entry, so this scores 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"hb": [-1.0] * T, "ha": [0.0] * (T - 1),
                  "zb": [0.0] * T, "za": [0.0] * T}))
