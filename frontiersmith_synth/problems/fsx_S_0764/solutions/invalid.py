# TIER: invalid
# Emit a malformed schedule: a negative battery-charge entry and a wrong-length
# fuel-discharge vector. The evaluator rejects any negative / non-finite / mis-shaped
# entry, so this scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({
    "bc": [-1.0] * T,
    "bd": [0.0] * T,
    "fc": [0.0] * T,
    "fd": [0.0] * (T - 1),
}))
