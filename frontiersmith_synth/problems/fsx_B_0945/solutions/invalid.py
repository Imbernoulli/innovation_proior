# TIER: invalid
# Off-by-one: treats paddock indices as 1..P instead of 0..P-1, so it always
# visits paddock index P -- one past the valid range 0..P-1 -- for the whole
# season. The evaluator rejects any answer containing an out-of-range index,
# so this scores 0.0 everywhere.
import sys, json

inst = json.load(sys.stdin)
R, C, T = inst["R"], inst["C"], inst["T"]
P = R * C  # bug: should stop at P-1, this is out of range

print(json.dumps({"visits": [P] * T}))
