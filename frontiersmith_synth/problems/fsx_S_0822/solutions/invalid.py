# TIER: invalid
# Every decoded type is a huge out-of-range sentinel, so the evaluator rejects
# the answer outright (a decode entry must lie in [0,K)) -> 0.0 on every
# instance, regardless of the target or K.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]

G = 1
Win = [[0] * T]
W = [[0]]
bias = [1]
decode = [99999, 99999]

print(json.dumps({"G": G, "Win": Win, "W": W, "bias": bias, "decode": decode}))
