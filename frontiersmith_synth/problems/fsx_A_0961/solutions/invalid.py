# TIER: invalid
# Sets every sensor's threshold to 0 (flags almost every event on every
# sensor) with full weight and a vote threshold of 1. This blows the 1%
# false-positive SLA on every instance (nearly all benign events get
# flagged), so the evaluator rejects it and scores 0.0.
import sys, json

inst = json.load(sys.stdin)
K = inst["channels"]

theta = [0.0] * K
w = [1.0] * K
tau = 1.0

print(json.dumps({"theta": theta, "w": w, "tau": tau}))
