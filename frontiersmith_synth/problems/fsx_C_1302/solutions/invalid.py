# TIER: invalid
# Malformed output: wrong length (missing the last round's bid) and a
# negative bid -- must be rejected by the feasibility check and score 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
bids = [1.0] * (T - 1) + [-5.0]
print(json.dumps({"bids": bids[:-1]}))  # also drops one entry -> wrong length
