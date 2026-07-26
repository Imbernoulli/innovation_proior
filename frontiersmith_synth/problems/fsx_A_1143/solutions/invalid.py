# TIER: invalid
# Malformed on purpose: wrong-length, non-numeric bids list. Must score 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"bids": ["a_lot_of_money", None, -5]}))
