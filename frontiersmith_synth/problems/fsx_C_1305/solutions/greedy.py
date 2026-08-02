# TIER: greedy
"""The obvious first approach: quote a tight, fixed, SYMMETRIC spread to
maximize fill rate. Ignore inventory, ignore order flow entirely. This wins
on pure-noise sessions but gets picked off whenever informed flow is present,
because the quote never moves out of the way of a predictable price move."""
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"half_spread": 0.06, "inv_coef": 0.0, "ofi_coef": 0.0}))
