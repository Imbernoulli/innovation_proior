# TIER: trivial
"""Never quote anything anyone would trade against: an absurdly wide spread,
no inventory skew, no order-flow skew. Effectively never fills -> ~0 PnL."""
import sys, json

json.load(sys.stdin)
print(json.dumps({"half_spread": 100000.0, "inv_coef": 0.0, "ofi_coef": 0.0}))
