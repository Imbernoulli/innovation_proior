# TIER: invalid
# Syntactically legal expression (passes the AST whitelist) that is
# guaranteed to blow up at evaluation time: dividing by the always-zero
# quantity (t-t). The checker must reject this as non-finite -> Ratio 0.
print("1 / ( t - t )")
