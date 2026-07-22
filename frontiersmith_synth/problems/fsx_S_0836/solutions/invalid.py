# TIER: invalid
# Syntactically legal expression (parses fine under the grammar) but blows up
# at rho=0.5, which the checker's finiteness probe grid always hits exactly
# -> must score 0 via the "reject non-finite" feasibility gate.
import sys

sys.stdin.read()
print("1.0/(rho-0.5)")
