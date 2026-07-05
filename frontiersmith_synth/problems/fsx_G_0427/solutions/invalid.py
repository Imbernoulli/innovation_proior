# TIER: invalid
# Emits the all-zero exponent vector: Pi = 1 is trivially "constant" but is not a
# real group (||a|| = 0).  The checker rejects it -> Ratio 0.0.
import sys
sys.stdin.read()
print("0 0 0 0 0 0")
