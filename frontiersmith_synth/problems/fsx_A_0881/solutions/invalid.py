# TIER: invalid
# Emits an expression whose ** exponent is not an integer literal (it's the
# variable T itself) -> the checker's grammar validator rejects it outright
# and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("T**T")
