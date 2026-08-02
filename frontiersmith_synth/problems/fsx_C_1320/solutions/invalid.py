# TIER: invalid
# Emits an expression referencing an unknown variable name -> the checker's
# strict whitelist rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("2.0 + bandgap_fudge_factor * x - dEN")
