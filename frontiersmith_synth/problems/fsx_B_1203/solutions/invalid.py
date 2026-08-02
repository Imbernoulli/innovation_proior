# TIER: invalid
# References a name outside {T, S, allowed functions}; the checker's
# strict expression validator rejects unknown identifiers -> Ratio: 0.0.
import sys

sys.stdin.read()
print("T + S + KAPPA_TIMES_TWO")
