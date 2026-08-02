# TIER: invalid
# References a name outside {G, H, allowed functions}; the checker's
# strict expression validator rejects unknown identifiers -> Ratio: 0.0.
import sys

sys.stdin.read()
print("G - BETA_TIMES_TWO * H")
