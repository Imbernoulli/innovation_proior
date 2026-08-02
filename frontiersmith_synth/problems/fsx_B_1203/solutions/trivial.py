# TIER: trivial
# Do-nothing baseline: reproduce the checker's own internal reference
# predictor exactly -- "assume it's an ordinary tide day, ignore the
# surge entirely". Never looks at S or kappa at all.
import sys

sys.stdin.read()
print("T")
