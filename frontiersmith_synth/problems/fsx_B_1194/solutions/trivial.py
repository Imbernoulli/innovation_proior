# TIER: trivial
# Do-nothing baseline: predict a constant amplitude of 0 ("the wake never
# sheds"), which reproduces the checker's own baseline exactly -> Ratio ~0.1.
# Note: fitting the training `amplitude` column directly (with anything --
# linear, sigmoid, whatever) lands here too, since that column is noise
# around zero end to end -- there is no post-onset signal in it.
import sys

sys.stdin.read()
print("0")
