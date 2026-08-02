# TIER: trivial
# Do-nothing baseline: predict a constant survival probability of 0.5 for every
# horizon and every cohort. Ignores the data entirely -> reproduces the
# checker's own constant-0.5 baseline (~0.1).
import sys

sys.stdin.read()
print("0.5")
