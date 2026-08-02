# TIER: trivial
# Do-nothing baseline: predict the constant 0.0 ground index (the checker's
# own baseline), ignoring the data entirely -> reproduces ~0.1.
import sys

sys.stdin.read()
print("OUT 0.0")
