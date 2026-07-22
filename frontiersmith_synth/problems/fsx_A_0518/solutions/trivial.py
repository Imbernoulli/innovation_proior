# TIER: trivial
# Do-nothing baseline: predict a constant heater value (the checker's own
# constant-0.5 baseline). Ignores the drive entirely -> reproduces ~0.1.
import sys

sys.stdin.read()
print("OUT 0.5")
