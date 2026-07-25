# TIER: trivial
# Do-nothing baseline: predict post = pre (nothing happens). Reproduces the
# checker's own identity baseline -> ~0.1.
import sys

sys.stdin.read()
print("V1 v1")
print("V2 v2")
