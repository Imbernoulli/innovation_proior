# TIER: invalid
# Emits garbage: declares rules whose tokens are not in the alphabet and whose
# count does not match -> infeasible, must score 0.
import sys

data = sys.stdin.read().split()
print(3)
print("zzzz")
print("#9#9")
print("a")
