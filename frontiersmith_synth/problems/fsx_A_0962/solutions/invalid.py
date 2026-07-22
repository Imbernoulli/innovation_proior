# TIER: invalid
# Deliberately broken: emits a negative price and the wrong length, must score 0.
import sys, json

inst = json.load(sys.stdin)
m = inst["m"]
prices = [-5.0] * (m - 1)  # wrong length AND negative
print(json.dumps({"prices": prices}))
