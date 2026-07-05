# TIER: invalid
# Emit a schedule of the WRONG length (a single step size regardless of N).  The
# evaluator requires exactly n_steps finite entries, so validation rejects this on
# every instance -> 0.0.  (Demonstrates the length/shape check.)
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"lr": [0.01]}))
