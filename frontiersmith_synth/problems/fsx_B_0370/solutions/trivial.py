# TIER: trivial
# Uniform split: divide the stockpile as evenly as possible across every node
# (depot + all districts), parking the remainder at the depot. Ignores the
# fact that districts differ wildly in demand -> this is exactly the baseline.
import sys, json
inst = json.load(sys.stdin)
N = inst["N"]; B = inst["B"]
base = B // (N + 1)
x = [base] * (N + 1)
x[0] += B - base * (N + 1)
print(json.dumps({"stock": x}))
