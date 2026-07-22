# TIER: invalid
# Malformed answer: targets a solver index that is out of range
# (solver id == k, which does not exist). The checker rejects the whole
# answer on the first out-of-range entry -> score 0.
import sys, json

inst = json.load(sys.stdin)
C, k = inst["n_cases"], inst["k"]
attempts = [[ci, k, 1000.0] for ci in range(C)]   # solver id k is invalid (valid ids are 0..k-1)
print(json.dumps({"attempts": attempts}))
