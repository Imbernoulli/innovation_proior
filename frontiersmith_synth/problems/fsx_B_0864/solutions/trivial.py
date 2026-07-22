# TIER: trivial
# "Don't think about it": every case gets the same fixed share of the
# budget (n_cases evenly), spent entirely on the SAME arbitrary default
# solver (index 1), attempted in the order the cases were given.
import sys, json

inst = json.load(sys.stdin)
C = inst["n_cases"]
T = inst["budget"]
share = T / C

attempts = [[ci, 1, share] for ci in range(C)]
print(json.dumps({"attempts": attempts}))
