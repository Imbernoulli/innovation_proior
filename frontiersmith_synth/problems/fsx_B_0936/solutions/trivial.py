# TIER: trivial
# Split the whole budget evenly across every station (remainder to the
# lowest-indexed stations), clipped to dock capacity. Ignores the trip
# schedule entirely.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
cap = inst["capacity"]
budget = inst["budget"]

base = budget // n
rem = budget - base * n
alloc = [base] * n
for i in range(rem):
    alloc[i] += 1
alloc = [min(a, c) for a, c in zip(alloc, cap)]

print(json.dumps({"init": alloc}))
