# TIER: greedy
# The obvious recipe: trust the most recent probe reading. Look at each
# solver's quality at its LAST (largest) probe checkpoint, pick whichever is
# currently highest, and pour the ENTIRE remaining budget into it. No regard
# for trend/shape -- this is exactly what falls for a fast-saturating decoy
# that looks best throughout the whole (short) probe window while a slower,
# still-accelerating solver would have paid off far more given the budget.
import sys, json

inst = json.load(sys.stdin)
heuristics = inst["heuristics"]
budget = inst["budget"]

best_id = None
best_val = None
for h in heuristics:
    v = h["probe"][-1]
    if best_val is None or v > best_val:
        best_val = v
        best_id = h["id"]

alloc = {}
if best_id is not None:
    alloc[best_id] = budget

print(json.dumps({"alloc": alloc}))
