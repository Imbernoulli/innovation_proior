# TIER: greedy
# Count-balance heuristic: call a cable well-spliced iff the number of opening
# tokens equals the number of closing tokens. This catches unbalanced (deleted-token)
# cables, but is blind to type mismatches and to the depth budget -- so it misclassifies
# every type-swapped and every too-deep cable as well-spliced. Beats the majority
# constant but leaves most of the structure on the table.
import sys, json

inst = json.load(sys.stdin)
opens = set(inst["open_symbols"])
closes = set(inst["close_symbols"])
q = inst["queries"]

labels = []
for s in q:
    no = sum(1 for ch in s if ch in opens)
    nc = sum(1 for ch in s if ch in closes)
    labels.append(1 if no == nc else 0)

print(json.dumps({"labels": labels}))
