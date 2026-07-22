# TIER: invalid
# Emit a malformed order: wrong length (one entry short) AND, where present, a
# negative quantity and a quantity exceeding maxOrder. The evaluator rejects any
# length mismatch / out-of-range / negative entry, so this scores 0.0.
import sys, json

inst = json.load(sys.stdin)
P = inst["P"]
bad = [inst["maxOrder"][j] + 1000 if j % 2 == 0 else -5 for j in range(P)]
bad = bad[:-1]  # drop the last entry -> wrong length
print(json.dumps({"order": bad}))
