# TIER: invalid
# Inoculate every cell with a species index one past the end of the catalogue.
# Every instance has a finite species list, so this index is always out of range
# -> the evaluator rejects the layout as infeasible -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
L = inst["L"]
S = len(inst["species"])

print(json.dumps({"assign": [S] * L}))
