# TIER: invalid
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

# Deliberately malformed: omega way outside the allowed [0,3] range, AND
# a "block" move whose groups do not partition {0,...,n-1} (missing the
# last index). Either defect alone must sink this to score 0.
ops = [
    {"type": "row", "omega": 999.0},
    {"type": "block", "axis": "row", "omega": 1.5, "groups": [[0, 1, 2]]},
]
print(json.dumps({"ops": ops}))
