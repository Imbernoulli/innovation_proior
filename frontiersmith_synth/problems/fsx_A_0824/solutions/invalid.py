# TIER: invalid
# Always "identifies" the target as item 0 without asking a single probe --
# wrong for every other item, so every instance's per-item correctness
# check fails and the whole instance scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"tree": {"guess": 0}}))
