# TIER: trivial
# Keep the predicates in the order they were given (identity layout). This is
# exactly the evaluator's baseline: sprinkle predicates keep their low indices and
# sort to the front of every check, so shared cores rarely form a common prefix.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
print(json.dumps({"order": list(range(M))}))
