# TIER: invalid
# Emits a wrong-shape answer (flat lists instead of one list per query manifest,
# and a berth code out of range) -> the evaluator rejects it -> 0.
import sys, json

inst = json.load(sys.stdin)
q = inst["queries"]
ans = {"predictions": {
    "id": [9999 for _ in q["id"]],
    "ood": [9999 for _ in q["ood"]],
}}
print(json.dumps(ans))
