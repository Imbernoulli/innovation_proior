# TIER: invalid
# Invalid: returns the wrong number of knot values (and one non-finite entry). The
# evaluator rejects the shape/finiteness -> every hall scores 0.0 -> Ratio 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"activation": [0.0, 1.0, float("nan")]}))
