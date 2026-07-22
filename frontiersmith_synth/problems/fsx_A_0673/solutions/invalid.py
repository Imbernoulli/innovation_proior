# TIER: invalid
# Deliberately broken answer: pin list far exceeds the budget (and contains
# out-of-range/duplicate ids), and the weights are the wrong types / out of range /
# non-finite. The evaluator must reject this and score every instance 0.0.
import sys, json

inst = json.load(sys.stdin)
M = inst.get("universe_size", 100)

bad_pin = list(range(M + 50))  # way over budget, and includes out-of-range ids
answer = {
    "pin": bad_pin,
    "w_lru": -5.0,          # negative, out of [0,8]
    "w_mru": None,          # wrong type
    "w_lfu": "not-a-number",  # wrong type
    "w_scan": float("nan"),  # non-finite
}
print(json.dumps(answer))
