# TIER: trivial
# Never move, never attack. Reproduces the evaluator's own baseline()
# exactly, so this always scores ratio == 0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
orders = [{"unit_id": u["id"], "move_to": None, "attack": None} for u in inst["friendly"]]
print(json.dumps({"orders": orders, "state": None}))
