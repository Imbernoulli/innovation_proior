# TIER: trivial
# Blind fixed-increment titration: add the SAME volume every round
# (V_max / max_rounds), completely ignoring the pH reading, and stop the
# first time the reading has reached or passed the target. This is the
# evaluator's own reference construction (a trivial candidate normalizes
# to ~0.10). No feedback, no adaptation -> wastes rounds crawling gentle
# regions and can overshoot badly through a steep one.
import sys, json

inst = json.load(sys.stdin)
V_max = float(inst["V_max"])
max_rounds = int(inst["max_rounds"])
pH = float(inst["pH"])
target = float(inst["target_pH"])
V = float(inst["V"])

max_add = float(inst.get("max_add", V_max))

if pH >= target:
    add = 0.0
else:
    step = V_max / max_rounds
    remaining = V_max - V
    add = min(step, remaining, max_add)
    if add <= 0.0:
        add = 0.0

print(json.dumps({"add": add}))
