# TIER: trivial
# Do-nothing baseline: never probe, predict a flat zero everywhere.  Reproduces the
# evaluator's predict-zero reference -> quality 0 -> normalized score ~= OFFSET (0.10).
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")
if phase == "query":
    print(json.dumps({"queries": []}))
else:
    G = int(inst["G"])
    print(json.dumps({"pred": [0.0] * G}))
