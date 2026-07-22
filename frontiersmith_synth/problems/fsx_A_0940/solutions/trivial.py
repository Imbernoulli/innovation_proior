# TIER: trivial
# Do-nothing baseline: run zero experiments, predict the energy-conservation-only weighted
# average (T_lin) for every test mix.  Reproduces the evaluator's zero-experiment reference
# exactly -> quality 0 -> normalized score == OFFSET (0.10).
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")

if phase == "query":
    print(json.dumps({"experiments": []}))
else:
    mat = inst["materials"]
    TA, TB0, TC = mat["TA"], mat["TB0"], mat["TC"]
    preds = []
    for m in inst["test_mixes"]:
        mA, mB, mC = m["mA"], m["mB"], m["mC"]
        M = mA + mB + mC
        if M <= 0:
            preds.append(0.0)
        else:
            preds.append((mA * TA + mB * TB0 + mC * TC) / M)
    print(json.dumps({"predictions": preds}))
