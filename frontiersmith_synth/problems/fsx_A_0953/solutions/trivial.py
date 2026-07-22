# TIER: trivial
# Do-nothing baseline: spend zero probes, predict the fixed-shape / constant-loudness naive
# reading using only the midpoints of the public hint windows.  Reproduces the evaluator's
# zero-probe reference exactly -> quality 0 -> normalized score == OFFSET (0.10).
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")

if phase == "query":
    print(json.dumps({"probes": []}))
else:
    hints = inst["hints"]
    w_g = 0.5 * (hints["w_lo"] + hints["w_hi"])
    f0_g = 0.5 * (hints["f0_0_lo"] + hints["f0_0_hi"])
    A_g = 0.5 * (hints["A_mid_lo"] + hints["A_mid_hi"])
    if w_g <= 1e-6:
        w_g = 1e-6

    def kernel(u, w):
        z = u / w
        v = 1.0 - z * z
        return v if v > 0.0 else 0.0

    preds = []
    for q in inst["test_queries"]:
        preds.append(A_g * kernel(q["x"] - f0_g, w_g))
    print(json.dumps({"predictions": preds}))
