# TIER: trivial
# Leave the raw model PoP untouched (identity map). This reproduces the evaluator's
# weak reference (BS_base) and therefore scores ~0.1 by construction.
import sys, json
inst = json.load(sys.stdin)
ts = inst["test_score"]
out = []
for s in ts:
    v = float(s)
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    out.append(v)
print(json.dumps({"prob": out}))
