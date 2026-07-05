# TIER: trivial
# Keep EVERY column -- hand the frozen scorer the whole wide fraud table.  This
# exactly reproduces the evaluator's weak all-columns baseline (a_cand == a_all),
# so it normalizes to ~0.1 on every dataset.  No feature selection at all.
import sys, json

inst = json.load(sys.stdin)
F = inst["n_features"]
print(json.dumps({"features": list(range(F))}))
