# TIER: trivial
# The identity / catalog canonical order [0, 1, ..., C-1]. This is exactly the
# evaluator's baseline order (no reasoning: emit clauses in their catalog index
# order), so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["n_clauses"]
print(json.dumps({"order": list(range(C))}))
