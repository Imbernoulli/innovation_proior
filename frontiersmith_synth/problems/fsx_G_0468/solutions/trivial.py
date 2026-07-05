# TIER: trivial
# Keep the retrieval order untouched: return the candidates exactly as presented.
# This reproduces the evaluator's weak reference (the as-presented order), so it
# scores ~0.1 on every session.
import sys, json

inst = json.load(sys.stdin)
m = len(inst["items"])
print(json.dumps({"ranking": list(range(m))}))
