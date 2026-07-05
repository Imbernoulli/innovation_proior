# TIER: trivial
# Pre-position each site to exactly its historical mean and hold no vessel reserve.
# This reproduces the evaluator's internal weak baseline, so it scores ~0.1 on every
# instance: it under-provisions relative to the high shortage penalty and meets only
# ~50% service, but it is always within budget.
import sys, json

inst = json.load(sys.stdin)
q = [s["mean"] for s in inst["sites"]]
print(json.dumps({"q": q, "q0": 0.0}))
