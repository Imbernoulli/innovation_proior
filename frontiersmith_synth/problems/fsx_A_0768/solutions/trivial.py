# TIER: trivial
# Pure literal pass, no dictionary at all: the whole sequence as ONE literal run. This
# is exactly the evaluator's baseline construction, so every instance scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

answer = {"dictionary": [], "segments": [{"type": "lit", "len": n}]}
print(json.dumps(answer))
