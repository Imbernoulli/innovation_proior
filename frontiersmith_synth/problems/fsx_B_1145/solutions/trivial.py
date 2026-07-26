# TIER: trivial
# Do nothing: submit an empty move sequence, leaving the term exactly as given. This is
# exactly the evaluator's baseline construction, so it maps to ratio ~0.1 on every
# instance by construction.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"moves": []}))
