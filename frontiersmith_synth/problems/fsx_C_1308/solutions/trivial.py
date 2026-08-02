# TIER: trivial
# Do nothing: send every item straight to auto-moderation (empty schedule).
# This reproduces the evaluator's own "cost_auto" anchor exactly, so it scores
# ~0.1 on every instance -- the do-nothing floor.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"schedule": {}}))
