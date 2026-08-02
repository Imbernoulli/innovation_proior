# TIER: invalid
# Submits a rule whose "next" state points outside the declared state
# budget (state_budget, e.g. 40) -- the evaluator validates every rule's
# state/next against the budget before simulating anything, so this whole
# controller is rejected and the instance's Ratio is forced to 0.0.
import sys, json

inst = json.load(sys.stdin)
budget = inst["state_budget"]

answer = {
    "start_state": 0,
    "rules": [{"state": 0, "see": "R", "action": "N", "next": budget}],
}
print(json.dumps(answer))
