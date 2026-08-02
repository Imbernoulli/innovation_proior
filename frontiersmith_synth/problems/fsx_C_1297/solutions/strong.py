# TIER: strong
# The insight: don't look at the grids at all -- decode the color codes the
# instance itself hands you.  Two modes only: mode 0 (before the relay
# beacon) interprets a floor color with phase0_code, mode 1 (after it) with
# phase1_code.  Stepping onto a beacon tile "K_<color>" always means: apply
# the POST-relay code to that color and switch to mode 1.  This 12-rule
# reactive table never looks at coordinates, never depends on which grid it
# is standing in, and is therefore exactly as good on grids it has never
# seen as on the 3 visible ones -- unlike a hardcoded path, it reacts to
# what it senses instead of memorizing a trajectory.
import sys, json

inst = json.load(sys.stdin)
phase0 = inst["phase0_code"]
phase1 = inst["phase1_code"]

rules = []
for color, act in phase0.items():
    rules.append({"state": 0, "see": color, "action": act, "next": 0})
for color, act in phase1.items():
    rules.append({"state": 0, "see": "K_" + color, "action": act, "next": 1})
for color, act in phase1.items():
    rules.append({"state": 1, "see": color, "action": act, "next": 1})

answer = {"start_state": 0, "rules": rules}
print(json.dumps(answer))
