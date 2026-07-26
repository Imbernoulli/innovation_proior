# TIER: greedy
# The obvious first-instinct recipe: repeatedly scan the term left to right; at the
# first position where SOME merge rule's (a,b) pattern matches, apply the FIRST such
# rule in the ruleset's given order (classic "first match wins", never look ahead);
# repeat full passes until nothing changes. NEVER consider split rules, since a split
# locally *increases* the term -- a sensible-looking pruning rule for someone minimizing
# size. This fully solves monotone instances (only one unambiguous rule ever matches any
# pattern there) but is trapped twice:
#   - canyon: the locked pockets' LOCK symbol matches no merge rule at all, only a split
#     rule this recipe never uses -> permanently stuck at "color LOCK color" per pocket.
#   - braided: whenever a pair matches two rules at once, "first in the list" always
#     picks the dead-end family-0 rule (deliberately listed before the bridge rule) ->
#     each chain stalls two symbols short of the bridge's extra collapse.
import sys, json


def full_scan_collapse(state, merges, budget, moves):
    changed = True
    while changed and len(moves) < budget:
        changed = False
        i = 0
        while i <= len(state) - 2:
            matched = None
            for ridx, r in enumerate(merges):
                if state[i] == r["a"] and state[i + 1] == r["b"]:
                    matched = ridx
                    break
            if matched is not None:
                moves.append({"op": "merge", "pos": i, "rule": matched})
                state = state[:i] + [merges[matched]["c"]] + state[i + 2:]
                changed = True
                if len(moves) >= budget:
                    break
            else:
                i += 1
    return state


def solve(inst):
    state = list(inst["term"])
    moves = []
    full_scan_collapse(state, inst["merges"], inst["budget"], moves)
    return moves


inst = json.load(sys.stdin)
print(json.dumps({"moves": solve(inst)}))
