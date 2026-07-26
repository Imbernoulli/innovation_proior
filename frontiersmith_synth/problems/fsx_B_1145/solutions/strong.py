# TIER: strong
# INSIGHT: no single fixed rewrite recipe is near-optimal across the mixture -- so first
# spend a few cheap, static checks DIAGNOSING which of the three planted landscapes this
# instance is (landscape-type-classification), THEN dispatch a tailored strategy
# (per-instance-strategy-dispatch):
#   - splits present               -> CANYON: a size-increasing split is worth taking
#     whenever it fuses two runs the merge rules alone could never join, so actively
#     apply every available split (turning a LOCK into its flanking color) before/while
#     collapsing, instead of refusing any move that grows the term.
#   - a pattern (a,b) matches TWO OR MORE distinct merge rules ("family" collisions)
#     -> BRAIDED: at every such fork, don't blindly take the first-listed rule -- cheaply
#     SIMULATE each candidate branch forward (full collapse) and commit to whichever one
#     actually finishes shorter. This is exploratory local search at the fork, not a
#     bigger canned recipe.
#   - neither                      -> MONOTONE: the ruleset is already confluent, so the
#     plain scan-and-collapse recipe (identical to `greedy` here) is already optimal --
#     no extra machinery buys anything on this landscape.
# This is a genuine reformulation (diagnose, then dispatch a matched strategy) rather
# than "greedy plus more iterations": on canyon/braided instances it takes moves a
# strictly-non-increasing recipe would never even consider.
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


def classify(inst):
    if inst["splits"]:
        return "canyon"
    seen = {}
    for r in inst["merges"]:
        key = (r["a"], r["b"])
        if key in seen and seen[key] != r["c"]:
            return "braided"
        seen[key] = r["c"]
    return "monotone"


def solve_monotone(inst):
    moves = []
    full_scan_collapse(list(inst["term"]), inst["merges"], inst["budget"], moves)
    return moves


def solve_canyon(inst):
    state = list(inst["term"])
    merges = inst["merges"]; splits = inst["splits"]; budget = inst["budget"]
    moves = []
    changed = True
    while changed and len(moves) < budget:
        changed = False
        for sidx, sr in enumerate(splits):
            if len(moves) >= budget:
                break
            if sr["a"] not in state:
                continue
            pos = state.index(sr["a"])
            moves.append({"op": "split", "pos": pos, "rule": sidx})
            state = state[:pos] + [sr["b"], sr["c"]] + state[pos + 1:]
            changed = True
        state = full_scan_collapse(state, merges, budget, moves)
    return moves


def solve_braided(inst):
    state = list(inst["term"])
    merges = inst["merges"]; budget = inst["budget"]
    moves = []
    changed = True
    while changed and len(moves) < budget:
        changed = False
        i = 0
        while i <= len(state) - 2:
            cands = [ridx for ridx, r in enumerate(merges)
                     if state[i] == r["a"] and state[i + 1] == r["b"]]
            if cands:
                if len(cands) == 1:
                    chosen = cands[0]
                else:
                    best_ridx, best_len = None, None
                    for ridx in cands:
                        trial = state[:i] + [merges[ridx]["c"]] + state[i + 2:]
                        trial_moves = []
                        remaining_budget = max(0, budget - len(moves) - 1)
                        final = full_scan_collapse(list(trial), merges, remaining_budget, trial_moves)
                        L = len(final)
                        if best_len is None or L < best_len:
                            best_len, best_ridx = L, ridx
                    chosen = best_ridx
                moves.append({"op": "merge", "pos": i, "rule": chosen})
                state = state[:i] + [merges[chosen]["c"]] + state[i + 2:]
                changed = True
                if len(moves) >= budget:
                    break
            else:
                i += 1
    return moves


def solve(inst):
    kind = classify(inst)
    if kind == "canyon":
        return solve_canyon(inst)
    if kind == "braided":
        return solve_braided(inst)
    return solve_monotone(inst)


inst = json.load(sys.stdin)
print(json.dumps({"moves": solve(inst)}))
