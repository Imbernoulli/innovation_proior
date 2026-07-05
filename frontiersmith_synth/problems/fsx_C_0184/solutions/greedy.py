# TIER: greedy
"""Well-nesting-only classifier.  Discovers that a valid mission must be a Dyck-K
word (RETURNs match the most recent same-depot LAUNCH, swarm empty at the end),
but IGNORES the hidden airspace-capacity constraint.  So it correctly rejects
structurally-broken missions and accepts genuinely valid ones, but wrongly
accepts 'too many drones aloft' missions (well-nested but over capacity),
capping its accuracy well below the capacity-aware solver."""
import sys
import json


def well_nested(seq, K):
    stack = []
    for t in seq:
        if not isinstance(t, int) or t < 0 or t >= 2 * K:
            return False
        if t % 2 == 0:
            stack.append(t // 2)
        else:
            typ = (t - 1) // 2
            if not stack or stack[-1] != typ:
                return False
            stack.pop()
    return len(stack) == 0


def main():
    inst = json.load(sys.stdin)
    K = inst["num_types"]
    preds = [1 if well_nested(q, K) else 0 for q in inst["queries"]]
    print(json.dumps(preds))


main()
