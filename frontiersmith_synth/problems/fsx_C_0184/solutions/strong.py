# TIER: strong
"""Capacity-aware rule inducer.  Recovers BOTH parts of the hidden rule from the
labelled train split:

  (a) validity requires a well-nested Dyck-K word (RETURN matches most-recent
      same-depot LAUNCH, empty swarm at end), and
  (b) an airspace capacity: a valid mission's maximum simultaneous airborne count
      must not exceed a threshold D.

It estimates the capacity as the largest max-depth seen among VALID (label==1)
well-nested training missions, then predicts valid iff a query is well-nested and
its max depth is within that estimate.  This generalizes to arbitrarily LONGER
(OOD) missions because the rule is length-independent.  Because the true capacity
boundary is never exhibited by the training valids (they stop one below it), the
estimate is slightly conservative and a few boundary-capacity queries are missed
-> strong but not perfect, and clearly above the well-nesting-only classifier."""
import sys
import json


def parse(seq, K):
    """Return (well_nested, max_depth)."""
    stack = []
    mx = 0
    for t in seq:
        if not isinstance(t, int) or t < 0 or t >= 2 * K:
            return False, 0
        if t % 2 == 0:
            stack.append(t // 2)
            if len(stack) > mx:
                mx = len(stack)
        else:
            typ = (t - 1) // 2
            if not stack or stack[-1] != typ:
                return False, 0
            stack.pop()
    return len(stack) == 0, mx


def main():
    inst = json.load(sys.stdin)
    K = inst["num_types"]

    cap = 0
    for r in inst["train"]:
        if r["label"] == 1:
            ok, d = parse(r["seq"], K)
            if ok and d > cap:
                cap = d

    preds = []
    for q in inst["queries"]:
        ok, d = parse(q, K)
        preds.append(1 if (ok and d <= cap) else 0)
    print(json.dumps(preds))


main()
