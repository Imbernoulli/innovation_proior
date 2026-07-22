# TIER: strong
# Insight: a bud's fate is decided at its OWN founding instant (the founding
# check), not by any blanket, contiguous suppression. So instead of a single
# switch time, spend the growth budget as a SCATTERED schedule: suppress
# auxin release for exactly one tick each -- precisely the founding instant
# of every bud that must stay arrested -- and leave every other tick
# released. Every arrested bud is caught at formation (cheap: 1 budget unit
# each, regardless of where in the timeline it sits), while every target
# bud's own founding tick is left released, so no target is collaterally
# sacrificed. This is exactly the temporal-schedule control the problem
# asks for: WHEN (not how strongly, in aggregate) auxin dips low is what
# sculpts the branching set.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); BUDGET = int(next(it)); K = int(next(it))
    c = [int(next(it)) for _ in range(H)]
    T = [int(next(it)) for _ in range(K)]
    Tset = set(T)

    r = [1] * H
    budget_remaining = BUDGET
    for i in range(1, H + 1):
        if i not in Tset and budget_remaining > 0:
            r[i - 1] = 0
            budget_remaining -= 1

    sys.stdout.write(" ".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
