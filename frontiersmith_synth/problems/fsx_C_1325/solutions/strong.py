# TIER: strong
# The insight: the objective is not "fewest steps", it is the DELIVERED cost of
# 1 unit of target, and yields multiply along whatever path connects a raw
# material to the target. Define per_unit(m) = exact rational cost to deliver
# ONE unit of molecule m. For a reaction with yield y and cost c consuming
# inputs i1..ik, delivering `a` units of its output costs
#   c*(a/y) + sum_j per_unit(i_j) * (a/y)
# i.e. per_unit(output) = (c + sum_j per_unit(i_j)) / y.  This recursion is
# exact and needs no search heuristic: solved bottom-up (the instance is
# acyclic by construction) it AUTOMATICALLY prefers convergent branching
# (shallow branches keep the 1/y multiplier small on each branch's own cost)
# over a long linear chain (every step's 1/y multiplies through everything
# upstream of it), and it automatically prefers the protect/react/deprotect
# detour whenever three high-yield steps beat one low-yield shortcut -- even
# though that detour uses MORE reaction steps.
import sys
from fractions import Fraction as Fr


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it)); P = int(next(it))
    purchasable = {}
    for _ in range(P):
        mol = int(next(it)); cost = int(next(it))
        purchasable[mol] = Fr(cost)
    reactions = {}
    reactions_by_output = {}
    for _ in range(M):
        rid = int(next(it)); output = int(next(it)); k = int(next(it))
        ins = [int(next(it)) for _ in range(k)]
        ypct = int(next(it)); cost = int(next(it))
        reactions[rid] = (output, ins, Fr(ypct, 100), Fr(cost))
        reactions_by_output.setdefault(output, []).append(rid)
    return T, purchasable, reactions, reactions_by_output


def main():
    T, purchasable, reactions, reactions_by_output = read_instance()

    best_choice = {}
    memo_cost = {}

    def solve(mol, stack):
        if mol in memo_cost:
            return memo_cost[mol]
        if mol in stack:
            raise RuntimeError("cycle")
        opts = reactions_by_output.get(mol)
        best_cost = None
        best_rid = None
        if mol in purchasable:
            best_cost = purchasable[mol]
        stack.add(mol)
        if opts:
            for rid in opts:
                _out, ins, yfrac, cost = reactions[rid]
                acc = cost
                for m in ins:
                    acc += solve(m, stack)
                cand = acc / yfrac
                if best_cost is None or cand < best_cost:
                    best_cost = cand
                    best_rid = rid
        stack.discard(mol)
        if best_cost is None:
            raise RuntimeError("no way to make molecule %d" % mol)
        memo_cost[mol] = best_cost
        best_choice[mol] = best_rid  # None means buy
        return best_cost

    solve(T, set())

    lines = []
    counter = [0]

    def new_id():
        counter[0] += 1
        return counter[0]

    def emit(mol):
        rid = best_choice.get(mol)
        if rid is None:
            iid = new_id()
            lines.append("BUY %d %d" % (mol, iid))
            return iid
        _out, ins, _y, _c = reactions[rid]
        input_iids = [emit(m) for m in ins]
        iid = new_id()
        lines.append("REACT %d %d %s" % (rid, iid, " ".join(map(str, input_iids))))
        return iid

    root = emit(T)
    lines.append("ROOT %d" % root)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
