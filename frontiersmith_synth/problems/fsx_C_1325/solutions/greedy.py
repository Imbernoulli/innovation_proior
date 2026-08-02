# TIER: greedy
# The "obvious" retrosynthesis planner: minimize total number of reaction
# steps (a textbook shortest-route search over the disconnection graph), tying
# broken by minimizing the plain additive reagent/step cost -- WITHOUT ever
# dividing by yield. This never notices that a shorter route can need far more
# raw material once you account for how yields compound multiplicatively along
# a chain, so it always prefers linear chains over convergent ones and always
# prefers the unprotected shortcut over the protect/react/deprotect detour.
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it)); P = int(next(it))
    purchasable = {}
    for _ in range(P):
        mol = int(next(it)); cost = int(next(it))
        purchasable[mol] = cost
    reactions = {}
    reactions_by_output = {}
    for _ in range(M):
        rid = int(next(it)); output = int(next(it)); k = int(next(it))
        ins = [int(next(it)) for _ in range(k)]
        ypct = int(next(it)); cost = int(next(it))
        reactions[rid] = (output, ins, ypct, cost)
        reactions_by_output.setdefault(output, []).append(rid)
    return T, purchasable, reactions, reactions_by_output


def main():
    T, purchasable, reactions, reactions_by_output = read_instance()

    best_choice = {}
    memo_steps = {}
    memo_naive_cost = {}

    def solve(mol, stack):
        if mol in memo_steps:
            return memo_steps[mol], memo_naive_cost[mol]
        if mol in stack:
            raise RuntimeError("cycle")
        opts = reactions_by_output.get(mol)
        best = None
        if mol in purchasable:
            best = (0, purchasable[mol], None)
        stack.add(mol)
        if opts:
            for rid in opts:
                _out, ins, _y, cost = reactions[rid]
                steps_sum = 1
                cost_sum = cost
                for m in ins:
                    s, c = solve(m, stack)
                    steps_sum += s
                    cost_sum += c
                cand = (steps_sum, cost_sum, rid)
                if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                    best = cand
        stack.discard(mol)
        if best is None:
            raise RuntimeError("no way to make molecule %d" % mol)
        memo_steps[mol] = best[0]
        memo_naive_cost[mol] = best[1]
        best_choice[mol] = best[2]  # rid or None (buy)
        return best[0], best[1]

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
