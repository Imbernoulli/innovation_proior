# TIER: trivial
# Reproduces the checker's own internal baseline: at every molecule that still
# needs to be made, always take the reaction with the SMALLEST reaction id.
# This is a "do nothing clever" reference construction, not a real strategy.
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
    for out in reactions_by_output:
        reactions_by_output[out].sort()
    return T, purchasable, reactions, reactions_by_output


def main():
    T, purchasable, reactions, reactions_by_output = read_instance()
    lines = []
    counter = [0]

    def new_id():
        counter[0] += 1
        return counter[0]

    def emit(mol):
        opts = reactions_by_output.get(mol)
        if not opts:
            iid = new_id()
            lines.append("BUY %d %d" % (mol, iid))
            return iid
        rid = opts[0]  # smallest id, naive
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
