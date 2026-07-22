# TIER: greedy
# The obvious de-obfuscation: a SYNTACTIC peephole simplifier + structural
# common-subexpression elimination + dead-code elimination, run to a fixpoint.
# It applies only local, structure-based identities:
#     ~~x -> x,  x&x -> x,  x|x -> x,  x^x -> 0,  x^0 -> x,  x&1 -> x, ...
#     and merges gates that are literally the same (op, operands).
# This strips the shallow identity padding, but it is BLIND to the planted
# dead logic (built from distant, structurally-different nodes so no x&~x rule
# fires) and to redundant recomputations (a subfunction cloned into a different
# shape). Those survive -> greedy plateaus well above the strong solution.
import sys


def parse(text):
    toks = text.split()
    it = iter(toks)
    nx = lambda: next(it)
    n = int(nx()); g = int(nx())
    gates = []
    for _ in range(g):
        op = nx()
        if op in ('AND', 'OR', 'XOR'):
            gates.append((op, int(nx()), int(nx())))
        elif op == 'NOT':
            gates.append((op, int(nx()), -1))
        else:
            gates.append((op, -1, -1))
    assert nx() == 'OUTPUT'
    out = int(nx())
    return n, gates, out


class Simpl:
    """Emits a new circuit, folding constants + structural CSE. Wires are ints
    (node ids) or the markers 'C0'/'C1'."""
    def __init__(self, n):
        self.n = n
        self.gates = []              # new gate list, id = n + index
        self.cse = {}                # (op,a,b) -> id
        self.op = {}                 # id -> op (for double-negation detection)

    def _emit(self, op, a, b):
        key = (op, a, b)
        got = self.cse.get(key)
        if got is not None:
            return got
        idd = self.n + len(self.gates)
        self.gates.append((op, a, b))
        self.cse[key] = idd
        self.op[idd] = op
        return idd

    def mk_not(self, x):
        if x == 'C0':
            return 'C1'
        if x == 'C1':
            return 'C0'
        if self.op.get(x) == 'NOT':
            return self.gates[x - self.n][1]   # ~~y -> y
        return self._emit('NOT', x, -1)

    def mk(self, op, x, y):
        if op == 'AND':
            if x == 'C0' or y == 'C0':
                return 'C0'
            if x == 'C1':
                return y
            if y == 'C1':
                return x
            if x == y:
                return x
            a, b = (x, y) if x <= y else (y, x)
            return self._emit('AND', a, b)
        if op == 'OR':
            if x == 'C1' or y == 'C1':
                return 'C1'
            if x == 'C0':
                return y
            if y == 'C0':
                return x
            if x == y:
                return x
            a, b = (x, y) if x <= y else (y, x)
            return self._emit('OR', a, b)
        # XOR
        if x == 'C0':
            return y
        if y == 'C0':
            return x
        if x == 'C1':
            return self.mk_not(y)
        if y == 'C1':
            return self.mk_not(x)
        if x == y:
            return 'C0'
        a, b = (x, y) if x <= y else (y, x)
        return self._emit('XOR', a, b)


def render(n, gates, out):
    lines = []
    if out in ('C0', 'C1'):
        lines.append("%d %d" % (n, 1))
        lines.append('CONST0' if out == 'C0' else 'CONST1')
        lines.append("OUTPUT %d" % n)
        return "\n".join(lines) + "\n"
    # DCE from out, keep reachable gates, renumber preserving order
    keep = set()
    stack = [out]
    while stack:
        v = stack.pop()
        if v < n or v in keep:
            continue
        keep.add(v)
        op, a, b = gates[v - n]
        stack.append(a)
        if op != 'NOT':
            stack.append(b)
    kept_ids = sorted(i for i in keep)
    remap = {i: i for i in range(n)}
    for new_idx, old in enumerate(kept_ids):
        remap[old] = n + new_idx
    lines.append("%d %d" % (n, len(kept_ids)))
    for old in kept_ids:
        op, a, b = gates[old - n]
        if op == 'NOT':
            lines.append("NOT %d" % remap[a])
        else:
            lines.append("%s %d %d" % (op, remap[a], remap[b]))
    lines.append("OUTPUT %d" % remap[out])
    return "\n".join(lines) + "\n"


def main():
    n, gates, out = parse(sys.stdin.read())
    S = Simpl(n)
    rep = {i: i for i in range(n)}
    for idx, (op, a, b) in enumerate(gates):
        idd = n + idx
        if op == 'CONST0':
            rep[idd] = 'C0'
        elif op == 'CONST1':
            rep[idd] = 'C1'
        elif op == 'NOT':
            rep[idd] = S.mk_not(rep[a])
        else:
            rep[idd] = S.mk(op, rep[a], rep[b])
    sys.stdout.write(render(n, S.gates, rep[out]))


if __name__ == '__main__':
    main()
