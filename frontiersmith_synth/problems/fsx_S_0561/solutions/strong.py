# TIER: strong
# INSIGHT: stop reading the circuit as text; read it as a FUNCTION.
# Compute the truth table (functional signature) of every wire over all 2^n
# inputs, then canonicalise by VALUE:
#   * any wire whose signature is constant 0/1 is folded away -- this catches the
#     planted dead logic that is globally 0 yet built from distant, structurally
#     different nodes (no local x&~x rule can see it);
#   * any two wires with the SAME signature are merged -- this catches the
#     redundant recomputations (a subfunction cloned into a different shape).
# Then dead-code-eliminate. The De Morgan / XOR structural blow-ups have genuinely
# distinct signatures, so they survive: strong does NOT reach the true optimum,
# it just removes everything a functional view proves redundant.
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
    return n, gates, int(nx())


def input_tts(n):
    N = 1 << n
    M = (1 << N) - 1
    tts = []
    for i in range(n):
        half = 1 << i
        col = ((1 << half) - 1) << half
        w = 2 * half
        while w < N:
            col |= col << w
            w <<= 1
        tts.append(col & M)
    return tts, M


class Strong:
    def __init__(self, n):
        self.n = n
        tts, self.M = input_tts(n)
        self.gates = []
        self.tt = {}              # id -> truth table
        self.byval = {}           # truth table -> id
        for i in range(n):
            self.tt[i] = tts[i]

    def _canon(self, op, a, b, t):
        if t == 0:
            return 'C0'
        if t == self.M:
            return 'C1'
        got = self.byval.get(t)
        if got is not None:
            return got            # merge distant equal-signature wire
        idd = self.n + len(self.gates)
        self.gates.append((op, a, b))
        self.tt[idd] = t
        self.byval[t] = idd
        return idd

    def mk_not(self, x):
        if x == 'C0':
            return 'C1'
        if x == 'C1':
            return 'C0'
        return self._canon('NOT', x, -1, self.tt[x] ^ self.M)

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
            return self._canon('AND', a, b, self.tt[x] & self.tt[y])
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
            return self._canon('OR', a, b, self.tt[x] | self.tt[y])
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
        return self._canon('XOR', a, b, self.tt[x] ^ self.tt[y])


def render(n, gates, out):
    lines = []
    if out in ('C0', 'C1'):
        lines.append("%d %d" % (n, 1))
        lines.append('CONST0' if out == 'C0' else 'CONST1')
        lines.append("OUTPUT %d" % n)
        return "\n".join(lines) + "\n"
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
    kept_ids = sorted(keep)
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
    S = Strong(n)
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
