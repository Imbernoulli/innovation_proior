# TIER: trivial
"""Naive construction == the checker baseline.

Materialize every stage densely (in the given order) and fold the chain
left-to-right.  Reproduces baseline B exactly  ->  Ratio ~ 0.1 .
"""
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    nxt = lambda: next(it)
    L = int(nxt())
    dims = [int(nxt()) for _ in range(L + 1)]
    inputs = 0
    stages = []

    def take(r, c):
        nonlocal inputs
        for _ in range(r * c):
            nxt()
        idx = inputs
        inputs += 1
        return idx

    for i in range(L):
        din, dout = dims[i], dims[i + 1]
        typ = nxt()
        if typ == "DENSE":
            di = take(din, dout)
            stages.append({"t": "DENSE", "din": din, "dout": dout, "D": di})
        elif typ == "LOWRANK":
            r = int(nxt())
            ui = take(din, r); vi = take(r, dout)
            stages.append({"t": "LOWRANK", "din": din, "dout": dout, "r": r, "U": ui, "V": vi})
        elif typ == "SUMLR":
            r = int(nxt())
            ai = take(din, dout); bi = take(din, r); ci = take(r, dout)
            stages.append({"t": "SUMLR", "din": din, "dout": dout, "r": r,
                           "A": ai, "B": bi, "C": ci})
    return L, dims, stages, inputs


class Circuit:
    def __init__(self, m):
        self.m = m
        self.ops = []

    def _new(self):
        return self.m + len(self.ops) - 1

    def mul(self, a, b):
        self.ops.append(("MUL", a, b)); return self._new()

    def add(self, a, b):
        self.ops.append(("ADD", a, b)); return self._new()

    def emit(self):
        out = [str(len(self.ops))]
        for op, a, b in self.ops:
            out.append("%s %d %d" % (op, a, b))
        sys.stdout.write("\n".join(out) + "\n")


def stage_node(circ, s):
    if s["t"] == "DENSE":
        return s["D"]
    if s["t"] == "LOWRANK":
        return circ.mul(s["U"], s["V"])
    t = circ.mul(s["B"], s["C"])
    return circ.add(s["A"], t)


def main():
    L, dims, stages, m = read_instance()
    circ = Circuit(m)
    nodes = [stage_node(circ, s) for s in stages]
    acc = nodes[0]
    for i in range(1, L):
        acc = circ.mul(acc, nodes[i])
    circ.emit()


if __name__ == "__main__":
    main()
