# TIER: invalid
"""Emits a well-formed but WRONG circuit: it returns (twice) the dense matrix
of stage 0, which is not the composite T.  Must score 0 (equivalence gate).
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
        idx = inputs; inputs += 1; return idx

    for i in range(L):
        din, dout = dims[i], dims[i + 1]
        typ = nxt()
        if typ == "DENSE":
            di = take(din, dout)
            stages.append({"t": "DENSE", "D": di})
        elif typ == "LOWRANK":
            r = int(nxt()); ui = take(din, r); vi = take(r, dout)
            stages.append({"t": "LOWRANK", "U": ui, "V": vi})
        elif typ == "SUMLR":
            r = int(nxt()); ai = take(din, dout); bi = take(din, r); ci = take(r, dout)
            stages.append({"t": "SUMLR", "A": ai, "B": bi, "C": ci})
    return L, dims, stages, inputs


def main():
    L, dims, stages, m = read_instance()
    ops = []

    def new():
        return m + len(ops) - 1

    s0 = stages[0]
    if s0["t"] == "DENSE":
        node = s0["D"]
    elif s0["t"] == "LOWRANK":
        ops.append(("MUL", s0["U"], s0["V"])); node = new()
    else:
        ops.append(("MUL", s0["B"], s0["C"])); t = new()
        ops.append(("ADD", s0["A"], t)); node = new()
    ops.append(("ADD", node, node))   # wrong result, guarantees >=1 op
    out = [str(len(ops))]
    for op, a, b in ops:
        out.append("%s %d %d" % (op, a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
