# TIER: strong
"""Exploit the low-rank factorizations AND reorder.

Flatten the chain into individual thin factors (LOWRANK stages contribute
their U,V separately instead of being pre-multiplied), keep DENSE stages as
single nodes, and materialize each SUMLR stage once as A+B.C .  Then run
matrix-chain DP over the FLATTENED factor list, so the optimizer can decide
where each low-rank factor is cheapest to absorb.  This is strictly at least
as good as the greedy (pre-multiplying U.V is one option in the DP), and much
better on low-rank-heavy chains.  It ignores sum-distribution, so it is NOT
provably optimal -- headroom remains.
"""
import sys
sys.setrecursionlimit(10000)


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
            stages.append({"t": "DENSE", "din": din, "dout": dout, "D": di})
        elif typ == "LOWRANK":
            r = int(nxt()); ui = take(din, r); vi = take(r, dout)
            stages.append({"t": "LOWRANK", "din": din, "dout": dout, "r": r, "U": ui, "V": vi})
        elif typ == "SUMLR":
            r = int(nxt()); ai = take(din, dout); bi = take(din, r); ci = take(r, dout)
            stages.append({"t": "SUMLR", "din": din, "dout": dout, "r": r, "A": ai, "B": bi, "C": ci})
    return L, dims, stages, inputs


class Circuit:
    def __init__(self, m):
        self.m = m; self.ops = []

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


def chain_dp(p):
    M = len(p) - 1
    INF = float("inf")
    dp = [[0] * M for _ in range(M)]
    split = [[0] * M for _ in range(M)]
    for length in range(2, M + 1):
        for i in range(0, M - length + 1):
            j = i + length - 1
            best = INF; bk = i
            for k in range(i, j):
                c = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if c < best:
                    best = c; bk = k
            dp[i][j] = best; split[i][j] = bk
    return split


def main():
    L, dims, stages, m = read_instance()
    circ = Circuit(m)

    node = []     # flattened factor node ids
    fdim = []     # flattened dims: factor t is fdim[t] x fdim[t+1]
    fdim.append(dims[0])
    for s in stages:
        if s["t"] == "DENSE":
            node.append(s["D"]); fdim.append(s["dout"])
        elif s["t"] == "LOWRANK":
            node.append(s["U"]); fdim.append(s["r"])
            node.append(s["V"]); fdim.append(s["dout"])
        else:  # SUMLR: materialize once
            t = circ.mul(s["B"], s["C"])
            nd = circ.add(s["A"], t)
            node.append(nd); fdim.append(s["dout"])

    split = chain_dp(fdim)

    def build(i, j):
        if i == j:
            return node[i]
        k = split[i][j]
        l = build(i, k); r = build(k + 1, j)
        return circ.mul(l, r)

    build(0, len(node) - 1)
    circ.emit()


if __name__ == "__main__":
    main()
