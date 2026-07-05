# TIER: trivial
"""Schoolbook GF(2^k) multiplier: all k*k partial products a_i & b_j, XOR-accumulate
into the length-(2k-1) product polynomial, then reduce mod f. Uses k*k AND gates =
the checker's baseline -> Ratio ~= 0.1."""
import sys


def build(k, m):
    gates = []
    nxt = [2 * k]

    def emit(op, i, j):
        gates.append((op, i, j))
        w = nxt[0]
        nxt[0] += 1
        return w

    def xor(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return emit("XOR", x, y)

    A = list(range(k))
    B = list(range(k, 2 * k))
    # schoolbook polynomial product
    d = [None] * (2 * k - 1)
    for i in range(k):
        for j in range(k):
            p = emit("AND", A[i], B[j])
            d[i + j] = xor(d[i + j], p)
    # reduction mod f
    for s in range(2 * k - 2, k - 1, -1):
        ds = d[s]
        if ds is None:
            continue
        for i in range(k):
            if m[i]:
                d[s - k + i] = xor(d[s - k + i], ds)
        d[s] = None
    outs = [d[i] for i in range(k)]
    return gates, outs


def main():
    toks = sys.stdin.read().split()
    k = int(toks[0])
    m = [int(x) for x in toks[1:2 + k]]
    gates, outs = build(k, m)
    lines = [str(len(gates))]
    for op, i, j in gates:
        lines.append("%s %d %d" % (op, i, j))
    lines.append(" ".join(str(w) for w in outs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
