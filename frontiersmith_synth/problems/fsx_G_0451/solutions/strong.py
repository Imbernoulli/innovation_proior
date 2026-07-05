# TIER: strong
"""Recursive Karatsuba GF(2^k) multiplier. Recurses all the way down (trying several
recursion base-cases and keeping the one with the FEWEST GF(2) multiplications), and
falls back to schoolbook if it would ever be worse -> minimizes the AND-gate count,
the scored objective. This is NOT a proven-optimal scheme: the multiplicative
complexity of GF(2^k) multiplication is open, so headroom remains."""
import sys


class Builder:
    def __init__(self, k):
        self.gates = []
        self.nxt = 2 * k

    def emit(self, op, i, j):
        self.gates.append((op, i, j))
        w = self.nxt
        self.nxt += 1
        return w

    def xor(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return self.emit("XOR", x, y)

    def andg(self, x, y):
        if x is None or y is None:
            return None
        return self.emit("AND", x, y)


def poly_school(bd, A, B):
    res = [None] * (len(A) + len(B) - 1)
    for i, ai in enumerate(A):
        if ai is None:
            continue
        for j, bj in enumerate(B):
            p = bd.andg(ai, bj)
            if p is not None:
                res[i + j] = bd.xor(res[i + j], p)
    return res


def poly_add(bd, A, B):
    n = max(len(A), len(B))
    res = [None] * n
    for i in range(n):
        x = A[i] if i < len(A) else None
        y = B[i] if i < len(B) else None
        res[i] = bd.xor(x, y)
    return res


def karatsuba(bd, A, B, base):
    n = max(len(A), len(B))
    A = A + [None] * (n - len(A))
    B = B + [None] * (n - len(B))
    if n <= base:
        return poly_school(bd, A, B)
    mm = (n + 1) // 2
    A0, A1 = A[:mm], A[mm:]
    B0, B1 = B[:mm], B[mm:]
    z0 = karatsuba(bd, A0, B0, base)
    z2 = karatsuba(bd, A1, B1, base)
    z1 = karatsuba(bd, poly_add(bd, A0, A1), poly_add(bd, B0, B1), base)
    zmid = poly_add(bd, poly_add(bd, z1, z0), z2)
    L = 2 * n - 1
    res = [None] * L
    for i, w in enumerate(z0):
        res[i] = bd.xor(res[i], w)
    for i, w in enumerate(zmid):
        res[i + mm] = bd.xor(res[i + mm], w)
    for i, w in enumerate(z2):
        res[i + 2 * mm] = bd.xor(res[i + 2 * mm], w)
    return res


def reduce_mod(bd, d, k, m):
    d = list(d)
    for s in range(2 * k - 2, k - 1, -1):
        ds = d[s]
        if ds is None:
            continue
        for i in range(k):
            if m[i]:
                d[s - k + i] = bd.xor(d[s - k + i], ds)
        d[s] = None
    return d[:k]


def build(k, m, base):
    bd = Builder(k)
    A = list(range(k))
    B = list(range(k, 2 * k))
    prod = karatsuba(bd, A, B, base)
    outs = reduce_mod(bd, prod, k, m)
    return bd.gates, outs


def count_ands(gates):
    return sum(1 for op, i, j in gates if op == "AND")


def main():
    toks = sys.stdin.read().split()
    k = int(toks[0])
    m = [int(x) for x in toks[1:2 + k]]
    best = None
    best_ands = None
    for base in (1, 2, 3, k):          # base=k -> pure schoolbook fallback
        gates, outs = build(k, m, base)
        a = count_ands(gates)
        if best_ands is None or a < best_ands:
            best_ands = a
            best = (gates, outs)
    gates, outs = best
    lines = [str(len(gates))]
    for op, i, j in gates:
        lines.append("%s %d %d" % (op, i, j))
    lines.append(" ".join(str(w) for w in outs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
